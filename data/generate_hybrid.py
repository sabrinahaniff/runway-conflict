import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import asdict
from .runways import get_runway
from .transform import flight_phase_from_position
from simulator.entities import Aircraft, AircraftPhase, GroundVehicle, VehicleType, Runway
from simulator.movement import update_vehicle
from simulator.conflict import assess_conflict
from simulator.generate import make_vehicle, SCENARIO_WEIGHTS

PHASE_MAP = {
    "final_approach": AircraftPhase.FINAL_APPROACH,
    "flare": AircraftPhase.FLARE,
    "rollout": AircraftPhase.ROLLOUT,
    "vacating": AircraftPhase.VACATING,
    "clear": AircraftPhase.CLEAR,
}


def row_to_aircraft(row, aircraft_id):
    phase = PHASE_MAP.get(row.get("aircraft_phase", "final_approach"), AircraftPhase.FINAL_APPROACH)
    return Aircraft(
        id=aircraft_id, x=float(row["aircraft_x"]), y=float(row["aircraft_y"]),
        speed=float(row["aircraft_speed"]), lateral_speed=float(row["aircraft_lateral_speed"]),
        altitude=float(row["aircraft_altitude"]), phase=phase,
        decel_rate=1.8, approach_speed=72.0, touchdown_speed=67.0,
    )


def build_hybrid_dataset(tracks_path, runway, vehicles_per_track=2, seed=42):
    rng = random.Random(seed)
    tracks_df = pd.read_parquet(tracks_path)
    unique_scenarios = tracks_df["scenario_id"].unique()
    print(f"Found {len(unique_scenarios)} real landing sequences.")

    scenario_types = list(SCENARIO_WEIGHTS.keys())
    scenario_probs = list(SCENARIO_WEIGHTS.values())
    all_rows = []

    for scenario_id in unique_scenarios:
        track = tracks_df[tracks_df["scenario_id"] == scenario_id].sort_values("timestamp")

        for v_idx in range(vehicles_per_track):
            scenario_type = rng.choices(scenario_types, weights=scenario_probs, k=1)[0]
            vehicle = make_vehicle(rng, runway, scenario_type)

            for _, row in track.iterrows():
                aircraft = row_to_aircraft(row, str(row["callsign"]))
                if aircraft.phase == AircraftPhase.CLEAR:
                    break
                assessment = assess_conflict(aircraft, vehicle, runway, float(row["timestamp"]))
                all_rows.append({
                    "scenario_id": f"{scenario_id}_v{v_idx}",
                    "scenario_type": scenario_type,
                    "timestamp": float(row["timestamp"]),
                    "data_source": "real_aircraft",
                    "aircraft_x": aircraft.x, "aircraft_y": aircraft.y,
                    "aircraft_speed": aircraft.speed,
                    "aircraft_lateral_speed": aircraft.lateral_speed,
                    "aircraft_altitude": aircraft.altitude,
                    "aircraft_phase": aircraft.phase.value,
                    "vehicle_x": vehicle.x, "vehicle_y": vehicle.y,
                    "vehicle_speed": vehicle.speed, "vehicle_heading": vehicle.heading,
                    "vehicle_type": vehicle.vehicle_type.value,
                    "current_separation": assessment.current_separation,
                    "cpa_distance": assessment.cpa_distance,
                    "time_to_cpa": assessment.time_to_cpa,
                    "aircraft_on_runway": int(assessment.aircraft_on_runway),
                    "vehicle_on_runway": int(assessment.vehicle_on_runway),
                    "co_occupancy_t15": int(assessment.co_occupancy_t15),
                    "co_occupancy_t30": int(assessment.co_occupancy_t30),
                    "co_occupancy_t60": int(assessment.co_occupancy_t60),
                    "aircraft_time_to_threshold": assessment.aircraft_time_to_threshold,
                    "vehicle_dist_to_runway_edge": assessment.vehicle_dist_to_runway_edge,
                    "risk_score": assessment.risk_score,
                    "risk_level": assessment.risk_level,
                })
                vehicle = update_vehicle(vehicle, dt=1.0)

    df = pd.DataFrame(all_rows)
    print(f"Total rows: {len(df):,}")
    print(df["risk_level"].value_counts(normalize=True).round(3))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--output", default="data/hybrid_dataset.parquet")
    parser.add_argument("--airport", default="KLAX")
    parser.add_argument("--runway", default="24L")
    parser.add_argument("--vehicles-per-track", type=int, default=2)
    args = parser.parse_args()

    runway_def = get_runway(args.airport, args.runway)
    runway = runway_def.to_runway()
    df = build_hybrid_dataset(args.tracks, runway, args.vehicles_per_track)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved → {args.output}")