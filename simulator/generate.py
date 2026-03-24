import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from .entities import Aircraft, AircraftPhase, GroundVehicle, VehicleType, Runway
from .movement import update_aircraft, update_vehicle
from .conflict import assess_conflict

DT = 1.0
MAX_SCENARIO_T = 120.0

SCENARIO_WEIGHTS = {
    "crossing": 0.15, "entering": 0.20,
    "parallel": 0.25, "clear": 0.40,
}


def make_aircraft(rng, runway):
    approach_speed = rng.uniform(65.0, 80.0)
    touchdown_speed = approach_speed - rng.uniform(3.0, 7.0)
    return Aircraft(
        id=f"AC{rng.randint(100,999)}",
        x=rng.uniform(-3000.0, -800.0),
        y=rng.uniform(-15.0, 15.0),
        speed=approach_speed,
        lateral_speed=0.0,
        altitude=0.0,
        phase=AircraftPhase.FINAL_APPROACH,
        decel_rate=rng.uniform(1.4, 2.5),
        approach_speed=approach_speed,
        touchdown_speed=touchdown_speed,
    )


def make_vehicle(rng, runway, scenario_type):
    speed = rng.uniform(3.0, 12.0)
    if scenario_type == "crossing":
        entry_x = rng.uniform(100.0, 800.0)
        start_y = -(runway.half_width + rng.uniform(20.0, 80.0))
        waypoints = [(entry_x, runway.half_width + rng.uniform(20.0, 80.0))]
        start_x = entry_x
    elif scenario_type == "entering":
        entry_x = rng.uniform(200.0, 1200.0)
        start_y = -(runway.half_width + rng.uniform(40.0, 120.0))
        waypoints = [(entry_x, 0.0), (entry_x, runway.half_width + 50.0)]
        start_x = entry_x
    elif scenario_type == "parallel":
        start_x = rng.uniform(-200.0, 500.0)
        start_y = runway.half_width + rng.uniform(10.0, 40.0)
        waypoints = [(rng.uniform(800.0, runway.length), start_y)]
    else:
        start_x = rng.uniform(-500.0, runway.length + 200.0)
        start_y = runway.half_width + rng.uniform(100.0, 300.0)
        waypoints = [(start_x + rng.uniform(-200.0, 200.0), start_y + rng.uniform(-50.0, 50.0))]

    heading = np.arctan2(waypoints[0][1] - start_y, waypoints[0][0] - start_x) if waypoints else 0.0
    return GroundVehicle(
        id=f"GV{rng.randint(100,999)}",
        vehicle_type=rng.choice(list(VehicleType)),
        x=start_x, y=start_y, speed=speed,
        heading=heading, waypoints=waypoints,
    )


def simulate_scenario(rng, runway, scenario_id):
    scenario_types = list(SCENARIO_WEIGHTS.keys())
    weights = list(SCENARIO_WEIGHTS.values())
    scenario_type = rng.choices(scenario_types, weights=weights, k=1)[0]
    aircraft = make_aircraft(rng, runway)
    vehicle = make_vehicle(rng, runway, scenario_type)
    rows = []
    t = 0.0
    while t < MAX_SCENARIO_T:
        if aircraft.phase == AircraftPhase.CLEAR:
            break
        assessment = assess_conflict(aircraft, vehicle, runway, t)
        rows.append({
            "scenario_id": scenario_id, "scenario_type": scenario_type,
            "timestamp": t, "aircraft_id": assessment.aircraft_id,
            "vehicle_id": assessment.vehicle_id,
            "aircraft_x": aircraft.x, "aircraft_y": aircraft.y,
            "aircraft_speed": aircraft.speed, "aircraft_lateral_speed": aircraft.lateral_speed,
            "aircraft_altitude": aircraft.altitude, "aircraft_phase": aircraft.phase.value,
            "vehicle_x": vehicle.x, "vehicle_y": vehicle.y,
            "vehicle_speed": vehicle.speed, "vehicle_heading": vehicle.heading,
            "vehicle_type": vehicle.vehicle_type.value,
            "current_separation": assessment.current_separation,
            "cpa_distance": assessment.cpa_distance, "time_to_cpa": assessment.time_to_cpa,
            "aircraft_on_runway": int(assessment.aircraft_on_runway),
            "vehicle_on_runway": int(assessment.vehicle_on_runway),
            "co_occupancy_t15": int(assessment.co_occupancy_t15),
            "co_occupancy_t30": int(assessment.co_occupancy_t30),
            "co_occupancy_t60": int(assessment.co_occupancy_t60),
            "aircraft_time_to_threshold": assessment.aircraft_time_to_threshold,
            "vehicle_dist_to_runway_edge": assessment.vehicle_dist_to_runway_edge,
            "risk_score": assessment.risk_score, "risk_level": assessment.risk_level,
        })
        aircraft = update_aircraft(aircraft, runway, DT)
        vehicle = update_vehicle(vehicle, DT)
        t += DT
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, default=2000)
    parser.add_argument("--output", type=str, default="data/scenarios.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    runway = Runway()
    all_rows = []
    for i in range(args.scenarios):
        if i % 500 == 0:
            print(f"Scenario {i}/{args.scenarios}...")
        all_rows.extend(simulate_scenario(rng, runway, i))
    df = pd.DataFrame(all_rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df):,} rows → {args.output}")