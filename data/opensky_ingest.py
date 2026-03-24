import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from .runways import RunwayDefinition, get_runway
from .transform import latlon_to_local, compute_local_velocity, compute_speed_ms, flight_phase_from_position

APPROACH_ALTITUDE_M = 5000 * 0.3048

AIRPORT_BOUNDING_BOXES = {
    "KLAX": (33.70, 34.10, -118.60, -118.20),
    "EGLL": (51.35, 51.60, -0.65,   -0.20),
    "KJFK": (40.50, 40.80, -74.00, -73.60),
    "KORD": (41.85, 42.10, -88.05, -87.75),
}


@dataclass
class ProcessedTrackPoint:
    callsign: str
    scenario_id: str
    timestamp: float
    aircraft_x: float
    aircraft_y: float
    aircraft_speed: float
    aircraft_lateral_speed: float
    aircraft_altitude: float
    aircraft_phase: str
    raw_lat: float
    raw_lon: float


def fetch_from_csv(csv_path, airport_icao, runway_id, max_flights=None):
    runway_def = get_runway(airport_icao, runway_id)
    bbox = AIRPORT_BOUNDING_BOXES.get(airport_icao)
    if not bbox:
        raise ValueError(f"No bounding box for {airport_icao}")

    print(f"Loading {csv_path}...")
    state_cols = [
        "time", "icao24", "lat", "lon", "velocity",
        "heading", "vertrate", "callsign", "onground",
        "alert", "spi", "squawk", "baroaltitude", "geoaltitude",
        "lastposupdate", "lastcontact"
    ]

    df_raw = pd.read_csv(csv_path, names=state_cols, header=0)
    print(f"Loaded {len(df_raw):,} raw state vectors.")

    df_raw = df_raw.dropna(subset=["lat", "lon", "baroaltitude", "velocity"])
    df_raw = df_raw[
        (df_raw["lat"] >= bbox[0]) & (df_raw["lat"] <= bbox[1]) &
        (df_raw["lon"] >= bbox[2]) & (df_raw["lon"] <= bbox[3]) &
        (df_raw["baroaltitude"] < APPROACH_ALTITUDE_M) &
        (df_raw["velocity"] > 30)
    ]

    print(f"After airport filter: {len(df_raw):,} state vectors")

    if len(df_raw) == 0:
        raise ValueError(f"No data found for {airport_icao}. Check the CSV covers this airport.")

    return _process_tracks(df_raw, runway_def, max_flights)


def _process_tracks(df, runway_def, max_flights=None):
    rows = []
    processed = 0
    skipped = 0
    runway = runway_def.to_runway()

    for callsign, group in df.groupby("callsign"):
        if max_flights and processed >= max_flights:
            break
        group = group.sort_values("time").reset_index(drop=True)
        track = _extract_landing_sequence(group, runway_def, runway)
        if track is None or len(track) < 10:
            skipped += 1
            continue

        scenario_id = f"{callsign}_{int(group['time'].iloc[0])}"
        for i, point in enumerate(track):
            point.scenario_id = scenario_id
            point.timestamp = float(i)
            rows.append(asdict(point))

        processed += 1
        if processed % 100 == 0:
            print(f"  Processed {processed} landings ({skipped} skipped)...")

    print(f"\nExtracted {processed} landing sequences ({skipped} skipped)")
    print(f"Total track points: {len(rows):,}")
    return pd.DataFrame(rows)


def _extract_landing_sequence(group, runway_def, runway):
    local_positions = []
    timestamps = []

    for _, row in group.iterrows():
        pos = latlon_to_local(
            lat=float(row["lat"]), lon=float(row["lon"]),
            altitude_m=float(row["baroaltitude"]),
            runway=runway_def,
        )
        local_positions.append(pos)
        timestamps.append(float(row["time"]))

    if len(local_positions) < 2:
        return None

    filtered = [
        (pos, ts) for pos, ts in zip(local_positions, timestamps)
        if -30000 <= pos.x <= runway.length and pos.altitude < 1500
    ]

    if len(filtered) < 10:
        return None

    xs = [p.x for p, _ in filtered]
    if not (min(xs) < -50 and max(xs) > 50):
        return None

    result = []
    for i, (pos, ts) in enumerate(filtered):
        if i == 0:
            vx, vy = 0.0, 0.0
        else:
            vx, vy = compute_local_velocity(
                [p for p, _ in filtered[:i+1]],
                [t for _, t in filtered[:i+1]]
            )
        speed = compute_speed_ms(vx, vy)
        phase = flight_phase_from_position(pos.x, pos.altitude, speed, runway.length)
        result.append(ProcessedTrackPoint(
            callsign=str(group["callsign"].iloc[0]).strip(),
            scenario_id="", timestamp=0.0,
            aircraft_x=pos.x, aircraft_y=pos.y,
            aircraft_speed=speed, aircraft_lateral_speed=vy,
            aircraft_altitude=pos.altitude, aircraft_phase=phase,
            raw_lat=float(group["lat"].iloc[i] if i < len(group) else 0),
            raw_lon=float(group["lon"].iloc[i] if i < len(group) else 0),
        ))
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["csv"], default="csv")
    parser.add_argument("--airport", default="KLAX")
    parser.add_argument("--runway", default="24L")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed_tracks.parquet")
    parser.add_argument("--max-flights", type=int, default=None)
    args = parser.parse_args()

    df = fetch_from_csv(args.input, args.airport, args.runway, args.max_flights)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved → {args.output}")