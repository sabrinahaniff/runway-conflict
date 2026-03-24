import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "aircraft_x",
    "aircraft_y",
    "aircraft_speed",
    "aircraft_lateral_speed",
    "aircraft_altitude",
    "vehicle_x",
    "vehicle_y",
    "vehicle_speed",
    "vehicle_heading",
    "current_separation",
    "cpa_distance",
    "time_to_cpa",
    "aircraft_on_runway",
    "vehicle_on_runway",
    "co_occupancy_t15",
    "co_occupancy_t30",
    "co_occupancy_t60",
    "aircraft_time_to_threshold",
    "vehicle_dist_to_runway_edge",
    "closing_speed",
    "time_to_threshold_x_dist_to_edge",
    "aircraft_phase_encoded",
]

PHASE_ENCODING = {
    "final_approach": 0,
    "flare": 1,
    "rollout": 2,
    "vacating": 3,
    "clear": 4,
}

LABEL_ENCODING = {
    "safe": 0,
    "warning": 1,
    "high_risk": 2,
}

LABEL_DECODING = {v: k for k, v in LABEL_ENCODING.items()}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    dx = df["vehicle_x"] - df["aircraft_x"]
    dy = df["vehicle_y"] - df["aircraft_y"]
    dist = np.sqrt(dx**2 + dy**2).clip(lower=1e-6)

    ux = dx / dist
    uy = dy / dist

    dvx = (df["vehicle_speed"] * np.cos(df["vehicle_heading"])
           - df["aircraft_speed"])
    dvy = (df["vehicle_speed"] * np.sin(df["vehicle_heading"])
           - df["aircraft_lateral_speed"])

    df["closing_speed"] = -(dvx * ux + dvy * uy)

    df["time_to_threshold_x_dist_to_edge"] = (
        df["aircraft_time_to_threshold"]
        * df["vehicle_dist_to_runway_edge"].clip(lower=0)
    )

    df["aircraft_phase_encoded"] = (
        df["aircraft_phase"].map(PHASE_ENCODING).fillna(0).astype(int)
    )

    return df


def prepare_dataset(df: pd.DataFrame):
    df = engineer_features(df)
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available].fillna(0).astype(float)
    y = df["risk_level"].map(LABEL_ENCODING).astype(int)
    return X, y