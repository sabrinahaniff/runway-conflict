import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

from .state_manager import EntityState
from ml.features import engineer_features, FEATURE_COLUMNS, LABEL_DECODING


class ConflictPredictor:
    """
    Loads the trained model and runs inference on a
    (aircraft, vehicle) state pair.

    Stateless — takes a snapshot, returns a score.
    """

    def __init__(self, model_dir: str = "ml/model"):
        print(f"Loading model from {model_dir}...")
        self.model = xgb.XGBClassifier()
        self.model.load_model(f"{model_dir}/xgboost.json")

        with open(f"{model_dir}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        with open(f"{model_dir}/feature_columns.pkl", "rb") as f:
            self.feature_columns = pickle.load(f)

        print("Model loaded.")

    def predict(self, aircraft: EntityState, vehicle: EntityState,
                runway_length: float = 3685.0,
                runway_half_width: float = 30.5) -> dict:
        """
        Given current aircraft + vehicle state, return risk prediction.
        """
        # Build raw feature row 
        dx = vehicle.x - aircraft.x
        dy = vehicle.y - aircraft.y
        dist = max(np.sqrt(dx**2 + dy**2), 1e-6)

        # CPA calculation
        rel_vx = (vehicle.speed * np.cos(vehicle.heading)) - aircraft.speed
        rel_vy = (vehicle.speed * np.sin(vehicle.heading)) - aircraft.lateral_speed
        rel_pos = np.array([dx, dy])
        rel_vel = np.array([rel_vx, rel_vy])
        v_sq = np.dot(rel_vel, rel_vel)

        if v_sq > 1e-9:
            t_cpa = float(np.clip(-np.dot(rel_pos, rel_vel) / v_sq, 0, 60))
        else:
            t_cpa = 0.0

        cpa_pos = rel_pos + rel_vel * t_cpa
        cpa_distance = float(np.linalg.norm(cpa_pos))

        # Occupancy projections
        def on_runway(x, y):
            return (0 <= x <= runway_length and
                    -runway_half_width <= y <= runway_half_width)

        def project(ex, ey, evx, evy, dt):
            return ex + evx * dt, ey + evy * dt

        ac_vx, ac_vy = aircraft.speed, aircraft.lateral_speed
        gv_vx = vehicle.speed * np.cos(vehicle.heading)
        gv_vy = vehicle.speed * np.sin(vehicle.heading)

        def co_occ(dt):
            ax, ay = project(aircraft.x, aircraft.y, ac_vx, ac_vy, dt)
            vx, vy = project(vehicle.x, vehicle.y, gv_vx, gv_vy, dt)
            return int(on_runway(ax, ay) and on_runway(vx, vy))

        # Time to threshold
        if aircraft.speed > 0 and aircraft.x < 0:
            time_to_threshold = abs(aircraft.x) / aircraft.speed
        else:
            time_to_threshold = 0.0

        # Vehicle distance to runway edge
        if on_runway(vehicle.x, vehicle.y):
            vdist = -min(runway_half_width - abs(vehicle.y),
                         vehicle.x, runway_length - vehicle.x)
        else:
            vdist = abs(abs(vehicle.y) - runway_half_width)

        phase_map = {
            "final_approach": 0, "flare": 1, "rollout": 2,
            "vacating": 3, "clear": 4
        }

        row = {
            "aircraft_x": aircraft.x,
            "aircraft_y": aircraft.y,
            "aircraft_speed": aircraft.speed,
            "aircraft_lateral_speed": aircraft.lateral_speed,
            "aircraft_altitude": aircraft.altitude,
            "vehicle_x": vehicle.x,
            "vehicle_y": vehicle.y,
            "vehicle_speed": vehicle.speed,
            "vehicle_heading": vehicle.heading,
            "current_separation": dist,
            "cpa_distance": cpa_distance,
            "time_to_cpa": t_cpa,
            "aircraft_on_runway": int(on_runway(aircraft.x, aircraft.y)),
            "vehicle_on_runway": int(on_runway(vehicle.x, vehicle.y)),
            "co_occupancy_t15": co_occ(15),
            "co_occupancy_t30": co_occ(30),
            "co_occupancy_t60": co_occ(60),
            "aircraft_time_to_threshold": time_to_threshold,
            "vehicle_dist_to_runway_edge": vdist,
            "aircraft_phase": aircraft.phase,
        }

        df = pd.DataFrame([row])
        df = engineer_features(df)

        available = [c for c in self.feature_columns if c in df.columns]
        X = df[available].fillna(0).astype(float)
        X_scaled = self.scaler.transform(X)

        proba = self.model.predict_proba(X_scaled)[0]
        pred_class = int(np.argmax(proba))
        label = LABEL_DECODING[pred_class]

        return {
            "aircraft_id": aircraft.id,
            "vehicle_id": vehicle.id,
            "risk_level": label,
            "risk_score": float(proba[pred_class]),
            "probabilities": {
                "safe": float(proba[0]),
                "warning": float(proba[1]),
                "high_risk": float(proba[2]),
            },
            "cpa_distance_m": round(cpa_distance, 1),
            "time_to_cpa_s": round(t_cpa, 1),
            "current_separation_m": round(dist, 1),
        }