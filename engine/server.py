"""
server.py — FastAPI inference server.

Endpoints:
  POST /update/aircraft   — update aircraft position
  POST /update/vehicle    — update vehicle position
  GET  /predict           — run model on all pairs, return alerts
  GET  /state             — current positions of all entities
  POST /reset             — clear all state
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time

from .state_manager import StateManager
from .predictor import ConflictPredictor
from .alert import make_alert

app = FastAPI(title="Runway Conflict Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

state = StateManager()
predictor = ConflictPredictor(model_dir="ml/model")


# ── Request schemas ───────────────────────────────────────────────

class AircraftUpdate(BaseModel):
    id: str
    x: float
    y: float
    speed: float
    heading: float = 0.0
    altitude: float = 0.0
    phase: str = "final_approach"
    lateral_speed: float = 0.0


class VehicleUpdate(BaseModel):
    id: str
    x: float
    y: float
    speed: float
    heading: float


# ── Endpoints ─────────────────────────────────────────────────────

@app.post("/update/aircraft")
def update_aircraft(data: AircraftUpdate):
    state.update_aircraft(
        id=data.id, x=data.x, y=data.y,
        speed=data.speed, heading=data.heading,
        altitude=data.altitude, phase=data.phase,
        lateral_speed=data.lateral_speed,
    )
    return {"status": "ok", "id": data.id}


@app.post("/update/vehicle")
def update_vehicle(data: VehicleUpdate):
    state.update_vehicle(
        id=data.id, x=data.x, y=data.y,
        speed=data.speed, heading=data.heading,
    )
    return {"status": "ok", "id": data.id}


@app.get("/predict")
def predict():
    pairs = state.get_all_pairs()
    if not pairs:
        return {"alerts": [], "message": "No entities in state"}

    results = []
    for aircraft, vehicle in pairs:
        prediction = predictor.predict(aircraft, vehicle)
        alert = make_alert(prediction)
        results.append({
            "prediction": prediction,
            "alert": {
                "risk_level": alert.risk_level,
                "color": alert.color,
                "message": alert.message,
                "should_alarm": alert.should_alarm,
            }
        })

    # Sort by risk — high risk first
    order = {"high_risk": 0, "warning": 1, "safe": 2}
    results.sort(key=lambda r: order[r["prediction"]["risk_level"]])

    return {
        "timestamp": time.time(),
        "n_pairs": len(results),
        "alerts": results,
    }


@app.get("/state")
def get_state():
    return {
        "aircraft": {k: vars(v) for k, v in state.aircraft.items()},
        "vehicles": {k: vars(v) for k, v in state.vehicles.items()},
    }


@app.post("/reset")
def reset():
    state.clear()
    return {"status": "cleared"}


@app.get("/health")
def health():
    return {"status": "ok"}