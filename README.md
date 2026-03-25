# Runway Conflict Detection System

<img width="1340" height="708" alt="IMG_4676" src="https://github.com/user-attachments/assets/612f4cbb-a0cd-4be3-87dd-6181247fe6a5" />


---

## What this system does

Every time a plane lands, it shares the runway with ground vehicles like fire trucks stationed at the runway end, maintenance crews, fuel trucks, and follow-me cars. Most of the time this is fine. But occasionally a vehicle is in the wrong place at the wrong time. These are called **runway incursions**, and they are one of the leading causes of fatal aviation accidents worldwide.

This system watches the positions of all aircraft and vehicles every second and answers one question: **are any of these about to be in the same place at the same time?**

It outputs one of three alerts:

| Alert | Meaning |
|---|---|
| GREEN — SAFE | No conflicts predicted in the next 60 seconds |
| ORANGE — WARNING | A potential conflict is developing — monitor closely |
| RED — HIGH RISK | Collision course predicted — immediate action needed |

---

## How to read the dashboard

### The top-down map (main view)

The large map shows a bird's-eye view of Los Angeles International Airport (KLAX), looking straight down as if from a helicopter.

**The two horizontal grey bars** are the runways — Runway 24L (top) and Runway 24R (bottom). Each is 3,685 meters long and 61 meters wide. The yellow bars at each end mark the **threshold** — the point where aircraft first touch down. The slightly brighter zone at the left end is the **touchdown zone**, the most critical 900 meters where most landings occur.

**The triangles** are aircraft. They point right because all aircraft are landing from left to right. Each shows its callsign (UAL232 = United Airlines 232, AAL456 = American Airlines 456, etc). Color shows landing phase:
- Blue = final approach (still in the air, descending)
- Yellow = flare (last 150m before touchdown, nose pitching up)
- Orange = rollout (on runway, braking hard)
- Purple = vacating (turning off onto a taxiway)

**The squares** are ground vehicles — fire trucks, maintenance crews, fuel trucks, follow-me cars. Their color shows risk level relative to landing aircraft: red = dangerous, orange = caution, natural color = safe.

**The dashed lines** show where each entity will be in the next 20 seconds. When a dashed line from an aircraft and a dashed line from a vehicle point toward the same spot, that's a potential conflict.

**The dashed circles** that appear on the runway mark the Closest Point of Approach (CPA) — where two entities will get closest to each other. The label shows distance in meters and how many seconds away. "21m · 8s" means two entities will pass within 21 meters of each other in 8 seconds — that's a crisis.

### The alert banner
The colored bar at the top shows the worst situation right now: which aircraft and vehicle are the most dangerous pair, how close they'll get, and how much time there is.

### The status panel (top right)
Three numbers: how many pairs are currently HIGH RISK, WARNING, and SAFE. With 4 aircraft and 6 vehicles, the system assesses 24 pairs simultaneously every second.

### The pairs table (bottom right)
Every aircraft-vehicle combination being monitored, sorted worst first. This gives controllers and researchers the full picture, not just the worst case.

### The risk timeline (bottom)
How overall risk has changed over the last 60 ticks (~24 seconds). The spike from green to red shows how quickly situations can develop and how much warning time the system provides.

---

## Why this matters

### Don't airports already have systems for this?

Yes, systems like RIMCAS and ASDE-X exist at major airports. They use rule-based detection: fixed geometric thresholds hardcoded by engineers. If an aircraft is within X meters of the runway and a vehicle is within Y meters of the centerline, alert.

This project takes a different approach: **let the model learn the thresholds from data rather than hardcoding them.**

The research question is whether a data-driven approach can generalize better to edge cases that rule-based systems miss, produce probabilistic risk scores that give more nuanced information, and adapt to different airports without manual re-tuning.

This is an active area of research. Runway incursions cause approximately 3 fatal accidents per year worldwide. The most deadly aviation accident in history — Tenerife 1977, 583 deaths — was a runway collision. Despite decades of work, incursion rates have not significantly declined.

---

## Technical architecture

```
OpenSky Network ADS-B data (real aircraft GPS positions)
         ↓
  data/opensky_ingest.py      (extracts real landing sequences)
  data/transform.py          (converts GPS coords to runway-local meters)
         ↓
  simulator/conflict.py      (CPA geometry engine (labels every tick))
  data/generate_hybrid.py    (pairs real aircraft + synthetic vehicles)
         ↓
  118,000 labeled training rows
         ↓
  ml/features.py             (22 engineered features per snapshot)
  ml/train.py                (XGBoost, split by scenario ID)
         ↓
  ml/model/xgboost.json      with 99.6% test accuracy
         ↓
  engine/server.py           FastAPI: POST positions → GET predictions
         ↓
  dashboard/app.py           Dash live visualization
```

### How the model works

Every second, for every (aircraft, vehicle) pair, the system extracts 22 features and runs XGBoost inference. The top features the model learned on its own:

1. `vehicle_dist_to_runway_edge` (22%) — how close the vehicle is to the runway
2. `cpa_distance` (16%) — the minimum predicted separation
3. `vehicle_y` (12%) — lateral position of the vehicle
4. `aircraft_on_runway` (10%) — is the plane already rolling out?
5. `co_occupancy_t15` (8%) — will both be on runway in 15 seconds?

### Training data

Real aircraft trajectories from OpenSky Network (KLAX, 2022) combined with synthetically generated ground vehicle movements. 118,000 labeled snapshots. Split by scenario ID (not by row) to prevent temporal leakage.

---

## Limitations

**Ground vehicle data is synthetic.** No public dataset of airport ground vehicle positions exists. A production system would require integration with the airport's Surface Movement Guidance and Control System.

**99.6% accuracy needs context.** The test set comes from the same simulation distribution as training. Real-world generalization requires evaluation on data from different airports and conditions.

---

## Running the system

```bash
pip install -r requirements.txt
brew install libomp  # Mac only
```

Three terminals:

```bash
# Terminal 1 — inference engine
python -m uvicorn engine.server:app --reload --port 8000

# Terminal 2 — dashboard
python dashboard/app.py

# Open http://127.0.0.1:8050
```

Test the API directly:
```bash
curl -X POST http://127.0.0.1:8000/update/aircraft \
  -H "Content-Type: application/json" \
  -d '{"id":"AC001","x":-400,"y":2,"speed":70,"altitude":35,"phase":"final_approach","lateral_speed":0,"heading":0}'

curl -X POST http://127.0.0.1:8000/update/vehicle \
  -H "Content-Type: application/json" \
  -d '{"id":"GV001","x":200,"y":-30,"speed":8,"heading":1.57}'

curl http://127.0.0.1:8000/predict
```

---

## Project structure

```
runway_conflict/
├── simulator/         # Physics engine + conflict labeler
├── data/              # ADS-B ingestion + hybrid dataset builder
├── ml/                # Feature engineering + XGBoost pipeline
├── engine/            # FastAPI inference server
├── dashboard/         # Dash visualization
└── tests/             # Geometry engine unit tests
```

---

## Built with

Python · XGBoost · FastAPI · Plotly Dash · Pandas · NumPy · OpenSky Network ADS-B data

## Data sources

Aircraft trajectories: OpenSky Network (opensky-network.org) — free historical ADS-B data.
Airport geometry: FAA Airport Diagrams.
Ground vehicle routes: Synthetically generated.
