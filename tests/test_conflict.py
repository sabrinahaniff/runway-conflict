import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from simulator.conflict import compute_cpa, assess_conflict
from simulator.entities import Aircraft, AircraftPhase, GroundVehicle, VehicleType, Runway

runway = Runway()
passed = 0
failed = 0

def check(name, condition, detail=''):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))
        failed += 1

# CPA tests
d, t = compute_cpa(np.array([0.,0.]), np.array([0.,0.]), np.array([100.,0.]), np.array([0.,0.]))
check("CPA stationary", abs(d-100) < 1e-4 and t == 0.0)

d, t = compute_cpa(np.array([0.,0.]), np.array([10.,0.]), np.array([100.,0.]), np.array([-10.,0.]))
check("CPA head-on", abs(d) < 1.0 and abs(t-5.0) < 0.1)

d, t = compute_cpa(np.array([0.,0.]), np.array([10.,0.]), np.array([0.,50.]), np.array([10.,0.]))
check("CPA parallel", abs(d-50) < 1.0)

# Runway geometry
check("on runway center", runway.is_on_runway(500., 0.) == True)
check("off runway lateral", runway.is_on_runway(500., 50.) == False)
check("off runway before threshold", runway.is_on_runway(-1., 0.) == False)

# Conflict labels
ac = Aircraft("AC1", x=-3000., y=0., speed=70., lateral_speed=0., altitude=156.,
              phase=AircraftPhase.FINAL_APPROACH, decel_rate=1.8, approach_speed=70., touchdown_speed=65.)
gv = GroundVehicle("GV1", VehicleType.FIRE_TRUCK, x=500., y=300., speed=5., heading=0., waypoints=[(600.,300.)])
r = assess_conflict(ac, gv, runway, 0.)
check("safe scenario", r.risk_level == "safe", f"got {r.risk_level}")

ac2 = Aircraft("AC2", x=300., y=0., speed=55., lateral_speed=0., altitude=0.,
               phase=AircraftPhase.ROLLOUT, decel_rate=1.8, approach_speed=70., touchdown_speed=65.)
gv2 = GroundVehicle("GV2", VehicleType.FIRE_TRUCK, x=350., y=0., speed=3., heading=0., waypoints=[(450.,0.)])
r2 = assess_conflict(ac2, gv2, runway, 0.)
check("both on runway = high_risk", r2.risk_level == "high_risk", f"got {r2.risk_level}")

print(f"\n{passed} passed, {failed} failed")
