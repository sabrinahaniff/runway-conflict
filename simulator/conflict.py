import numpy as np
from dataclasses import dataclass
from .entities import Aircraft, AircraftPhase, GroundVehicle, Runway


@dataclass
class ConflictAssessment:
    aircraft_id: str
    vehicle_id: str
    timestamp: float
    current_separation: float
    cpa_distance: float
    time_to_cpa: float
    aircraft_on_runway: bool
    vehicle_on_runway: bool
    co_occupancy_t15: bool
    co_occupancy_t30: bool
    co_occupancy_t60: bool
    risk_level: str
    risk_score: float
    aircraft_time_to_threshold: float
    vehicle_dist_to_runway_edge: float


CPA_HIGH_RISK_M = 50.0
CPA_WARNING_M = 150.0


def compute_cpa(pos_a, vel_a, pos_b, vel_b, t_horizon=60.0):
    r = pos_a - pos_b
    v = vel_a - vel_b
    v_sq = np.dot(v, v)
    if v_sq < 1e-9:
        return float(np.linalg.norm(r)), 0.0
    t_cpa = float(np.clip(-np.dot(r, v) / v_sq, 0.0, t_horizon))
    cpa_distance = float(np.linalg.norm(r + v * t_cpa))
    return cpa_distance, t_cpa


def project_runway_occupancy(pos, vel, runway, dt):
    px = pos[0] + vel[0] * dt
    py = pos[1] + vel[1] * dt
    return runway.is_on_runway(px, py)


def assess_conflict(aircraft, vehicle, runway, timestamp, t_horizon=60.0):
    current_sep = float(np.linalg.norm(aircraft.position - vehicle.position))
    cpa_distance, time_to_cpa = compute_cpa(
        aircraft.position, aircraft.velocity,
        vehicle.position, vehicle.velocity, t_horizon
    )
    co_t15 = (project_runway_occupancy(aircraft.position, aircraft.velocity, runway, 15.0) and
               project_runway_occupancy(vehicle.position, vehicle.velocity, runway, 15.0))
    co_t30 = (project_runway_occupancy(aircraft.position, aircraft.velocity, runway, 30.0) and
               project_runway_occupancy(vehicle.position, vehicle.velocity, runway, 30.0))
    co_t60 = (project_runway_occupancy(aircraft.position, aircraft.velocity, runway, 60.0) and
               project_runway_occupancy(vehicle.position, vehicle.velocity, runway, 60.0))

    if aircraft.speed > 0 and aircraft.x < 0:
        time_to_threshold = abs(aircraft.x) / aircraft.speed
    else:
        time_to_threshold = 0.0

    vehicle_on_rwy = runway.is_on_runway(vehicle.x, vehicle.y)
    vdist = _vehicle_dist_to_runway(vehicle, runway)
    risk_score, risk_level = _compute_risk(
        aircraft, vehicle, runway, cpa_distance, time_to_cpa,
        co_t15, co_t30, co_t60, time_to_threshold, current_sep
    )

    return ConflictAssessment(
        aircraft_id=aircraft.id, vehicle_id=vehicle.id,
        timestamp=timestamp, current_separation=current_sep,
        cpa_distance=cpa_distance, time_to_cpa=time_to_cpa,
        aircraft_on_runway=aircraft.is_on_runway,
        vehicle_on_runway=vehicle_on_rwy,
        co_occupancy_t15=co_t15, co_occupancy_t30=co_t30, co_occupancy_t60=co_t60,
        risk_level=risk_level, risk_score=risk_score,
        aircraft_time_to_threshold=time_to_threshold,
        vehicle_dist_to_runway_edge=vdist,
    )


def _vehicle_dist_to_runway(vehicle, runway):
    if runway.is_on_runway(vehicle.x, vehicle.y):
        return -min(runway.half_width - abs(vehicle.y),
                    vehicle.x - runway.rollout_start,
                    runway.rollout_end - vehicle.x)
    return min(abs(abs(vehicle.y) - runway.half_width),
               abs(vehicle.x - runway.rollout_start),
               abs(vehicle.x - runway.rollout_end))


def _compute_risk(aircraft, vehicle, runway, cpa_distance, time_to_cpa,
                  co_t15, co_t30, co_t60, time_to_threshold, current_sep):
    on_runway_both = (aircraft.is_on_runway and runway.is_on_runway(vehicle.x, vehicle.y))
    near_runway = (aircraft.phase in (AircraftPhase.FINAL_APPROACH, AircraftPhase.FLARE)
                   and time_to_threshold < 30.0)

    if on_runway_both:
        return max(0.85, _cpa_to_score(cpa_distance, time_to_cpa, 1.0)), "high_risk"
    if co_t15:
        return max(0.80, _cpa_to_score(cpa_distance, time_to_cpa, 0.9)), "high_risk"
    if cpa_distance < CPA_HIGH_RISK_M and time_to_cpa < 30.0 and near_runway:
        return max(0.75, _cpa_to_score(cpa_distance, time_to_cpa, 0.85)), "high_risk"
    if co_t30:
        return max(0.55, _cpa_to_score(cpa_distance, time_to_cpa, 0.6)), "warning"
    if cpa_distance < CPA_WARNING_M and time_to_cpa < 60.0:
        return max(0.45, _cpa_to_score(cpa_distance, time_to_cpa, 0.5)), "warning"
    if time_to_threshold < 30.0 and _vehicle_dist_to_runway(vehicle, runway) < 100.0:
        return 0.4 + (30.0 - time_to_threshold) / 30.0 * 0.2, "warning"
    return min(0.35, _cpa_to_score(cpa_distance, time_to_cpa, 0.2)), "safe"


def _cpa_to_score(cpa_distance, time_to_cpa, weight=1.0):
    d = max(0.0, 1.0 - cpa_distance / 500.0)
    t = max(0.0, 1.0 - time_to_cpa / 60.0)
    return weight * (0.6 * d + 0.4 * t)