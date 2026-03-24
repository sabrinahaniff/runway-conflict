import numpy as np
from .entities import Aircraft, AircraftPhase, GroundVehicle, Runway


def update_aircraft(aircraft: Aircraft, runway: Runway, dt: float) -> Aircraft:
    phase = aircraft.phase
    if phase == AircraftPhase.FINAL_APPROACH:
        return _update_approach(aircraft, runway, dt)
    elif phase == AircraftPhase.FLARE:
        return _update_flare(aircraft, runway, dt)
    elif phase == AircraftPhase.ROLLOUT:
        return _update_rollout(aircraft, runway, dt)
    elif phase == AircraftPhase.VACATING:
        return _update_vacating(aircraft, runway, dt)
    else:
        return aircraft


def _update_approach(aircraft: Aircraft, runway: Runway, dt: float) -> Aircraft:
    glide_slope = np.tan(np.radians(3.0))
    new_x = aircraft.x + aircraft.speed * dt
    new_altitude = max(0.0, -new_x * glide_slope)
    lateral_correction = -aircraft.y * 0.3
    new_y = aircraft.y + lateral_correction * dt
    new_phase = AircraftPhase.FLARE if new_x >= -150.0 else AircraftPhase.FINAL_APPROACH
    return Aircraft(
        id=aircraft.id, x=new_x, y=new_y,
        speed=aircraft.speed, lateral_speed=lateral_correction,
        altitude=new_altitude, phase=new_phase,
        decel_rate=aircraft.decel_rate,
        approach_speed=aircraft.approach_speed,
        touchdown_speed=aircraft.touchdown_speed,
    )


def _update_flare(aircraft: Aircraft, runway: Runway, dt: float) -> Aircraft:
    new_x = aircraft.x + aircraft.speed * dt
    flare_progress = (aircraft.x + 150.0) / 150.0
    new_altitude = max(0.0, aircraft.altitude * (1.0 - flare_progress * 0.8))
    new_speed = max(aircraft.touchdown_speed, aircraft.speed - 0.5 * dt)
    lateral_correction = -aircraft.y * 0.8
    new_y = aircraft.y + lateral_correction * dt
    new_phase = AircraftPhase.ROLLOUT if new_x >= 0.0 else AircraftPhase.FLARE
    return Aircraft(
        id=aircraft.id, x=new_x, y=new_y,
        speed=new_speed, lateral_speed=lateral_correction,
        altitude=new_altitude, phase=new_phase,
        decel_rate=aircraft.decel_rate,
        approach_speed=aircraft.approach_speed,
        touchdown_speed=aircraft.touchdown_speed,
    )


def _update_rollout(aircraft: Aircraft, runway: Runway, dt: float) -> Aircraft:
    new_x = aircraft.x + aircraft.speed * dt
    taxi_speed = 15.0
    new_speed = max(taxi_speed, aircraft.speed - aircraft.decel_rate * dt)
    lateral_correction = -aircraft.y * 1.2
    new_y = aircraft.y + lateral_correction * dt
    vacating = new_speed <= taxi_speed + 2.0 and new_x > runway.length * 0.4
    return Aircraft(
        id=aircraft.id, x=new_x, y=new_y,
        speed=new_speed, lateral_speed=lateral_correction,
        altitude=0.0,
        phase=AircraftPhase.VACATING if vacating else AircraftPhase.ROLLOUT,
        decel_rate=aircraft.decel_rate,
        approach_speed=aircraft.approach_speed,
        touchdown_speed=aircraft.touchdown_speed,
    )


def _update_vacating(aircraft: Aircraft, runway: Runway, dt: float) -> Aircraft:
    new_x = aircraft.x + aircraft.speed * dt
    new_speed = max(5.0, aircraft.speed - 1.0 * dt)
    target_y = runway.half_width + 20.0
    lateral_correction = (target_y - aircraft.y) * 0.5
    new_y = aircraft.y + lateral_correction * dt
    is_clear = not runway.is_on_runway(new_x, new_y)
    return Aircraft(
        id=aircraft.id, x=new_x, y=new_y,
        speed=new_speed, lateral_speed=lateral_correction,
        altitude=0.0,
        phase=AircraftPhase.CLEAR if is_clear else AircraftPhase.VACATING,
        decel_rate=aircraft.decel_rate,
        approach_speed=aircraft.approach_speed,
        touchdown_speed=aircraft.touchdown_speed,
    )


WAYPOINT_ARRIVAL_THRESHOLD = 5.0


def update_vehicle(vehicle: GroundVehicle, dt: float) -> GroundVehicle:
    if not vehicle.has_waypoints:
        return vehicle
    target = vehicle.current_target
    direction = target - vehicle.position
    distance = np.linalg.norm(direction)
    if distance < WAYPOINT_ARRIVAL_THRESHOLD:
        return GroundVehicle(
            id=vehicle.id, vehicle_type=vehicle.vehicle_type,
            x=vehicle.x, y=vehicle.y, speed=vehicle.speed,
            heading=vehicle.heading, waypoints=vehicle.waypoints,
            current_waypoint_idx=vehicle.current_waypoint_idx + 1,
        )
    new_heading = np.arctan2(direction[1], direction[0])
    new_x = vehicle.x + vehicle.speed * np.cos(new_heading) * dt
    new_y = vehicle.y + vehicle.speed * np.sin(new_heading) * dt
    return GroundVehicle(
        id=vehicle.id, vehicle_type=vehicle.vehicle_type,
        x=new_x, y=new_y, speed=vehicle.speed,
        heading=new_heading, waypoints=vehicle.waypoints,
        current_waypoint_idx=vehicle.current_waypoint_idx,
    )


def project_position(x, y, vx, vy, dt):
    return x + vx * dt, y + vy * dt