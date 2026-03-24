import numpy as np
from dataclasses import dataclass

EARTH_RADIUS_M = 6_371_000.0


@dataclass
class LocalPosition:
    x: float
    y: float
    altitude: float


def latlon_to_local(lat, lon, altitude_m, runway, ground_elevation_m=0.0):
    d_lat = np.radians(lat - runway.threshold_lat)
    d_lon = np.radians(lon - runway.threshold_lon)
    cos_lat = np.cos(np.radians(runway.threshold_lat))
    north_m = d_lat * EARTH_RADIUS_M
    east_m = d_lon * EARTH_RADIUS_M * cos_lat
    heading_rad = np.radians(runway.true_heading)
    x = north_m * np.cos(heading_rad) + east_m * np.sin(heading_rad)
    y = -north_m * np.sin(heading_rad) + east_m * np.cos(heading_rad)
    altitude_agl = max(0.0, altitude_m - ground_elevation_m)
    return LocalPosition(x=x, y=y, altitude=altitude_agl)


def compute_local_velocity(positions, timestamps):
    if len(positions) < 2:
        return 0.0, 0.0
    dt = timestamps[-1] - timestamps[-2]
    if dt < 1e-6:
        return 0.0, 0.0
    vx = (positions[-1].x - positions[-2].x) / dt
    vy = (positions[-1].y - positions[-2].y) / dt
    return vx, vy


def compute_speed_ms(vx, vy):
    return float(np.sqrt(vx**2 + vy**2))


def flight_phase_from_position(x, altitude, speed, runway_length):
    if x < -150.0:
        return "final_approach"
    elif x < 0.0:
        return "flare"
    elif 0.0 <= x <= runway_length and altitude < 5.0:
        if speed < 20.0:
            return "vacating"
        return "rollout"
    else:
        return "clear"