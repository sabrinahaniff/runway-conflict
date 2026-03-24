from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class AircraftPhase(Enum):
    FINAL_APPROACH = "final_approach"
    FLARE = "flare"
    ROLLOUT = "rollout"
    VACATING = "vacating"
    CLEAR = "clear"


class VehicleType(Enum):
    FIRE_TRUCK = "fire_truck"
    MAINTENANCE = "maintenance"
    FOLLOW_ME = "follow_me"
    FUEL_TRUCK = "fuel_truck"


@dataclass
class Aircraft:
    id: str
    x: float
    y: float
    speed: float
    lateral_speed: float = 0.0
    altitude: float = 0.0
    phase: AircraftPhase = AircraftPhase.FINAL_APPROACH
    decel_rate: float = 1.8
    approach_speed: float = 72.0
    touchdown_speed: float = 67.0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.speed, self.lateral_speed])

    @property
    def is_on_runway(self) -> bool:
        return self.phase in (AircraftPhase.FLARE, AircraftPhase.ROLLOUT)


@dataclass
class GroundVehicle:
    id: str
    vehicle_type: VehicleType
    x: float
    y: float
    speed: float
    heading: float
    waypoints: list = field(default_factory=list)
    current_waypoint_idx: int = 0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([
            self.speed * np.cos(self.heading),
            self.speed * np.sin(self.heading)
        ])

    @property
    def has_waypoints(self) -> bool:
        return self.current_waypoint_idx < len(self.waypoints)

    @property
    def current_target(self):
        if self.has_waypoints:
            return np.array(self.waypoints[self.current_waypoint_idx])
        return None


@dataclass
class Runway:
    id: str = "RW24L"
    length: float = 3200.0
    width: float = 46.0
    touchdown_zone_end: float = 900.0
    rollout_start: float = 0.0
    rollout_end: float = 3200.0

    @property
    def half_width(self) -> float:
        return self.width / 2.0

    def is_on_runway(self, x: float, y: float) -> bool:
        return (
            self.rollout_start <= x <= self.rollout_end
            and -self.half_width <= y <= self.half_width
        )

    def distance_to_centerline(self, y: float) -> float:
        return abs(y)

    def distance_to_threshold(self, x: float) -> float:
        return x