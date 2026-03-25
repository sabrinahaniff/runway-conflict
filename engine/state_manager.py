import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EntityState:
    id: str
    x: float
    y: float
    speed: float
    heading: float
    altitude: float = 0.0
    phase: str = "final_approach"
    lateral_speed: float = 0.0
    last_updated: float = field(default_factory=time.time)


class StateManager:
    """
    Holds the current position of every aircraft and vehicle.
    Updated every tick. Thread-safe for FastAPI.
    """

    def __init__(self):
        self.aircraft: Dict[str, EntityState] = {}
        self.vehicles: Dict[str, EntityState] = {}
        self._lock = False

    def update_aircraft(self, id: str, x: float, y: float,
                        speed: float, heading: float,
                        altitude: float = 0.0, phase: str = "final_approach",
                        lateral_speed: float = 0.0):
        self.aircraft[id] = EntityState(
            id=id, x=x, y=y, speed=speed, heading=heading,
            altitude=altitude, phase=phase, lateral_speed=lateral_speed,
        )

    def update_vehicle(self, id: str, x: float, y: float,
                       speed: float, heading: float):
        self.vehicles[id] = EntityState(
            id=id, x=x, y=y, speed=speed, heading=heading,
        )

    def get_all_pairs(self):
        """Return every (aircraft, vehicle) pair for conflict checking."""
        pairs = []
        for ac in self.aircraft.values():
            for gv in self.vehicles.values():
                pairs.append((ac, gv))
        return pairs

    def clear(self):
        self.aircraft.clear()
        self.vehicles.clear()