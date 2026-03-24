from dataclasses import dataclass
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from simulator.entities import Runway


@dataclass
class RunwayDefinition:
    airport_icao: str
    runway_id: str
    threshold_lat: float
    threshold_lon: float
    true_heading: float
    length_m: float
    width_m: float

    def to_runway(self) -> Runway:
        return Runway(
            id=f"{self.airport_icao}_{self.runway_id}",
            length=self.length_m,
            width=self.width_m,
        )


RUNWAYS = {
    "KLAX_24L": RunwayDefinition("KLAX", "24L", 33.9425, -118.4081, 249.0, 3685.0, 61.0),
    "KLAX_24R": RunwayDefinition("KLAX", "24R", 33.9286, -118.3953, 249.0, 3382.0, 61.0),
    "EGLL_27L": RunwayDefinition("EGLL", "27L", 51.4775, -0.4614,   270.0, 3902.0, 61.0),
    "EGLL_27R": RunwayDefinition("EGLL", "27R", 51.4641, -0.4328,   270.0, 3660.0, 61.0),
    "KJFK_22L": RunwayDefinition("KJFK", "22L", 40.6268, -73.7799,  224.0, 4423.0, 61.0),
    "KORD_10L": RunwayDefinition("KORD", "10L", 41.9958, -87.9344,  100.0, 3963.0, 61.0),
}


def get_runway(airport_icao: str, runway_id: str) -> RunwayDefinition:
    key = f"{airport_icao}_{runway_id}"
    if key not in RUNWAYS:
        raise ValueError(f"Runway {key} not found. Available: {list(RUNWAYS.keys())}")
    return RUNWAYS[key]