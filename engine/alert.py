from dataclasses import dataclass


THRESHOLDS = {
    "high_risk": 0.5,
    "warning": 0.3,
}

COLORS = {
    "high_risk": "#FF3B30",
    "warning":   "#FF9500",
    "safe":      "#34C759",
}


@dataclass
class Alert:
    risk_level: str
    risk_score: float
    color: str
    message: str
    should_alarm: bool


def make_alert(prediction: dict) -> Alert:
    level = prediction["risk_level"]
    score = prediction["risk_score"]
    ac = prediction["aircraft_id"]
    gv = prediction["vehicle_id"]
    cpa = prediction["cpa_distance_m"]
    t = prediction["time_to_cpa_s"]

    if level == "high_risk":
        return Alert(
            risk_level="high_risk",
            risk_score=score,
            color=COLORS["high_risk"],
            message=f"HIGH RISK — {ac} and {gv} — CPA {cpa}m in {t}s",
            should_alarm=True,
        )
    elif level == "warning":
        return Alert(
            risk_level="warning",
            risk_score=score,
            color=COLORS["warning"],
            message=f"WARNING — {ac} and {gv} — CPA {cpa}m in {t}s",
            should_alarm=False,
        )
    else:
        return Alert(
            risk_level="safe",
            risk_score=score,
            color=COLORS["safe"],
            message=f"SAFE — {ac} and {gv} — separation {prediction['current_separation_m']}m",
            should_alarm=False,
        )