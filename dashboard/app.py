import sys
import math
import requests
import numpy as np
from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go

app = Dash(__name__)

API_URL = "http://127.0.0.1:8000"

# runway dimensions (KLAX 24L)
RUNWAY_LENGTH = 3685.0
RUNWAY_WIDTH = 61.0
RUNWAY_HALF_W = RUNWAY_WIDTH / 2

# simulation state (moves entities each tick)
sim = {
    "aircraft": {"x": -3000.0, "y": 5.0, "speed": 72.0,
                 "altitude": 280.0, "phase": "final_approach",
                 "lateral_speed": -0.5, "heading": 0.0},
    "vehicle":  {"x": -50.0, "y": -(RUNWAY_HALF_W + 60),
                 "speed": 7.0, "heading": math.pi / 2},
    "tick": 0,
    "scenario": "crossing",
}

def _btn_style(bg="#2a2a4a"):
    return {
        "background": bg, "color": "#fff", "border": "none",
        "padding": "8px 14px", "borderRadius": "6px",
        "cursor": "pointer", "fontFamily": "monospace",
        "fontSize": "13px",
    }

# layout

app.layout = html.Div([

    html.Div([
        html.H2("Runway Conflict Detection",
                style={"margin": "0", "color": "#fff",
                       "fontFamily": "monospace", "fontSize": "20px"}),
        html.Span("LIVE", style={
            "background": "#34C759", "color": "#fff",
            "padding": "2px 10px", "borderRadius": "4px",
            "fontSize": "12px", "fontWeight": "bold",
            "fontFamily": "monospace",
        }),
    ], style={
        "background": "#1a1a2e", "padding": "14px 24px",
        "display": "flex", "alignItems": "center", "gap": "16px",
    }),

    # alert banner
    html.Div(id="alert-banner", style={
        "padding": "14px 24px", "fontSize": "16px",
        "fontFamily": "monospace", "fontWeight": "bold",
        "textAlign": "center", "transition": "background 0.3s",
    }),

    # main content
    html.Div([

        # left — runway view
        html.Div([
            dcc.Graph(id="runway-plot",
                      style={"height": "520px"},
                      config={"displayModeBar": False}),
        ], style={"flex": "2"}),

        # right — stats panel
        html.Div([
            html.Div(id="risk-gauge"),
            html.Div(id="stats-panel", style={"marginTop": "20px"}),
            html.Div([
                html.Button("Crossing scenario", id="btn-crossing", n_clicks=0,
                            style=_btn_style()),
                html.Button("Safe scenario", id="btn-safe", n_clicks=0,
                            style=_btn_style()),
                html.Button("Reset", id="btn-reset", n_clicks=0,
                            style=_btn_style("#555")),
            ], style={"marginTop": "20px", "display": "flex",
                      "flexDirection": "column", "gap": "8px"}),
        ], style={
            "flex": "1", "padding": "20px",
            "background": "#0f0f1a", "borderRadius": "8px",
            "margin": "0 0 0 16px",
        }),

    ], style={
        "display": "flex", "padding": "16px 24px",
        "background": "#12122a", "minHeight": "560px",
    }),

    # ticker
    dcc.Interval(id="ticker", interval=500, n_intervals=0),

    # hidden stores
    dcc.Store(id="scenario-store", data="crossing"),
    dcc.Store(id="prediction-store", data={}),

], style={"background": "#12122a", "minHeight": "100vh"})


def _btn_style(bg="#2a2a4a"):
    return {
        "background": bg, "color": "#fff", "border": "none",
        "padding": "8px 14px", "borderRadius": "6px",
        "cursor": "pointer", "fontFamily": "monospace",
        "fontSize": "13px",
    }


# scenario control

@callback(
    Output("scenario-store", "data"),
    Input("btn-crossing", "n_clicks"),
    Input("btn-safe", "n_clicks"),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True,
)
def set_scenario(cross, safe, reset):
    from dash import ctx
    btn = ctx.triggered_id
    if btn == "btn-crossing":
        _init_crossing()
        return "crossing"
    elif btn == "btn-safe":
        _init_safe()
        return "safe"
    else:
        _init_crossing()
        return "crossing"


def _init_crossing():
    sim["aircraft"] = {"x": -3000.0, "y": 5.0, "speed": 72.0,
                       "altitude": 280.0, "phase": "final_approach",
                       "lateral_speed": -0.5, "heading": 0.0}
    sim["vehicle"] = {"x": 300.0, "y": -(RUNWAY_HALF_W + 60),
                      "speed": 7.0, "heading": math.pi / 2}
    sim["tick"] = 0
    sim["scenario"] = "crossing"


def _init_safe():
    sim["aircraft"] = {"x": -3000.0, "y": 5.0, "speed": 72.0,
                       "altitude": 280.0, "phase": "final_approach",
                       "lateral_speed": -0.5, "heading": 0.0}
    sim["vehicle"] = {"x": 1800.0, "y": RUNWAY_HALF_W + 150,
                      "speed": 4.0, "heading": 0.0}
    sim["tick"] = 0
    sim["scenario"] = "safe"


#main tick callback

@callback(
    Output("runway-plot", "figure"),
    Output("alert-banner", "children"),
    Output("alert-banner", "style"),
    Output("risk-gauge", "children"),
    Output("stats-panel", "children"),
    Output("prediction-store", "data"),
    Input("ticker", "n_intervals"),
    Input("scenario-store", "data"),
)
def update(n, scenario):
    # advance simulation
    _tick_simulation()

    # send positions to API
    ac = sim["aircraft"]
    gv = sim["vehicle"]

    try:
        requests.post(f"{API_URL}/update/aircraft", json={
            "id": "AC001", "x": ac["x"], "y": ac["y"],
            "speed": ac["speed"], "heading": ac["heading"],
            "altitude": ac["altitude"], "phase": ac["phase"],
            "lateral_speed": ac["lateral_speed"],
        }, timeout=0.5)

        requests.post(f"{API_URL}/update/vehicle", json={
            "id": "GV001", "x": gv["x"], "y": gv["y"],
            "speed": gv["speed"], "heading": gv["heading"],
        }, timeout=0.5)

        resp = requests.get(f"{API_URL}/predict", timeout=0.5)
        pred_data = resp.json()
        alert_info = pred_data["alerts"][0]["alert"] if pred_data["alerts"] else None
        pred_info = pred_data["alerts"][0]["prediction"] if pred_data["alerts"] else None
    except Exception:
        alert_info = {"risk_level": "safe", "color": "#34C759",
                      "message": "API not reachable", "should_alarm": False}
        pred_info = None

    # build runway figure
    fig = _build_figure(ac, gv, pred_info)

    # alert banner
    color = alert_info["color"] if alert_info else "#34C759"
    msg = alert_info["message"] if alert_info else "No data"
    banner_style = {
        "padding": "14px 24px", "fontSize": "16px",
        "fontFamily": "monospace", "fontWeight": "bold",
        "textAlign": "center", "background": color,
        "color": "#fff", "transition": "background 0.3s",
    }

    # risk gauge
    level = alert_info["risk_level"] if alert_info else "safe"
    score = pred_info["risk_score"] if pred_info else 0.0
    gauge = _build_gauge(level, score)

    # stats panel
    stats = _build_stats(ac, gv, pred_info)

    return fig, msg, banner_style, gauge, stats, pred_info or {}


# simulation physics

def _tick_simulation():
    dt = 0.5
    ac = sim["aircraft"]
    gv = sim["vehicle"]

    # Aircraft movement
    if ac["phase"] == "final_approach":
        ac["x"] += ac["speed"] * dt
        ac["y"] += ac["lateral_speed"] * dt
        ac["altitude"] = max(0, -ac["x"] * math.tan(math.radians(3)))
        if ac["x"] >= -150:
            ac["phase"] = "flare"

    elif ac["phase"] == "flare":
        ac["x"] += ac["speed"] * dt
        ac["altitude"] = max(0, ac["altitude"] * 0.85)
        ac["y"] *= 0.98
        if ac["x"] >= 0:
            ac["phase"] = "rollout"

    elif ac["phase"] == "rollout":
        ac["speed"] = max(15, ac["speed"] - 1.8 * dt)
        ac["x"] += ac["speed"] * dt
        ac["y"] *= 0.99
        if ac["speed"] <= 16 and ac["x"] > RUNWAY_LENGTH * 0.4:
            ac["phase"] = "vacating"

    elif ac["phase"] == "vacating":
        ac["speed"] = max(5, ac["speed"] - 1.0 * dt)
        ac["x"] += ac["speed"] * dt
        ac["y"] += 2 * dt
        if ac["y"] > RUNWAY_HALF_W + 30:
            ac["phase"] = "clear"
            _reset_aircraft()

    # Vehicle movement
    gv["x"] += gv["speed"] * math.cos(gv["heading"]) * dt
    gv["y"] += gv["speed"] * math.sin(gv["heading"]) * dt

    # Wrap vehicle if it goes too far
    if abs(gv["y"]) > RUNWAY_HALF_W + 120:
        if sim["scenario"] == "crossing":
            gv["heading"] = -gv["heading"]
        else:
            gv["x"] = min(gv["x"] + 5 * dt, RUNWAY_LENGTH - 100)

    sim["tick"] += 1


def _reset_aircraft():
    sim["aircraft"] = {
        "x": -3000.0, "y": 5.0, "speed": 72.0,
        "altitude": 280.0, "phase": "final_approach",
        "lateral_speed": -0.5, "heading": 0.0,
    }


# figure builder

def _build_figure(ac, gv, pred):
    fig = go.Figure()

    # runway surface
    fig.add_shape(type="rect",
                  x0=0, x1=RUNWAY_LENGTH,
                  y0=-RUNWAY_HALF_W, y1=RUNWAY_HALF_W,
                  fillcolor="#2a2a2a", line=dict(color="#555", width=1))

    # runway centerline
    fig.add_shape(type="line",
                  x0=0, x1=RUNWAY_LENGTH, y0=0, y1=0,
                  line=dict(color="#fff", width=1, dash="dash"))

    # threshold markers
    for rx in [0, RUNWAY_LENGTH]:
        fig.add_shape(type="line",
                      x0=rx, x1=rx,
                      y0=-RUNWAY_HALF_W, y1=RUNWAY_HALF_W,
                      line=dict(color="#ffcc00", width=2))

    # touchdown zone
    fig.add_shape(type="rect",
                  x0=0, x1=900,
                  y0=-RUNWAY_HALF_W, y1=RUNWAY_HALF_W,
                  fillcolor="rgba(255,200,0,0.06)",
                  line=dict(color="rgba(255,200,0,0.2)", width=1))

    # aircraft projected path (dashed line)
    proj_x = ac["x"] + ac["speed"] * 30
    fig.add_shape(type="line",
                  x0=ac["x"], x1=proj_x,
                  y0=ac["y"], y1=ac["y"],
                  line=dict(color="#4fc3f7", width=1, dash="dot"))

    # vehicle projected path
    vproj_x = gv["x"] + gv["speed"] * math.cos(gv["heading"]) * 20
    vproj_y = gv["y"] + gv["speed"] * math.sin(gv["heading"]) * 20
    fig.add_shape(type="line",
                  x0=gv["x"], x1=vproj_x,
                  y0=gv["y"], y1=vproj_y,
                  line=dict(color="#ff9800", width=1, dash="dot"))

    # risk color
    risk_color = "#FF3B30" if (pred and pred.get("risk_level") == "high_risk") \
        else "#FF9500" if (pred and pred.get("risk_level") == "warning") \
        else "#34C759"

    # aircraft marker
    fig.add_trace(go.Scatter(
        x=[ac["x"]], y=[ac["y"]],
        mode="markers+text",
        marker=dict(symbol="triangle-right", size=18,
                    color="#4fc3f7", line=dict(color="#fff", width=1)),
        text=["AC001"], textposition="top center",
        textfont=dict(color="#4fc3f7", size=11),
        name="Aircraft", showlegend=False,
    ))

    # vehicle marker
    fig.add_trace(go.Scatter(
        x=[gv["x"]], y=[gv["y"]],
        mode="markers+text",
        marker=dict(symbol="square", size=14,
                    color=risk_color, line=dict(color="#fff", width=1)),
        text=["GV001"], textposition="top center",
        textfont=dict(color=risk_color, size=11),
        name="Vehicle", showlegend=False,
    ))

    # cpa point if warning/high_risk
    if pred and pred.get("risk_level") in ("warning", "high_risk"):
        cpa_x = ac["x"] + ac["speed"] * pred.get("time_to_cpa_s", 0)
        fig.add_trace(go.Scatter(
            x=[cpa_x], y=[0],
            mode="markers",
            marker=dict(symbol="x", size=14,
                        color="#FF3B30", line=dict(width=2)),
            name="CPA", showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor="#12122a",
        plot_bgcolor="#1a1a2e",
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(
            range=[-500, RUNWAY_LENGTH + 200],
            showgrid=False, zeroline=False,
            tickfont=dict(color="#888", size=10),
            title=dict(text="meters along runway", font=dict(color="#888")),
        ),
        yaxis=dict(
            range=[-120, 120],
            showgrid=False, zeroline=False,
            tickfont=dict(color="#888", size=10),
            scaleanchor="x", scaleratio=1,
        ),
        title=dict(
            text="KLAX Runway 24L — Live View",
            font=dict(color="#aaa", size=13, family="monospace"),
            x=0.5,
        ),
    )

    return fig


#gauge and stats

def _build_gauge(level, score):
    color = {"high_risk": "#FF3B30",
             "warning": "#FF9500",
             "safe": "#34C759"}.get(level, "#34C759")
    label = {"high_risk": "HIGH RISK",
             "warning": "WARNING",
             "safe": "SAFE"}.get(level, "SAFE")

    return html.Div([
        html.Div(label, style={
            "background": color, "color": "#fff",
            "padding": "16px", "borderRadius": "8px",
            "textAlign": "center", "fontSize": "22px",
            "fontWeight": "bold", "fontFamily": "monospace",
            "letterSpacing": "2px",
        }),
        html.Div([
            html.Div(style={
                "width": f"{score * 100:.0f}%",
                "height": "8px",
                "background": color,
                "borderRadius": "4px",
                "transition": "width 0.4s",
            }),
        ], style={
            "background": "#2a2a4a", "borderRadius": "4px",
            "marginTop": "10px", "overflow": "hidden",
        }),
        html.Div(f"Confidence: {score:.1%}", style={
            "color": "#888", "fontSize": "11px",
            "fontFamily": "monospace", "marginTop": "4px",
        }),
    ])


def _build_stats(ac, gv, pred):
    def row(label, value, color="#ccc"):
        return html.Div([
            html.Span(label, style={"color": "#666",
                                    "fontFamily": "monospace",
                                    "fontSize": "12px"}),
            html.Span(value, style={"color": color,
                                    "fontFamily": "monospace",
                                    "fontSize": "12px",
                                    "float": "right"}),
        ], style={"borderBottom": "1px solid #1a1a3a",
                  "padding": "6px 0"})

    phase_colors = {
        "final_approach": "#4fc3f7",
        "flare": "#ffcc00",
        "rollout": "#ff9800",
        "vacating": "#ab47bc",
        "clear": "#34C759",
    }

    items = [
        html.Div("AIRCRAFT", style={"color": "#4fc3f7",
                                     "fontFamily": "monospace",
                                     "fontSize": "11px",
                                     "marginBottom": "4px"}),
        row("Position X", f"{ac['x']:.0f}m"),
        row("Altitude", f"{ac['altitude']:.0f}m"),
        row("Speed", f"{ac['speed']:.1f} m/s"),
        row("Phase", ac["phase"],
            phase_colors.get(ac["phase"], "#ccc")),
        html.Div("VEHICLE", style={"color": "#ff9800",
                                    "fontFamily": "monospace",
                                    "fontSize": "11px",
                                    "marginTop": "12px",
                                    "marginBottom": "4px"}),
        row("Position X", f"{gv['x']:.0f}m"),
        row("Position Y", f"{gv['y']:.0f}m"),
        row("Speed", f"{gv['speed']:.1f} m/s"),
    ]

    if pred:
        items += [
            html.Div("PREDICTION", style={"color": "#888",
                                           "fontFamily": "monospace",
                                           "fontSize": "11px",
                                           "marginTop": "12px",
                                           "marginBottom": "4px"}),
            row("Separation", f"{pred.get('current_separation_m', 0):.0f}m"),
            row("CPA distance", f"{pred.get('cpa_distance_m', 0):.0f}m"),
            row("Time to CPA", f"{pred.get('time_to_cpa_s', 0):.1f}s"),
        ]

    return html.Div(items)


if __name__ == "__main__":
    app.run(debug=True, port=8050)