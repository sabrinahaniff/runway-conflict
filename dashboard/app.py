import math
import random
import requests
from dash import Dash, dcc, html, Input, Output, callback, ctx
import plotly.graph_objects as go

app = Dash(__name__, title="Runway Conflict Monitor")
API_URL = "http://127.0.0.1:8000"

RUNWAY_LENGTH = 3685.0
RUNWAY_HALF_W = 30.5
RUNWAYS = [
    {"id": "24L", "y_center": 0,    "label": "Runway 24L"},
    {"id": "24R", "y_center": -420, "label": "Runway 24R"},
]
VTYPES = [
    {"type": "fire_truck",  "color": "#ff5252"},
    {"type": "maintenance", "color": "#ffab40"},
    {"type": "fuel_truck",  "color": "#ce93d8"},
    {"type": "follow_me",   "color": "#80cbc4"},
]

def make_ac(callsign, rwy_y, xoff=0, spd=0):
    x = -3500 + xoff
    return {"id": callsign, "x": x, "y": rwy_y,
            "speed": 72+spd, "lateral_speed": random.uniform(-0.4,0.4),
            "altitude": max(0, abs(x)*math.tan(math.radians(3))),
            "phase": "final_approach", "heading": 0.0, "runway": rwy_y}

def make_gv(vid, x, y, speed, heading, ti):
    vt = VTYPES[ti % len(VTYPES)]
    return {"id": vid, "x": x, "y": y, "speed": speed,
            "heading": heading, "type": vt["type"], "color": vt["color"]}

def init_cross():
    return {
        "aircraft": [
            make_ac("UAL232", 0,    xoff=0,     spd=0),
            make_ac("AAL456", 0,    xoff=-1800, spd=3),
            make_ac("DAL789", -420, xoff=0,     spd=-2),
            make_ac("SWA101", -420, xoff=-2000, spd=1),
        ],
        "vehicles": [
            make_gv("GV001", 300,  -RUNWAY_HALF_W-40,      7.0, math.pi/2,   0),
            make_gv("GV002", 900,   RUNWAY_HALF_W+40,      5.0, -math.pi/2,  1),
            make_gv("GV003", 200,  -420-RUNWAY_HALF_W,     6.0, math.pi/2,   0),
            make_gv("GV004", 1500,  RUNWAY_HALF_W+30,      4.0, math.pi,     2),
            make_gv("GV005", 2800, -420+RUNWAY_HALF_W,     3.0, -math.pi/2,  3),
            make_gv("GV006", 600,  -420-RUNWAY_HALF_W-20,  5.5, math.pi/2,   1),
        ],
        "history": [], "scenario": "crossing", "tick": 0,
    }

def init_safe():
    return {
        "aircraft": [
            make_ac("UAL890", 0,    xoff=0,     spd=0),
            make_ac("SKW342", 0,    xoff=-1800, spd=2),
            make_ac("ASA567", -420, xoff=0,     spd=-1),
        ],
        "vehicles": [
            make_gv("GV001", 3000,  RUNWAY_HALF_W+80,       3.0, 0,       2),
            make_gv("GV002", 3200, -420-RUNWAY_HALF_W-60,   2.0, math.pi, 3),
            make_gv("GV003", 2800,  RUNWAY_HALF_W+100,      2.0, 0,       1),
            make_gv("GV004", 3100, -420+RUNWAY_HALF_W+80,   3.0, math.pi, 0),
        ],
        "history": [], "scenario": "safe", "tick": 0,
    }

sim = init_cross()

RISK_COLOR = {"high_risk": "#ff5252", "warning": "#ffab40", "safe": "#69f0ae"}
RISK_BG    = {"high_risk": "rgba(255,82,82,0.08)", "warning": "rgba(255,171,64,0.08)", "safe": "rgba(105,240,174,0.05)"}
RISK_LABEL = {"high_risk": "HIGH RISK", "warning": "WARNING", "safe": "ALL SAFE"}

def hbtn(bg, border):
    return {"background": bg, "color": "#ccc", "border": f"1px solid {border}",
            "padding": "6px 12px", "borderRadius": "5px", "cursor": "pointer",
            "fontFamily": "monospace", "fontSize": "11px"}

def leg(sym, color, label):
    return html.Div([
        html.Span(sym,   style={"color": color, "marginRight": "8px", "fontSize": "14px"}),
        html.Span(label, style={"color": "#555", "fontFamily": "monospace", "fontSize": "11px"}),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "4px"})

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Span("✈ ", style={"fontSize": "20px"}),
            html.Span("RUNWAY CONFLICT MONITOR",
                      style={"fontFamily": "monospace", "fontSize": "16px",
                             "fontWeight": "bold", "letterSpacing": "3px", "color": "#e0e0e0"}),
        ]),
        html.Div([
            html.Span("● LIVE  ", style={"color": "#69f0ae", "fontFamily": "monospace", "fontSize": "12px"}),
            html.Span("Los Angeles International · KLAX",
                      style={"color": "#666", "fontFamily": "monospace", "fontSize": "12px"}),
        ]),
        html.Div([
            html.Button("⚠ Conflict", id="btn-cross", n_clicks=0, style=hbtn("#3a1010","#ff5252")),
            html.Button("✓ Safe",     id="btn-safe",  n_clicks=0, style=hbtn("#0a2010","#69f0ae")),
            html.Button("↺ Reset",    id="btn-reset", n_clicks=0, style=hbtn("#1a1a2a","#555")),
        ], style={"display":"flex","gap":"8px"}),
    ], style={"background":"#080818","padding":"12px 24px","display":"flex",
              "justifyContent":"space-between","alignItems":"center",
              "borderBottom":"1px solid #1a1a3a"}),

    html.Div(id="alert-bar"),

    html.Div([
        html.Div([
            dcc.Graph(id="airport-map", style={"height":"480px"},
                      config={"displayModeBar":False,"scrollZoom":True}),
            html.Div([
                html.Div("RISK TIMELINE",
                         style={"color":"#444","fontFamily":"monospace","fontSize":"10px","marginBottom":"4px"}),
                dcc.Graph(id="timeline", style={"height":"70px"},
                          config={"displayModeBar":False}),
            ], style={"background":"#080818","borderTop":"1px solid #111","padding":"8px 12px"}),
        ], style={"flex":"2.2","background":"#0a0a1a","borderRadius":"8px","overflow":"hidden"}),

        html.Div([
            html.Div(id="status-panel", style={"marginBottom":"10px"}),
            html.Div([
                html.Div("LEGEND", style={"color":"#444","fontFamily":"monospace",
                                           "fontSize":"10px","marginBottom":"8px","letterSpacing":"1px"}),
                leg("▶","#4fc3f7","Aircraft on approach"),
                leg("■","#ff5252","Vehicle — HIGH RISK"),
                leg("■","#ffab40","Vehicle — WARNING"),
                leg("■","#69f0ae","Vehicle — SAFE"),
                leg("○","#ff5252","Projected collision zone"),
                html.Div(style={"height":"1px","background":"#111","margin":"8px 0"}),
                html.Div("Dashed lines = next 20 seconds",
                         style={"color":"#333","fontSize":"10px","fontFamily":"monospace"}),
            ], style={"background":"#0a0a1a","padding":"12px","borderRadius":"6px",
                      "marginBottom":"10px","border":"1px solid #1a1a3a"}),
            html.Div([
                html.Div("ALL MONITORED PAIRS",
                         style={"color":"#444","fontFamily":"monospace",
                                "fontSize":"10px","marginBottom":"8px","letterSpacing":"1px"}),
                html.Div(id="pairs-table"),
            ], style={"background":"#0a0a1a","padding":"12px","borderRadius":"6px",
                      "border":"1px solid #1a1a3a","flex":"1","overflowY":"auto"}),
        ], style={"flex":"1","marginLeft":"10px","display":"flex","flexDirection":"column","maxHeight":"560px"}),
    ], style={"display":"flex","padding":"10px 12px","gap":"10px","background":"#060614"}),

    dcc.Interval(id="tick", interval=400, n_intervals=0),
    dcc.Store(id="sc-store", data="crossing"),
], style={"background":"#060614","minHeight":"100vh"})


@callback(Output("sc-store","data"),
          Input("btn-cross","n_clicks"), Input("btn-safe","n_clicks"), Input("btn-reset","n_clicks"),
          prevent_initial_call=True)
def switch(c1,c2,c3):
    if ctx.triggered_id == "btn-safe":
        sim.update(init_safe()); return "safe"
    else:
        sim.update(init_cross()); return "crossing"


@callback(
    Output("airport-map","figure"), Output("timeline","figure"),
    Output("alert-bar","children"), Output("status-panel","children"),
    Output("pairs-table","children"),
    Input("tick","n_intervals"), Input("sc-store","data"),
)
def update(n, sc):
    advance()
    preds = fetch()
    order = {"high_risk":2,"warning":1,"safe":0}
    worst = max(preds, key=lambda p: order.get(p["prediction"]["risk_level"],0), default=None)
    wlvl  = worst["prediction"]["risk_level"] if worst else "safe"
    sim["history"].append({"safe":0.05,"warning":0.5,"high_risk":1.0}[wlvl])
    if len(sim["history"]) > 120: sim["history"].pop(0)
    return draw_map(preds), draw_timeline(), draw_alert(wlvl,worst), draw_status(wlvl,preds), draw_table(preds)


def advance():
    dt = 0.4
    for ac in sim["aircraft"]:
        ry = ac["runway"]
        ph = ac["phase"]
        if ph == "final_approach":
            ac["x"] += ac["speed"]*dt
            ac["y"] += ac.get("lateral_speed",0)*dt
            ac["altitude"] = max(0, -ac["x"]*math.tan(math.radians(3)))
            ac["y"] += (ry - ac["y"])*0.05
            if ac["x"] >= -150: ac["phase"] = "flare"
        elif ph == "flare":
            ac["x"] += ac["speed"]*dt
            ac["altitude"] = max(0, ac["altitude"]*0.80)
            ac["y"] += (ry - ac["y"])*0.15
            if ac["x"] >= 0: ac["phase"] = "rollout"; ac["altitude"] = 0
        elif ph == "rollout":
            ac["speed"] = max(15, ac["speed"]-1.8*dt)
            ac["x"] += ac["speed"]*dt
            ac["y"] += (ry - ac["y"])*0.2
            if ac["speed"] <= 16 and ac["x"] > RUNWAY_LENGTH*0.4: ac["phase"] = "vacating"
        elif ph == "vacating":
            ac["speed"] = max(5, ac["speed"]-1.0*dt)
            ac["x"] += ac["speed"]*dt
            ac["y"] += (ry+RUNWAY_HALF_W+50-ac["y"])*0.05
            if ac["x"] > RUNWAY_LENGTH+500:
                ac["x"] = -3500+random.uniform(-200,200)
                ac["y"] = ry+random.uniform(-5,5)
                ac["speed"] = 72+random.uniform(-4,4)
                ac["altitude"] = abs(ac["x"])*math.tan(math.radians(3))
                ac["phase"] = "final_approach"
                ac["lateral_speed"] = random.uniform(-0.4,0.4)
    for gv in sim["vehicles"]:
        gv["x"] += gv["speed"]*math.cos(gv["heading"])*dt
        gv["y"] += gv["speed"]*math.sin(gv["heading"])*dt
        if abs(gv["y"]) > 550: gv["heading"] = math.pi - gv["heading"]
        if gv["x"] > RUNWAY_LENGTH+200: gv["x"] = -200.0
        elif gv["x"] < -200: gv["x"] = RUNWAY_LENGTH+200.0
    sim["tick"] += 1


def fetch():
    try:
        requests.post(f"{API_URL}/reset", headers={"Content-Type":"application/json"}, timeout=0.3)
        for ac in sim["aircraft"]:
            requests.post(f"{API_URL}/update/aircraft", json={
                "id":ac["id"],"x":ac["x"],"y":ac["y"],"speed":ac["speed"],
                "heading":ac["heading"],"altitude":ac["altitude"],
                "phase":ac["phase"],"lateral_speed":ac.get("lateral_speed",0),
            }, timeout=0.3)
        for gv in sim["vehicles"]:
            requests.post(f"{API_URL}/update/vehicle", json={
                "id":gv["id"],"x":gv["x"],"y":gv["y"],"speed":gv["speed"],"heading":gv["heading"],
            }, timeout=0.3)
        return requests.get(f"{API_URL}/predict", timeout=0.5).json().get("alerts",[])
    except:
        return []


def draw_map(preds):
    fig = go.Figure()

    # Background
    fig.add_shape(type="rect", x0=-600, x1=RUNWAY_LENGTH+500, y0=-620, y1=200,
                  fillcolor="#080810", line=dict(width=0))

    # Runways
    for rwy in RUNWAYS:
        cy = rwy["y_center"]
        fig.add_shape(type="rect", x0=0, x1=RUNWAY_LENGTH,
                      y0=cy-RUNWAY_HALF_W, y1=cy+RUNWAY_HALF_W,
                      fillcolor="#1e1e1e", line=dict(color="#333",width=1))
        for i in range(0, int(RUNWAY_LENGTH), 150):
            fig.add_shape(type="line", x0=i, x1=i+80, y0=cy, y1=cy,
                          line=dict(color="rgba(255,255,255,0.13)",width=2))
        for tx in [0, RUNWAY_LENGTH]:
            fig.add_shape(type="rect", x0=tx-8, x1=tx+8,
                          y0=cy-RUNWAY_HALF_W, y1=cy+RUNWAY_HALF_W,
                          fillcolor="#ffcc00", line=dict(width=0))
        fig.add_shape(type="rect", x0=0, x1=900,
                      y0=cy-RUNWAY_HALF_W, y1=cy+RUNWAY_HALF_W,
                      fillcolor="rgba(255,204,0,0.04)",
                      line=dict(color="rgba(255,204,0,0.1)",width=1))
        fig.add_annotation(x=RUNWAY_LENGTH+60, y=cy, text=rwy["label"],
                           font=dict(size=10,color="#555",family="monospace"),
                           showarrow=False, xanchor="left")
        for sign in [-1,1]:
            fig.add_shape(type="line", x0=-600, x1=0,
                          y0=cy+sign*RUNWAY_HALF_W, y1=cy+sign*RUNWAY_HALF_W,
                          line=dict(color="rgba(255,255,255,0.03)",width=1))

    # Worst risk per vehicle
    veh_risk = {}
    order = {"high_risk":2,"warning":1,"safe":0}
    for p in preds:
        vid = p["prediction"]["vehicle_id"]
        lvl = p["prediction"]["risk_level"]
        if vid not in veh_risk or order[lvl] > order[veh_risk[vid]]:
            veh_risk[vid] = lvl

    # CPA zones
    for p in preds:
        lvl = p["prediction"]["risk_level"]
        if lvl not in ("warning","high_risk"): continue
        pred = p["prediction"]
        ac = next((a for a in sim["aircraft"] if a["id"]==pred["aircraft_id"]),None)
        if not ac: continue
        t = pred["time_to_cpa_s"]
        cx = ac["x"] + ac["speed"]*t
        cy2 = ac["y"]
        r = 80 if lvl=="high_risk" else 50
        fc = "rgba(255,82,82,0.15)" if lvl=="high_risk" else "rgba(255,171,64,0.12)"
        bc = "#ff5252" if lvl=="high_risk" else "#ffab40"
        fig.add_shape(type="circle", x0=cx-r, x1=cx+r, y0=cy2-r*0.6, y1=cy2+r*0.6,
                      fillcolor=fc, line=dict(color=bc,width=1.5,dash="dash"))
        fig.add_annotation(x=cx, y=cy2-r*0.6-18,
                           text=f"⚠ {pred['cpa_distance_m']:.0f}m · {t:.0f}s",
                           font=dict(size=10,color=bc,family="monospace"),
                           showarrow=False,
                           bgcolor="rgba(8,8,20,0.85)",
                           bordercolor=bc, borderwidth=1)

    # Vehicles
    for gv in sim["vehicles"]:
        lvl   = veh_risk.get(gv["id"],"safe")
        color = RISK_COLOR.get(lvl, gv["color"]) if lvl != "safe" else gv["color"]
        px = gv["x"] + gv["speed"]*math.cos(gv["heading"])*20
        py = gv["y"] + gv["speed"]*math.sin(gv["heading"])*20
        fig.add_shape(type="line", x0=gv["x"], x1=px, y0=gv["y"], y1=py,
                      line=dict(color="rgba(180,180,180,0.25)",width=1,dash="dot"))
        fig.add_trace(go.Scatter(
            x=[gv["x"]], y=[gv["y"]], mode="markers+text",
            marker=dict(symbol="square",size=10,color=color,line=dict(color="#fff",width=0.8)),
            text=[gv["id"]], textposition="top center",
            textfont=dict(color=color,size=9,family="monospace"),
            showlegend=False,
            hovertemplate=(f"<b>{gv['id']}</b><br>Type: {gv['type']}<br>"
                           f"X: {gv['x']:.0f}m<br>Speed: {gv['speed']:.1f} m/s<br>"
                           f"Risk: <b>{lvl}</b><extra></extra>"),
        ))

    # Aircraft
    for ac in sim["aircraft"]:
        pc = {"flare":"#ffcc00","rollout":"#ff9800","vacating":"#ab47bc"}
        color = pc.get(ac["phase"],"#4fc3f7")
        px = ac["x"] + ac["speed"]*20
        fig.add_shape(type="line", x0=ac["x"], x1=px, y0=ac["y"], y1=ac["y"],
                      line=dict(color="rgba(79,195,247,0.25)",width=1,dash="dot"))
        fig.add_trace(go.Scatter(
            x=[ac["x"]], y=[ac["y"]], mode="markers+text",
            marker=dict(symbol="triangle-right",size=14,color=color,line=dict(color="#fff",width=1)),
            text=[ac["id"]], textposition="top center",
            textfont=dict(color=color,size=9,family="monospace"),
            showlegend=False,
            hovertemplate=(f"<b>{ac['id']}</b><br>Phase: {ac['phase']}<br>"
                           f"X: {ac['x']:.0f}m<br>Alt: {ac['altitude']:.0f}m<br>"
                           f"Speed: {ac['speed']:.1f} m/s<extra></extra>"),
        ))

    fig.update_layout(
        paper_bgcolor="#080810", plot_bgcolor="#080810",
        margin=dict(l=10,r=10,t=10,b=10),
        xaxis=dict(range=[-600,RUNWAY_LENGTH+500], showgrid=False, zeroline=False,
                   tickfont=dict(color="#333",size=9,family="monospace"),
                   title=dict(text="distance along runway (meters)",font=dict(color="#333",size=10))),
        yaxis=dict(range=[-620,200], showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
        hovermode="closest", showlegend=False,
    )
    return fig


def draw_timeline():
    h = sim["history"] or [0]
    v = h[-1]
    color = "#ff5252" if v>=0.9 else "#ffab40" if v>=0.4 else "#69f0ae"
    fc = "rgba(255,82,82,0.13)" if v>=0.9 else "rgba(255,171,64,0.13)" if v>=0.4 else "rgba(105,240,174,0.13)"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(h))), y=h,
                             fill="tozeroy", fillcolor=fc,
                             line=dict(color=color,width=1.5),
                             mode="lines", showlegend=False))
    fig.add_shape(type="line", x0=0, x1=120, y0=0.4, y1=0.4,
                  line=dict(color="rgba(255,171,64,0.3)",width=1,dash="dot"))
    fig.add_shape(type="line", x0=0, x1=120, y0=0.9, y1=0.9,
                  line=dict(color="rgba(255,82,82,0.3)",width=1,dash="dot"))
    fig.update_layout(
        paper_bgcolor="#080818", plot_bgcolor="#080818",
        margin=dict(l=0,r=0,t=0,b=0),
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False,range=[0,1.05]),
    )
    return fig


def draw_alert(lvl, worst):
    color = RISK_COLOR.get(lvl,"#69f0ae")
    bgs = {"high_risk":"rgba(255,82,82,0.08)","warning":"rgba(255,171,64,0.08)","safe":"rgba(105,240,174,0.05)"}
    bg = bgs.get(lvl,"rgba(105,240,174,0.05)")
    labels = {"high_risk":"⚠ HIGH RISK DETECTED","warning":"⚠ CAUTION — CONFLICT WARNING","safe":"● ALL CLEAR"}
    title = labels.get(lvl,"● ALL CLEAR")
    if worst and lvl != "safe":
        p = worst["prediction"]
        detail = (f"{p['aircraft_id']} ↔ {p['vehicle_id']}  ·  "
                  f"CPA {p['cpa_distance_m']:.0f}m  ·  {p['time_to_cpa_s']:.0f}s  ·  "
                  f"{len(sim['aircraft'])} aircraft · {len(sim['vehicles'])} vehicles")
    else:
        detail = (f"Monitoring {len(sim['aircraft'])*len(sim['vehicles'])} pairs — "
                  f"{len(sim['aircraft'])} aircraft · {len(sim['vehicles'])} vehicles")
    return html.Div([
        html.Span(f"{title}  ", style={"fontWeight":"bold"}),
        html.Span(detail, style={"opacity":"0.8"}),
    ], style={"padding":"10px 24px","fontFamily":"monospace","fontSize":"13px",
              "color":color,"background":bg,"borderBottom":f"1px solid {color}"})


def draw_status(lvl, preds):
    color = RISK_COLOR.get(lvl,"#69f0ae")
    label = RISK_LABEL.get(lvl,"ALL SAFE")
    n_h = sum(1 for p in preds if p["prediction"]["risk_level"]=="high_risk")
    n_w = sum(1 for p in preds if p["prediction"]["risk_level"]=="warning")
    n_s = sum(1 for p in preds if p["prediction"]["risk_level"]=="safe")
    def badge(count, lbl, c):
        return html.Div([
            html.Div(count, style={"fontSize":"20px","fontWeight":"bold","color":c,"fontFamily":"monospace"}),
            html.Div(lbl,   style={"fontSize":"9px","color":"#444","fontFamily":"monospace"}),
        ], style={"background":RISK_BG.get(lbl.lower().replace(" ","_"),"rgba(100,100,100,0.05)"),
                  "border":f"1px solid {c}",
                  "borderRadius":"5px","padding":"6px 10px","textAlign":"center","flex":"1"})
    return html.Div([
        html.Div(label, style={"fontFamily":"monospace","fontSize":"22px","fontWeight":"bold",
                               "color":color,"letterSpacing":"2px","marginBottom":"10px"}),
        html.Div([badge(str(n_h),"HIGH RISK","#ff5252"),
                  badge(str(n_w),"WARNING","#ffab40"),
                  badge(str(n_s),"SAFE","#69f0ae")],
                 style={"display":"flex","gap":"6px"}),
    ], style={"background":"#0a0a1a","padding":"14px","borderRadius":"8px",
              "border":f"1px solid {color}"})


def draw_table(preds):
    if not preds:
        return html.Div("Waiting for data...",
                        style={"color":"#333","fontFamily":"monospace","fontSize":"12px"})
    def th(t): return html.Div(t, style={"flex":"1","fontFamily":"monospace","fontSize":"9px",
                                          "color":"#333","letterSpacing":"1px"})
    def td(t,c): return html.Div(t, style={"flex":"1","fontFamily":"monospace","fontSize":"11px",
                                            "color":c,"padding":"0 4px"})
    rows = [html.Div([th("Aircraft"),th("Vehicle"),th("Risk"),th("CPA"),th("Time")],
                     style={"display":"flex","borderBottom":"1px solid #111",
                            "paddingBottom":"6px","marginBottom":"4px"})]
    order = {"high_risk":0,"warning":1,"safe":2}
    for p in sorted(preds, key=lambda p: order[p["prediction"]["risk_level"]]):
        pred = p["prediction"]
        lvl  = pred["risk_level"]
        c    = RISK_COLOR.get(lvl,"#69f0ae")
        rows.append(html.Div([
            td(pred["aircraft_id"],"#888"),
            td(pred["vehicle_id"],"#666"),
            td(lvl.replace("_"," ").upper(), c),
            td(f"{pred['cpa_distance_m']:.0f}m","#888"),
            td(f"{pred['time_to_cpa_s']:.0f}s","#888"),
        ], style={"display":"flex","borderBottom":"1px solid #0d0d1a","padding":"5px 0",
                  "background":RISK_BG.get(lvl,"rgba(100,100,100,0.03)"),
                  "borderRadius":"3px","marginBottom":"2px"}))
    return html.Div(rows)


if __name__ == "__main__":
    app.run(debug=True, port=8050)