"""
CsSnI3 Perovskite Solar Cell – Degradation Simulator
=====================================================
Side-by-side, schematic comparison of

    (1) a Standard CsSnI3 solar cell, and
    (2) a PTM-treated CsSnI3 solar cell (localized electron density engineering)

under simulated sunlight + ambient air exposure.

Everything is drawn with Plotly shapes / scatter traces (no external images).

Run locally :  streamlit run app.py
Deploy      :  push app.py + requirements.txt (streamlit, plotly, numpy) to GitHub
               and point Streamlit Community Cloud at app.py.

NOTE: This is an educational *schematic*. The two headline efficiencies
(10.1 % for PTM-treated, 5.5 % for standard after exposure) are the scenario
values requested for this demo; the other indicators (conductivity, Sn4+ share,
distortion) are illustrative model quantities, not measured data.
"""

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# 0. PAGE CONFIG  (must be the first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CsSnI₃ Degradation Simulator",
    page_icon="☀️",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
PCE_FRESH = 10.1          # % – fresh device (both cells start identical)
PCE_STANDARD_FINAL = 5.5  # % – standard cell after sunlight + air
PCE_PTM_FINAL = 10.1      # % – PTM-treated cell after sunlight + air

N_FRAMES = 32             # animation frames
FRAME_DELAY = 0.05        # seconds between frames

# Colour palette (black / yellow = perovskite phases, red = defects,
# blue / green = PTM shield)
C_BLACK_PHASE = (20, 20, 24)      # black γ-phase CsSnI3
C_YELLOW_PHASE = (242, 197, 0)    # yellow phase / Cs2SnI6-rich
C_SN2 = "#4FC3F7"                 # Sn2+
C_SN4 = "#E040FB"                 # Sn4+
C_IODIDE = "#B388FF"
C_CS = "#FFFFFF"
C_OXYGEN = "#FF5252"
C_DEFECT = "#FF1744"
C_PTM_GREEN = "#00E5A0"
C_PTM_BLUE = "#2979FF"
C_SUN = "#FFD400"
C_STD = "#FF5252"
C_PTM = "#00E5A0"
FIG_BG = "#0B1220"

PLOT_CONFIG = {"displayModeBar": False}

# ──────────────────────────────────────────────────────────────────────────────
# 2. STATIC GEOMETRY  (computed once per script run, deterministic via seed)
# ──────────────────────────────────────────────────────────────────────────────
# Device stack, light enters from the top. (label, y0, y1, fill colour)
LAYERS = [
    ("Glass / TCO", 10.8, 12.0, "rgba(127,219,255,0.35)"),
    ("ETL", 9.8, 10.8, "#3949AB"),
    ("CsSnI₃ absorber", 4.6, 9.8, None),   # colour is dynamic (black → yellow)
    ("HTL", 3.6, 4.6, "#6D4C41"),
    ("Metal electrode", 2.8, 3.6, "#CFD8DC"),
]
PEROV_Y0, PEROV_Y1 = 4.6, 9.8
X0, X1 = 0.0, 10.0

# Lattice of corner-sharing SnI6 octahedra, seen in 2-D:
# Sn at nodes, I⁻ at bond mid-points, Cs⁺ in the cage centres.
NX, NY = 8, 4
_gx, _gy = np.meshgrid(np.linspace(1.0, 9.0, NX), np.linspace(5.5, 8.9, NY))
SN_X, SN_Y = _gx.ravel(), _gy.ravel()          # index = row * NX + col
N_SN = NX * NY

_ix, _iy, _nb = [], [], []
for r in range(NY):                             # horizontal Sn–I–Sn links
    for c in range(NX - 1):
        a, b = r * NX + c, r * NX + c + 1
        _ix.append((SN_X[a] + SN_X[b]) / 2)
        _iy.append(SN_Y[a])
        _nb.append((a, b))
for r in range(NY - 1):                         # vertical Sn–I–Sn links
    for c in range(NX):
        a, b = r * NX + c, (r + 1) * NX + c
        _ix.append(SN_X[a])
        _iy.append((SN_Y[a] + SN_Y[b]) / 2)
        _nb.append((a, b))
I_X, I_Y, I_NB = np.array(_ix), np.array(_iy), _nb
N_I = len(I_X)

_cx, _cy = [], []
for r in range(NY - 1):
    for c in range(NX - 1):
        _cx.append((SN_X[r * NX + c] + SN_X[r * NX + c + 1]) / 2)
        _cy.append((SN_Y[r * NX + c] + SN_Y[(r + 1) * NX + c]) / 2)
CS_X, CS_Y = np.array(_cx), np.array(_cy)
N_CS = len(CS_X)

_rng = np.random.default_rng(11)
# Each Sn oxidises when the global oxidation progress passes its threshold
# (thresholds up to 1.15 → roughly 85 % of Sn ends up as Sn4+).
SN_THRESH = _rng.uniform(0.05, 1.15, N_SN)
# A few oxidised Sn sites leave the lattice → Sn vacancies.
_eligible = np.where(SN_THRESH < 0.8)[0]
IS_VAC_SN = np.zeros(N_SN, dtype=bool)
IS_VAC_SN[_rng.choice(_eligible, size=min(6, len(_eligible)), replace=False)] = True
# Some iodide sites empty out as the lattice collapses.
I_VAC_ONSET = np.full(N_I, np.inf)
_vi = _rng.choice(N_I, size=9, replace=False)
I_VAC_ONSET[_vi] = _rng.uniform(0.35, 0.95, size=9)
# Random displacement directions used for crystal distortion.
SN_DISP = np.clip(_rng.normal(0, 1, (N_SN, 2)), -1.6, 1.6)
I_DISP = np.clip(_rng.normal(0, 1, (N_I, 2)), -1.6, 1.6)
CS_DISP = np.clip(_rng.normal(0, 1, (N_CS, 2)), -1.6, 1.6)

# Oxygen molecules entering from the ambient-air side (left edge).
N_O2 = 9
_rows = np.array([0, 1, 2, 3, 0, 1, 2, 3, 1])
O2_Y = SN_Y[_rows * NX]
O2_TX = SN_X[_rng.integers(0, 3, N_O2)] - 0.5      # where each O2 attacks
O2_T0 = _rng.permutation(np.linspace(0.12, 0.60, N_O2))  # start times
AIR_X = -2.6                                         # O2 start position


# ──────────────────────────────────────────────────────────────────────────────
# 3. SIMULATION MODEL  (progress p ∈ [0, 1]; p < 0 means "not started")
# ──────────────────────────────────────────────────────────────────────────────
def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3 - 2 * x)


def sim_state(treated: bool, p: float) -> dict:
    """Return the normalised degradation variables for a given cell/progress."""
    if p < 0:
        return dict(ox=0.0, dist=0.0, phase=0.0, s=0.0)
    s = smoothstep((p - 0.2) / 0.7)                 # overall degradation driver
    if treated:                                     # PTM: nothing structural happens
        return dict(ox=0.0, dist=0.0, phase=0.0, s=s)
    return dict(
        ox=float(np.clip((p - 0.2) / 0.6, 0, 1)),    # Sn2+ → Sn4+
        dist=float(np.clip((p - 0.4) / 0.5, 0, 1)),  # lattice distortion
        phase=smoothstep((p - 0.25) / 0.55),         # black → yellow phase
        s=s,
    )


def indicators(treated: bool, p: float) -> dict:
    """Numbers shown in the metric cards."""
    st_ = sim_state(treated, p)
    s = st_["s"]
    if treated:
        return dict(
            pce=PCE_PTM_FINAL,                 # stays at 10.1 %
            cond=100.0 - 2.0 * s,              # high conductivity maintained
            sn4=0.0,                           # Sn2+ remains stable
            distort=0.0,                       # crystal intact
            s=s,
        )
    pce = PCE_FRESH + (PCE_STANDARD_FINAL - PCE_FRESH) * s
    if p >= 1:
        pce = PCE_STANDARD_FINAL               # exactly 5.5 % at the end
    return dict(
        pce=pce,
        cond=100.0 - 86.0 * s,                 # conductivity drops heavily
        sn4=100.0 * float(np.mean(st_["ox"] >= SN_THRESH)),
        distort=100.0 * st_["dist"],
        s=s,
    )


def status_message(treated: bool, p: float):
    """(kind, text) describing the current stage for the caption box."""
    if p < 0:
        return "info", ("Pristine cell. Press **Simulate Sunlight & Air Exposure**."
                        if not treated else
                        "Pristine cell with the PTM shield already anchored on the Sn sites.")
    if p >= 1:
        if treated:
            return "success", ("**Final state:** Sn²⁺ intact, lattice intact, conductivity "
                               "maintained. PCE stays at **10.1 %**.")
        return "error", ("**Final state:** Sn⁴⁺-rich, distorted lattice, vacancies, yellow "
                         "phase. PCE crashes to **5.5 %**.")
    if p < 0.2:
        return "info", "**Stage 1 – Sunlight:** photons cross the glass/TCO and ETL and reach the absorber."
    if p < 0.45:
        return ("info", "**Stage 2 – Air ingress:** O₂ approaches the absorber edge.") if treated else \
               ("warning", "**Stage 2 – Air ingress:** O₂ diffuses into the lattice and attacks Sn²⁺ sites.")
    if p < 0.75:
        return ("info", "**Stage 3 – Shield holds:** O₂ is repelled; localized electron density keeps Sn²⁺ stable.") \
            if treated else \
               ("warning", "**Stage 3 – Oxidation:** Sn²⁺ → Sn⁴⁺, the octahedra distort, the film darkens → yellow.")
    return ("info", "**Stage 4 – Steady state:** crystal structure unchanged, carriers still flow.") \
        if treated else \
           ("warning", "**Stage 4 – Collapse:** Sn / I vacancies form, conductivity falls, PCE drops.")


# ──────────────────────────────────────────────────────────────────────────────
# 4. PLOTLY DRAWING
# ──────────────────────────────────────────────────────────────────────────────
def lerp_rgb(c0, c1, t: float) -> str:
    return "rgb({},{},{})".format(*[int(round(a + (b - a) * t)) for a, b in zip(c0, c1)])


def build_figure(treated: bool, p: float) -> go.Figure:
    """Draw one solar cell (layers + atoms + sunlight + oxygen) for progress p."""
    running = p >= 0
    s = sim_state(treated, p)
    ox, dist, phase = s["ox"], s["dist"], s["phase"]

    fig = go.Figure()
    shapes, notes = [], []

    # ── 4a. Layer rectangles ────────────────────────────────────────────────
    perov_fill = lerp_rgb(C_BLACK_PHASE, C_YELLOW_PHASE, phase)
    for label, y0, y1, color in LAYERS:
        shapes.append(dict(
            type="rect", x0=X0, x1=X1, y0=y0, y1=y1, layer="below",
            fillcolor=color or perov_fill,
            line=dict(color="rgba(255,255,255,0.35)", width=1),
        ))
        if color is not None:       # perovskite label is composed separately below
            notes.append(dict(x=X1 + 0.45, y=(y0 + y1) / 2, text=label,
                              font=dict(size=12, color="#C9D1E0")))

    # Sunlight bar
    shapes.append(dict(
        type="rect", x0=X0, x1=X1, y0=14.7, y1=15.4, layer="below",
        fillcolor="rgba(255,212,0,0.85)" if running else "rgba(255,212,0,0.25)",
        line=dict(color="#FFA000", width=1),
    ))
    notes.append(dict(x=5, y=15.05, text="<b>Sunlight (1 sun)</b>", xanchor="center",
                      font=dict(size=12, color="#1A1A1A" if running else "#FFE082")))

    # Ambient-air zone on the left edge
    shapes.append(dict(
        type="rect", x0=-3.1, x1=-0.3, y0=PEROV_Y0, y1=PEROV_Y1, layer="below",
        fillcolor="rgba(100,181,246,0.10)", line=dict(color="rgba(144,202,249,0.35)", width=1, dash="dot"),
    ))
    notes.append(dict(
        x=-1.7, y=10.35, xanchor="center", font=dict(size=11, color="#90CAF9"),
        text="Ambient air<br>(O₂ repelled)" if treated else "Ambient air<br>(O₂ attacks)",
    ))

    # PTM shield envelope around the absorber (treated cell only)
    if treated:
        shapes.append(dict(
            type="rect", x0=-0.25, x1=X1 + 0.25, y0=PEROV_Y0 - 0.1, y1=PEROV_Y1 + 0.1, layer="below",
            fillcolor="rgba(0,229,160,0.10)", line=dict(color=C_PTM_GREEN, width=3),
        ))

    # ── 4b. Lattice state ───────────────────────────────────────────────────
    sn_x = SN_X + dist * 0.20 * SN_DISP[:, 0]
    sn_y = SN_Y + dist * 0.20 * SN_DISP[:, 1]
    i_x = I_X + dist * 0.28 * I_DISP[:, 0]
    i_y = I_Y + dist * 0.28 * I_DISP[:, 1]
    cs_x = CS_X + dist * 0.30 * CS_DISP[:, 0]
    cs_y = CS_Y + dist * 0.30 * CS_DISP[:, 1]

    healthy = ox < SN_THRESH                                # still Sn2+
    sn_vac = IS_VAC_SN & (ox >= SN_THRESH + 0.1)            # Sn has left its site
    oxidised = (~healthy) & (~sn_vac)                       # Sn4+
    i_vac = ox >= I_VAC_ONSET                               # missing iodide

    # Cs⁺
    fig.add_trace(go.Scatter(
        x=cs_x, y=cs_y, mode="markers", name="Cs⁺", hoverinfo="name",
        marker=dict(size=11, color=C_CS, line=dict(color="#555", width=1)),
    ))

    # Bonds (Sn–I–Sn); skip any bond touching a vacancy so the lattice visibly breaks
    bx, by = [], []
    for k in range(N_I):
        if i_vac[k]:
            continue
        for nb in I_NB[k]:
            if sn_vac[nb]:
                continue
            bx += [i_x[k], sn_x[nb], None]
            by += [i_y[k], sn_y[nb], None]
    fig.add_trace(go.Scatter(
        x=bx, y=by, mode="lines", hoverinfo="skip", showlegend=False,
        line=dict(color="#9AA0A6", width=1.5),
    ))

    # PTM halos + molecules (treated only): drawn under the atoms
    if treated:
        fig.add_trace(go.Scatter(
            x=SN_X, y=SN_Y, mode="markers", name="PTM shield (e⁻ density)", hoverinfo="name",
            marker=dict(size=34, color="rgba(0,229,160,0.28)", line=dict(color=C_PTM_GREEN, width=1.5)),
        ))
        fig.add_trace(go.Scatter(
            x=SN_X - 0.3, y=SN_Y + 0.3, mode="markers", name="PTM molecule", hoverinfo="name",
            marker=dict(size=9, symbol="diamond", color=C_PTM_BLUE, line=dict(color="white", width=1)),
        ))

    # I⁻
    keep_i = ~i_vac
    fig.add_trace(go.Scatter(
        x=i_x[keep_i], y=i_y[keep_i], mode="markers", name="I⁻", hoverinfo="name",
        marker=dict(size=9, color=C_IODIDE, line=dict(color="#2A1A55", width=1)),
    ))

    # Sn²⁺ (healthy) and Sn⁴⁺ (oxidised)
    fig.add_trace(go.Scatter(
        x=sn_x[healthy], y=sn_y[healthy], mode="markers+text", name="Sn²⁺", hoverinfo="name",
        text=["2+"] * int(healthy.sum()), textfont=dict(size=9, color="#0B1220"),
        marker=dict(size=21, color=C_SN2, line=dict(color="white", width=1)),
    ))
    fig.add_trace(go.Scatter(
        x=sn_x[oxidised], y=sn_y[oxidised], mode="markers+text", name="Sn⁴⁺", hoverinfo="name",
        text=["4+"] * int(oxidised.sum()), textfont=dict(size=9, color="#0B1220"),
        marker=dict(size=21, color=C_SN4, line=dict(color="white", width=1)),
    ))

    # Vacancies / defects (red open symbols)
    n_sv, n_iv = int(sn_vac.sum()), int(i_vac.sum())
    fig.add_trace(go.Scatter(
        x=np.concatenate([sn_x[sn_vac], i_x[i_vac]]),
        y=np.concatenate([sn_y[sn_vac], i_y[i_vac]]),
        mode="markers", name="Vacancy / defect", hoverinfo="name",
        marker=dict(
            symbol=["square-open"] * n_sv + ["circle-open"] * n_iv,
            size=[22] * n_sv + [15] * n_iv,
            color=C_DEFECT, line=dict(color=C_DEFECT, width=2.5),
        ),
    ))

    # ── 4c. Oxygen molecules ────────────────────────────────────────────────
    ox_x, ox_y = [], []
    if running:
        for j in range(N_O2):
            t0 = O2_T0[j]
            if p < t0:
                continue
            if treated:
                # approach the shield, bounce back, repeat
                cyc = ((p - t0) * 2.2) % 1.0
                x = AIR_X + 2.0 * (1 - abs(2 * cyc - 1))
            else:
                if p > t0 + 0.30:          # consumed in the oxidation reaction
                    continue
                frac = float(np.clip((p - t0) / 0.22, 0, 1))
                x = AIR_X + (O2_TX[j] - AIR_X) * frac
            ox_x += [x, x]
            ox_y += [O2_Y[j] + 0.17, O2_Y[j] - 0.17]
    fig.add_trace(go.Scatter(
        x=ox_x, y=ox_y, mode="markers", name="O₂", hoverinfo="name",
        marker=dict(size=10, color=C_OXYGEN, line=dict(color="white", width=1)),
    ))

    # ── 4d. Photons ─────────────────────────────────────────────────────────
    beam_x = np.linspace(1.0, 9.0, 6)
    ph_x, ph_y = [], []
    if running:
        bxs, bys = [], []
        for k, x in enumerate(beam_x):
            bxs += [x, x, None]
            bys += [14.7, 5.0, None]
            y = 14.6 - ((p * 3.0 + k / 6.0) % 1.0) * (14.6 - 5.2)
            ph_x.append(x)
            ph_y.append(y)
        fig.add_trace(go.Scatter(
            x=bxs, y=bys, mode="lines", hoverinfo="skip", showlegend=False,
            line=dict(color="rgba(255,212,0,0.35)", width=1.5, dash="dot"),
        ))
    fig.add_trace(go.Scatter(
        x=ph_x, y=ph_y, mode="markers", name="Photon", hoverinfo="name",
        marker=dict(symbol="triangle-down", size=12, color="#FFEB3B", line=dict(color="#FF8F00", width=1)),
    ))

    # ── 4e. Right-hand labels & status tags ─────────────────────────────────
    if phase > 0.5:
        phase_txt = f"<span style='color:{lerp_rgb(C_YELLOW_PHASE, C_YELLOW_PHASE, 0)}'>yellow phase (degraded)</span>"
    else:
        phase_txt = "black γ-phase"
    notes.append(dict(x=X1 + 0.45, y=9.3, text=f"CsSnI₃ absorber<br><sub>{phase_txt}</sub>",
                      font=dict(size=12, color="#C9D1E0")))

    def tag(y, text, color):
        notes.append(dict(x=X1 + 0.45, y=y, text=f"<b>{text}</b>", font=dict(size=11, color=color)))

    if treated:
        if running:
            tag(7.6, "No oxidation (Sn²⁺ stable)", C_PTM_GREEN)
            tag(6.9, "Lattice intact", C_PTM_GREEN)
        tag(6.2, "PTM shield on Sn", C_PTM_BLUE if not running else "#82B1FF")
        footer, footer_color = "PTM shield · localized electron density on Sn²⁺", C_PTM_GREEN
    else:
        if ox > 0.1:
            tag(7.6, "Sn²⁺ → Sn⁴⁺ oxidation", C_SN4)
        if dist > 0.15:
            tag(6.9, "Crystal distortion", C_DEFECT)
        if n_sv + n_iv > 0:
            tag(6.2, "Vacancies / defects", C_DEFECT)
        if ox > 0.35:
            footer, footer_color = "2 CsSnI₃ + O₂ → Cs₂SnI₆ + SnO₂", "#FF8A80"
        else:
            footer, footer_color = "Unprotected Sn²⁺ lattice", "#90A4AE"
    notes.append(dict(x=5, y=1.9, xanchor="center", text=footer, font=dict(size=12, color=footer_color)))

    # ── 4f. Layout ──────────────────────────────────────────────────────────
    for a in notes:
        a.setdefault("xref", "x")
        a.setdefault("yref", "y")
        a.setdefault("xanchor", "left")
        a.setdefault("showarrow", False)

    fig.update_layout(
        shapes=shapes, annotations=notes, height=560,
        margin=dict(l=4, r=4, t=6, b=62),
        paper_bgcolor=FIG_BG, plot_bgcolor=FIG_BG,
        font=dict(color="#E8ECF4"),
        xaxis=dict(range=[-3.2, 14.8], visible=False, fixedrange=True),
        yaxis=dict(range=[1.3, 15.7], visible=False, fixedrange=True),
        legend=dict(orientation="h", yanchor="top", y=-0.01, x=0.5, xanchor="center",
                    font=dict(size=11, color="#DDE3EE")),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 5. STREAMLIT RENDER HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def render_metrics(slot, treated: bool, ind: dict, other: dict) -> None:
    """Three st.metric cards with red/green delta arrows."""
    with slot.container():
        c1, c2, c3 = st.columns(3)
        started = ind["s"] > 0.01 or other["s"] > 0.01

        if treated:   # compare the PTM cell against the standard cell
            d_pce = f"{ind['pce'] - other['pce']:+.1f} pp vs. standard" if started else None
            d_cond = f"{ind['cond'] - other['cond']:+.0f} pts vs. standard" if started else None
            d_sn = f"{ind['sn4'] - other['sn4']:+.0f} pts vs. standard" if started else None
        else:         # compare the standard cell against its own fresh state
            d_pce = f"{ind['pce'] - PCE_FRESH:+.1f} pp vs. fresh" if started else None
            d_cond = f"{ind['cond'] - 100:+.0f} pts vs. fresh" if started else None
            d_sn = f"{ind['sn4']:+.0f} pts vs. fresh" if started else None

        c1.metric("PCE", f"{ind['pce']:.1f}%", delta=d_pce,
                  help="Power conversion efficiency (headline scenario values).")
        c2.metric("Conductivity", f"{ind['cond']:.0f}%", delta=d_cond,
                  help="Relative electrical conductivity (illustrative index, fresh = 100 %).")
        c3.metric("Sn⁴⁺ share", f"{ind['sn4']:.0f}%", delta=d_sn, delta_color="inverse",
                  help="Fraction of tin oxidised from Sn²⁺ to Sn⁴⁺ (lower is better).")


def render_status(slot, treated: bool, p: float) -> None:
    kind, text = status_message(treated, p)
    getattr(slot, kind)(text)


def render_frame(p: float, slots: dict) -> None:
    """Draw both cells + captions + metrics for progress p."""
    ind = {False: indicators(False, p), True: indicators(True, p)}
    for treated, key in ((False, "std"), (True, "ptm")):
        slots[key]["chart"].plotly_chart(build_figure(treated, p), config=PLOT_CONFIG)
        render_status(slots[key]["status"], treated, p)
        render_metrics(slots[key]["metrics"], treated, ind[treated], ind[not treated])


def render_summary() -> None:
    """Head-to-head charts shown after the simulation has finished."""
    st.markdown("### Head-to-head summary")
    left, right = st.columns(2, gap="large")

    # PCE vs normalised exposure
    ps = np.linspace(0, 1, 41)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=ps, y=[indicators(False, q)["pce"] for q in ps], mode="lines",
                              name="Standard CsSnI₃", line=dict(color=C_STD, width=4)))
    fig1.add_trace(go.Scatter(x=ps, y=[indicators(True, q)["pce"] for q in ps], mode="lines",
                              name="PTM-treated CsSnI₃", line=dict(color=C_PTM, width=4)))
    fig1.add_annotation(x=1, y=PCE_STANDARD_FINAL, text="<b>5.5 %</b>", showarrow=False,
                        xanchor="left", font=dict(color=C_STD, size=14))
    fig1.add_annotation(x=1, y=PCE_PTM_FINAL, text="<b>10.1 %</b>", showarrow=False,
                        xanchor="left", font=dict(color=C_PTM, size=14))
    fig1.update_layout(
        title="PCE during sunlight + air exposure", template="plotly_dark",
        paper_bgcolor=FIG_BG, plot_bgcolor=FIG_BG, height=380,
        xaxis=dict(title="Normalised exposure (a.u.)", range=[0, 1.12]),
        yaxis=dict(title="PCE (%)", range=[0, 12]),
        legend=dict(orientation="h", y=-0.25), margin=dict(l=10, r=10, t=50, b=10),
    )
    left.plotly_chart(fig1, config=PLOT_CONFIG)

    # Final-state indicators (all as % so they share one axis)
    a, b = indicators(False, 1.0), indicators(True, 1.0)
    cats = ["PCE retained", "Conductivity retained", "Sn⁴⁺ share", "Lattice distortion"]
    std_vals = [100 * a["pce"] / PCE_FRESH, a["cond"], a["sn4"], a["distort"]]
    ptm_vals = [100 * b["pce"] / PCE_FRESH, b["cond"], b["sn4"], b["distort"]]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=cats, y=std_vals, name="Standard", marker_color=C_STD,
                          text=[f"{v:.0f}" for v in std_vals], textposition="outside"))
    fig2.add_trace(go.Bar(x=cats, y=ptm_vals, name="PTM-treated", marker_color=C_PTM,
                          text=[f"{v:.0f}" for v in ptm_vals], textposition="outside"))
    fig2.update_layout(
        title="Final state after exposure (%)", barmode="group", template="plotly_dark",
        paper_bgcolor=FIG_BG, plot_bgcolor=FIG_BG, height=380,
        yaxis=dict(range=[0, 125]), legend=dict(orientation="h", y=-0.25),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    right.plotly_chart(fig2, config=PLOT_CONFIG)


# ──────────────────────────────────────────────────────────────────────────────
# 6. STYLING
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .hero {background:#111317; border-left:8px solid #FFD400; padding:1.2rem 1.6rem;
           border-radius:4px; margin-bottom:1rem;}
    .hero-title {color:#FFD400; font-size:2.1rem; font-weight:800; line-height:1.15;}
    .hero-sub {color:#CFD3DC; margin-top:.45rem; font-size:1.02rem;}
    .cell-tag {padding:.5rem .9rem; border-radius:4px; font-weight:700; font-size:1.05rem;
               margin-bottom:.5rem; border-left:6px solid;}
    .cell-std {background:rgba(229,57,53,.13); border-color:#E53935;}
    .cell-ptm {background:rgba(0,200,150,.13); border-color:#00C896;}
    button[kind="primary"], [data-testid="stBaseButton-primary"] {
        background:#FFD400 !important; color:#111 !important; border:none !important;
        font-weight:800 !important; font-size:1.1rem !important;}
    button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
        background:#FFE44D !important; color:#000 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# 7. PAGE: HEADER + INTRO
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
      <div class="hero-title">Lead-free CsSnI₃ perovskite solar cell: sunlight and air, side by side</div>
      <div class="hero-sub">What happens to the tin lattice when an untreated cell and a PTM-treated cell
      meet oxygen under illumination.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

i1, i2, i3 = st.columns(3, gap="large")
i1.markdown(
    "**Why CsSnI₃?**  \n"
    "Cesium tin iodide replaces toxic lead with tin. Its black γ-phase has a narrow band gap "
    "(≈ 1.3 eV) and high carrier mobility, which suits photovoltaics."
)
i2.markdown(
    "**The problem**  \n"
    "Sn²⁺ oxidises to Sn⁴⁺ very easily in air. That creates Sn vacancies (self-doping), "
    "distorts the lattice, converts black CsSnI₃ into yellow, Sn⁴⁺-rich phases such as Cs₂SnI₆, and "
    "collapses conductivity and efficiency."
)
i3.markdown(
    "**The PTM approach**  \n"
    "PTM treatment localizes electron density on the Sn sites, forming a protective shield that keeps "
    "Sn²⁺ from reacting with O₂, so the crystal and its conductivity survive."
)

with st.expander("How to read the schematic"):
    st.markdown(
        "- Layers from top to bottom: **Glass/TCO → ETL → CsSnI₃ absorber → HTL → Metal electrode**.\n"
        "- Inside the absorber: **Sn** (blue = Sn²⁺, magenta = Sn⁴⁺), **I⁻** (violet), **Cs⁺** (white).\n"
        "- **Red open symbols** are vacancies/defects; **red dumbbells** are O₂ molecules entering from the "
        "air side (edges, grain boundaries, pinholes).\n"
        "- **Black → yellow** absorber colour shows the black γ-phase degrading.\n"
        "- **Green halos + blue diamonds** are the PTM shield and PTM molecules.\n"
        "- Only the PCE values (10.1 % and 5.5 %) are the scenario targets; conductivity, Sn⁴⁺ share and "
        "distortion are illustrative model indicators, and the lattice is a 2-D cartoon, not a DFT result."
    )

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# 8. CONTROLS
# ──────────────────────────────────────────────────────────────────────────────
if "done" not in st.session_state:
    st.session_state["done"] = False

b_run, b_reset = st.columns([4, 1])
run_clicked = b_run.button("☀️  Simulate Sunlight & Air Exposure", type="primary",
                           use_container_width=True)
reset_clicked = b_reset.button("Reset", use_container_width=True)
if reset_clicked:
    st.session_state["done"] = False

progress_slot = st.empty()

# ──────────────────────────────────────────────────────────────────────────────
# 9. SIDE-BY-SIDE CELLS  (placeholders so the animation can overwrite them)
# ──────────────────────────────────────────────────────────────────────────────
col_std, col_ptm = st.columns(2, gap="large")
slots = {}
with col_std:
    st.markdown('<div class="cell-tag cell-std">Standard CsSnI₃ solar cell</div>', unsafe_allow_html=True)
    slots["std"] = dict(chart=st.empty(), status=st.empty(), metrics=st.empty())
with col_ptm:
    st.markdown('<div class="cell-tag cell-ptm">PTM-treated CsSnI₃ solar cell '
                '(localized electron density engineering)</div>', unsafe_allow_html=True)
    slots["ptm"] = dict(chart=st.empty(), status=st.empty(), metrics=st.empty())

# ──────────────────────────────────────────────────────────────────────────────
# 10. RUN / SHOW STATE
# ──────────────────────────────────────────────────────────────────────────────
if run_clicked:
    bar = progress_slot.progress(0, text="Exposing both cells to sunlight and air…")
    for i in range(N_FRAMES + 1):
        p = i / N_FRAMES
        render_frame(p, slots)
        bar.progress(int(round(p * 100)), text=f"Exposure {int(round(p * 100))} %")
        time.sleep(FRAME_DELAY)
    progress_slot.empty()
    st.session_state["done"] = True
    render_summary()
elif st.session_state["done"]:
    render_frame(1.0, slots)       # keep the final state on screen after reruns
    render_summary()
else:
    render_frame(-1.0, slots)      # pristine, before exposure

st.caption(
    "Educational schematic built with Streamlit and Plotly. Headline PCE values follow the scenario "
    "(standard 10.1 % → 5.5 %, PTM-treated 10.1 % → 10.1 %); other indicators are illustrative."
)
