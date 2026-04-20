"""
TacticsAI — Premium Dashboard (Week 1 + 2 + 3)

Design system (UI-UX Pro Max):
  - Dark navy #0a0f1e base + #0d1328 sidebar
  - Electric blue #4B7BF5 primary accent
  - Glassmorphism cards + frosted borders
  - Inter font (900 weight hero, 500 labels)
  - Semantic color tokens — no raw hex in components
  - 8px spacing rhythm throughout
  - Duration 150-300ms micro-interactions via CSS
  - SVG-style unicode symbols (no decorative emoji)

Week 3: Formation labels, heatmap tab, tactical metrics panel.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from config import ADS_STRIP_HEIGHT, CALIBRATION_FRAMES, DEFAULT_MODEL, DEFAULT_SRC_POINTS
from src.pitch.calibration import (
    auto_detect_pitch_corners,
    draw_corners_on_frame,
    validate_transform,
)
from src.pipeline import get_video_info, process_video

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TacticsAI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system (UI-UX Pro Max: dark navy, electric blue, Inter) ─────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Base ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.stApp { background: #0a0f1e; color: #e8eaf0; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1328;
    border-right: 1px solid rgba(75,123,245,0.15);
}
section[data-testid="stSidebar"] * { font-size: 0.85rem; }

/* ── Metric cards (glassmorphism) ── */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(75,123,245,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    backdrop-filter: blur(12px);
    transition: border-color 150ms ease;
}
div[data-testid="metric-container"]:hover {
    border-color: rgba(75,123,245,0.5);
}
div[data-testid="metric-container"] label {
    color: rgba(232,234,240,0.45) !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    color: #e8eaf0 !important;
}

/* ── File uploader ── */
div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(75,123,245,0.35) !important;
    border-radius: 14px;
    background: rgba(75,123,245,0.04);
    transition: border-color 200ms ease;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(75,123,245,0.7) !important;
}

/* ── Primary button ── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #4B7BF5 0%, #7B5CF5 100%);
    border: none;
    border-radius: 10px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.04em;
    padding: 0.65rem 1.5rem;
    transition: opacity 150ms ease, transform 150ms ease;
    box-shadow: 0 4px 24px rgba(75,123,245,0.3);
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    opacity: 0.88;
    transform: translateY(-1px);
}

/* ── Progress bar ── */
div[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #4B7BF5, #7B5CF5) !important;
    border-radius: 4px;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    color: rgba(232,234,240,0.5) !important;
    border-bottom: 2px solid transparent !important;
    transition: color 150ms ease, border-color 150ms ease;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #4B7BF5 !important;
    border-bottom-color: #4B7BF5 !important;
}
div[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(75,123,245,0.12) !important;
    gap: 0 !important;
}

/* ── Expander ── */
details[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(75,123,245,0.12) !important;
    border-radius: 10px;
}

/* ── Select / radio / slider ── */
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(75,123,245,0.2) !important;
    border-radius: 8px;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #4B7BF5 !important;
}

/* ── Divider ── */
hr { border-color: rgba(75,123,245,0.1) !important; margin: 1rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(75,123,245,0.3); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:4px 0 12px">
      <div style="width:32px;height:32px;background:linear-gradient(135deg,#4B7BF5,#7B5CF5);
                  border-radius:8px;display:flex;align-items:center;justify-content:center;
                  font-size:18px;">⚽</div>
      <div>
        <div style="font-weight:800;font-size:1rem;color:#e8eaf0;line-height:1.1;">TacticsAI</div>
        <div style="font-size:0.65rem;color:rgba(232,234,240,0.4);letter-spacing:0.05em;">
          WEEK 1 · 2 · 3
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown('<p style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;color:rgba(232,234,240,0.35);text-transform:uppercase;margin-bottom:8px;">Model</p>', unsafe_allow_html=True)
    model_opts = {
        "yolov8x  (base, max accuracy)": "yolov8x.pt",
        "yolov8l  (large, balanced)":    "yolov8l.pt",
        "yolov8m  (medium, fast)":       "yolov8m.pt",
        "best.pt  (fine-tuned)":         "best.pt",
    }
    model_path = model_opts[st.selectbox("Weights", list(model_opts), label_visibility="collapsed")]
    device = st.radio("Device", ["cuda", "cpu"], horizontal=True)

    st.divider()
    st.markdown('<p style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;color:rgba(232,234,240,0.35);text-transform:uppercase;margin-bottom:8px;">Processing</p>', unsafe_allow_html=True)
    skip_frames = st.slider("Frame skip", 0, 5, 0, help="0 = every frame")
    raw_max = st.number_input("Max frames (0 = full)", 0, 10000, 300)
    max_frames = None if raw_max == 0 else int(raw_max)

    st.divider()
    st.markdown('<p style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;color:rgba(232,234,240,0.35);text-transform:uppercase;margin-bottom:8px;">Analytics</p>', unsafe_allow_html=True)
    enable_teams  = st.toggle("Team classification", value=True)
    enable_birds  = st.toggle("Bird\'s-eye + Heatmap", value=True)
    calib_frames  = st.slider("Calibration frames", 20, 120, CALIBRATION_FRAMES, disabled=not enable_teams)

    st.divider()
    # Roadmap checklist
    st.markdown('<p style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;color:rgba(232,234,240,0.35);text-transform:uppercase;margin-bottom:8px;">Roadmap</p>', unsafe_allow_html=True)
    for label, done in [
        ("Detection (YOLOv8)",    True),
        ("Tracking (ByteTrack)",  True),
        ("Team classification",   True),
        ("Bird\'s-eye view",      True),
        ("Formation detection",   True),
        ("Heatmaps",              True),
        ("Tactical metrics",      True),
        ("Pass network",          True),
        ("Pressing triggers",     True),
        ("Speed tracking",        True),
        ("Player re-ID",          True),
        ("Fine-tuned YOLOv8",     False),
    ]:
        icon = '<span style="color:#4B7BF5;">✓</span>' if done else '<span style="color:rgba(232,234,240,0.25);">○</span>'
        color = "#e8eaf0" if done else "rgba(232,234,240,0.35)"
        st.markdown(f'<div style="display:flex;gap:8px;align-items:center;padding:2px 0;">'
                    f'{icon}<span style="font-size:0.78rem;color:{color};">{label}</span></div>',
                    unsafe_allow_html=True)


# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:2.5rem 0 1.5rem">
  <div style="display:inline-flex;align-items:center;gap:8px;
              background:rgba(75,123,245,0.1);border:1px solid rgba(75,123,245,0.25);
              border-radius:20px;padding:4px 14px;margin-bottom:16px;">
    <span style="width:7px;height:7px;background:#4B7BF5;border-radius:50%;
                 box-shadow:0 0 8px #4B7BF5;display:inline-block;"></span>
    <span style="font-size:0.72rem;font-weight:600;letter-spacing:0.08em;color:#4B7BF5;">
      WEEK 3 — FORMATION · HEATMAPS · TACTICAL METRICS
    </span>
  </div>
  <h1 style="font-size:3.2rem;font-weight:900;letter-spacing:-0.03em;margin:0;line-height:1.05;
             background:linear-gradient(135deg,#e8eaf0 0%,#4B7BF5 60%,#7B5CF5 100%);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    TacticsAI
  </h1>
  <p style="font-size:1rem;color:rgba(232,234,240,0.45);margin-top:8px;font-weight:400;">
    Real-time football tactical analysis — detection, tracking, teams, formations &amp; heatmaps
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload match clip", type=["mp4", "avi", "mov"],
                            label_visibility="collapsed")

if uploaded is None:
    st.markdown("""
    <div style="background:rgba(75,123,245,0.05);border:1px solid rgba(75,123,245,0.15);
                border-radius:14px;padding:2rem 2.5rem;margin-top:1rem;">
      <p style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;
                color:rgba(75,123,245,0.8);text-transform:uppercase;margin:0 0 12px;">
        What's new in Week 3
      </p>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(75,123,245,0.12);
                    border-radius:10px;padding:14px 16px;">
          <div style="font-size:0.82rem;font-weight:700;color:#e8eaf0;margin-bottom:4px;">
            Formation Detection
          </div>
          <div style="font-size:0.75rem;color:rgba(232,234,240,0.45);">
            K-Means row clustering + rolling vote smoothing
          </div>
        </div>
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(75,123,245,0.12);
                    border-radius:10px;padding:14px 16px;">
          <div style="font-size:0.82rem;font-weight:700;color:#e8eaf0;margin-bottom:4px;">
            Gaussian Heatmaps
          </div>
          <div style="font-size:0.75rem;color:rgba(232,234,240,0.45);">
            Cumulative per-team movement density maps
          </div>
        </div>
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(75,123,245,0.12);
                    border-radius:10px;padding:14px 16px;">
          <div style="font-size:0.82rem;font-weight:700;color:#e8eaf0;margin-bottom:4px;">
            Tactical Metrics
          </div>
          <div style="font-size:0.75rem;color:rgba(232,234,240,0.45);">
            Width / depth / pressing intensity / space control
          </div>
        </div>
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(75,123,245,0.12);
                    border-radius:10px;padding:14px 16px;">
          <div style="font-size:0.82rem;font-weight:700;color:#e8eaf0;margin-bottom:4px;">
            Voronoi Space Control
          </div>
          <div style="font-size:0.75rem;color:rgba(232,234,240,0.45);">
            Pitch area dominated by each team per frame
          </div>
        </div>
      </div>
      <p style="margin:16px 0 0;font-size:0.75rem;color:rgba(232,234,240,0.3);">
        Quick start: upload <code style="background:rgba(75,123,245,0.15);padding:2px 6px;
        border-radius:4px;color:#4B7BF5;">data/tryolabs_demo.mp4</code>
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Save + video info ─────────────────────────────────────────────────────────
with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
    tmp.write(uploaded.read())
    video_path = tmp.name

try:
    info = get_video_info(video_path)
except FileNotFoundError as exc:
    st.error(str(exc)); st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Duration",      f"{info['duration_s']:.1f} s")
c2.metric("FPS",           f"{info['fps']:.0f}")
c3.metric("Resolution",    f"{info['width']}×{info['height']}")
c4.metric("Total frames",  f"{info['total_frames']:,}")


# ── Pitch calibration ─────────────────────────────────────────────────────────
corners: list = DEFAULT_SRC_POINTS   # fallback

if enable_birds:
    cap_prev = cv2.VideoCapture(video_path)
    ok, first_frame = cap_prev.read()
    cap_prev.release()

    if ok:
        fH, fW = first_frame.shape[:2]

        # Auto-detect corners from this specific frame
        auto_corners = auto_detect_pitch_corners(first_frame, ads_strip_height=ADS_STRIP_HEIGHT)

        # Show calibration expander
        with st.expander("Pitch Corner Calibration — auto-detected, adjust if needed",
                         expanded=True):

            # ── Two-column layout: controls | warp preview ────────────────────
            ctrl_col, prev_col = st.columns([1, 2], gap="medium")

            with ctrl_col:
                st.markdown("""
                <p style="font-size:0.78rem;color:rgba(232,234,240,0.6);margin:0 0 10px;">
                  Corners auto-detected from green grass + pitch-line Hough.
                  Adjust if the warped preview (right) doesn't look correct.
                  <br><strong>Order: TL → TR → BR → BL</strong>
                </p>
                """, unsafe_allow_html=True)

                labels_cal = ["Top-Left (far)", "Top-Right (far)",
                              "Bottom-Right (near)", "Bottom-Left (near)"]
                corners = []
                for i, (lbl, default) in enumerate(zip(labels_cal, auto_corners)):
                    st.caption(lbl)
                    c_row = st.columns(2)
                    with c_row[0]:
                        cx = st.number_input(f"X", 0, fW, int(default[0]),
                                             key=f"cal_x{i}", label_visibility="collapsed")
                    with c_row[1]:
                        cy = st.number_input(f"Y", 0, fH, int(default[1]),
                                             key=f"cal_y{i}", label_visibility="collapsed")
                    corners.append([cx, cy])
                    st.markdown("")

                # Ads strip control
                st.caption("Exclude top N px (scoreboard/stands)")
                ads_strip = st.slider(
                    "Top exclude", 0, 300, ADS_STRIP_HEIGHT,
                    key="ads_strip", label_visibility="collapsed",
                )

            with prev_col:
                # Draw corners + polygon on first frame
                corners_vis = draw_corners_on_frame(first_frame, corners)
                st.image(cv2.cvtColor(corners_vis, cv2.COLOR_BGR2RGB),
                         caption="Corners on original frame", width='stretch')

                # Warped preview — this is the key validation
                warped_prev = validate_transform(
                    first_frame, corners,
                    dst_width=1040, dst_height=680,
                )
                st.image(cv2.cvtColor(warped_prev, cv2.COLOR_BGR2RGB),
                         caption="Warped preview (should show flat grass)",
                         width='stretch')

                st.markdown("""
                <p style="font-size:0.72rem;color:rgba(232,234,240,0.4);margin:4px 0 0;">
                  If the warped preview shows a rectangular grass area with players
                  in correct relative positions, calibration is good.
                </p>
                """, unsafe_allow_html=True)
    else:
        corners = DEFAULT_SRC_POINTS

st.divider()

# ── Run button ────────────────────────────────────────────────────────────────
if not st.button("Run Analysis", type="primary", width='stretch'):
    st.stop()


# ── Main layout ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="medium")

with col_left:
    # Tab: Live video / Heatmap
    tab_video, tab_heat = st.tabs(["Annotated Output", "Cumulative Heatmap"])
    with tab_video:
        video_ph = st.empty()
    with tab_heat:
        heat_ph = st.empty()

with col_right:
    # Tab: Bird's-eye / Tactical metrics
    tab_pitch, tab_metrics = st.tabs(["Bird\'s-eye View", "Tactical Metrics"])
    with tab_pitch:
        pitch_ph = st.empty()
    with tab_metrics:
        metrics_ph = st.empty()

# ── Stats row ─────────────────────────────────────────────────────────────────
st.markdown('<p style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;'
            'color:rgba(232,234,240,0.35);text-transform:uppercase;margin:1.5rem 0 0.5rem;">Live Stats</p>',
            unsafe_allow_html=True)

stat_cols = st.columns(8)
prog_bar  = st.progress(0.0, text="Starting…")

m_frame   = stat_cols[0].empty()
m_players = stat_cols[1].empty()
m_ball    = stat_cols[2].empty()
m_ts      = stat_cols[3].empty()
m_team_a  = stat_cols[4].empty()
m_team_b  = stat_cols[5].empty()
m_fa      = stat_cols[6].empty()
m_fb      = stat_cols[7].empty()

st.markdown('<p style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;'
            'color:rgba(232,234,240,0.35);text-transform:uppercase;margin:1.5rem 0 0.5rem;">Player count history</p>',
            unsafe_allow_html=True)
chart_ph = st.empty()

# ── State ─────────────────────────────────────────────────────────────────────
player_history: list = []
last_heatmap   = None


# ── Processing stream ─────────────────────────────────────────────────────────
for result in process_video(
    video_path,
    model_path=model_path,
    device=device,
    skip_frames=skip_frames,
    max_frames=max_frames,
    show_progress=False,
    src_points=corners,
    calibration_frames=calib_frames,
    enable_team_classification=enable_teams,
    enable_bird_eye=enable_birds and enable_teams,
):
    # ── Video frame ───────────────────────────────────────────────────────────
    rgb = cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB)
    with tab_video:
        video_ph.image(rgb, width='stretch')

    # ── Heatmap tab ───────────────────────────────────────────────────────────
    if result.get("heatmap_view") is not None:
        last_heatmap = result["heatmap_view"]
    if last_heatmap is not None:
        with tab_heat:
            heat_ph.image(cv2.cvtColor(last_heatmap, cv2.COLOR_BGR2RGB),
                          width='stretch')

    # ── Bird's-eye ────────────────────────────────────────────────────────────
    if result.get("pitch_view") is not None:
        with tab_pitch:
            pitch_ph.image(cv2.cvtColor(result["pitch_view"], cv2.COLOR_BGR2RGB),
                           width='stretch')

    # ── Tactical metrics panel ────────────────────────────────────────────────
    tm = result.get("tactical_metrics", {})
    formations = result.get("formations", {})
    speed_summary = result.get("team_speed_summary", {})
    pressing_evts = result.get("pressing_events", [])
    pressing_st   = result.get("pressing_state", {})

    if tm:
        sc = tm.get("space_control", {})
        pi = tm.get("pressing_intensity", {})
        teams_tm = tm.get("teams", {})

        rows = []
        for tid in range(2):
            t   = teams_tm.get(tid, {})
            spd = speed_summary.get(tid, {})
            is_pressing = pressing_st.get(tid, False)
            press_icon  = " 🔴 **PRESSING**" if is_pressing else ""

            rows.append(
                f"**Team {'AB'[tid]}**{press_icon} &nbsp;|&nbsp; "
                f"Formation: `{formations.get(tid, '—')}` &nbsp;|&nbsp; "
                f"Width: **{t.get('width_m', 0)} m** &nbsp;|&nbsp; "
                f"Depth: **{t.get('depth_m', 0)} m** &nbsp;|&nbsp; "
                f"Press dist: **{pi.get(tid, 0)} m** &nbsp;|&nbsp; "
                f"Space: **{sc.get(tid, 0)}%**\n\n"
                f"&nbsp;&nbsp;&nbsp;&nbsp;"
                f"Dist run: **{spd.get('total_dist_m', 0)} m** &nbsp;|&nbsp; "
                f"Max speed: **{spd.get('team_max_spd_kmh', 0)} km/h** &nbsp;|&nbsp; "
                f"Top sprinter ID: `{spd.get('top_sprinter_id', '—')}`"
            )
        ball_zone = tm.get("ball_zone", "—")

        # Pressing events log
        evt_lines = []
        for evt in pressing_evts:
            evt_lines.append(
                f"- t={evt.timestamp:.1f}s — "
                f"Team {'AB'[evt.team_id]} pressed @ avg {evt.avg_dist_m} m"
            )
        evt_block = ("\n\n**Pressing events:**\n" + "\n".join(evt_lines)) if evt_lines else ""

        metrics_content = (
            "\n\n---\n\n".join(rows)
            + f"\n\n---\n\nBall zone: **{ball_zone}**"
            + evt_block
        )
        with tab_metrics:
            metrics_ph.markdown(metrics_content)

    # ── Progress ──────────────────────────────────────────────────────────────
    prog_bar.progress(
        float(result["progress"]),
        text=f"Frame {result['frame_idx']} / {result['total_frames']}",
    )

    # ── Stats metrics ─────────────────────────────────────────────────────────
    tc = result["team_counts"]
    m_frame.metric("Frame",    result["frame_idx"])
    m_players.metric("Players", result["player_count"])
    m_ball.metric("Ball",      "Yes" if result["ball_detected"] else "No")
    m_ts.metric("Time",        f"{result['timestamp']:.1f}s")
    m_team_a.metric("Team A",  tc.get(0, 0))
    m_team_b.metric("Team B",  tc.get(1, 0))
    m_fa.metric("Form. A",     formations.get(0, "—"))
    m_fb.metric("Form. B",     formations.get(1, "—"))

    # ── Chart ─────────────────────────────────────────────────────────────────
    player_history.append({
        "Total": result["player_count"],
        "Team A": tc.get(0, 0),
        "Team B": tc.get(1, 0),
    })
    if len(player_history) > 1:
        chart_ph.line_chart(player_history, height=160)


prog_bar.progress(1.0, text="Analysis complete")
st.success(f"Done — {len(player_history)} frames processed.")
