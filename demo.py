import streamlit as st
import threading, time, json, os
from pathlib import Path
from statistics import mean

try:
    from pylsl import resolve_byprop, StreamInlet
    HAS_LSL = True
except ImportError:
    HAS_LSL = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ── Hardware layer (module level, no Streamlit) ──────────────────────
class HW:
    def __init__(self):
        self.connected = False
        self.baseline_mm = 4.0
        self.pupil  = []   # (t, left, right, mean)
        self.gaze   = []   # (t, x, y)
        self.eyelid = []   # (t, left, right)
        self.blinks = []   # (t, duration_ms)
        self._bstate = "OPEN"
        self._bopen  = None
        self.lock = threading.Lock()
        self._thread_started = False

@st.cache_resource
def _get_hw():
    return HW()

hw = _get_hw()

def _read(inlet):
    while True:
        s, _ = inlet.pull_sample(timeout=0.05)
        if not s:
            continue
        t = time.time()
        pl, pr = s[2], s[9]
        el, er = s[18], s[21]
        em = (el + er) / 2
        with hw.lock:
            hw.pupil.append((t, pl, pr, (pl + pr) / 2))
            hw.gaze.append((t, s[0], s[1]))
            hw.eyelid.append((t, el, er))
            if hw._bstate == "OPEN" and em < 2.0:
                hw._bstate = "CLOSING"
                hw._bopen = t
            elif hw._bstate == "CLOSING" and em > 5.0:
                dur = (t - hw._bopen) * 1000
                if dur > 50:
                    hw.blinks.append((t, dur))
                hw._bstate = "OPEN"
            cutoff = t - 60
            hw.pupil  = [x for x in hw.pupil  if x[0] > cutoff]
            hw.gaze   = [x for x in hw.gaze   if x[0] > cutoff]
            hw.eyelid = [x for x in hw.eyelid if x[0] > cutoff]

def window(buf, t0, t1):
    with hw.lock:
        return [x for x in buf if t0 <= x[0] <= t1]

# ── Sentences ────────────────────────────────────────────────────────
SENTENCES = [
    "The patient should take two tablets with water twice daily.",
    "Aspirin irreversibly inhibits COX-1 and COX-2 enzymes, reducing prostaglandin synthesis and platelet aggregation.",
    "Click save before closing the application.",
    "Stochastic gradient descent with momentum converges faster than vanilla gradient descent on non-convex loss surfaces.",
    "The meeting is confirmed for Thursday at two in the afternoon.",
    "Pharmacokinetic variability across patients necessitates individualised dosimetric recalibration for optimal outcomes.",
]

# ── Streamlit UI ─────────────────────────────────────────────────────
st.set_page_config(page_title="Implicit Annotator", layout="wide")

for k, v in {"phase": "start", "idx": 0, "results": [], "t_start": 0.0, "t_baseline": 0.0}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── PHASE: start ─────────────────────────────────────────────────────
if st.session_state.phase == "start":
    st.title("Implicit — Neural Annotation")
    if st.button("Connect and Start"):
        if not hw._thread_started:
            connected = False
            if HAS_LSL:
                try:
                    streams = resolve_byprop("type", "Gaze", timeout=8)
                    if streams:
                        inlet = StreamInlet(streams[0])
                        threading.Thread(target=_read, args=(inlet,), daemon=True).start()
                        hw._thread_started = True
                        connected = True
                except Exception:
                    connected = False
            hw.connected = connected
        st.session_state.phase = "baseline"
        st.session_state.t_baseline = time.time()
        st.rerun()

# ── PHASE: baseline ─────────────────────────────────────────────────
elif st.session_state.phase == "baseline":
    elapsed = time.time() - st.session_state.t_baseline
    remaining = max(0, 10 - elapsed)
    st.subheader("Relax and look at the screen")
    st.progress(min(elapsed / 10, 1.0))
    st.caption(f"{remaining:.1f}s remaining")
    with hw.lock:
        recent = [x[3] for x in hw.pupil if x[0] > time.time() - 0.5]
    st.metric("Live Pupil (mm)", f"{mean(recent):.2f}" if recent else "—")
    if elapsed >= 10:
        with hw.lock:
            vals = [x[3] for x in hw.pupil]
        hw.baseline_mm = mean(vals) if vals else 4.0
        st.session_state.phase = "annotate"
        st.session_state.idx = 0
        st.session_state.t_start = time.time()
        st.rerun()
    else:
        time.sleep(0.3)
        st.rerun()

# ── PHASE: annotate ─────────────────────────────────────────────────
elif st.session_state.phase == "annotate":
    idx = st.session_state.idx
    sentence = SENTENCES[idx]
    st.markdown(
        f"<div style='background:#1e1e1e;color:#fff;padding:40px;border-radius:12px;"
        f"font-size:1.5rem;text-align:center;margin:20px 0'>{sentence}</div>",
        unsafe_allow_html=True,
    )
    with hw.lock:
        recent = [x[3] for x in hw.pupil if x[0] > time.time() - 0.5]
    delta = mean(recent) - hw.baseline_mm if recent else 0.0
    color = "green" if abs(delta) < 0.3 else "orange" if abs(delta) < 0.6 else "red"
    bw_live = [b for b in hw.blinks if b[0] >= st.session_state.t_start]
    with hw.lock:
        scount = len([x for x in hw.pupil if x[0] >= st.session_state.t_start])
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Pupil Δ:** :{color}[{delta:+.3f} mm]")
    c2.metric("Blinks", len(bw_live))
    c3.metric("Samples", scount)

    col1, col2 = st.columns(2)
    yes_clicked = col1.button("YES", type="primary", use_container_width=True)
    no_clicked = col2.button("NO", type="secondary", use_container_width=True)

    if yes_clicked or no_clicked:
        t_end = time.time()
        t0 = st.session_state.t_start
        pw = window(hw.pupil, t0, t_end)
        gw = window(hw.gaze, t0, t_end)
        ew = window(hw.eyelid, t0, t_end)
        bw = [b for b in hw.blinks if t0 <= b[0] <= t_end]
        means = [x[3] for x in pw]
        mean_d = mean(means) - hw.baseline_mm if means else 0.0
        peak_d = max(means) - hw.baseline_mm if means else 0.0
        if means:
            pi = means.index(max(means))
            frac = pi / len(means)
            when = "early" if frac < 0.33 else "middle" if frac < 0.66 else "late"
        else:
            when = "—"
        xs = [x[1] for x in gw]
        ys = [x[2] for x in gw]
        st.session_state.results.append({
            "sentence":     sentence[:50] + "...",
            "annotation":   "YES" if yes_clicked else "NO",
            "mean_delta":   round(mean_d, 3),
            "peak_delta":   round(peak_d, 3),
            "peak_when":    when,
            "eyelid_mm":    round(mean([(x[1]+x[2])/2 for x in ew]), 2) if ew else 0,
            "blinks":       len(bw),
            "gaze_x_range": round(max(xs) - min(xs), 0) if xs else 0,
            "gaze_y_range": round(max(ys) - min(ys), 0) if ys else 0,
            "read_secs":    round(t_end - t0, 1),
            "samples":      len(pw),
            "confused":     mean_d > 0.4,
            "timeseries":   [{"t": round(x[0]-t0, 2), "pupil": round(x[3], 3)} for x in pw],
        })
        if idx + 1 < len(SENTENCES):
            st.session_state.idx = idx + 1
            st.session_state.t_start = time.time()
            st.rerun()
        else:
            st.session_state.phase = "results"
            st.rerun()
    else:
        time.sleep(0.3)
        st.rerun()

# ── PHASE: results ───────────────────────────────────────────────────
elif st.session_state.phase == "results":
    results = st.session_state.results
    display = [{k: v for k, v in r.items() if k != "timeseries"} for r in results]
    st.dataframe(display, use_container_width=True)

    if HAS_PLOTLY:
        fig = go.Figure()
        colors = ["#636EFA","#EF553B","#00CC96","#AB63FA","#FFA15A","#19D3F3"]
        for i, r in enumerate(results):
            ts = r["timeseries"]
            if ts:
                fig.add_trace(go.Scatter(
                    x=[p["t"] for p in ts], y=[p["pupil"] for p in ts],
                    mode="lines", name=r["sentence"][:30], line=dict(color=colors[i % len(colors)]),
                ))
        fig.add_hline(y=hw.baseline_mm, line_dash="dash", line_color="gray",
                      annotation_text="baseline", annotation_position="top left")
        fig.update_layout(title="Pupil Dilation Over Reading Time",
                          xaxis_title="Time (s)", yaxis_title="Pupil Mean (mm)")
        st.plotly_chart(fig, use_container_width=True)

    deltas = [r["mean_delta"] for r in results]
    total_blinks = sum(r["blinks"] for r in results)
    yes_n = sum(1 for r in results if r["annotation"] == "YES")
    no_n = len(results) - yes_n
    st.markdown(f"**Mean Δ across session:** {mean(deltas):.3f} mm" if deltas else "")
    st.markdown(f"**Total blinks:** {total_blinks}")
    st.markdown(f"**YES:** {yes_n} / **NO:** {no_n}")
    st.markdown(f"**Hardware:** {'NEON Live' if hw.connected else 'Mock Mode'}")

    outpath = Path("data/demo_output.json")
    outpath.parent.mkdir(exist_ok=True)
    outpath.write_text(json.dumps(results, indent=2))
    st.caption(f"Saved to {outpath}")

    if st.button("New Session"):
        st.session_state.phase = "start"
        st.session_state.idx = 0
        st.session_state.results = []
        st.session_state.t_start = 0.0
        st.session_state.t_baseline = 0.0
        st.rerun()
