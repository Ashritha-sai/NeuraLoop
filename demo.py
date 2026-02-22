import streamlit as st
import threading, time, json, os, random
from pathlib import Path
import statistics
from statistics import mean
from dotenv import load_dotenv
load_dotenv()

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

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
        self._thread = None

@st.cache_resource
def _get_hw():
    return HW()

hw = _get_hw()

@st.fragment(run_every=0.5)
def live_pupil_metrics():
    with hw.lock:
        recent = [x[3] for x in hw.pupil if x[0] > time.time() - 0.5]
    raw = mean(recent) if recent else 0.0
    delta = raw - hw.baseline_mm if recent else 0.0
    bw_live = [b for b in hw.blinks if b[0] >= st.session_state.t_start]
    with hw.lock:
        scount = len([x for x in hw.pupil if x[0] >= st.session_state.t_start])
    c1, c2, c3 = st.columns(3)
    c1.metric("Live Pupil (mm)", f"{raw:.2f}", delta=f"{delta:+.3f} mm")
    c2.metric("Blinks", len(bw_live))
    c3.metric("Samples", scount)

def _read(inlet):
    while True:
        try:
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
        except Exception:
            time.sleep(0.1)

def window(buf, t0, t1):
    with hw.lock:
        return [x for x in buf if t0 <= x[0] <= t1]

# ── IMPLICIT SIGNAL LAYER ──────────────────────────────

def compute_confusion_score(pupil_window, gaze_window, baseline_mm):
    means = [x[3] for x in pupil_window]
    if not means:
        return 0.5, 0.0, 0, 0.5
    mean_delta = statistics.mean(means) - baseline_mm
    pupil_component = min(max(mean_delta / 0.6, 0.0), 1.0)
    xs = [g[1] for g in gaze_window]
    regression_count = 0
    if len(xs) > 10:
        x_range = max(xs) - min(xs) if max(xs) != min(xs) else 1
        for i in range(2, len(xs)):
            moved_forward = xs[i-2] < xs[i-1]
            went_back = xs[i] < xs[i-1] - (0.05 * x_range)
            if moved_forward and went_back:
                regression_count += 1
    regression_component = min(regression_count / 3.0, 1.0)
    confusion = (0.60 * pupil_component) + (0.40 * regression_component)
    return round(confusion, 3), round(mean_delta, 3), regression_count, round(pupil_component, 3)


def compute_attention_score(blink_window, read_secs, sample_count):
    if read_secs <= 0 or sample_count == 0:
        return 0.5
    blink_rate = (len(blink_window) / read_secs) * 60
    if blink_rate < 4:
        score = 0.6
    elif blink_rate < 8:
        score = 1.0
    elif blink_rate < 20:
        score = 0.75
    elif blink_rate < 25:
        score = 0.4
    else:
        score = 0.2
    expected_samples = read_secs * 195
    coverage = sample_count / max(expected_samples, 1)
    if coverage < 0.5:
        score *= 0.7
    return round(score, 3)


def compute_annotation_confidence(annotation, confusion_score, attention_score,
                                   read_secs, mean_delta):
    explicit = 1.0 if annotation in ("YES", "NO") else 0.0
    if annotation == "YES":
        implicit_consistency = 1.0 - (confusion_score * 0.5)
    elif annotation == "NO":
        implicit_consistency = 0.5 + (confusion_score * 0.5)
    else:
        implicit_consistency = 0.3
    if read_secs < 1.5:
        speed_penalty = 0.15
    elif read_secs < 2.5:
        speed_penalty = 0.05
    else:
        speed_penalty = 0.0
    confidence = (
        0.45 * explicit +
        0.30 * implicit_consistency +
        0.25 * attention_score
    ) - speed_penalty
    return round(max(min(confidence, 1.0), 0.0), 3)


def compute_session_score(results):
    if not results:
        return 0.0, {}
    annotated = [r for r in results if r["annotation"] in ("YES", "NO")]
    coverage = len(annotated) / len(results)
    mean_confidence = statistics.mean(
        r["annotation_confidence"] for r in results
    ) if results else 0.0
    attention_consistency = statistics.mean(
        r["attention_score"] for r in results
    ) if results else 0.0
    signal_quality = sum(
        1 for r in results if r["samples"] > 50
    ) / len(results)
    session_score = (
        0.40 * mean_confidence +
        0.20 * coverage +
        0.20 * attention_consistency +
        0.20 * signal_quality
    )
    grade = (
        "EXCELLENT" if session_score > 0.85 else
        "GOOD"      if session_score > 0.70 else
        "ACCEPTABLE" if session_score > 0.55 else
        "LOW QUALITY"
    )
    return round(session_score, 3), {
        "grade": grade,
        "coverage": round(coverage, 3),
        "mean_confidence": round(mean_confidence, 3),
        "attention_consistency": round(attention_consistency, 3),
        "signal_quality": round(signal_quality, 3)
    }

# ── Sentences ────────────────────────────────────────────────────────
FALLBACK_SENTENCES = [
    "Octopuses have three hearts and blue blood.",
    "The ancient Romans used concrete that grew stronger over centuries from seawater exposure.",
    "A single bolt of lightning carries enough energy to toast about a hundred thousand slices of bread.",
    "She finished the marathon despite the rain.",
    "Honey never spoils because its low moisture and acidic pH create an inhospitable environment for bacteria.",
]

def _fetch_sentences(result_list):
    if not HAS_GROQ:
        result_list.append(FALLBACK_SENTENCES)
        return
    try:
        topics = ["astronomy", "ocean life", "ancient history", "cooking", "architecture",
                  "weather patterns", "music theory", "mathematics", "animal behavior",
                  "world travel", "medicine", "literature", "geology", "photography",
                  "economics", "botany", "mythology", "robotics", "linguistics", "sports"]
        chosen = random.sample(topics, 3)
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Return ONLY a JSON array of exactly 5 standalone English sentences. No markdown, no explanation, no code fences — just the raw JSON array. Each sentence must be completely self-contained and meaningful on its own — a reader should fully understand it without any prior context. Vary the length: one short (5-8 words), two medium (10-18 words), one long detailed sentence (20-30 words), and one more short. Each sentence should teach something interesting or paint a vivid picture. Never produce vague filler like 'close the door' or 'it will rain tomorrow'."
                },
                {
                    "role": "user",
                    "content": f"Generate 5 unique sentences touching on these topics: {', '.join(chosen)}."
                }
            ],
            max_tokens=300,
            temperature=0.9,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        sentences = json.loads(text)
        if isinstance(sentences, list) and len(sentences) == 5 and all(isinstance(s, str) for s in sentences):
            result_list.append(sentences)
        else:
            result_list.append(FALLBACK_SENTENCES)
    except Exception:
        result_list.append(FALLBACK_SENTENCES)

# ── Streamlit UI ─────────────────────────────────────────────────────
st.set_page_config(page_title="Implicit Annotator", layout="wide")

for k, v in {"phase": "start", "idx": 0, "results": [], "t_start": 0.0, "t_baseline": 0.0, "sentences": [], "_sentence_result": [], "_sentence_thread": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── PHASE: start ─────────────────────────────────────────────────────
if st.session_state.phase == "start":
    st.title("Implicit — Neural Annotation")
    if st.button("Connect and Start"):
        connected = False
        if HAS_LSL:
            try:
                streams = resolve_byprop("type", "Gaze", timeout=8)
                if streams:
                    inlet = StreamInlet(streams[0])
                    t = threading.Thread(target=_read, args=(inlet,), daemon=True)
                    t.start()
                    hw._thread = t
                    connected = True
            except Exception:
                connected = False
        hw.connected = connected
        st.session_state.phase = "baseline"
        st.session_state.t_baseline = time.time()
        st.rerun()

# ── PHASE: baseline ─────────────────────────────────────────────────
elif st.session_state.phase == "baseline":
    if st.session_state["_sentence_thread"] is None:
        result_list = []
        st.session_state["_sentence_result"] = result_list
        t = threading.Thread(target=_fetch_sentences, args=(result_list,), daemon=True)
        st.session_state["_sentence_thread"] = t
        t.start()
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
        thread = st.session_state["_sentence_thread"]
        if thread is not None:
            thread.join(timeout=5)
        result_list = st.session_state["_sentence_result"]
        if result_list and isinstance(result_list[0], list) and len(result_list[0]) == 5:
            st.session_state["sentences"] = result_list[0]
        else:
            st.session_state["sentences"] = FALLBACK_SENTENCES
        assert len(st.session_state["sentences"]) == 5
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
    sentence = st.session_state["sentences"][idx]
    st.markdown(
        f"<div style='background:#1e1e1e;color:#fff;padding:40px;border-radius:12px;"
        f"font-size:1.5rem;text-align:center;margin:20px 0'>{sentence}</div>",
        unsafe_allow_html=True,
    )
    live_pupil_metrics()

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
        confusion, mean_d, reg_count, pupil_comp = compute_confusion_score(pw, gw, hw.baseline_mm)
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
            # ── implicit signal scores ──
            "confusion_score":        confusion,
            "pupil_component":        pupil_comp,
            "regression_count":       reg_count,
            "attention_score":        compute_attention_score(bw, t_end-t0, len(pw)),
            "annotation_confidence":  0.0,
            "rewrite_triggered":      confusion > 0.55,
        })
        results = st.session_state.results
        att = results[-1]["attention_score"]
        conf = compute_annotation_confidence(
            results[-1]["annotation"], confusion, att,
            results[-1]["read_secs"], results[-1]["mean_delta"]
        )
        results[-1]["annotation_confidence"] = conf
        if idx + 1 < len(st.session_state["sentences"]):
            st.session_state.idx = idx + 1
            st.session_state.t_start = time.time()
            st.rerun()
        else:
            st.session_state.phase = "results"
            st.rerun()
    else:
        pass  # fragment auto-refreshes live metrics

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

    # ── Session confidence score ─────────────────────────────────────
    session_score, breakdown = compute_session_score(results)
    st.divider()
    st.metric("Session Confidence Score", f"{session_score:.2f}",
              delta=breakdown.get("grade", ""))
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Coverage", f"{breakdown.get('coverage', 0):.3f}")
    sc2.metric("Mean Confidence", f"{breakdown.get('mean_confidence', 0):.3f}")
    sc3.metric("Attention", f"{breakdown.get('attention_consistency', 0):.3f}")
    sc4.metric("Signal Quality", f"{breakdown.get('signal_quality', 0):.3f}")

    # ── Attention warnings ───────────────────────────────────────────
    low_att = [r for r in results if r.get("attention_score", 1) < 0.5]
    if low_att:
        st.warning(f"Low attention on {len(low_att)} sentence(s):")
        for r in low_att:
            st.markdown(f"- {r['sentence']}")

    # ── Rewrite candidates ───────────────────────────────────────────
    rewrites = [r for r in results if r.get("rewrite_triggered")]
    if rewrites:
        st.info(f"{len(rewrites)} sentence(s) would be rewritten in the live LLM loop:")
        for r in rewrites:
            st.markdown(f"- {r['sentence']} (confusion: {r['confusion_score']:.3f})")

    # ── DPO pair quality ─────────────────────────────────────────────
    high_conf = [r for r in results if r.get("annotation_confidence", 0) > 0.72]
    st.markdown(f"**{len(high_conf)} high-quality DPO pairs generated this session**")

    outpath = Path("data/demo_output.json")
    outpath.parent.mkdir(exist_ok=True)
    output = {"results": results, "session_score": session_score, "breakdown": breakdown}
    outpath.write_text(json.dumps(output, indent=2))
    st.caption(f"Saved to {outpath}")

    if st.button("New Session"):
        with hw.lock:
            hw.pupil.clear()
            hw.gaze.clear()
            hw.eyelid.clear()
            hw.blinks.clear()
            hw._bstate = "OPEN"
            hw._bopen = None
        st.session_state.phase = "start"
        st.session_state.idx = 0
        st.session_state.results = []
        st.session_state.t_start = 0.0
        st.session_state.t_baseline = 0.0
        st.session_state["sentences"] = []
        st.session_state["_sentence_result"] = []
        st.session_state["_sentence_thread"] = None
        st.rerun()
