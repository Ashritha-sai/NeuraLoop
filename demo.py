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

def _generate_neural_diagnosis(result_dict):
    parts = []
    cs = result_dict["confusion_score"]
    md = result_dict.get("pupil_mean_delta", result_dict["mean_delta"])
    pd_ = result_dict.get("pupil_peak_delta", result_dict["peak_delta"])
    pw = result_dict["peak_when"]
    rc = result_dict["regression_count"]
    ann = result_dict["annotation"]

    if ann == "YES" and cs < 0.3:
        return f"Low confusion ({cs}), steady pupil response. Sentence processed fluently."
    if ann == "YES" and cs >= 0.3:
        return f"User accepted despite moderate confusion ({cs}). Pupil delta {md:+.3f}mm with {rc} regressions — possible tolerance of complexity."
    # ann == "NO" cases:
    if pd_ > 0.5:
        parts.append(f"High pupil dilation (peak {pd_:+.3f}mm at {pw} of sentence)")
    if rc >= 3:
        parts.append(f"{rc} gaze regressions indicating re-reading")
    if cs > 0.55:
        parts.append(f"confusion score {cs}")
    if not parts:
        parts.append(f"Moderate neural difficulty signals (confusion {cs}, delta {md:+.3f}mm)")
    return ". ".join(parts) + " — suggests cognitive processing difficulty."

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

def _rewrite_sentences(sentences_with_data, result_list):
    """Background thread: ask Groq to rewrite unclear sentences using neural signals."""
    fallback = [
        {"diagnosis": "Rewrite generated without neural analysis (API unavailable).",
         "rewrite": "In simpler terms, " + s["sentence"]}
        for s in sentences_with_data
    ]
    if not HAS_GROQ:
        result_list.append(fallback)
        return
    try:
        lines = []
        for i, s in enumerate(sentences_with_data, 1):
            lines.append(
                f"Sentence {i}: '{s['sentence']}'\n"
                f"- Pupil dilation: mean {s['pupil_mean_delta']:+.3f}mm, "
                f"peak {s['pupil_peak_delta']:+.3f}mm at {s['peak_when']} of sentence\n"
                f"- Gaze regressions: {s['regression_count']} "
                f"(eyes jumped backward {s['regression_count']} times)\n"
                f"- Confusion score: {s['confusion_score']}\n"
                f"- Reading time: {s['read_secs']}s"
            )
        user_msg = "\n\n".join(lines)
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentence rewriting assistant for a neural annotation system. "
                        "You will receive sentences that a human reader flagged as unclear or confusing, "
                        "along with their eye-tracking neural signals. Your job:\n"
                        "1. Analyze WHY the sentence was confusing using the neural data provided.\n"
                        "2. Rewrite each sentence to be clearer while preserving the core meaning.\n"
                        "3. Return ONLY a JSON array of objects, one per sentence, each with exactly "
                        "two keys: 'diagnosis' (string: 1-2 sentence explanation of what was likely "
                        "confusing, referencing the neural signals) and 'rewrite' (string: the improved "
                        "sentence). No markdown, no code fences, just raw JSON."
                    ),
                },
                {"role": "user", "content": user_msg},
            ],
            max_tokens=600,
            temperature=0.7,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        parsed = json.loads(text)
        if (isinstance(parsed, list)
                and len(parsed) == len(sentences_with_data)
                and all(isinstance(d, dict) and "diagnosis" in d and "rewrite" in d for d in parsed)):
            result_list.append(parsed)
        else:
            result_list.append(fallback)
    except Exception:
        result_list.append(fallback)

# ── Streamlit UI ─────────────────────────────────────────────────────
st.set_page_config(page_title="Implicit Annotator", layout="wide")

for k, v in {
    "phase": "start", "idx": 0, "results": [], "t_start": 0.0, "t_baseline": 0.0,
    "sentences": [], "_sentence_result": [], "_sentence_thread": None,
    "rewrite_round": 0, "rewrite_queue": [], "rewrite_sentences": [],
    "rewrite_results": [], "rewrite_idx": 0, "dpo_log": [],
    "_rewrite_thread": None, "_rewrite_result": [],
}.items():
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

    # ── Build initial DPO log from first-pass results ────────────────
    if not st.session_state["dpo_log"]:
        dpo_log = []
        for i, r in enumerate(results):
            full_sentence = st.session_state["sentences"][i]
            diag = _generate_neural_diagnosis(r)
            dpo_log.append({
                "original_sentence": full_sentence,
                "final_sentence": full_sentence if r["annotation"] == "YES" else None,
                "total_rewrites": 0,
                "accepted": r["annotation"] == "YES",
                "original_index": i,
                "history": [{
                    "version": 0,
                    "sentence": full_sentence,
                    "annotation": r["annotation"],
                    "confusion_score": r["confusion_score"],
                    "attention_score": r["attention_score"],
                    "annotation_confidence": r["annotation_confidence"],
                    "pupil_mean_delta": r["mean_delta"],
                    "pupil_peak_delta": r["peak_delta"],
                    "peak_when": r["peak_when"],
                    "regression_count": r["regression_count"],
                    "read_secs": r["read_secs"],
                    "samples": r["samples"],
                    "neural_diagnosis": diag,
                }],
            })
        st.session_state["dpo_log"] = dpo_log

    # ── Rewrite loop or finish ───────────────────────────────────────
    no_indices = [i for i, r in enumerate(results) if r["annotation"] == "NO"]
    if no_indices:
        st.info(f"{len(no_indices)} sentence(s) marked NO — eligible for rewrite.")
        col_rw1, col_rw2 = st.columns(2)
        if col_rw1.button("Start Rewrite Loop", type="primary", use_container_width=True):
            queue = []
            for i in no_indices:
                r = results[i]
                queue.append({
                    "sentence": st.session_state["sentences"][i],
                    "confusion_score": r["confusion_score"],
                    "pupil_mean_delta": r["mean_delta"],
                    "pupil_peak_delta": r["peak_delta"],
                    "peak_when": r["peak_when"],
                    "regression_count": r["regression_count"],
                    "read_secs": r["read_secs"],
                    "dpo_index": i,
                })
            st.session_state["rewrite_queue"] = queue
            st.session_state["rewrite_round"] = 1
            st.session_state["_rewrite_thread"] = None
            st.session_state["_rewrite_result"] = []
            st.session_state["rewrite_results"] = []
            st.session_state.phase = "rewriting"
            st.rerun()
        if col_rw2.button("Skip Rewrites — Finish Session", use_container_width=True):
            st.session_state.phase = "final_results"
            st.rerun()
    else:
        st.success("All sentences accepted! No rewrites needed.")
        st.session_state.phase = "final_results"
        st.rerun()

# ── PHASE: rewriting ─────────────────────────────────────────────────
elif st.session_state.phase == "rewriting":
    queue = st.session_state["rewrite_queue"]
    rnd = st.session_state["rewrite_round"]
    if st.session_state["_rewrite_thread"] is None:
        result_list = []
        st.session_state["_rewrite_result"] = result_list
        t = threading.Thread(target=_rewrite_sentences, args=(queue, result_list), daemon=True)
        st.session_state["_rewrite_thread"] = t
        t.start()
    thread = st.session_state["_rewrite_thread"]
    with st.spinner(f"Rewriting {len(queue)} sentence(s) using neural signals... (round {rnd})"):
        thread.join(timeout=0.1)
    if not thread.is_alive():
        result_list = st.session_state["_rewrite_result"]
        if result_list:
            parsed = result_list[0]
        else:
            parsed = [
                {"diagnosis": "Rewrite generated without neural analysis (API unavailable).",
                 "rewrite": "In simpler terms, " + s["sentence"]}
                for s in queue
            ]
        rw_sentences = []
        for i, s in enumerate(queue):
            rw_sentences.append({
                "original": s["sentence"],
                "rewrite": parsed[i]["rewrite"],
                "diagnosis": parsed[i]["diagnosis"],
                "dpo_index": s["dpo_index"],
            })
        st.session_state["rewrite_sentences"] = rw_sentences
        st.session_state["rewrite_idx"] = 0
        st.session_state["rewrite_results"] = []
        st.session_state.t_start = time.time()
        st.session_state.phase = "rewrite_annotate"
        st.rerun()
    else:
        time.sleep(0.3)
        st.rerun()

# ── PHASE: rewrite_annotate ──────────────────────────────────────────
elif st.session_state.phase == "rewrite_annotate":
    rw_idx = st.session_state["rewrite_idx"]
    rw_sentences = st.session_state["rewrite_sentences"]
    rnd = st.session_state["rewrite_round"]
    rw = rw_sentences[rw_idx]

    st.subheader(f"Rewrite Round {rnd} — Sentence {rw_idx + 1} of {len(rw_sentences)}")
    st.markdown(
        f"<div style='background:#333;color:#aaa;padding:15px;border-radius:8px;"
        f"font-size:1rem;margin:10px 0'><b>Original:</b> {rw['original']}</div>",
        unsafe_allow_html=True,
    )
    st.info(f"**LLM Diagnosis:** {rw['diagnosis']}")
    st.markdown(
        f"<div style='background:#1e1e1e;color:#fff;padding:40px;border-radius:12px;"
        f"font-size:1.5rem;text-align:center;margin:20px 0'>{rw['rewrite']}</div>",
        unsafe_allow_html=True,
    )
    live_pupil_metrics()

    col1, col2 = st.columns(2)
    yes_clicked = col1.button("YES", type="primary", use_container_width=True, key=f"rw_yes_{rnd}_{rw_idx}")
    no_clicked = col2.button("NO", type="secondary", use_container_width=True, key=f"rw_no_{rnd}_{rw_idx}")

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
        att = compute_attention_score(bw, t_end - t0, len(pw))
        ann_label = "YES" if yes_clicked else "NO"
        conf = compute_annotation_confidence(ann_label, confusion, att, t_end - t0, mean_d)
        result_dict = {
            "sentence":     rw["rewrite"][:50] + "...",
            "annotation":   ann_label,
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
            "confusion_score":        confusion,
            "pupil_component":        pupil_comp,
            "regression_count":       reg_count,
            "attention_score":        att,
            "annotation_confidence":  conf,
            "rewrite_triggered":      confusion > 0.55,
            "rewrite_round":          rnd,
            "llm_diagnosis":          rw["diagnosis"],
        }
        st.session_state["rewrite_results"].append({
            "result": result_dict,
            "dpo_index": rw["dpo_index"],
            "full_sentence": rw["rewrite"],
        })
        if rw_idx + 1 < len(rw_sentences):
            st.session_state["rewrite_idx"] = rw_idx + 1
            st.session_state.t_start = time.time()
            st.rerun()
        else:
            st.session_state.phase = "rewrite_check"
            st.rerun()
    else:
        pass  # fragment auto-refreshes live metrics

# ── PHASE: rewrite_check ─────────────────────────────────────────────
elif st.session_state.phase == "rewrite_check":
    rnd = st.session_state["rewrite_round"]
    rewrite_results = st.session_state["rewrite_results"]
    dpo_log = st.session_state["dpo_log"]
    still_no = []

    for rr in rewrite_results:
        dpo_idx = rr["dpo_index"]
        r = rr["result"]
        entry = dpo_log[dpo_idx]
        diag = _generate_neural_diagnosis(r)
        entry["history"].append({
            "version": rnd,
            "sentence": rr["full_sentence"],
            "annotation": r["annotation"],
            "confusion_score": r["confusion_score"],
            "attention_score": r["attention_score"],
            "annotation_confidence": r["annotation_confidence"],
            "pupil_mean_delta": r["mean_delta"],
            "pupil_peak_delta": r["peak_delta"],
            "peak_when": r["peak_when"],
            "regression_count": r["regression_count"],
            "read_secs": r["read_secs"],
            "samples": r["samples"],
            "neural_diagnosis": diag,
        })
        if r["annotation"] == "YES":
            entry["accepted"] = True
            entry["final_sentence"] = rr["full_sentence"]
            entry["total_rewrites"] = rnd
        else:
            still_no.append({
                "sentence": rr["full_sentence"],
                "confusion_score": r["confusion_score"],
                "pupil_mean_delta": r["mean_delta"],
                "pupil_peak_delta": r["peak_delta"],
                "peak_when": r["peak_when"],
                "regression_count": r["regression_count"],
                "read_secs": r["read_secs"],
                "dpo_index": dpo_idx,
            })

    if still_no and rnd < 3:
        st.session_state["rewrite_queue"] = still_no
        st.session_state["rewrite_round"] = rnd + 1
        st.session_state["_rewrite_thread"] = None
        st.session_state["_rewrite_result"] = []
        st.session_state["rewrite_results"] = []
        st.session_state.phase = "rewriting"
        st.rerun()
    else:
        for entry in dpo_log:
            if not entry["accepted"]:
                entry["total_rewrites"] = rnd
        st.session_state.phase = "final_results"
        st.rerun()

# ── PHASE: final_results ─────────────────────────────────────────────
elif st.session_state.phase == "final_results":
    st.title("Session Complete — DPO Training Log")
    dpo_log = st.session_state["dpo_log"]
    all_results = st.session_state.results

    for i, entry in enumerate(dpo_log):
        status_icon = "YES" if entry["accepted"] else "NO"
        with st.expander(f"{status_icon} Sentence {i+1}: {entry['original_sentence'][:60]}..."):
            st.markdown(f"**Total rewrites:** {entry['total_rewrites']}")
            if entry["accepted"] and entry.get("final_sentence"):
                st.success(f"**Final accepted:** {entry['final_sentence']}")
            else:
                st.error("**Not accepted** after all rewrite rounds.")
            for h in entry["history"]:
                label = "Original" if h["version"] == 0 else f"Rewrite round {h['version']}"
                st.markdown(f"---\n**{label}:** {h['sentence']}")
                st.markdown(
                    f"Annotation: **{h['annotation']}** | "
                    f"Confusion: {h['confusion_score']} | "
                    f"Attention: {h['attention_score']} | "
                    f"Confidence: {h['annotation_confidence']}"
                )
                st.caption(f"Neural diagnosis: {h['neural_diagnosis']}")

    # ── Summary metrics ──────────────────────────────────────────────
    st.divider()
    total = len(dpo_log)
    accepted_first = sum(1 for e in dpo_log if e["accepted"] and e["total_rewrites"] == 0)
    required_rewrites = sum(1 for e in dpo_log if e["total_rewrites"] > 0)
    max_rounds = max((e["total_rewrites"] for e in dpo_log), default=0)
    rewrites_to_accept = [e["total_rewrites"] for e in dpo_log if e["accepted"] and e["total_rewrites"] > 0]
    avg_rewrites = mean(rewrites_to_accept) if rewrites_to_accept else 0.0
    unresolved = sum(1 for e in dpo_log if not e["accepted"])

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Total Sentences", total)
    mc2.metric("Accepted First Pass", accepted_first)
    mc3.metric("Required Rewrites", required_rewrites)
    mc4, mc5, mc6 = st.columns(3)
    mc4.metric("Max Rounds Used", max_rounds)
    mc5.metric("Avg Rewrites to Accept", f"{avg_rewrites:.1f}")
    mc6.metric("Unresolved", unresolved)

    # ── Session score on all results ─────────────────────────────────
    session_score, breakdown = compute_session_score(all_results)
    st.divider()
    st.metric("Session Confidence Score", f"{session_score:.2f}",
              delta=breakdown.get("grade", ""))

    # ── Save DPO log ─────────────────────────────────────────────────
    outpath = Path("data/dpo_log.json")
    outpath.parent.mkdir(exist_ok=True)
    dpo_output = {
        "dpo_log": dpo_log,
        "session_score": session_score,
        "breakdown": breakdown,
        "rewrite_stats": {
            "total_sentences": total,
            "accepted_first_pass": accepted_first,
            "required_rewrites": required_rewrites,
            "max_rounds_used": max_rounds,
            "still_unresolved": unresolved,
        },
    }
    outpath.write_text(json.dumps(dpo_output, indent=2))
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
        st.session_state["rewrite_round"] = 0
        st.session_state["rewrite_queue"] = []
        st.session_state["rewrite_sentences"] = []
        st.session_state["rewrite_results"] = []
        st.session_state["rewrite_idx"] = 0
        st.session_state["dpo_log"] = []
        st.session_state["_rewrite_thread"] = None
        st.session_state["_rewrite_result"] = []
        st.rerun()
