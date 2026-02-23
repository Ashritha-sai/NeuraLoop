# Implicit &mdash; Neural Annotation

> Real-time implicit annotation system that fuses eye-tracking biometrics with LLM-powered sentence rewriting to build high-quality preference datasets.

**Implicit** captures pupil dilation, gaze regressions, eyelid aperture, and blink patterns from a [Pupil Labs Neon](https://pupil-labs.com/products/neon) eye tracker while a user reads and annotates sentences. When the neural signals reveal confusion, an LLM automatically rewrites the sentence and the user re-annotates -- producing a DPO (Direct Preference Optimization) training corpus grounded in real cognitive data.

---

## Features

- **Real-time biometric acquisition** -- Pupil diameter, gaze coordinates, eyelid aperture, and blink detection at 200 Hz via Lab Streaming Layer (LSL)
- **Baseline calibration** -- Quiet-screen period to establish resting pupil diameter before annotation begins
- **Implicit confusion scoring** -- Combines pupil dilation (60%) and gaze regressions (40%) into a single confusion metric
- **Attention monitoring** -- Blink-rate analysis detects deep focus, normal reading, and fatigue states
- **Annotation confidence fusion** -- Merges explicit user labels with implicit neural signals and attention quality
- **LLM sentence generation** -- Groq-hosted Llama 3 generates novel sentences from random topics
- **Neural-guided rewriting** -- Confused sentences are rewritten up to 3 rounds with full neural diagnosis context
- **DPO corpus generation** -- Every original/rewrite pair is logged with neural metrics for preference training
- **Graceful degradation** -- Falls back to Mock Mode (no hardware) and fallback sentences (no API key)

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Streamlit Web UI                       │
│  ┌──────────┐  ┌───────────┐  ┌────────────────────────┐ │
│  │ Baseline │→ │ Annotate  │→ │ Results & Rewrite Loop │ │
│  └──────────┘  └───────────┘  └────────────────────────┘ │
├──────────────────────────────────────────────────────────┤
│              Implicit Signal Processing                   │
│  ┌────────────────┐ ┌────────────┐ ┌──────────────────┐  │
│  │ Confusion Score│ │ Attention  │ │ Confidence Fusion│  │
│  │ pupil + gaze   │ │ blink rate │ │ explicit+implicit│  │
│  └────────────────┘ └────────────┘ └──────────────────┘  │
├──────────────────────────────────────────────────────────┤
│  Hardware Layer (HW)         │  LLM Layer (Groq)         │
│  Pupil Labs Neon via LSL     │  Sentence gen & rewrite   │
│  200 Hz, 22-channel stream   │  Llama 3.1 8B Instant     │
└──────────────────────────────┴───────────────────────────┘
```

---

## How It Works

| Phase | What happens |
|-------|-------------|
| **1. Connect** | Resolves the Neon Gaze LSL stream (falls back to Mock Mode after 8 s) |
| **2. Baseline** | 10-second quiet screen to measure resting pupil diameter; sentences fetched in parallel |
| **3. Annotate** | 5 sentences shown one at a time with live pupil delta, blink count, and sample metrics; user clicks **Yes** / **No** |
| **4. Results** | Summary table, per-sentence pupil dilation chart, and session score |
| **5. Rewrite** | LLM rewrites sentences where `confusion_score > 0.55`, citing neural diagnosis |
| **6. Re-annotate** | User reads rewritten sentences and re-labels; loop repeats up to 3 rounds |
| **7. Final** | Full DPO log with version history, neural metrics, and session statistics saved to disk |

---

## Quick Start

### Prerequisites

- Python 3.10+
- (Optional) [Pupil Labs Neon](https://pupil-labs.com/products/neon) with Neon Companion app streaming via LSL
- (Optional) [Groq API key](https://console.groq.com/keys) for LLM features

### Installation

```bash
git clone https://github.com/<your-username>/implicit-neural-annotation.git
cd implicit-neural-annotation

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

### Run

```bash
streamlit run demo.py
```

> **No hardware?** The app detects the missing LSL stream and switches to **Mock Mode** automatically -- all biometric values read zero, but the full UI and LLM pipeline still work.

---

## Project Structure

```
implicit-neural-annotation/
├── demo.py                      # Single-file Streamlit app (UI + hardware + scoring + LLM)
├── implicit_annotator/
│   └── config.py                # All tunable constants (thresholds, weights, paths)
├── data/                        # Runtime outputs (git-ignored)
│   ├── demo_output.json         # Per-session results with timeseries pupil data
│   └── dpo_log.json             # DPO training corpus with rewrite history
├── requirements.txt
├── .env.example                 # API key template
├── .gitignore
└── LICENSE
```

---

## Configuration Reference

All parameters live in [`implicit_annotator/config.py`](implicit_annotator/config.py). Key groups:

| Group | Examples | Purpose |
|-------|----------|---------|
| **Stream** | `SAMPLE_RATE_HZ`, `LSL_TIMEOUT_S` | Neon LSL connection settings |
| **Pupil** | `DILATION_BASE_THRESHOLD_MM`, `BASELINE_DURATION_S` | Dilation detection thresholds |
| **Gaze** | `REGRESSION_COOLDOWN_MS`, `ANCHOR_FIXATION_MS` | Regression and fixation analysis |
| **Blink** | `BLINK_CLOSE_THRESHOLD_MM`, `FOCUS_RATE_THRESHOLD` | Blink detection state machine |
| **Scoring** | `W_EXPLICIT_AGREEMENT`, `CONFUSION_THRESHOLD` | Confidence fusion weights |
| **Output** | `DPO_OUTPUT`, `SESSION_LOG` | File paths for generated data |

---

## Scoring

**Confusion Score** (per sentence)
- Pupil dilation component (60%): mean pupil delta vs. baseline
- Gaze regression component (40%): backward eye movements during reading

**Attention Score** (per sentence)
- Optimal blink rate: 4-8 blinks/min = full score
- Extremes (very low or very high) penalized

**Session Score** (overall)
- Mean annotation confidence (40%) + coverage (20%) + attention consistency (20%) + signal quality (20%)
- Grades: `EXCELLENT` > 0.85 | `GOOD` > 0.70 | `ACCEPTABLE` > 0.55 | `LOW QUALITY`

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | [Streamlit](https://streamlit.io/) |
| Eye tracking | [Pupil Labs Neon](https://pupil-labs.com/products/neon) via [pylsl](https://github.com/labstreaminglayer/pylsl) |
| Visualization | [Plotly](https://plotly.com/python/) |
| LLM | [Groq](https://groq.com/) (Llama 3.1 8B Instant) |
| Language | Python 3.10+ |

---

## License

This project is licensed under the [MIT License](LICENSE).
