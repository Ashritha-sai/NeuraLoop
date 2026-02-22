# Implicit — Neural Annotation

Real-time implicit annotation tool that captures pupil dilation, gaze, eyelid aperture, and blink data from a **Pupil Labs Neon** eye tracker via LSL while a user reads and annotates sentences.

## How It Works

1. **Connect** — resolves the Neon Gaze LSL stream (falls back to mock mode if unavailable).
2. **Baseline** — 10-second quiet screen period to establish resting pupil diameter.
3. **Annotate** — six sentences are shown one at a time. Live pupil delta, blink count, and sample count are displayed. The user clicks **YES** or **NO** for each.
4. **Results** — a summary table, per-sentence pupil dilation chart, and session statistics. Output is saved to `data/demo_output.json`.

## Requirements

- Python 3.10+
- [Streamlit](https://streamlit.io/)
- [pylsl](https://github.com/labstreaminglayer/pylsl)
- [Plotly](https://plotly.com/python/)

```
pip install streamlit pylsl plotly
```

## Run

```
streamlit run demo.py
```

## Project Structure

```
demo.py                  # single-file app (UI + hardware layer)
implicit_annotator/
  config.py              # tunable constants for the full system
data/                    # runtime outputs (git-ignored)
```

## Hardware

Requires a **Pupil Labs Neon** streaming via the Neon Companion app with LSL enabled. If no stream is found within 8 seconds, the app runs in **Mock Mode** (all biometric values will be zero).
