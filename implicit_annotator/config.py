"""
Central configuration for the implicit annotation system.
Every tunable constant lives here so no module ever hardcodes a magic number.
"""


class Config:
    """Single source of truth for all system parameters."""

    # ── NEON Gaze stream (single 22-channel stream) ─────────────────────
    STREAM_NAME: str = "Neon Companion_Neon Gaze"
    STREAM_TYPE: str = "Gaze"
    STREAM_CHANNELS: int = 22
    GAZE_X: int = 0           # gaze x screen coordinate
    GAZE_Y: int = 1           # gaze y screen coordinate
    PUPIL_LEFT: int = 2       # left pupil diameter mm
    PUPIL_RIGHT: int = 9      # right pupil diameter mm
    EYELID_APERTURE_LEFT: int = 18   # left eyelid aperture mm
    EYELID_APERTURE_RIGHT: int = 21  # right eyelid aperture mm

    # ── NEON IMU stream (separate when present) ──────────────────────
    LSL_IMU_STREAM: str = "Neon IMU"
    LSL_TIMEOUT_S: int = 8               # seconds to wait before falling back to mock mode
    SAMPLE_RATE_HZ: int = 200            # native sampling rate of the Neon eye tracker
    BUFFER_DURATION_S: int = 10          # rolling buffer length for real-time analysis

    # ── PUPIL dilation parameters ───────────────────────────────────────
    BASELINE_DURATION_S: int = 30        # quiet-screen period for baseline pupil measurement
    BASELINE_COLLECTION_INTERVAL_S: float = 0.005  # 200 Hz collection during baseline
    DILATION_BASE_THRESHOLD_MM: float = 0.40       # minimum change from baseline to count as dilation
    DILATION_FK_SCALE: float = 0.02      # scales Flesch-Kincaid grade into threshold adjustment
    PEAK_SEARCH_WINDOW_MS: int = 2000    # window after stimulus onset to search for peak dilation
    RECOVERY_WINDOW_MS: int = 500        # post-peak window to measure pupil recovery slope

    # ── GAZE analysis parameters ────────────────────────────────────────
    SENTENCE_LEFT_FRAC: float = 0.33     # left boundary of the sentence region (fraction of screen)
    SENTENCE_RIGHT_FRAC: float = 0.66    # right boundary of the sentence region
    ANCHOR_FIXATION_MS: int = 1500       # fixation duration that signals deliberate re-reading
    REGRESSION_COOLDOWN_MS: int = 300    # dead-zone after a regression to suppress duplicates

    # ── GESTURE detection parameters ────────────────────────────────────
    WINDOW_OPEN_DELAY_MS: int = 500      # delay after prompt before gesture window opens
    WINDOW_DURATION_MS: int = 4000       # how long the gesture detection window stays open
    FREEZE_DURATION_MS: int = 800        # post-gesture freeze to avoid double-counting
    NOD_ANGULAR_VELOCITY_THRESHOLD: int = 60   # deg/s to classify a head movement as a nod
    SHAKE_ANGULAR_VELOCITY_THRESHOLD: int = 50  # deg/s to classify a head movement as a shake
    NOD_AXIS: str = "pitch"              # IMU axis corresponding to nodding
    SHAKE_AXIS: str = "yaw"             # IMU axis corresponding to shaking
    NEUTRAL_RECAL_INTERVAL_S: int = 60   # seconds between neutral-pose recalibrations
    NOD_BIPHASIC_WINDOW_MS: int = 800    # max time for a biphasic nod reversal (down-then-up)
    SHAKE_BIPHASIC_WINDOW_MS: int = 600  # max time for a biphasic shake reversal (left-right)

    # ── BLINK detection (eyelid-aperture state machine) ─────────────────
    BLINK_CLOSE_THRESHOLD_MM: float = 2.0   # eyelid aperture below this = eye closed
    BLINK_OPEN_THRESHOLD_MM: float = 5.0    # eyelid must recover above this to confirm blink
    BLINK_MIN_DURATION_MS: float = 50.0     # minimum closure duration to count as a blink

    # ── BLINK rate analysis ─────────────────────────────────────────────
    RATE_WINDOW_S: int = 30              # sliding window for computing blink rate
    NORMAL_RATE_MIN: int = 15            # lower bound of normal blink rate (blinks/min)
    NORMAL_RATE_MAX: int = 20            # upper bound of normal blink rate (blinks/min)
    FOCUS_RATE_THRESHOLD: int = 8        # below this rate signals deep focus / cognitive load
    FATIGUE_RATE_THRESHOLD: int = 25     # above this rate signals fatigue
    SUPPRESSION_DROP_FRACTION: float = 0.50  # fractional drop from baseline that flags blink suppression

    # ── READABILITY scaling ─────────────────────────────────────────────
    FK_EASY_THRESHOLD: int = 8           # Flesch-Kincaid grade below which text is "easy"
    FK_HARD_THRESHOLD: int = 14          # Flesch-Kincaid grade above which text is "hard"

    # ── FUSION weights and thresholds ───────────────────────────────────
    W_EXPLICIT_AGREEMENT: float = 0.45   # weight of explicit gesture agreement in final score
    W_ATTENTION: float = 0.25            # weight of gaze-derived attention signal
    W_IMPLICIT_CONSISTENCY: float = 0.30  # weight of pupil/blink implicit consistency signal
    CONFUSION_THRESHOLD: float = 0.55    # fusion score below which the system flags confusion
    CONFUSION_SUSTAIN_S: float = 2.0     # seconds confusion must persist before triggering re-prompt
    ATTENTION_WARN_SENTENCES: int = 3    # consecutive low-attention sentences before warning
    ATTENTION_WARN_THRESHOLD: float = 0.40  # attention score below which a sentence counts as low
    W_PUPIL: float = 0.50                    # pupil component weight in confusion score
    CONSISTENCY_AGREE_CEILING: float = 0.35  # confusion below this = fully consistent with YES gesture
    FAST_LATENCY_MS: int = 800               # gesture response faster than this triggers a penalty

    # ── FATIGUE monitoring ──────────────────────────────────────────────
    BLOCK_DURATION_S: int = 300          # annotation block length for fatigue tracking (5 min)
    DEGRADATION_THRESHOLD: float = 0.30  # cross-block quality drop that triggers a break suggestion
    CONFIDENCE_PENALTY: float = 0.15     # penalty applied to confidence when fatigue is detected

    # ── ANNOTATOR PROFILE calibration ───────────────────────────────────
    PROFILE_PATH: str = "data/annotator_profile.json"  # where the per-annotator profile is stored
    RECAL_DAYS: int = 30                 # days before profile recalibration is recommended
    EASY_FK_MAX: int = 8                 # FK grade ceiling for easy calibration sentences
    HARD_FK_MIN: int = 14               # FK grade floor for hard calibration sentences
    CALIBRATION_SENTENCES_EACH: int = 5  # number of sentences per difficulty tier during calibration

    # ── SCORING weights ─────────────────────────────────────────────────
    ANNOTATION_COVERAGE_WEIGHT: float = 0.20   # fraction of session quality from annotation coverage
    ATTENTION_CONSISTENCY_WEIGHT: float = 0.20  # fraction from gaze attention consistency
    SIGNAL_QUALITY_WEIGHT: float = 0.20        # fraction from raw signal quality checks
    MEAN_CONFIDENCE_WEIGHT: float = 0.40       # fraction from mean annotation confidence
    MIN_SIGNAL_QUALITY: float = 0.85           # minimum signal quality to accept a session
    DPO_LOW_CONFIDENCE_MARGIN: float = 0.10   # confusion-score gap below which a DPO pair is flagged

    # ── OUTPUT PATHS ────────────────────────────────────────────────────
    DPO_OUTPUT: str = "data/neural_preference_corpus.jsonl"  # DPO training pairs output file
    SESSION_LOG: str = "data/session_log.jsonl"              # per-session metadata log


# Module-level singleton so every import gets the same object
cfg = Config()
