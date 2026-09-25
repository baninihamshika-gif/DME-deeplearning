"""
Canonical configuration for the DME OCT Detection project.

Every hyperparameter, path, and expected-dataset value used elsewhere in the
codebase is defined here and nowhere else — see PROJECT_BRIEF.md Section 8
for provenance and Section 6 for the hard constraints (C1-C8) these values
support. Nothing in this file touches data or trains anything (Phase 0
scope); it only declares values Phase 1+ will read.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Dataset — Kermany et al. (Cell, 2018), OCT2017 / Mendeley V2
# This is the 83,484-training-image release. A second release exists with
# 108,312 images and NORMAL = 51,140 — loading that one silently invalidates
# every value below (class weights, imbalance ratio, test-set size).
# ---------------------------------------------------------------------------
EXPECTED_TRAIN_COUNTS = {"DME": 11_348, "NORMAL": 26_315}
EXPECTED_TEST_COUNTS = {"DME": 242, "NORMAL": 242}
EXPECTED_IMBALANCE_RATIO = 2.32  # NORMAL : DME

CLASS_NAMES = ["Normal", "DME"]  # index order is load-bearing — see C5
CLASS_WEIGHTS = [0.72, 1.66]  # inverse frequency, same order as CLASS_NAMES

IMAGE_SIZE = 300  # EfficientNet-B3 native input resolution
VAL_SPLIT = 0.20  # carved from train/ — the provided val/ folder is unusable (8 images/class)

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
RETINAL_FLATTENING_ENABLED = True  # exposed as a flag so Phase 3's ablation can test with it off

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE_CLS = 32
BATCH_SIZE_SEG = 16

LR_FROZEN = 1e-3  # Phase A — conv_head + bn2 + classifier only
LR_FINETUNE = 1e-5  # Phase B — full network unfrozen
LR_SEG = 1e-4  # Phase 4 (extended) segmentation

EPOCHS_PHASE_A = 5
EPOCHS_PHASE_B = 25

EARLY_STOP_PATIENCE = 5
PLATEAU_FACTOR = 0.3
PLATEAU_PATIENCE = 3

DROP_RATE = 0.3

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
SENSITIVITY_TARGET = 0.97

# ---------------------------------------------------------------------------
# Thickness index (extended objectives, Phase 5)
# ETDRS central subfield is a 1 mm circle; Spectralis macular B-scans
# typically span ~6 mm, so the foveal window is ~1/6 of scan width. Applied
# to retina CONTENT width, never the padded IMAGE_SIZE frame (C2).
# ---------------------------------------------------------------------------
FOVEAL_WINDOW_FRACTION = 0.167

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent

# NEEDS CONFIRMATION: no local path for the Kermany dataset is specified in
# the brief. Defaulting to <repo>/data/raw for local tooling (e.g. curating
# data/sample/ in Phase 1); on Kaggle, DATA_DIR below is used instead and
# this value is irrelevant. Update if the team keeps the raw dataset
# somewhere else on the local machine.
LOCAL_DATA_DIR = REPO_ROOT / "data" / "raw"

ON_KAGGLE = Path("/kaggle/input").exists()
DATA_DIR = Path("/kaggle/input/kermany2018") if ON_KAGGLE else LOCAL_DATA_DIR

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
SPLITS_DIR = REPO_ROOT / "data" / "splits"
SAMPLE_DIR = REPO_ROOT / "data" / "sample"

# ---------------------------------------------------------------------------
# Kaggle automation (Phase 2.5)
# ---------------------------------------------------------------------------
KAGGLE_USERNAME = "baninihamshika"

KAGGLE_SRC_DATASET_SLUG = f"{KAGGLE_USERNAME}/dme-oct-src"
KAGGLE_KERMANY_DATASET_SLUG = "paultimothymooney/kermany2018"  # verified Phase 1 -- 83,484-train-image release
KAGGLE_KERNEL_SLUG = f"{KAGGLE_USERNAME}/dme-oct-train"

KAGGLE_ARTIFACTS_DIR = ARTIFACTS_DIR / "kaggle"
KAGGLE_QUOTA_LOG = KAGGLE_ARTIFACTS_DIR / "quota_log.json"
KAGGLE_LAST_SMOKE = KAGGLE_ARTIFACTS_DIR / "last_smoke.json"

KAGGLE_POLL_INTERVAL_SEC = 60
KAGGLE_GPU_WEEKLY_BUDGET_MIN = 30 * 60  # ~30 GPU hrs/week, free tier
KAGGLE_PUSH_BUDGET_DEFAULT_MIN = 6 * 60  # refuse a single push projected to exceed this
KAGGLE_SMOKE_TIME_BUDGET_MIN = 10  # brief: smoke run must complete in under 10 minutes
KAGGLE_MAX_POLL_MIN = 6 * 60  # give up polling (classified as an infra timeout) after this long
KAGGLE_MAX_INFRA_RETRIES = 2

# `kaggle kernels push` itself is a separate failure point from poll()'s own
# infra-retry loop (which only ever sees a kernel that has already started
# running) -- confirmed the hard way (2026-09-23): a real push failed
# outright with "503 Server Error: Service Unavailable" from Kaggle's own
# SaveKernel API endpoint, before any kernel was ever created, so poll()
# never got a chance to help. _run_kaggle_cli(..., infra_retry=True) now
# retries a push call that fails with an infra-looking message (same keyword
# list classify_failure() already uses) up to KAGGLE_MAX_INFRA_RETRIES times,
# waiting this long between attempts.
KAGGLE_PUSH_INFRA_RETRY_DELAY_SEC = 20

# `kaggle datasets create`/`version` return as soon as the upload finishes,
# not once Kaggle has finished processing the new version -- confirmed the
# hard way (2026-09): the first real smoke push uploaded fine, then pushed
# the kernel immediately after, and the kernel failed with
# FileNotFoundError on /kaggle/input/dme-oct-src because that version
# wasn't attachable yet. kaggle/run.py now polls dataset status and blocks
# on "ready" before pushing the kernel -- these bound that wait.
KAGGLE_DATASET_READY_POLL_SEC = 10
KAGGLE_DATASET_READY_TIMEOUT_MIN = 5

# ---------------------------------------------------------------------------
# Kaggle automation (Phase 4, extended -- segmentation)
# ---------------------------------------------------------------------------
# Duke DME dataset (Chiu et al. 2015) is redistribution-restricted (research/
# educational use only, no redistribution -- see segment.py's module
# docstring) and has no verified-structure public Kaggle mirror we could
# confirm from this session (no live Kaggle API access to inspect one), so
# it's uploaded as a NEW PRIVATE dataset under this account -- the same
# already-verified local copy segment.py's rasterization gate was checked
# against -- rather than an unreviewed public re-host.
KAGGLE_DUKE_DATASET_SLUG = f"{KAGGLE_USERNAME}/dme-oct-duke-src"
# The Phase 2 classifier checkpoint (needed to initialise the U-Net encoder)
# only exists on the local machine (fetched from the classifier's own Kaggle
# run) -- also uploaded as its own small private dataset rather than
# guessed-at via kernel_sources' kernel-output-mount behaviour, which this
# session has no way to verify against a live account either.
KAGGLE_CLASSIFIER_CKPT_DATASET_SLUG = f"{KAGGLE_USERNAME}/dme-oct-classifier-ckpt"
KAGGLE_SEGMENT_KERNEL_SLUG = f"{KAGGLE_USERNAME}/dme-oct-segment"

KAGGLE_SEGMENT_LAST_SMOKE = KAGGLE_ARTIFACTS_DIR / "segment_last_smoke.json"
# Deliberately NOT a separate quota log -- the 7-day/30h GPU cap is a single
# real Kaggle-account-wide resource shared by every kernel on the account,
# classifier and segmentation alike, so both pipelines must account against
# the SAME KAGGLE_QUOTA_LOG for the guardrail to mean anything real. Each
# entry's kernel_slug field is what estimate_segment_run_minutes() filters
# on to avoid conflating the two pipelines' very different per-epoch costs.

# segment.py's --smoke trains 1 epoch on the FULL annotated Duke set (110
# scans total, no subsampling -- unlike the classifier's smoke, which trains
# on a ~500-image SUBSET of a ~38k-image dataset). That matters for
# estimate_segment_run_minutes(): a segmentation smoke run's epoch_seconds
# is already representative of a real full-run epoch's cost, so the
# classifier's ~9.7x-undershoot problem (smoke-subset vs full-dataset
# per-epoch cost) should not recur here -- flagged as "should", not
# confirmed, until a real smoke + real full run both exist to compare
# (C6: no invented numbers).
KAGGLE_SEGMENT_SMOKE_TIME_BUDGET_MIN = 15  # generous first-guess ceiling; tighten once a real smoke duration exists
