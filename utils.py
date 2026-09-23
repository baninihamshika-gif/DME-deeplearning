"""
Shared utilities for the DME OCT Detection project.

Phase 0: set_seed().
Phase 1: patient ID parsing, patient-grouped splitting, image preprocessing
(denoise -> retinal flattening -> CLAHE -> aspect-preserving letterbox
resize), and a disk cache for the deterministic part of that pipeline.
"""

import hashlib
import os
import random
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import (
    CLASS_NAMES,
    EXPECTED_TEST_COUNTS,
    EXPECTED_TRAIN_COUNTS,
    IMAGE_SIZE,
    RETINAL_FLATTENING_ENABLED,
    SEED,
    VAL_SPLIT,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = SEED) -> None:
    """
    Seed every source of randomness this project touches and force cuDNN
    into deterministic mode.

    Determinism has a real cost (cuDNN autotuning is disabled, so training
    is slower), but a project whose split (C1) and headline accuracy (C7)
    both depend on catching leakage needs runs that are exactly
    reproducible, not just "close."

    torch is imported lazily here, not at module level: per
    PROJECT_BRIEF.md Section 4, the local machine has no torch installed
    by design (code editing and git only; training runs on Kaggle). A
    hard top-level `import torch` would make this whole module
    (including the Phase 1 patient-parsing/splitting/preprocessing code,
    none of which touches torch) unimportable locally. When torch isn't
    present, Python/NumPy seeding still happens and the torch-specific
    seeding is skipped silently; on Kaggle, where torch is always
    installed, nothing changes.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Patient ID parsing (C1)
# ---------------------------------------------------------------------------

# Kermany OCT2017 filenames: "<LABEL>-<patientID>-<imageIndex>.<ext>",
# e.g. "DME-1072015-11.jpeg", "NORMAL-956449-1.jpeg".
_PATIENT_ID_PATTERN = re.compile(
    r"^(?P<label>CNV|DME|DRUSEN|NORMAL)-(?P<patient>\d+)-(?P<image>\d+)\.(?P<ext>jpe?g|png)$",
    re.IGNORECASE,
)


def parse_patient_id(path) -> str:
    """
    Extract the patient ID encoded in a Kermany OCT2017 filename, e.g.
    "DME-1072015-11.jpeg" -> "1072015".

    Raises ValueError on any filename that doesn't match the expected
    pattern. A silent fallback here (e.g. treating the whole filename as a
    unique "patient") would corrupt the patient-grouped split (C1) without
    any visible symptom, so this refuses to guess.
    """
    name = Path(path).name
    match = _PATIENT_ID_PATTERN.match(name)
    if match is None:
        raise ValueError(
            "Filename does not match the expected Kermany OCT2017 pattern "
            f"'<LABEL>-<patientID>-<imageIndex>.<ext>': {name!r}"
        )
    return match.group("patient")


# ---------------------------------------------------------------------------
# Patient-grouped splitting (C1, C8)
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}

# Folder name (as it appears on disk) -> label value used in CLASS_NAMES.
_FOLDER_TO_LABEL = {"DME": "DME", "NORMAL": "Normal"}


@dataclass
class SplitResult:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    discovered_counts: dict = field(default_factory=dict)
    train_val_patient_overlap: int = 0
    test_patient_overlap: int = 0
    excluded_test_overlap_patients: int = 0
    excluded_test_overlap_images: int = 0


def _scan_class_folders(root: Path) -> pd.DataFrame:
    """Scan the DME/ and NORMAL/ subfolders of `root` into a path/label/patient frame."""
    records = []
    discovered_counts = {}
    for folder_name, label in _FOLDER_TO_LABEL.items():
        class_dir = root / folder_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class folder not found: {class_dir}")
        image_paths = sorted(
            p for p in class_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        discovered_counts[folder_name] = len(image_paths)
        for p in image_paths:
            records.append({"path": str(p), "label": label, "patient": parse_patient_id(p)})
    return pd.DataFrame.from_records(records, columns=["path", "label", "patient"]), discovered_counts


def build_splits(
    train_dir,
    test_dir=None,
    val_split: float = VAL_SPLIT,
    seed: int = SEED,
    expected_counts: Optional[dict] = EXPECTED_TRAIN_COUNTS,
    ratio_tolerance: float = 0.05,
    exclude_test_overlap_patients: bool = True,
) -> SplitResult:
    """
    Scan the DME/ and NORMAL/ folders under `train_dir`, group images by
    patient, and produce a patient-disjoint train/validation split (C1).

    `expected_counts` defaults to EXPECTED_TRAIN_COUNTS (the canonical
    83,484-image Kermany release, see PROJECT_BRIEF.md Section 5) and is
    asserted against the discovered per-folder counts. Pass a different
    dict (or None to skip the check) only for testing against a synthetic
    fixture -- real invocations must use the default so a wrong dataset
    release or an incomplete download is caught immediately, not silently
    trained on.

    Also asserts: zero patient overlap between train and validation; both
    splits contain both classes; each split's class ratio is within
    `ratio_tolerance` of the overall ratio. If `test_dir` is given, checks
    patient overlap between train+val and the official test split (C8): on
    the real Kermany data this is severe (305 of 368 test patients, 84% of
    the 484 test images, also appear in train/val), so a violation is
    always surfaced as a loud warning, and -- since the team decided to fix
    it from the training side rather than alter the official 484-image
    test set (keeps published-benchmark comparability, per PROJECT_BRIEF.md
    discussion) -- by default (`exclude_test_overlap_patients=True`) the
    overlapping patients' images are dropped from `train_df`/`val_df`
    before they're returned. The *pre-exclusion* finding is still reported
    via `test_patient_overlap`/the warning; `excluded_test_overlap_*`
    fields on the result report what was actually removed. Pass
    `exclude_test_overlap_patients=False` to get the diagnostic-only
    behavior (report the overlap, change nothing).

    Every check prints its own PASSED line on success, not just an
    exception on failure -- "it didn't error" and "it ran and passed" must
    stay distinguishable in the log (the same principle as C5).

    Splitting note: on the real Kermany data, 339 patient IDs turn out to
    appear under *both* the DME and NORMAL folders (the same patient graded
    differently across visits/eyes) -- so patient-grouping cannot be done
    per class independently; a patient straddling both classes must still
    land entirely in one split. This uses StratifiedGroupKFold (group =
    patient, over the whole DME+Normal pool at once, `round(1/val_split)`
    folds, one held out as validation) rather than a single combined
    GroupShuffleSplit: a plain GroupShuffleSplit only targets the overall
    train/val *sample-count* ratio and has no notion of class, so whichever
    patients land in val can -- and on this data, does -- skew the class
    ratio well past a 5% tolerance. StratifiedGroupKFold keeps every
    patient's rows together while balancing class ratio across folds as
    well as the grouping constraint allows.
    """
    train_dir = Path(train_dir)
    df, discovered_counts = _scan_class_folders(train_dir)

    if expected_counts is not None:
        for folder_name, expected in expected_counts.items():
            found = discovered_counts.get(folder_name, 0)
            if found != expected:
                raise AssertionError(
                    f"Discovered {found} images in '{folder_name}/', expected {expected} "
                    "(expected_counts). This usually means the wrong dataset release is "
                    "loaded (see PROJECT_BRIEF.md Section 5) or the folder is incomplete."
                )
        print(f"[build_splits] count assertion vs expected_counts: PASSED {discovered_counts}")
    else:
        print(f"[build_splits] count assertion skipped (expected_counts=None); discovered {discovered_counts}")

    cross_class_patients = set(df[df["label"] == CLASS_NAMES[0]]["patient"]) & set(
        df[df["label"] == CLASS_NAMES[1]]["patient"]
    )
    if cross_class_patients:
        print(
            f"[build_splits] note: {len(cross_class_patients)} patient(s) have images in "
            f"both classes; they will be kept whole in one split (train or val)."
        )

    n_splits = max(2, round(1 / val_split))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(skf.split(df, y=df["label"], groups=df["patient"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    print(
        f"[build_splits] StratifiedGroupKFold: n_splits={n_splits} (target val fraction "
        f"~{1/n_splits:.1%} vs requested {val_split:.1%}); actual val fraction "
        f"{len(val_df) / len(df):.1%}"
    )

    train_patients = set(train_df["patient"])
    val_patients = set(val_df["patient"])
    train_val_overlap = train_patients & val_patients
    if train_val_overlap:
        raise AssertionError(
            f"{len(train_val_overlap)} patient(s) appear in both train and validation "
            f"splits (C1 violation): {sorted(train_val_overlap)[:5]}"
        )
    print(f"[build_splits] patient overlap train<->val: 0 (PASSED)")

    for name, split_df in (("train", train_df), ("validation", val_df)):
        present = set(split_df["label"])
        missing = set(CLASS_NAMES) - present
        if missing:
            raise AssertionError(f"{name} split is missing class(es): {missing}")
    print("[build_splits] both splits contain both classes: PASSED")

    overall_ratio = df["label"].value_counts(normalize=True)
    for name, split_df in (("train", train_df), ("validation", val_df)):
        split_ratio = split_df["label"].value_counts(normalize=True)
        for cls in CLASS_NAMES:
            overall = overall_ratio.get(cls, 0.0)
            this = split_ratio.get(cls, 0.0)
            if overall <= 0:
                continue
            drift = abs(this - overall) / overall
            if drift > ratio_tolerance:
                raise AssertionError(
                    f"{name} split class ratio for {cls!r} drifted {drift:.1%} from the "
                    f"overall ratio ({this:.3f} vs {overall:.3f}) — exceeds the "
                    f"{ratio_tolerance:.0%} tolerance."
                )
    print(f"[build_splits] class ratio within {ratio_tolerance:.0%} of overall: PASSED")

    test_patient_overlap = 0
    excluded_patients = 0
    excluded_images = 0
    if test_dir is not None:
        test_dir = Path(test_dir)
        test_df, _ = _scan_class_folders(test_dir)
        test_patients = set(test_df["patient"])
        leaked = (train_patients | val_patients) & test_patients
        test_patient_overlap = len(leaked)
        if leaked:
            warnings.warn(
                f"(C8) {test_patient_overlap} patient(s) appear in both train/val and the "
                f"official test split: {sorted(leaked)[:5]} — this compromises the "
                "evaluation regardless of our own split hygiene. Report before proceeding.",
                stacklevel=2,
            )
            print(f"[build_splits] train/val <-> test patient overlap: {test_patient_overlap} (C8 VIOLATION)")

            if exclude_test_overlap_patients:
                pre_train, pre_val = len(train_df), len(val_df)
                train_df = train_df[~train_df["patient"].isin(leaked)].reset_index(drop=True)
                val_df = val_df[~val_df["patient"].isin(leaked)].reset_index(drop=True)
                excluded_patients = test_patient_overlap
                excluded_images = (pre_train - len(train_df)) + (pre_val - len(val_df))
                print(
                    f"[build_splits] excluded {excluded_patients} overlapping patient(s) "
                    f"({excluded_images} images: train {pre_train}->{len(train_df)}, "
                    f"val {pre_val}->{len(val_df)}) from train/val to keep the official "
                    "484-image test set clean. test_patient_overlap above reports what was "
                    "found before this fix, not the current (now zero) state."
                )
            else:
                print(
                    "[build_splits] exclude_test_overlap_patients=False: train/val returned "
                    "unmodified -- the overlap above is a diagnostic only."
                )
        else:
            print("[build_splits] train/val <-> test patient overlap: 0 (PASSED)")

    return SplitResult(
        train_df=train_df,
        val_df=val_df,
        discovered_counts=discovered_counts,
        train_val_patient_overlap=len(train_val_overlap),
        test_patient_overlap=test_patient_overlap,
        excluded_test_overlap_patients=excluded_patients,
        excluded_test_overlap_images=excluded_images,
    )


def load_class_folder_df(root, expected_counts: Optional[dict] = EXPECTED_TEST_COUNTS) -> pd.DataFrame:
    """
    Scan `root`'s DME/ and NORMAL/ class folders into a path/label/patient
    frame, with no splitting.

    This is for the held-out test set, which build_splits() only ever
    touches indirectly (to check the C8 train/val<->test patient overlap
    when `test_dir` is passed) -- it never returns the test set itself.
    Phase 3 needs the actual test rows to run inference on, hence this.

    `expected_counts` defaults to EXPECTED_TEST_COUNTS (the canonical
    484-image Kermany test release) and is asserted against the discovered
    per-folder counts, same principle and same reason as build_splits's own
    guard on the train counts: catch a wrong/incomplete download immediately
    rather than silently evaluating on the wrong data. Pass None to skip the
    check (e.g. against a synthetic test fixture).
    """
    df, discovered_counts = _scan_class_folders(Path(root))
    if expected_counts is not None:
        for folder_name, expected in expected_counts.items():
            found = discovered_counts.get(folder_name, 0)
            if found != expected:
                raise AssertionError(
                    f"Discovered {found} images in '{folder_name}/', expected {expected} "
                    "(expected_counts). This usually means the wrong dataset release is "
                    "loaded (see PROJECT_BRIEF.md Section 5) or the folder is incomplete."
                )
        print(f"[load_class_folder_df] count assertion vs expected_counts: PASSED {discovered_counts}")
    else:
        print(f"[load_class_folder_df] count assertion skipped (expected_counts=None); discovered {discovered_counts}")
    return df


# ---------------------------------------------------------------------------
# Preprocessing (C2, C4)
# ---------------------------------------------------------------------------


def _letterbox_resize(image: np.ndarray, target_size: int = IMAGE_SIZE):
    """
    Resize a 2D grayscale array so its longest side equals `target_size`,
    preserving aspect ratio (C2), then zero-pad the shorter side to make it
    square. Returns (canvas, content_bbox) where content_bbox =
    (y0, y1, x0, x1) locates the real (non-padded) pixels — Phase 5's
    thickness index must exclude padding from "retina content width".
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size), dtype=resized.dtype)
    y0 = (target_size - new_h) // 2
    x0 = (target_size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    content_bbox = (y0, y0 + new_h, x0, x0 + new_w)
    return canvas, content_bbox


def flatten_retina(gray: np.ndarray, enabled: bool = True):
    """
    Estimate and correct retinal tilt/curvature so the retinal band sits at
    a consistent vertical position across images.

    Otsu-thresholds the image to isolate the retinal band, fits a degree-2
    polynomial (column -> lower-boundary row) to it, and shifts each column
    vertically to flatten that boundary. If the fit looks unreliable (too
    little boundary coverage, or a fit implying an implausibly large shift),
    the *original* image is returned unchanged rather than a distorted one —
    a silent bad flatten is worse than no flatten. Returns
    (image, fallback_triggered).
    """
    if not enabled:
        return gray, False

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = gray.shape
    lower_boundary = np.full(w, np.nan)
    for col in range(w):
        rows = np.nonzero(binary[:, col])[0]
        if rows.size > 0:
            lower_boundary[col] = rows.max()

    valid = ~np.isnan(lower_boundary)
    if valid.sum() < 0.5 * w:
        return gray, True

    xs = np.nonzero(valid)[0]
    ys = lower_boundary[valid]
    coeffs = np.polyfit(xs, ys, deg=2)
    fitted = np.polyval(coeffs, np.arange(w))

    target_row = np.median(fitted)
    shifts = np.round(target_row - fitted).astype(int)

    if np.max(np.abs(shifts)) > h // 2:
        return gray, True

    flattened = np.zeros_like(gray)
    for col in range(w):
        shift = int(shifts[col])
        if shift >= 0:
            if shift < h:
                flattened[shift:, col] = gray[: h - shift, col]
        else:
            if -shift < h:
                flattened[: h + shift, col] = gray[-shift:, col]

    return flattened, False


def preprocess_image(image: np.ndarray, image_size: int = IMAGE_SIZE, flatten: bool = RETINAL_FLATTENING_ENABLED):
    """
    Full deterministic preprocessing pipeline for one OCT B-scan: grayscale
    -> NLM denoise -> retinal flattening -> CLAHE -> aspect-preserving
    resize + letterbox pad -> 3-channel replication.

    `image` may already be grayscale (2D) or BGR/RGB (3D). Returns
    (rgb_uint8_image, content_bbox, flatten_fallback_triggered).
    Normalisation (mean/std) is deliberately NOT applied here — it happens
    after augmentation, at load time.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    flattened, fallback_triggered = flatten_retina(denoised, enabled=flatten)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(flattened)

    letterboxed, content_bbox = _letterbox_resize(enhanced, image_size)

    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_GRAY2RGB)

    return rgb, content_bbox, fallback_triggered


# ---------------------------------------------------------------------------
# Preprocessing cache
# ---------------------------------------------------------------------------


class PreprocessCache:
    """
    Disk cache for the deterministic, expensive part of preprocess_image()
    (denoise + flatten + CLAHE + letterbox). Augmentation and normalisation
    are NOT cached — they must vary per epoch and happen at load time.

    The cache key covers the source path plus every parameter that affects
    the cached output, so changing image_size or the flatten flag can't
    silently serve stale results computed under the old settings.
    """

    def __init__(self, cache_dir, image_size: int = IMAGE_SIZE, flatten: bool = RETINAL_FLATTENING_ENABLED):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
        self.flatten = flatten

    def _key(self, source_path) -> str:
        raw = f"{source_path}|{self.image_size}|{self.flatten}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, source_path) -> Path:
        return self.cache_dir / f"{self._key(source_path)}.npz"

    def get(self, source_path):
        """Return the cached (rgb, content_bbox, fallback_triggered), or None on a miss."""
        cache_path = self._cache_path(source_path)
        if not cache_path.exists():
            return None
        with np.load(cache_path) as data:
            rgb = data["rgb"]
            content_bbox = tuple(int(v) for v in data["content_bbox"])
            fallback_triggered = bool(data["fallback_triggered"])
        return rgb, content_bbox, fallback_triggered

    def put(self, source_path, rgb, content_bbox, fallback_triggered) -> None:
        cache_path = self._cache_path(source_path)
        tmp_path = cache_path.with_name(cache_path.name + ".tmp")
        # Write through an open file handle, not a bare path -- np.savez_compressed
        # silently appends ".npz" to a path that doesn't already end with it, which
        # would turn "<key>.npz.tmp" into "<key>.npz.tmp.npz" and break the rename below.
        with open(tmp_path, "wb") as fh:
            np.savez_compressed(
                fh,
                rgb=rgb,
                content_bbox=np.array(content_bbox, dtype=np.int64),
                fallback_triggered=np.array(fallback_triggered),
            )
        tmp_path.replace(cache_path)  # atomic on POSIX — a crash mid-write can't corrupt an entry

    def get_or_compute(self, source_path):
        cached = self.get(source_path)
        if cached is not None:
            return cached
        image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {source_path}")
        result = preprocess_image(image, image_size=self.image_size, flatten=self.flatten)
        self.put(source_path, *result)
        return result
