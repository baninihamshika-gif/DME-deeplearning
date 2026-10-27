"""
Phase 4 (extended objective E1): U-Net segmentation of intraretinal fluid.

Confirm all Phase 3 artifacts are saved before starting (PROJECT_BRIEF.md
Phase 4 preamble) -- this project's Phase 3 sign-off is already recorded in
phase3-ablation-and-signoff.md before this file was written.

--------------------------------------------------------------------------
Duke DME dataset (Chiu et al., Biomedical Optics Express 2015)
--------------------------------------------------------------------------
10 Subject_XX.mat files. Each holds 61 B-scans per subject, but only a
subset (typically 11) are manually graded -- the exact indices are
DETECTED per subject at load time (never hardcoded), by checking which
scans are not entirely NaN in manualLayers1. Across all 10 subjects this
totals 110 manually annotated scans, matching PROJECT_BRIEF.md Section 5's
"110 B-scans from 10 DME patients" exactly.

.mat structure, inspected directly with scipy.io.loadmat rather than
assumed from the paper text:

    images            (496, 768, 61) uint8    -- grayscale B-scans
    manualFluid1/2    (496, 768, 61) float64  -- per-pixel INTEGER region
                                                 labels (0, 1..13), NaN
                                                 where that scan wasn't
                                                 graded by this rater.
                                                 Already a raster, not
                                                 coordinates -- binarized
                                                 here as (label > 0).
    manualLayers1/2   (8, 768, 61)  float64   -- y-COORDINATE per column
                                                 per boundary (NaN = un-
                                                 graded column/scan). 8
                                                 boundaries, confirmed
                                                 top-to-bottom by mean y
                                                 (boundary 0 = ILM,
                                                 boundary 7 = deepest).
    automaticFluidDME, automaticLayersDME/Normal -- the paper's own
                                                 algorithm output, not
                                                 human ground truth. Not
                                                 used here.

Judgment calls, stated up front (same "explain before coding" convention
as explain.py / app.py):

  - The training target is the FLUID mask only, not the 8 layer
    boundaries. The brief's own model line is explicit -- `classes=1` --
    and fluid is E1's clinically central target: it is what actually
    drives the Phase 6 referral decision ("DME detected"). The 8 layer
    boundaries ARE rasterized in this file, but only to build the
    mandatory pre-training verification overlay the brief requires
    ("show a visual overlay of a rasterised mask on its source B-scan
    before proceeding") -- see verify_rasterization(). They are not a
    second training head. Retinal-layer boundary extraction for Phase 5's
    thickening index is a separate, later concern, and may use a
    different (non-U-Net) approach.

  - Grader 1 (manualFluid1) is used as ground truth. Grader 2
    (manualFluid2) is not used for training; it is the inter-grader
    reference reported next to the model (split_report()).

  - "boundary dist" in the brief's Phase 4 sign-off template is read as
    the standard image-segmentation metric -- mean symmetric surface
    distance, in pixels, between the predicted and true FLUID mask
    contours -- rather than a retinal-layer boundary distance, since this
    model has no layer-boundary output to measure that against.

  - No retinal flattening (the classifier's C2 preprocessing step) is
    applied here. Flattening shifts image columns vertically; doing that
    to the B-scan without applying the identical per-column shift to the
    fluid mask would desynchronise the two. Denoise + CLAHE (which do not
    move pixels) and the aspect-preserving letterbox resize (C2, applied
    identically to image and mask) are used, matching the classifier's
    preprocessing everywhere it is safe to reuse.

  - The letterbox resize here targets IMAGE_SIZE_SEG (320), not the
    classifier's IMAGE_SIZE (300). Discovered empirically, not assumed:
    running the real training loop (`--smoke`) raised
    `RuntimeError: Wrong input shape ... Expected image height and width
    divisible by 32` from smp.Unet -- its encoder has 5 downsampling
    stages (2**5 = 32), a constraint the classifier-only 300px choice
    never had to satisfy. 320 is the nearest multiple of 32 at or above
    300 (matching smp's own suggested fix in that error message). This
    does not affect transfer_encoder_weights(): convolution weights carry
    no spatial-resolution dependency, so a classifier trained at 300px
    transfers cleanly into a U-Net run at 320px -- confirmed by the same
    --smoke run (572/572 encoder keys still matched).

  - Patient = subject file, by construction: Subject_01.mat is one
    patient's 61 scans, so C1's patient-grouping requirement is trivially
    satisfied per file. The brief explicitly asks to split by patient
    ("10 patients only, so this matters more, not less") -- done here as
    a 3-way patient-grouped split (train/val/test) so the sign-off
    template's separate Val and Test metrics are both real, not one
    number duplicated into both slots.

Usage:
    python segment.py --duke-dir data/raw/duke_dme/2015_BOE_Chiu \\
        --classifier-checkpoint artifacts/kaggle/<run>/artifacts/checkpoint_best.pt \\
        [--smoke] [--artifacts-dir artifacts/segmentation_retrain]

The training/evaluation section was rewritten on 2026-10-28 after the
research audit found the first trained model non-functional; see the
"Train / eval" block comment and research_outputs/segmentation_retrain_report.md.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from utils import set_seed

MODEL_NAME_SEG = "timm-efficientnet-b3"

# ---------------------------------------------------------------------------
# Judgment-call constants (see module docstring). Not in PROJECT_BRIEF.md
# Section 8 -- the brief's config list predates the segmentation phase --
# added here rather than silently hardcoded inline.
# ---------------------------------------------------------------------------
EPOCHS_SEG = 60  # small dataset (~66 train images); generous budget, early stopping does the real work
VAL_PATIENTS = 2
TEST_PATIENTS = 2
# smp.Unet's encoder has 5 downsampling stages -> input H/W must be divisible by 2**5=32.
# config.IMAGE_SIZE (300, chosen for the classifier) is not; 320 is the nearest multiple
# of 32 at or above 300. See module docstring judgment-call list for how this was found.
IMAGE_SIZE_SEG = 320
# fraction of the 768-column width a run of NaN-free columns must span to be
# drawn as one connected boundary segment, rather than jumping across a real gap
MAX_BOUNDARY_GAP_COLS = 3


# ---------------------------------------------------------------------------
# Duke ingestion
# ---------------------------------------------------------------------------


def find_subject_files(duke_dir) -> List[Path]:
    duke_dir = Path(duke_dir)
    files = sorted(duke_dir.glob("Subject_*.mat"))
    if not files:
        raise FileNotFoundError(
            f"No Subject_*.mat files found under {duke_dir}. Expected the extracted "
            "2015_BOE_Chiu2.zip contents (10 files, Subject_01.mat .. Subject_10.mat)."
        )
    return files


def _subject_id(mat_path: Path) -> str:
    # "Subject_01.mat" -> "01" -- used directly as the patient id for the
    # patient-grouped split (C1); one .mat file is one patient by construction.
    return mat_path.stem.replace("Subject_", "")


def detect_annotated_scans(layers1: np.ndarray) -> List[int]:
    """layers1: (8, 768, n_scans). A scan counts as annotated if it is not
    entirely NaN across all 8 boundaries and all 768 columns -- detected per
    subject, never hardcoded, since the paper's own text doesn't guarantee
    every subject used the identical scan indices."""
    not_all_nan = ~np.all(np.isnan(layers1), axis=(0, 1))
    return [int(i) for i in np.where(not_all_nan)[0]]


def load_duke_subject(mat_path: Path) -> dict:
    """Loads one Subject_XX.mat and returns only the manually-annotated scans,
    as parallel lists aligned by index. Grader 2's arrays are used only as
    the inter-grader reference (see module docstring), never in training.

    variable_names restricts scipy.io.loadmat to the 5 arrays actually used
    (images + both graders' fluid/layers). Discovered empirically while
    running the real --smoke pipeline: an unrestricted loadmat() pulls in
    automaticFluidDME/automaticLayersDME/automaticLayersNormal too -- the
    paper's own algorithm output, never used anywhere in this file -- and
    automaticFluidDME alone is ~186MB per subject (same shape/dtype as
    manualFluid1). Across all 10 subjects that is >1.8GB of dead weight
    that contributed to a real cgroup OOM-kill during local smoke testing.
    """
    import scipy.io as sio

    m = sio.loadmat(
        str(mat_path),
        variable_names=["images", "manualFluid1", "manualFluid2", "manualLayers1", "manualLayers2"],
    )
    layers1 = m["manualLayers1"]  # (8, 768, 61)
    annotated = detect_annotated_scans(layers1)
    if not annotated:
        raise ValueError(f"{mat_path}: no manually-annotated scans detected (all-NaN manualLayers1).")

    subject = _subject_id(mat_path)
    images, fluid1, fluid2, layers1_out, layers2_out, scan_indices = [], [], [], [], [], []
    for idx in annotated:
        images.append(m["images"][:, :, idx])
        fluid1.append(m["manualFluid1"][:, :, idx])
        fluid2.append(m["manualFluid2"][:, :, idx])
        layers1_out.append(layers1[:, :, idx])
        layers2_out.append(m["manualLayers2"][:, :, idx])
        scan_indices.append(idx)

    return {
        "subject": subject,
        "scan_indices": scan_indices,
        "images": images,
        "fluid1": fluid1,
        "fluid2": fluid2,
        "layers1": layers1_out,
        "layers2": layers2_out,
    }


def binarize_fluid(fluid_raw: np.ndarray) -> np.ndarray:
    """manualFluid1/2 holds integer region labels (0 = none, 1..13 = distinct
    fluid pockets) with NaN where ungraded. Any labeled region is fluid."""
    return (np.nan_to_num(fluid_raw, nan=0.0) > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Mandatory pre-training verification (brief: "show a visual overlay of a
# rasterised mask on its source B-scan before proceeding" -- a wrong
# rasterisation is nearly undetectable from Dice alone).
# ---------------------------------------------------------------------------

_BOUNDARY_COLORS = [
    (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0),
    (255, 128, 0), (128, 0, 255), (0, 128, 255), (255, 255, 255),
]


def draw_boundary_overlay(image_gray: np.ndarray, layers: np.ndarray) -> np.ndarray:
    """layers: (8, W) y-per-column, NaN where ungraded. Draws all 8 boundaries
    as thin polylines over the (grayscale -> RGB) image, skipping NaN columns
    and not bridging gaps wider than MAX_BOUNDARY_GAP_COLS columns."""
    rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB).copy()
    w = image_gray.shape[1]
    for b in range(layers.shape[0]):
        ys = layers[b]
        pts = [(x, int(round(ys[x]))) for x in range(w) if not np.isnan(ys[x])]
        for i in range(1, len(pts)):
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i]
            if x1 - x0 <= MAX_BOUNDARY_GAP_COLS:
                cv2.line(rgb, (x0, y0), (x1, y1), _BOUNDARY_COLORS[b % len(_BOUNDARY_COLORS)], 1, lineType=cv2.LINE_AA)
    return rgb


def draw_fluid_overlay(image_gray: np.ndarray, fluid_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
    red = np.zeros_like(rgb)
    red[:, :, 0] = 255
    mask3 = np.repeat(fluid_mask[:, :, None].astype(bool), 3, axis=2)
    out = rgb.copy()
    out[mask3] = (1 - alpha) * rgb[mask3] + alpha * red[mask3]
    return out.astype(np.uint8)


def verify_rasterization(duke_dir, artifacts_dir, subject: Optional[str] = None, scan_pos: int = 3) -> Path:
    """Generates artifacts/segmentation_rasterization_check.png: original,
    +fluid mask overlay, +boundary overlay, for one real annotated scan.
    Does not require torch -- run this before any training, per the brief."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = find_subject_files(duke_dir)
    mat_path = files[0] if subject is None else next(f for f in files if _subject_id(f) == subject)
    sample = load_duke_subject(mat_path)
    pos = min(scan_pos, len(sample["images"]) - 1)

    img = sample["images"][pos]
    fluid_mask = binarize_fluid(sample["fluid1"][pos])
    layers = sample["layers1"][pos]
    scan_idx = sample["scan_indices"][pos]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"Subject_{sample['subject']}, scan {scan_idx} (0-based) -- original")
    axes[0].axis("off")
    axes[1].imshow(draw_fluid_overlay(img, fluid_mask))
    axes[1].set_title("+ manualFluid1 (red) rasterized")
    axes[1].axis("off")
    axes[2].imshow(draw_boundary_overlay(img, layers))
    axes[2].set_title("+ manualLayers1, 8 boundaries (rasterized from coords)")
    axes[2].axis("off")
    plt.tight_layout()

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "segmentation_rasterization_check.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[segment] rasterization verification figure written: {out_path}")
    print(f"[segment]   fluid pixels: {int(fluid_mask.sum())} / {fluid_mask.size}")
    print(f"[segment]   boundary NaN columns: {int(np.isnan(layers).any(axis=0).sum())} / {layers.shape[1]}")
    return out_path


# ---------------------------------------------------------------------------
# Preprocessing shared by image + mask (must be applied identically to both,
# unlike the classifier's preprocess_image() which only has an image to move)
# ---------------------------------------------------------------------------


def _letterbox_pair(image_gray: np.ndarray, mask: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image_gray.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    img_resized = cv2.resize(image_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left

    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    mask_padded = cv2.copyMakeBorder(mask_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return img_padded, mask_padded


def preprocess_pair(image_gray: np.ndarray, mask: np.ndarray, image_size: int = IMAGE_SIZE_SEG) -> Tuple[np.ndarray, np.ndarray]:
    """grayscale -> NLM denoise -> CLAHE (image only; mask is untouched by
    either, both are pixel-value operations that don't move anything) ->
    aspect-preserving letterbox resize (image AND mask, identically) -> 3ch."""
    if image_gray.dtype != np.uint8:
        image_gray = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(image_gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    img_padded, mask_padded = _letterbox_pair(enhanced, mask.astype(np.uint8), image_size)
    rgb = cv2.cvtColor(img_padded, cv2.COLOR_GRAY2RGB)
    return rgb, mask_padded


# ---------------------------------------------------------------------------
# Graded region
# ---------------------------------------------------------------------------


def graded_region(fluid_raw: np.ndarray, layers: np.ndarray) -> np.ndarray:
    """Pixels the grader actually assessed for fluid, as a uint8 (H, W) mask.

    Checked directly on all 110 annotated scans (retrain, 2026-10-28) rather
    than assumed: manualFluid1/2 contain NO NaN on the annotated scans -- the
    ungraded lateral margins are stored as 0, i.e. indistinguishable from
    "graded, no fluid". The graded extent is instead recorded in
    manualLayers: only a central band of ~500-550 of the 768 columns carries
    boundary coordinates, the two graders' column bands are identical, and
    zero fluid pixels from either grader fall outside that band. So graded =
    (fluid not NaN) AND (column has any manual layer coordinate). The NaN
    term is kept so a scan that IS NaN-marked is still handled correctly.
    Letterbox padding is excluded downstream because the padded border of
    this mask is 0.
    """
    cols = ~np.all(np.isnan(layers), axis=0)  # (W,)
    graded = ~np.isnan(fluid_raw)
    graded &= cols[None, :]
    return graded.astype(np.uint8)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_duke_index(duke_dir) -> List[dict]:
    """Returns one entry per manually-annotated scan across all 10 subjects:
    {mat_path, subject, scan_idx, pos} -- 'pos' indexes into that subject's
    already-filtered annotated-scan lists (see load_duke_subject)."""
    entries = []
    for mat_path in find_subject_files(duke_dir):
        sample = load_duke_subject(mat_path)
        for pos, scan_idx in enumerate(sample["scan_indices"]):
            entries.append({"mat_path": mat_path, "subject": sample["subject"], "scan_idx": scan_idx, "pos": pos})
    print(f"[segment] Duke index: {len(entries)} manually-annotated scans across {len(set(e['subject'] for e in entries))} subjects")
    return entries


def build_splits_duke(entries: List[dict], val_n: int = VAL_PATIENTS, test_n: int = TEST_PATIENTS, seed: int = config.SEED):
    subjects = sorted(set(e["subject"] for e in entries))
    rng = np.random.default_rng(seed)
    shuffled = list(subjects)
    rng.shuffle(shuffled)

    if len(shuffled) <= val_n + test_n:
        raise AssertionError(
            f"Only {len(shuffled)} Duke patients found, need > {val_n + test_n} (val_n={val_n} + test_n={test_n}) "
            "to hold out a non-empty train set. Check --duke-dir."
        )

    test_subjects = set(shuffled[:test_n])
    val_subjects = set(shuffled[test_n : test_n + val_n])
    train_subjects = set(shuffled[test_n + val_n :])
    assert not (train_subjects & val_subjects) and not (train_subjects & test_subjects) and not (val_subjects & test_subjects)

    train = [e for e in entries if e["subject"] in train_subjects]
    val = [e for e in entries if e["subject"] in val_subjects]
    test = [e for e in entries if e["subject"] in test_subjects]
    print(
        f"[segment] patient-grouped split (seed={seed}): "
        f"train={len(train_subjects)} patients/{len(train)} scans {sorted(train_subjects)}, "
        f"val={len(val_subjects)} patients/{len(val)} scans {sorted(val_subjects)}, "
        f"test={len(test_subjects)} patients/{len(test)} scans {sorted(test_subjects)}"
    )
    print("[segment] patient overlap train<->val<->test: 0 (PASSED, by construction -- disjoint subject sets)")
    return train, val, test


def load_scan(entry: dict, cache: dict) -> dict:
    """Native-resolution arrays for one annotated scan: image, grader-1 and
    grader-2 binary fluid masks, and the graded region (shared by both
    graders -- see graded_region). `cache` maps mat_path -> subject dict."""
    if entry["mat_path"] not in cache:
        cache[entry["mat_path"]] = load_duke_subject(entry["mat_path"])
    sample = cache[entry["mat_path"]]
    pos = entry["pos"]
    return {
        "image": sample["images"][pos],
        "gt1": binarize_fluid(sample["fluid1"][pos]),
        "gt2": binarize_fluid(sample["fluid2"][pos]),
        "graded": graded_region(sample["fluid1"][pos], sample["layers1"][pos])
        & graded_region(sample["fluid2"][pos], sample["layers2"][pos]),
    }


class DukeFluidDataset:
    """torch.utils.data.Dataset subclassed lazily (see main()) so this module
    imports cleanly without torch installed, matching train.py's pattern of
    keeping torch imports inside functions/__main__ where practical.

    Returns (image, fluid mask, graded mask), all at IMAGE_SIZE_SEG. The
    deterministic preprocessing (NLM denoise is the slow part) is computed
    once per scan and cached; only the random augmentation runs per epoch."""

    def __init__(self, entries: List[dict], image_size: int = IMAGE_SIZE_SEG, augment: bool = False):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        self.entries = entries
        self.image_size = image_size
        self.augment = augment
        subject_cache = {}
        self._pre = []
        for e in entries:
            scan = load_scan(e, subject_cache)
            rgb, mask = preprocess_pair(scan["image"], scan["gt1"], image_size)
            _, graded = preprocess_pair(scan["image"], scan["graded"], image_size)
            self._pre.append((rgb, mask, graded))

        norm = [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
        if augment:
            # No vertical flip (C4). Constant-0 border so a shifted/rotated frame
            # brings in ungraded (graded=0) pixels, never reflected fake labels.
            self.tfm = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.7,
                        border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0,
                    ),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                ]
                + norm
            )
        else:
            self.tfm = A.Compose(norm)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        rgb, mask, graded = self._pre[i]
        out = self.tfm(image=rgb, masks=[mask, graded])
        mask_t, graded_t = (m.float().unsqueeze(0) for m in out["masks"])  # (1, H, W) each
        return out["image"], mask_t, graded_t


# ---------------------------------------------------------------------------
# Model + encoder transfer
# ---------------------------------------------------------------------------


def build_seg_model():
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder_name=MODEL_NAME_SEG, encoder_weights=None, in_channels=3, classes=1)


def transfer_encoder_weights(unet, classifier_checkpoint_path: Path) -> Tuple[int, int]:
    """Initialises the U-Net's encoder from the Phase 2 classifier checkpoint
    via load_state_dict(..., strict=False), per the brief. Returns
    (matched, total_encoder_keys) using torch's own IncompatibleKeys
    accounting -- not a guess.

    Note: segmentation_models_pytorch's EfficientNetEncoder overrides
    load_state_dict() and does not return the IncompatibleKeys namedtuple
    (verified against segmentation-models-pytorch==0.3.* -- it calls
    super().load_state_dict(...) but discards the result). Calling
    torch.nn.Module.load_state_dict directly on the encoder bypasses that
    override so the real match count can be read and reported, as the
    brief requires ("report how many encoder keys matched -- a near-zero
    match means the transfer silently failed").
    """
    import torch
    from train import load_checkpoint

    ckpt = load_checkpoint(classifier_checkpoint_path, map_location="cpu")
    clf_state = ckpt["model_state_dict"]
    # classifier.* has no place in a segmentation encoder (classification-specific
    # final layer); smp's own override also strips these, done explicitly here too
    # so the reported "total" below is the real intended transfer set.
    filtered = {k: v for k, v in clf_state.items() if not k.startswith("classifier.")}

    result = torch.nn.Module.load_state_dict(unet.encoder, filtered, strict=False)
    total_encoder_keys = len(unet.encoder.state_dict())
    matched = total_encoder_keys - len(result.missing_keys)
    print(f"[segment] encoder transfer from {classifier_checkpoint_path}:")
    print(f"[segment]   matched: {matched}/{total_encoder_keys} encoder keys")
    if result.missing_keys:
        print(f"[segment]   missing (random init): {result.missing_keys[:10]}{' ...' if len(result.missing_keys) > 10 else ''}")
    if result.unexpected_keys:
        print(f"[segment]   unexpected (ignored): {result.unexpected_keys[:10]}{' ...' if len(result.unexpected_keys) > 10 else ''}")
    if matched == 0:
        raise RuntimeError("Encoder transfer matched 0 keys -- transfer silently failed. Refusing to proceed with a randomly-initialised encoder mislabeled as transferred.")
    return matched, total_encoder_keys


# ---------------------------------------------------------------------------
# Metrics (pure numpy, testable without torch)
# ---------------------------------------------------------------------------


def dice_coefficient(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    return float((2.0 * intersection + eps) / (pred.sum() + target.sum() + eps))


def iou_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float((intersection + eps) / (union + eps))


def boundary_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean symmetric surface distance (pixels) between predicted and true
    mask contours -- the standard image-segmentation "boundary distance"
    metric (see module docstring for why this reading was chosen over a
    retinal-layer interpretation). Returns nan if either mask is empty
    (no contour to measure)."""
    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan")

    pred_dist = cv2.distanceTransform(1 - pred, cv2.DIST_L2, 5)
    target_dist = cv2.distanceTransform(1 - target, cv2.DIST_L2, 5)

    pred_boundary = pred - cv2.erode(pred, np.ones((3, 3), np.uint8))
    target_boundary = target - cv2.erode(target, np.ones((3, 3), np.uint8))

    d1 = target_dist[pred_boundary.astype(bool)]
    d2 = pred_dist[target_boundary.astype(bool)]
    all_d = np.concatenate([d1, d2]) if (len(d1) + len(d2)) > 0 else np.array([0.0])
    return float(all_d.mean())


# ---------------------------------------------------------------------------
# Train / eval
#
# Retrain (2026-10-28) after the research audit found the first run
# non-functional (fluid predicted on ~100% of pixels, best checkpoint at
# epoch 1). Changes, each tied to an audit finding:
#   - loss and every metric are restricted to the graded region
#     (graded_region(): manually graded columns, letterbox padding excluded);
#   - class imbalance (fluid ~1% of graded pixels): BCE with a fixed
#     pos_weight plus a soft Dice term pooled over the batch, both on graded
#     pixels only -- fixed a priori, not tuned;
#   - checkpoint AND probability threshold are selected jointly on val
#     only: each epoch, val pooled Dice is computed over THRESHOLD_GRID and
#     its maximum is the epoch's score (pos_weight shifts calibration, so a
#     fixed 0.5 would rank epochs by calibration rather than separation);
#     patience is long enough to get past the first epochs;
#   - metrics are computed at native 496x768 resolution (probabilities are
#     un-letterboxed and resized back), mean and pooled, all images and
#     fluid-positive images only, next to all-foreground / all-background
#     baselines and grader-2-vs-grader-1 agreement;
#   - test is evaluated exactly once, after everything above is fixed, and
#     the script refuses to overwrite an existing final summary.
# ---------------------------------------------------------------------------

EPOCHS_SEG_MAX = 100
EARLY_STOP_PATIENCE_SEG = 15
BATCH_SIZE_SEG_CPU = 4  # 66 train scans -> ~17 optimiser steps/epoch (config.BATCH_SIZE_SEG=16 gave ~5)
LR_SEG_RETRAIN = 3e-4
WEIGHT_DECAY_SEG = 1e-4
BCE_POS_WEIGHT = 10.0  # fixed a priori; Dice term carries the rest of the imbalance handling
DICE_LOSS_WEIGHT = 1.0
THRESHOLD_GRID = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]


def threshold_sweep(probs: List[np.ndarray], gts: List[np.ndarray], gradeds: List[np.ndarray]) -> Tuple[float, dict]:
    """Pooled graded-pixel Dice at every THRESHOLD_GRID value; returns
    (best threshold, {threshold: Dice}). Ties go to the threshold nearer 0.5."""
    ms = [m.astype(bool) for m in gradeds]
    p_g = np.concatenate([p[m] for p, m in zip(probs, ms)])
    g_g = np.concatenate([g[m].astype(bool) for g, m in zip(gts, ms)])
    sweep = {}
    for t in THRESHOLD_GRID:
        pred = p_g > t
        denom = pred.sum() + g_g.sum()
        sweep[t] = float(2 * np.logical_and(pred, g_g).sum() / denom) if denom else float("nan")
    best = max(THRESHOLD_GRID, key=lambda t: (np.nan_to_num(sweep[t], nan=-1.0), -abs(t - 0.5)))
    return best, sweep


def masked_seg_loss(logits, mask, graded, pos_weight: float = BCE_POS_WEIGHT, dice_weight: float = DICE_LOSS_WEIGHT):
    """BCE(pos_weight) + soft Dice, both computed on graded pixels only.
    Dice is pooled over the whole batch so fluid-free scans still contribute
    false-positive penalty without a 0/0 per-image term."""
    import torch
    import torch.nn.functional as F

    pw = torch.tensor(pos_weight, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=pw, reduction="none")
    bce = (bce * graded).sum() / graded.sum().clamp_min(1.0)
    prob = torch.sigmoid(logits) * graded
    tgt = mask * graded
    dice = 1.0 - (2.0 * (prob * tgt).sum() + 1.0) / (prob.sum() + tgt.sum() + 1.0)
    return bce + dice_weight * dice


def unletterbox(prob: np.ndarray, orig_hw: Tuple[int, int], size: int = IMAGE_SIZE_SEG) -> np.ndarray:
    """Inverse of _letterbox_pair for a (size, size) probability map: crop
    the padding back off and resize to the native (H, W) bilinearly."""
    h, w = orig_hw
    scale = size / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    top, left = (size - new_h) // 2, (size - new_w) // 2
    crop = prob[top : top + new_h, left : left + new_w]
    return cv2.resize(crop.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def native_eval_set(entries: List[dict]) -> List[dict]:
    """Native-resolution ground truth for evaluation, one dict per scan."""
    cache, out = {}, []
    for e in entries:
        scan = load_scan(e, cache)
        out.append({"subject": e["subject"], "scan_idx": e["scan_idx"], **{k: scan[k] for k in ("gt1", "gt2", "graded")},
                    "hw": scan["image"].shape[:2]})
    return out


def predict_native(model, ds: "DukeFluidDataset", eval_set: List[dict], device, batch_size: int = BATCH_SIZE_SEG_CPU) -> List[np.ndarray]:
    import torch

    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            batch = torch.stack([ds[j][0] for j in range(i, min(i + batch_size, len(ds)))]).to(device)
            probs.extend(torch.sigmoid(model(batch))[:, 0].cpu().numpy())
    return [unletterbox(p, s["hw"]) for p, s in zip(probs, eval_set)]


def segmentation_metrics(preds: List[np.ndarray], gts: List[np.ndarray], gradeds: List[np.ndarray]) -> dict:
    """Dice/IoU on graded pixels only. Per-image Dice of an image with empty
    ground truth AND empty prediction is 1 (dice_coefficient's eps
    convention); that is why the fluid-positive-only mean is reported too.
    Pooled = pixel counts summed over all images before the ratio."""
    dices, ious, pos_dices, pos_ious, bdists = [], [], [], [], []
    inter = psum = gsum = union = 0
    pred_frac, fp_on_empty, n_empty = [], 0, 0
    for p, g, m in zip(preds, gts, gradeds):
        m = m.astype(bool)
        p = p.astype(bool) & m
        g = g.astype(bool) & m
        d, j = dice_coefficient(p[m], g[m]), iou_score(p[m], g[m])
        dices.append(d)
        ious.append(j)
        inter += int(np.logical_and(p, g).sum())
        psum += int(p.sum())
        gsum += int(g.sum())
        union += int(np.logical_or(p, g).sum())
        pred_frac.append(float(p[m].mean()))
        if g.any():
            pos_dices.append(d)
            pos_ious.append(j)
            bdists.append(boundary_distance(p, g))
        else:
            n_empty += 1
            fp_on_empty += int(p.any())
    finite_bd = [b for b in bdists if not np.isnan(b)]
    return {
        "n_images": len(dices),
        "n_fluid_positive": len(pos_dices),
        "n_empty_gt": n_empty,
        "n_empty_gt_with_any_predicted_fluid": fp_on_empty,
        "mean_dice": float(np.mean(dices)),
        "mean_iou": float(np.mean(ious)),
        "mean_dice_fluid_positive": float(np.mean(pos_dices)) if pos_dices else float("nan"),
        "mean_iou_fluid_positive": float(np.mean(pos_ious)) if pos_ious else float("nan"),
        "pooled_dice": float(2 * inter / (psum + gsum)) if (psum + gsum) else float("nan"),
        "pooled_iou": float(inter / union) if union else float("nan"),
        "mean_boundary_dist_px_fluid_positive": float(np.mean(finite_bd)) if finite_bd else float("nan"),
        "mean_pred_frac_of_graded": float(np.mean(pred_frac)),
        "gt_frac_of_graded_pooled": float(gsum / max(1, sum(int(m.sum()) for m in gradeds))),
    }


def split_report(probs: Optional[List[np.ndarray]], eval_set: List[dict], threshold: float) -> dict:
    """Model (if probs given), both trivial baselines and inter-grader
    agreement on one split, plus per-subject pooled Dice for the model."""
    gts = [s["gt1"] for s in eval_set]
    gradeds = [s["graded"] for s in eval_set]
    out = {
        "subjects": sorted(set(s["subject"] for s in eval_set)),
        "baseline_all_foreground": segmentation_metrics([np.ones_like(g) for g in gts], gts, gradeds),
        "baseline_all_background": segmentation_metrics([np.zeros_like(g) for g in gts], gts, gradeds),
        "inter_grader_g2_vs_g1": segmentation_metrics([s["gt2"] for s in eval_set], gts, gradeds),
    }
    if probs is not None:
        preds = [(p > threshold).astype(np.uint8) for p in probs]
        out["model"] = segmentation_metrics(preds, gts, gradeds)
        out["model_per_subject"] = {
            subj: segmentation_metrics(*zip(*[(pr, s["gt1"], s["graded"]) for pr, s in zip(preds, eval_set) if s["subject"] == subj]))
            for subj in out["subjects"]
        }
    return out


def per_image_rows(split: str, probs: List[np.ndarray], eval_set: List[dict], threshold: float) -> List[dict]:
    rows = []
    for p, s in zip(probs, eval_set):
        m = s["graded"].astype(bool)
        pred = (p > threshold) & m
        g1, g2 = s["gt1"].astype(bool) & m, s["gt2"].astype(bool) & m
        rows.append({
            "split": split, "subject": s["subject"], "scan_idx": s["scan_idx"],
            "graded_pixels": int(m.sum()), "gt1_fluid_pixels": int(g1.sum()), "gt2_fluid_pixels": int(g2.sum()),
            "pred_fluid_pixels": int(pred.sum()),
            "dice_model_vs_g1": dice_coefficient(pred[m], g1[m]), "iou_model_vs_g1": iou_score(pred[m], g1[m]),
            "dice_model_vs_g2": dice_coefficient(pred[m], g2[m]),
            "dice_g2_vs_g1": dice_coefficient(g2[m], g1[m]),
            "mean_prob_graded": float(p[m].mean()),
        })
    return rows


def _log_epoch_row(log_path: Path, row: dict) -> None:
    write_header = not Path(log_path).exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_seg_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total, n = 0.0, 0
    for images, masks, graded in loader:
        images, masks, graded = images.to(device), masks.to(device), graded.to(device)
        optimizer.zero_grad()
        loss = masked_seg_loss(model(images), masks, graded)
        loss.backward()
        optimizer.step()
        total += loss.item() * images.size(0)
        n += images.size(0)
    return total / max(1, n)


def save_examples_figure(out_path: Path, ds: "DukeFluidDataset", probs, eval_set, threshold: float, n: int = 4) -> None:
    """Native-res overlays for the n scans with the most grader-1 fluid
    (plus their grader-2 mask) -- qualitative only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(range(len(eval_set)), key=lambda i: -int((eval_set[i]["gt1"] & eval_set[i]["graded"]).sum()))[:n]
    cache = {}
    fig, axes = plt.subplots(len(order), 3, figsize=(15, 3.2 * len(order)))
    for row, i in enumerate(order):
        s = eval_set[i]
        img = load_scan(ds.entries[i], cache)["image"]
        pred = ((probs[i] > threshold) & s["graded"].astype(bool)).astype(np.uint8)
        for col, (mask, title) in enumerate(((s["gt1"], "grader 1"), (s["gt2"], "grader 2"), (pred, f"model (t={threshold})"))):
            axes[row, col].imshow(draw_fluid_overlay(img, mask))
            axes[row, col].set_title(f"Subject_{s['subject']} scan {s['scan_idx']} -- {title}", fontsize=9)
            axes[row, col].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Phase 4 (extended, E1): U-Net fluid segmentation, Duke DME dataset.")
    parser.add_argument("--duke-dir", type=Path, required=True, help="Path to the extracted 2015_BOE_Chiu folder (10 Subject_XX.mat files).")
    parser.add_argument("--classifier-checkpoint", type=Path, required=True, help="Phase 2 checkpoint_best.pt to initialise the encoder from.")
    parser.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR / "segmentation_retrain")
    parser.add_argument("--verify-only", action="store_true", help="Only generate the rasterization verification figure and exit -- no torch/GPU needed.")
    parser.add_argument("--smoke", action="store_true", help="2 epochs through the real path (training, val selection, threshold choice); never touches test.")
    parser.add_argument("--max-epochs", type=int, default=EPOCHS_SEG_MAX)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(config.SEED)
    if args.smoke:
        args.artifacts_dir = args.artifacts_dir / "smoke"
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.artifacts_dir / "segmentation_summary.json"
    if summary_path.exists() and not args.smoke:
        raise SystemExit(f"{summary_path} exists: test has already been evaluated for this run. Use a new --artifacts-dir.")

    # Mandatory gate, per the brief: verify rasterization BEFORE anything else.
    verify_rasterization(args.duke_dir, args.artifacts_dir)
    if args.verify_only:
        print("[segment] --verify-only: stopping after the rasterization check, as requested.")
        return

    import torch
    from torch.utils.data import DataLoader
    from train import save_checkpoint, load_checkpoint

    entries = build_duke_index(args.duke_dir)
    train_entries, val_entries, test_entries = build_splits_duke(entries)

    train_ds = DukeFluidDataset(train_entries, augment=True)
    val_ds = DukeFluidDataset(val_entries, augment=False)
    val_set = native_eval_set(val_entries)
    gen = torch.Generator().manual_seed(config.SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_SEG_CPU, shuffle=True, num_workers=args.num_workers, generator=gen)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_seg_model().to(device)
    matched, total_keys = transfer_encoder_weights(model, args.classifier_checkpoint)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_SEG_RETRAIN, weight_decay=WEIGHT_DECAY_SEG)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    epochs_total = 2 if args.smoke else args.max_epochs
    best_ckpt_path = args.artifacts_dir / "checkpoint_best.pt"
    log_path = args.artifacts_dir / "segmentation_training_log.csv"
    if log_path.exists():
        log_path.unlink()

    best_val, best_epoch, epochs_no_improve = -1.0, 0, 0
    for epoch in range(1, epochs_total + 1):
        t0 = time.time()
        train_loss = train_seg_epoch(model, train_loader, optimizer, device)
        val_probs = predict_native(model, val_ds, val_set, device)
        val_gts, val_gradeds = [s["gt1"] for s in val_set], [s["graded"] for s in val_set]
        epoch_t, epoch_sweep = threshold_sweep(val_probs, val_gts, val_gradeds)
        vm = segmentation_metrics([(p > epoch_t).astype(np.uint8) for p in val_probs], val_gts, val_gradeds)
        val_score = np.nan_to_num(epoch_sweep[epoch_t], nan=0.0)
        scheduler.step(val_score)
        epoch_time = time.time() - t0

        improved = val_score > best_val
        if improved:
            best_val, best_epoch, epochs_no_improve = val_score, epoch, 0
            save_checkpoint(best_ckpt_path, {
                "model_state_dict": model.state_dict(), "model_name": MODEL_NAME_SEG, "epoch": epoch,
                "val_pooled_dice": val_score, "val_threshold": epoch_t,
                "selection": "max over threshold grid of val pooled Dice (graded pixels, native res)",
            })
        else:
            epochs_no_improve += 1

        _log_epoch_row(log_path, {
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": train_loss,
            "val_best_threshold": epoch_t, "val_pooled_dice_at_0.5": epoch_sweep[0.5], "val_pooled_dice": vm["pooled_dice"], "val_mean_dice_fluid_positive": vm["mean_dice_fluid_positive"],
            "val_mean_dice": vm["mean_dice"], "val_pooled_iou": vm["pooled_iou"],
            "val_mean_pred_frac_of_graded": vm["mean_pred_frac_of_graded"], "epoch_time_sec": epoch_time,
        })
        print(
            f"[segment epoch {epoch}/{epochs_total}] train_loss={train_loss:.4f} val_pooled_dice={vm['pooled_dice']:.4f} (t={epoch_t}) "
            f"val_dice_fluid+={vm['mean_dice_fluid_positive']:.4f} pred_frac={vm['mean_pred_frac_of_graded']:.4f} "
            f"{'*best*' if improved else ''} ({epoch_time:.1f}s)", flush=True,
        )
        if epochs_no_improve >= EARLY_STOP_PATIENCE_SEG:
            print(f"[segment] early stopping: no val pooled Dice improvement in {EARLY_STOP_PATIENCE_SEG} epochs")
            break
    last_epoch = epoch

    # Best checkpoint; recompute its val sweep (identical to the selection-time one).
    model.load_state_dict(load_checkpoint(best_ckpt_path, map_location=device)["model_state_dict"])
    val_probs = predict_native(model, val_ds, val_set, device)
    gts, gradeds = [s["gt1"] for s in val_set], [s["graded"] for s in val_set]
    threshold, sweep = threshold_sweep(val_probs, gts, gradeds)
    print(f"[segment] best epoch {best_epoch}; val threshold sweep (pooled Dice): " + ", ".join(f"{t}:{d:.3f}" for t, d in sweep.items()))
    print(f"[segment] chosen threshold (val only): {threshold}")

    summary = {
        "run": "segmentation retrain (smoke)" if args.smoke else "segmentation retrain",
        "encoder_init": str(args.classifier_checkpoint), "encoder_keys_matched": f"{matched}/{total_keys}",
        "split_seed": config.SEED, "best_epoch": best_epoch, "last_epoch": last_epoch,
        "selection": "epoch and threshold jointly: max over threshold grid of val pooled Dice (graded pixels, native 496x768)",
        "threshold_grid": THRESHOLD_GRID, "val_threshold_sweep_pooled_dice": {str(t): d for t, d in sweep.items()},
        "chosen_threshold": threshold,
        "hyperparameters": {"batch_size": BATCH_SIZE_SEG_CPU, "lr": LR_SEG_RETRAIN, "weight_decay": WEIGHT_DECAY_SEG,
                            "bce_pos_weight": BCE_POS_WEIGHT, "dice_loss_weight": DICE_LOSS_WEIGHT,
                            "max_epochs": epochs_total, "early_stop_patience": EARLY_STOP_PATIENCE_SEG, "image_size": IMAGE_SIZE_SEG},
        "val": split_report(val_probs, val_set, threshold),
        "val_at_0.5": split_report(val_probs, val_set, 0.5)["model"],
    }
    rows = per_image_rows("val", val_probs, val_set, threshold)

    if args.smoke:
        print("[segment] --smoke: test split NOT evaluated.")
    else:
        # Single test evaluation -- everything above is frozen at this point.
        test_ds = DukeFluidDataset(test_entries, augment=False)
        test_set = native_eval_set(test_entries)
        test_probs = predict_native(model, test_ds, test_set, device)
        summary["test"] = split_report(test_probs, test_set, threshold)
        summary["test_at_0.5"] = split_report(test_probs, test_set, 0.5)["model"]
        rows += per_image_rows("test", test_probs, test_set, threshold)
        save_examples_figure(args.artifacts_dir / "segmentation_test_examples.png", test_ds, test_probs, test_set, threshold)
    save_examples_figure(args.artifacts_dir / "segmentation_val_examples.png", val_ds, val_probs, val_set, threshold)

    with open(args.artifacts_dir / "segmentation_per_image.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)

    for split in ("val", "test"):
        if split not in summary:
            continue
        r = summary[split]
        print(f"\n[{split}] subjects {r['subjects']}")
        for name in ("model", "baseline_all_foreground", "baseline_all_background", "inter_grader_g2_vs_g1"):
            m = r[name]
            print(f"  {name:26s} pooled Dice {m['pooled_dice']:.4f}  mean Dice {m['mean_dice']:.4f}  "
                  f"mean Dice fluid+ {m['mean_dice_fluid_positive']:.4f}  pooled IoU {m['pooled_iou']:.4f}")
    print(f"\n[segment] summary written: {summary_path}")


if __name__ == "__main__":
    main()
