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
    (manualFluid2) is loaded and available on every sample but not used
    for training -- reserved for a future inter-rater agreement check,
    not implemented here.

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
        --smoke
"""

import argparse
import csv
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
    as parallel lists aligned by index. Grader 2's arrays are included for a
    possible future inter-rater comparison (see module docstring) but unused
    in training below.

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


class DukeFluidDataset:
    """torch.utils.data.Dataset subclassed lazily (see main()) so this module
    imports cleanly without torch installed, matching train.py's pattern of
    keeping torch imports inside functions/__main__ where practical."""

    def __init__(self, entries: List[dict], image_size: int = IMAGE_SIZE_SEG, augment: bool = False):
        self.entries = entries
        self.image_size = image_size
        self.augment = augment
        self._cache = {}  # mat_path -> load_duke_subject() result; ~20MB/subject, 10 subjects fits comfortably

    def _get_subject(self, mat_path: Path) -> dict:
        if mat_path not in self._cache:
            self._cache[mat_path] = load_duke_subject(mat_path)
        return self._cache[mat_path]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        import torch
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        e = self.entries[i]
        sample = self._get_subject(e["mat_path"])
        pos = e["pos"]
        img = sample["images"][pos]
        fluid_mask = binarize_fluid(sample["fluid1"][pos])

        img_rgb, mask = preprocess_pair(img, fluid_mask, self.image_size)

        if self.augment:
            # No vertical flip (C4). Same augmentation family as the classifier's
            # training transform, applied jointly to image+mask via albumentations.
            tfm = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.7),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            tfm = A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        out = tfm(image=img_rgb, mask=mask)
        image_t = out["image"]
        mask_t = out["mask"].float().unsqueeze(0)  # (1, H, W)
        return image_t, mask_t


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
# ---------------------------------------------------------------------------


def _log_epoch_row(log_path: Path, row: dict) -> None:
    write_header = not Path(log_path).exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_seg_epoch(model, loader, dice_loss, bce_loss, device, optimizer=None):
    import torch

    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, n = 0.0, 0
    dices, ious, bdists = [], [], []

    torch.set_grad_enabled(train_mode)
    try:
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            if train_mode:
                optimizer.zero_grad()
            logits = model(images)
            loss = 0.5 * dice_loss(logits, masks) + 0.5 * bce_loss(logits, masks)
            if train_mode:
                loss.backward()
                optimizer.step()

            batch_n = images.size(0)
            total_loss += loss.item() * batch_n
            n += batch_n

            preds = (torch.sigmoid(logits) > 0.5).float().detach().cpu().numpy()
            targets = masks.detach().cpu().numpy()
            for p, t in zip(preds, targets):
                p2d, t2d = p[0], t[0]
                dices.append(dice_coefficient(p2d, t2d))
                ious.append(iou_score(p2d, t2d))
                bdists.append(boundary_distance(p2d, t2d))
    finally:
        torch.set_grad_enabled(True)

    avg_loss = total_loss / max(1, n)
    mean_dice = float(np.mean(dices)) if dices else float("nan")
    mean_iou = float(np.mean(ious)) if ious else float("nan")
    finite_bdists = [d for d in bdists if not np.isnan(d)]
    mean_bdist = float(np.mean(finite_bdists)) if finite_bdists else float("nan")
    return avg_loss, mean_dice, mean_iou, mean_bdist


def main():
    parser = argparse.ArgumentParser(description="Phase 4 (extended, E1): U-Net fluid segmentation, Duke DME dataset.")
    parser.add_argument("--duke-dir", type=Path, required=True, help="Path to the extracted 2015_BOE_Chiu folder (10 Subject_XX.mat files).")
    parser.add_argument("--classifier-checkpoint", type=Path, required=True, help="Phase 2 checkpoint_best.pt to initialise the encoder from.")
    parser.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR / "segmentation")
    parser.add_argument("--verify-only", action="store_true", help="Only generate the rasterization verification figure and exit -- no torch/GPU needed.")
    parser.add_argument("--smoke", action="store_true", help="Train for 1 epoch, log/checkpoint through the real path, to sanity-check before a full Kaggle push.")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(config.SEED)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Mandatory gate, per the brief: verify rasterization BEFORE anything else.
    verify_rasterization(args.duke_dir, args.artifacts_dir)
    if args.verify_only:
        print("[segment] --verify-only: stopping after the rasterization check, as requested.")
        return

    import torch
    from torch.utils.data import DataLoader
    import segmentation_models_pytorch as smp
    from train import save_checkpoint, load_checkpoint

    entries = build_duke_index(args.duke_dir)
    train_entries, val_entries, test_entries = build_splits_duke(entries)

    train_ds = DukeFluidDataset(train_entries, augment=True)
    val_ds = DukeFluidDataset(val_entries, augment=False)
    test_ds = DukeFluidDataset(test_entries, augment=False)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE_SEG, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE_SEG, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE_SEG, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_seg_model().to(device)
    transfer_encoder_weights(model, args.classifier_checkpoint)

    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR_SEG)

    epochs_total = 1 if args.smoke else EPOCHS_SEG
    best_ckpt_path = args.artifacts_dir / "checkpoint_best.pt"
    last_ckpt_path = args.artifacts_dir / "checkpoint_last.pt"
    log_path = args.artifacts_dir / "segmentation_training_log.csv"

    best_val_dice, best_epoch, epochs_no_improve = -1.0, 0, 0
    for epoch in range(1, epochs_total + 1):
        t0 = time.time()
        train_loss, train_dice, train_iou, _ = run_seg_epoch(model, train_loader, dice_loss, bce_loss, device, optimizer=optimizer)
        val_loss, val_dice, val_iou, val_bdist = run_seg_epoch(model, val_loader, dice_loss, bce_loss, device, optimizer=None)
        epoch_time = time.time() - t0

        improved = val_dice > best_val_dice
        if improved:
            best_val_dice, best_epoch, epochs_no_improve = val_dice, epoch, 0
        else:
            epochs_no_improve += 1

        _log_epoch_row(
            log_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_dice": train_dice,
                "train_iou": train_iou,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "val_boundary_dist_px": val_bdist,
                "epoch_time_sec": epoch_time,
            },
        )

        if improved:
            save_checkpoint(
                best_ckpt_path,
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": MODEL_NAME_SEG,
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                },
            )

        save_checkpoint(
            last_ckpt_path,
            {"model_state_dict": model.state_dict(), "model_name": MODEL_NAME_SEG, "epoch": epoch, "best_val_dice": best_val_dice, "best_epoch": best_epoch},
        )

        print(
            f"[segment epoch {epoch}/{epochs_total}] train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f} "
            f"{'*best*' if improved else ''} ({epoch_time:.1f}s)"
        )

        if not args.smoke and epochs_no_improve >= config.EARLY_STOP_PATIENCE:
            print(f"[segment] early stopping: no val_dice improvement in {config.EARLY_STOP_PATIENCE} epochs")
            break

    # Final val + test metrics from the BEST checkpoint, not the last epoch's weights.
    best_ckpt = load_checkpoint(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    _, final_val_dice, final_val_iou, final_val_bdist = run_seg_epoch(model, val_loader, dice_loss, bce_loss, device, optimizer=None)
    _, final_test_dice, final_test_iou, final_test_bdist = run_seg_epoch(model, test_loader, dice_loss, bce_loss, device, optimizer=None)

    verdict = "ship" if final_val_dice >= 0.6 else "future work"

    print("\nPHASE 4 COMPLETE" if verdict == "ship" else "\nPHASE 4 TIMEBOXED")
    print(f"Duke: {len(entries)} scans rasterized, overlay verified: yes")
    print("Encoder transfer: see 'encoder transfer' log line above")
    print(f"Val Dice {final_val_dice:.4f} IoU {final_val_iou:.4f} boundary dist {final_val_bdist:.2f} px | "
          f"Test Dice {final_test_dice:.4f} IoU {final_test_iou:.4f} boundary dist {final_test_bdist:.2f} px")
    print(f"Verdict: {verdict}")
    print("Blocking questions: none")


if __name__ == "__main__":
    main()
