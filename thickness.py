"""
Phase 5 (extended objective E2): relative retinal thickening index.

Confirm Phase 4 artifacts are saved before starting (PROJECT_BRIEF.md Phase 5
preamble) -- Phase 4's sign-off is recorded in phase4-segmentation-signoff.md
before this file was written.

--------------------------------------------------------------------------
What this computes, and on what data
--------------------------------------------------------------------------
Unlike Phase 4 (which trained on the small, separately-licensed Duke DME
dataset), Phase 5 runs directly on the SAME Kermany OCT2017 dataset the
classifier was trained/evaluated on (data/raw/OCT2017/{train,test}) --
there is no Duke involvement here. Per the brief: extract an ILM/RPE
boundary pair per column, take per-column thickness, find the foveal
column, take a windowed mean around it, and normalise by the median of
that same measurement over the Normal class in the TRAINING split (never
micrometres -- C3).

Judgment calls, stated up front (same "explain before coding" convention
as segment.py / explain.py / app.py):

  - Boundary extraction is NOT the Phase 4 U-Net. That model outputs a
    binary fluid mask (classes=1), not the 8-boundary layer coordinates,
    and segment.py's own docstring flags this explicitly: "Retinal-layer
    boundary extraction for Phase 5's thickening index is a separate,
    later concern, and may use a different (non-U-Net) approach." There
    are also no ILM/RPE ground-truth labels for the Kermany dataset at
    all (Duke's manualLayers1/2 only cover its own 110 scans, a different
    dataset). So this uses a classical, unsupervised approach: Otsu-
    threshold the retina tissue away from background, per column, and
    take the topmost/bottommost tissue row as an ILM/RPE PROXY. This is
    the same technique utils.flatten_retina() already uses (and has
    already been running in production since Phase 1) to find the
    retina's lower boundary for flattening -- extended here to also keep
    the upper boundary, which flatten_retina() computes internally but
    doesn't expose. It is a proxy for the true anatomical ILM/RPE
    boundaries, not a validated layer segmentation -- reported as a
    limitation, not hidden.

  - Preprocessing reuses utils.preprocess_image() / PreprocessCache
    exactly as the classifier does (denoise -> flatten -> CLAHE ->
    aspect-preserving letterbox resize), for two reasons: consistency
    (the same pipeline the classifier and its Normal/DME split were built
    on), and correctness (retinal flattening is a per-column RIGID
    vertical shift -- it moves a column's content up/down as a block but
    never stretches it, so per-column thickness in pixels is unchanged by
    flattening; safe to reuse rather than reimplementing an unflattened
    path).

  - All measurement happens in CONTENT-region-local column coordinates
    (i.e. after cropping to content_bbox, excluding the letterbox
    padding) -- per the brief: "window = FOVEAL_WINDOW_FRACTION x retina
    CONTENT width (excluding letterbox padding)". content_bbox comes
    straight from preprocess_image(), which _letterbox_resize()'s own
    docstring already flags as being for exactly this purpose.

  - "Elevated" (the third referral condition's trigger) is read literally
    from the brief's own point 5, "1.0 = typical normal": index > 1.0.
    "Peak thickness sits away from the detected foveal column" is
    operationalised as: the column of maximum thickness across the whole
    content width falls OUTSIDE the same windowed region already used to
    compute the index (ties the two computations together rather than
    inventing an unrelated distance threshold). Both are judgment calls,
    not brief-specified constants -- documented here, not buried in code.

  - "Separation: clear / partial / heavy overlap" (the brief's own sign-
    off template) is read as roc_auc_score(is_DME, index) on the TEST set
    -- the same metric this project already uses everywhere else to
    quantify class separability (Phase 3, Phase 3g). Bucketed: >=0.85
    clear, 0.65-0.85 partial, <0.65 heavy overlap. A judgment call on the
    cutoffs, stated here rather than left implicit.

  - The Normal reference (point 5, "median ... of the Normal class in the
    training split") is computed from build_splits()'s train_df, NOT the
    raw train_dir/NORMAL folder -- this must be the exact same
    patient-grouped train split (minus any C8 test-overlap exclusion) the
    classifier itself trained on, reconstructed by calling build_splits()
    with the same --train-dir/--test-dir the rest of the project uses,
    not re-derived independently.

Usage:
    python thickness.py --train-dir data/raw/OCT2017/train --test-dir data/raw/OCT2017/test --verify-only
    python thickness.py --train-dir data/raw/OCT2017/train --test-dir data/raw/OCT2017/test --smoke
    python thickness.py --train-dir data/raw/OCT2017/train --test-dir data/raw/OCT2017/test
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd

import config
from utils import PreprocessCache, build_splits, load_class_folder_df, set_seed

FOVEAL_WINDOW_FRACTION = config.FOVEAL_WINDOW_FRACTION
ELEVATED_THRESHOLD = 1.0  # index > this triggers the "elevated" half of the third referral condition -- see module docstring

# roc_auc_score buckets for the brief's "clear / partial / heavy overlap" sign-off line -- see module docstring.
SEPARATION_CLEAR_AUC = 0.85
SEPARATION_PARTIAL_AUC = 0.65

# A per-column thickness reading this large relative to the content region's
# own height is anatomically implausible -- see compute_thickness_profile()'s
# docstring for how this was picked and why (a documented judgment call, not
# a value fit to any one image).
MAX_PLAUSIBLE_THICKNESS_FRACTION = 0.5


# ---------------------------------------------------------------------------
# Pure, testable geometry (no file I/O, no torch)
# ---------------------------------------------------------------------------


def _smoothed_lower_boundary_anchor(gray_content: np.ndarray) -> Optional[np.ndarray]:
    """
    Per-column lower-tissue-boundary curve, deliberately identical to
    utils.flatten_retina()'s own internal technique: raw (unblurred) Otsu
    threshold, per-column bottommost foreground row, then a degree-2
    polynomial fit across columns. flatten_retina() has been running in
    production since Phase 1 without complaint, and its robustness comes
    specifically from that polynomial fit smoothing over noisy individual
    columns -- not from the per-column values being clean on their own.
    Reused here (rather than reimplemented differently) as a trustworthy
    RPE-ish anchor for _isolate_retina_mask() to seed from. Returns None if
    coverage is too sparse to fit (same 50%-of-width bar flatten_retina()
    itself uses before falling back).
    """
    _, binary = cv2.threshold(gray_content, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = gray_content.shape
    lower_boundary = np.full(w, np.nan)
    for col in range(w):
        rows = np.nonzero(binary[:, col])[0]
        if rows.size > 0:
            lower_boundary[col] = rows.max()

    valid = ~np.isnan(lower_boundary)
    if valid.sum() < 0.5 * w:
        return None

    xs = np.nonzero(valid)[0]
    ys = lower_boundary[valid]
    coeffs = np.polyfit(xs, ys, deg=2)
    fitted = np.polyval(coeffs, np.arange(w))
    return np.clip(fitted, 0, h - 1)


def _isolate_retina_mask(gray_content: np.ndarray) -> np.ndarray:
    """
    Gaussian-blur, Otsu-threshold, then walk outward from a trusted
    per-column anchor to keep only the contiguous tissue band around it.

    Three earlier versions of this were each caught failing by
    verify_thickness_extraction()'s own visual check before any corpus run
    -- worth recording, since none of the failures are obvious from the
    numbers alone:

    v1: raw Otsu threshold on the sharp image, per-column min/max nonzero
    row as the ILM/RPE proxy. Isolated bright noise specks in the dark
    vitreous cavity above the retina got picked up as "tissue," inflating
    thickness almost everywhere.

    v2: added a morphological opening + largest-connected-component step
    to discard those specks. This exposed that OCT retina isn't uniformly
    bright -- a single global Otsu threshold on the sharp image only ever
    separated the single brightest sub-band (the RPE) from everything
    else, undershooting real retina thickness by roughly 3-5x.

    v3: a strong Gaussian blur before thresholding smeared the brighter
    and duller layers into one smooth envelope, fixing v2's undershoot for
    most images. But picking "the largest bright connected component" as
    "the retina" is a purely brightness-based guess with no anatomical
    grounding, and it failed outright on 2/20 broader sample images: both
    had an unusually noisy/grainy vitreous cavity (visually confirmed
    against the raw source JPEGs -- genuine sensor speckle, not a
    preprocessing artifact) that, after blurring, averaged out to a
    brightness comparable to or exceeding the real retina band's own
    blurred brightness, with no intensity dip between them in some
    columns. Otsu + largest-component then picked the noisy vitreous
    region itself as "tissue" (confirmed by inspecting the raw per-pixel
    binary mask directly, not just the final overlay) -- not a bridging
    problem, an outright wrong-region selection that no amount of
    morphological cleanup of the SAME mask can fix, because for these
    columns the real retina was never above threshold in the first place.

    v4 (this version): stop trusting "largest bright blob = retina"
    unconditionally. Instead, get a trustworthy per-column RPE-ish anchor
    from _smoothed_lower_boundary_anchor() (the same technique
    flatten_retina() already relies on in production), then, in the
    blurred+thresholded+opened binary, take only the contiguous run of
    foreground pixels in that column that actually CONTAINS the anchor row
    -- not the full min/max span of whatever component the anchor happens
    to touch. This is robust to a brighter, disconnected (or even
    same-component-but-not-contiguous-in-this-column) region elsewhere in
    the frame, since that region simply isn't part of the run seeded at
    the anchor. Falls back to v3's largest-connected-component behaviour
    only if the anchor itself can't be computed (sparse Otsu coverage,
    same rare condition flatten_retina() itself bails out on).

    sigmaX/sigmaY were chosen by visual inspection against real Normal and
    DME samples (see verify_thickness_extraction()'s output) -- large
    enough to bridge between retinal layers of different brightness (and
    across small fluid pockets in a DME scan, which should NOT be read as
    "not retina"), not so large it smears the genuine top/bottom envelope
    into the background. This remains a proxy for the true anatomical
    ILM/RPE boundaries, not a validated layer segmentation -- reported as
    a limitation, not hidden (see module docstring), and
    compute_thickness_profile() additionally sanity-checks the resulting
    per-column readings rather than trusting them unconditionally.
    """
    blurred = cv2.GaussianBlur(gray_content, (0, 0), sigmaX=6, sigmaY=10)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    h, w = gray_content.shape
    anchor = _smoothed_lower_boundary_anchor(gray_content)
    if anchor is None:
        # Rare fallback: no trustworthy anchor -- same behaviour as v3.
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
        if n_labels <= 1:
            return opened  # background only -- nothing to isolate
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return np.where(labels == largest_label, 255, 0).astype(np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    for col in range(w):
        row = int(round(anchor[col]))
        col_fg = opened[:, col] > 0
        if not col_fg[row]:
            # Anchor itself isn't foreground in this column (rare) -- seed
            # from the nearest foreground row instead.
            fg_rows = np.nonzero(col_fg)[0]
            if fg_rows.size == 0:
                continue
            row = int(fg_rows[np.argmin(np.abs(fg_rows - row))])
        top = row
        while top - 1 >= 0 and col_fg[top - 1]:
            top -= 1
        bottom = row
        while bottom + 1 < h and col_fg[bottom + 1]:
            bottom += 1
        mask[top : bottom + 1, col] = 255
    return mask


def compute_thickness_profile(rgb_uint8: np.ndarray, content_bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Per-column retina thickness (px) within content_bbox = (y0, y1, x0, x1),
    the non-padded region preprocess_image() returns -- see
    _isolate_retina_mask() for how the tissue region is isolated (the
    ILM/RPE proxy; see module docstring for why this isn't a trained
    layer segmentation).

    Returns an array of length (x1 - x0); NaN in any column with no
    isolated tissue (e.g. a fully-padded or corrupt column), OR where the
    isolated span exceeds MAX_PLAUSIBLE_THICKNESS_FRACTION of the content
    region's own height. That cap was set by inspecting the per-column
    readings on a broader real sample (20 images): genuine columns --
    Normal and DME alike, including a visibly edematous DME scan -- topped
    out at 32% of content height, while the two columns _isolate_retina_mask
    still gets wrong even with the v4 anchor fix (see its docstring) sat at
    69-72%, a >2x gap with nothing in between. 50% is a deliberately
    generous cutoff inside that gap: it exists to catch a mask that is
    unambiguously wrong (would mean "the whole B-scan is retina, with no
    visible vitreous or choroid at all"), not to bound normal biological
    variation or edema severity, and per-column so that one bad stretch in
    an otherwise-good image doesn't discard the image's real signal.
    """
    y0, y1, x0, x1 = content_bbox
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)[y0:y1, x0:x1]
    if gray.size == 0:
        return np.array([])

    mask = _isolate_retina_mask(gray)
    h, w = mask.shape
    thickness = np.full(w, np.nan)
    for col in range(w):
        rows = np.nonzero(mask[:, col])[0]
        if rows.size > 0:
            span = float(rows.max() - rows.min())
            if span <= MAX_PLAUSIBLE_THICKNESS_FRACTION * h:
                thickness[col] = span
    return thickness


def find_foveal_column(thickness: np.ndarray) -> Optional[int]:
    """Foveal column = minimum thickness within the central third of content width (brief point 3)."""
    w = len(thickness)
    if w == 0:
        return None
    third = w // 3
    window = thickness[third : 2 * third]
    if window.size == 0 or np.all(np.isnan(window)):
        return None
    return third + int(np.nanargmin(window))


def windowed_region(content_width: int, center_col: int, window_fraction: float = FOVEAL_WINDOW_FRACTION) -> Tuple[int, int]:
    """[lo, hi) column range: window_fraction * content_width wide, centred on center_col, clipped to content bounds."""
    half = max(1, int(round(window_fraction * content_width / 2)))
    lo = max(0, center_col - half)
    hi = min(content_width, center_col + half + 1)
    return lo, hi


def windowed_mean_thickness(thickness: np.ndarray, center_col: int, window_fraction: float = FOVEAL_WINDOW_FRACTION) -> Tuple[float, Tuple[int, int]]:
    """Brief point 4: windowed mean thickness around the foveal column. Returns (mean_px, (lo, hi)); mean_px is NaN if the window has no valid columns."""
    lo, hi = windowed_region(len(thickness), center_col, window_fraction)
    window = thickness[lo:hi]
    if window.size == 0 or np.all(np.isnan(window)):
        return float("nan"), (lo, hi)
    return float(np.nanmean(window)), (lo, hi)


def find_peak_column(thickness: np.ndarray) -> Optional[int]:
    """Column of maximum thickness across the WHOLE content width (not just the central third) -- used by the third referral condition."""
    if thickness.size == 0 or np.all(np.isnan(thickness)):
        return None
    return int(np.nanargmax(thickness))


def is_centre_involvement_indeterminate(
    index: float, peak_col: Optional[int], window_bounds: Tuple[int, int], elevated_threshold: float = ELEVATED_THRESHOLD
) -> bool:
    """
    Brief's third referral condition: index elevated but peak thickness
    sits away from the foveal window -> "fluid present, centre involvement
    indeterminate -- volumetric imaging recommended." See module docstring
    for why "away from" is read as "outside the windowed region."
    """
    if index is None or np.isnan(index) or peak_col is None:
        return False
    lo, hi = window_bounds
    return index > elevated_threshold and not (lo <= peak_col < hi)


# ---------------------------------------------------------------------------
# Mandatory pre-run verification (brief: show a visual check before
# proceeding -- same convention as segment.py's verify_rasterization())
# ---------------------------------------------------------------------------


def verify_thickness_extraction(sample_paths: list, cache: PreprocessCache, artifacts_dir: Path) -> Path:
    """
    Generates artifacts/thickness_verification_check.png: for a few real
    images, the preprocessed scan with the detected upper/lower boundary
    per column, the foveal column, and the windowed region overlaid. A
    wrong boundary detection is nearly undetectable from an index number
    alone -- this is the same "verify before scaling to thousands of
    images" principle Phase 4 applied to rasterization.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(sample_paths)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, path in zip(axes, sample_paths):
        rgb, content_bbox, _fallback = cache.get_or_compute(path)
        y0, y1, x0, x1 = content_bbox
        thickness = compute_thickness_profile(rgb, content_bbox)
        fcol = find_foveal_column(thickness)
        pcol = find_peak_column(thickness)

        overlay = rgb.copy()
        content_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[y0:y1, x0:x1]
        mask = _isolate_retina_mask(content_gray)
        for col in range(mask.shape[1]):
            rows = np.nonzero(mask[:, col])[0]
            if rows.size > 0:
                overlay[y0 + rows.min(), x0 + col] = (255, 255, 0)  # ILM proxy, yellow
                overlay[y0 + rows.max(), x0 + col] = (0, 255, 255)  # RPE proxy, cyan

        ax.imshow(overlay)
        if fcol is not None:
            ax.axvline(x0 + fcol, color="lime", linewidth=1, label="foveal col")
            _, (lo, hi) = windowed_mean_thickness(thickness, fcol)
            ax.axvspan(x0 + lo, x0 + hi, color="lime", alpha=0.15, label="window")
        if pcol is not None:
            ax.axvline(x0 + pcol, color="red", linewidth=1, linestyle="--", label="peak col")
        ax.set_title(Path(path).name)
        ax.legend(loc="lower right", fontsize=7)
        ax.axis("off")

    out_path = artifacts_dir / "thickness_verification_check.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[thickness] verification figure written: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Per-image and corpus-level pipeline
# ---------------------------------------------------------------------------


def analyze_scan(path, cache: PreprocessCache) -> dict:
    """Full per-image pipeline. 'valid': False (with a reason) if no tissue was detected in the central third at all."""
    rgb, content_bbox, fallback = cache.get_or_compute(path)
    thickness = compute_thickness_profile(rgb, content_bbox)
    fcol = find_foveal_column(thickness)
    if fcol is None:
        return {"path": str(path), "valid": False, "reason": "no retina tissue detected in central third", "flatten_fallback": fallback}

    windowed_px, window_bounds = windowed_mean_thickness(thickness, fcol)
    pcol = find_peak_column(thickness)
    return {
        "path": str(path),
        "valid": not np.isnan(windowed_px),
        "reason": None if not np.isnan(windowed_px) else "windowed region had no valid columns",
        "windowed_thickness_px": windowed_px,
        "foveal_col": fcol,
        "peak_col": pcol,
        "window_bounds": window_bounds,
        "content_width": len(thickness),
        "flatten_fallback": fallback,
    }


def compute_normal_reference(normal_train_df: pd.DataFrame, cache: PreprocessCache) -> Tuple[float, int, int]:
    """Brief point 5: median windowed thickness (px) over the Normal TRAINING split. Returns (median_px, n_valid, n_total)."""
    vals = []
    for path in normal_train_df["path"]:
        r = analyze_scan(path, cache)
        if r["valid"]:
            vals.append(r["windowed_thickness_px"])
    if not vals:
        raise RuntimeError(
            "Normal reference computation found 0 valid images -- cannot normalise the index without a "
            "reference (C6: refusing to fabricate one). Check --train-dir and the boundary-detection logic."
        )
    return float(np.median(vals)), len(vals), len(normal_train_df)


def compute_indices_for_df(df: pd.DataFrame, cache: PreprocessCache, reference_px: float, elevated_threshold: float = ELEVATED_THRESHOLD) -> pd.DataFrame:
    """Per-row thickening index + third-condition flag for every image in df (columns: path, label, patient)."""
    rows = []
    for _, row in df.iterrows():
        r = analyze_scan(row["path"], cache)
        index = r["windowed_thickness_px"] / reference_px if r["valid"] else float("nan")
        indeterminate = (
            is_centre_involvement_indeterminate(index, r.get("peak_col"), r.get("window_bounds", (0, 0)), elevated_threshold)
            if r["valid"]
            else False
        )
        rows.append({"path": row["path"], "label": row["label"], "valid": r["valid"], "index": index, "centre_involvement_indeterminate": indeterminate})
    return pd.DataFrame(rows)


def summarize_separation(results_df: pd.DataFrame) -> dict:
    """DME vs Normal index distributions + roc_auc_score-based separation bucket (see module docstring)."""
    from sklearn.metrics import roc_auc_score

    valid = results_df[results_df["valid"]]
    normal_idx = valid[valid["label"] == "Normal"]["index"]
    dme_idx = valid[valid["label"] == "DME"]["index"]

    summary = {
        "n_valid": int(len(valid)),
        "n_total": int(len(results_df)),
        "normal_mean": float(normal_idx.mean()) if len(normal_idx) else float("nan"),
        "normal_std": float(normal_idx.std()) if len(normal_idx) else float("nan"),
        "dme_mean": float(dme_idx.mean()) if len(dme_idx) else float("nan"),
        "dme_std": float(dme_idx.std()) if len(dme_idx) else float("nan"),
        "centre_involvement_indeterminate_frac": float(valid["centre_involvement_indeterminate"].mean()) if len(valid) else float("nan"),
    }

    if len(normal_idx) and len(dme_idx):
        y = (valid["label"] == "DME").astype(int)
        auc = roc_auc_score(y, valid["index"])
        summary["separation_auc"] = float(auc)
        if auc >= SEPARATION_CLEAR_AUC:
            summary["separation"] = "clear"
        elif auc >= SEPARATION_PARTIAL_AUC:
            summary["separation"] = "partial"
        else:
            summary["separation"] = "heavy overlap"
    else:
        summary["separation_auc"] = float("nan")
        summary["separation"] = "undetermined (one class had 0 valid images)"

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Phase 5 (extended, E2): relative retinal thickening index.")
    parser.add_argument("--train-dir", type=Path, required=True, help="Same --train-dir used for training, to reconstruct the identical train split (C1) for the Normal reference.")
    parser.add_argument("--test-dir", type=Path, required=True, help="Official 484-image test set -- where DME vs Normal separation is reported.")
    parser.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR / "thickness")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Defaults to <artifacts-dir>/preprocess_cache.")
    parser.add_argument("--verify-only", action="store_true", help="Only generate the verification figure and exit -- no corpus run.")
    parser.add_argument("--smoke", action="store_true", help="Run on a small patient-grouped subset (~200 images) to sanity-check before the full run.")
    parser.add_argument("--elevated-threshold", type=float, default=ELEVATED_THRESHOLD)
    args = parser.parse_args()

    set_seed(config.SEED)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.artifacts_dir / "preprocess_cache")
    cache = PreprocessCache(cache_dir=cache_dir, image_size=config.IMAGE_SIZE, flatten=config.RETINAL_FLATTENING_ENABLED)

    result = build_splits(args.train_dir, test_dir=args.test_dir)
    train_df, test_df = result.train_df, load_class_folder_df(args.test_dir)

    sample_paths = [
        train_df[train_df["label"] == "Normal"]["path"].iloc[0],
        train_df[train_df["label"] == "DME"]["path"].iloc[0],
    ]
    verify_thickness_extraction(sample_paths, cache, args.artifacts_dir)
    if args.verify_only:
        print("[thickness] --verify-only: stopping after the verification check, as requested.")
        return

    normal_train_df = train_df[train_df["label"] == "Normal"]
    eval_test_df = test_df

    if args.smoke:
        rng = np.random.RandomState(config.SEED)
        normal_patients = normal_train_df["patient"].unique()
        rng.shuffle(normal_patients)
        picked, count = [], 0
        for p in normal_patients:
            if count >= 150:
                break
            picked.append(p)
            count += int((normal_train_df["patient"] == p).sum())
        normal_train_df = normal_train_df[normal_train_df["patient"].isin(picked)].reset_index(drop=True)
        eval_test_df = test_df.groupby("label", group_keys=False).head(20).reset_index(drop=True)
        print(f"[thickness] --smoke: reference from {len(normal_train_df)} Normal train images, evaluating {len(eval_test_df)} test images.")

    t0 = time.time()
    reference_px, n_valid_ref, n_total_ref = compute_normal_reference(normal_train_df, cache)
    print(f"[thickness] Normal reference (median windowed thickness): {reference_px:.2f}px, from {n_valid_ref}/{n_total_ref} valid train images ({time.time() - t0:.1f}s)")

    t0 = time.time()
    results_df = compute_indices_for_df(eval_test_df, cache, reference_px, args.elevated_threshold)
    print(f"[thickness] computed indices for {len(results_df)} test images ({time.time() - t0:.1f}s)")

    summary = summarize_separation(results_df)
    summary["reference_px"] = reference_px
    summary["reference_n_valid"] = n_valid_ref
    summary["reference_n_total"] = n_total_ref
    summary["elevated_threshold"] = args.elevated_threshold
    summary["smoke"] = args.smoke

    results_path = args.artifacts_dir / "thickness_results.csv"
    results_df.to_csv(results_path, index=False)
    summary_path = args.artifacts_dir / "thickness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[thickness] wrote {results_path} and {summary_path}")

    _plot_distributions(results_df, args.artifacts_dir)

    blocking_questions = []
    if summary["n_valid"] < 0.9 * summary["n_total"]:
        blocking_questions.append(
            f"only {summary['n_valid']}/{summary['n_total']} test images produced a valid index -- "
            "boundary detection may be failing on a meaningful fraction of scans."
        )

    print()
    print("PHASE 5 COMPLETE" if not args.smoke else "PHASE 5 SMOKE RUN (not a real result)")
    print(f"Normal median reference: {summary['reference_px']:.2f} px")
    print(f"Index — Normal: {summary['normal_mean']:.3f} ± {summary['normal_std']:.3f} | DME: {summary['dme_mean']:.3f} ± {summary['dme_std']:.3f}")
    print(f"Separation: {summary['separation']} (AUC {summary['separation_auc']:.3f})" if not np.isnan(summary["separation_auc"]) else f"Separation: {summary['separation']}")
    print(f"Blocking questions: {'; '.join(blocking_questions) if blocking_questions else 'none'}")


def _plot_distributions(results_df: pd.DataFrame, artifacts_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = results_df[results_df["valid"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, color in (("Normal", "tab:green"), ("DME", "tab:red")):
        vals = valid[valid["label"] == label]["index"]
        if len(vals):
            ax.hist(vals, bins=30, alpha=0.5, label=f"{label} (n={len(vals)})", color=color)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="reference (1.0)")
    ax.set_xlabel("Relative thickening index")
    ax.set_ylabel("Count")
    ax.set_title("Phase 5: thickening index distribution, test set")
    ax.legend()
    out_path = artifacts_dir / "thickness_distribution.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[thickness] distribution figure written: {out_path}")


if __name__ == "__main__":
    main()
