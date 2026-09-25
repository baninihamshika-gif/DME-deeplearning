"""
Phase 6: Gradio clinical triage interface for DME screening.

Serves the Phase 2/3 primary checkpoint (tf_efficientnet_b3, signed off in
Phase 3 -- see the project's phase3-ablation-and-signoff.md) behind a
single-screen triage UI. Per PROJECT_BRIEF.md Phase 6: ground the aesthetic
in OCT reporting itself (printed-report white, grayscale scan as anchor,
two semantic colours used only on state indicators), load the model once
at startup, and read the threshold/temperature from the checkpoint --
never hardcode them.

Design decisions made here, stated up front (same "explain before coding"
convention evaluate.py/explain.py follow -- see e.g. explain.py's
RANDOMIZATION_CORRELATION_THRESHOLD docstring for the same pattern):

  - The brief specifies three mutually exclusive referral states but the
    model outputs one continuous calibrated probability. "Below confidence
    threshold -- refer for manual review" is triggered for probabilities
    that land just under the tuned threshold (itself tuned for
    config.SENSITIVITY_TARGET = 0.97, see evaluate.py sweep_threshold) --
    i.e. cases the tuned operating point called Normal by only a small
    margin. LOW_CONFIDENCE_MARGIN below is that margin, in probability
    space. It is a judgment call, not fitted on any data or specified in
    the brief -- easy to change, always disclosed here rather than left
    implicit in the code.

  - Grad-CAM (reused directly from explain.py's generate_cam) targets
    model.conv_head, which exists on the EfficientNet family this
    project's primary checkpoint uses, but not on every architecture this
    codebase can train (e.g. resnet50, Phase 3g baseline, has no
    conv_head). This app is built to serve the signed-off primary
    checkpoint; if a checkpoint without conv_head is loaded, Grad-CAM is
    disabled with an explicit UI message instead of crashing or faking a
    heatmap.

  - At deployment the true label is unknown (unlike explain.py's offline
    grid, which explains the TRUE class because ground truth is known from
    the test set). Here Grad-CAM explains the PREDICTED class -- "what did
    the model attend to for the class it called."

  - Phase 4 (segmentation / fluid mask) and Phase 5 (thickening index,
    centre involvement) ARE wired in as of this revision, run through
    the exact same real, signed-off pipelines as segment.py/thickness.py
    (segment.build_seg_model() + the real trained checkpoint; the
    thickness.py pure functions directly, with the Normal reference read
    from the real corpus run's thickness_summary.json rather than
    recomputed live -- recomputing it here would mean re-running the
    ~2-hour full-corpus pass on every app launch). Neither result is
    clinically reliable on its own -- E1's real test Dice is 0.0090 (see
    phase4-segmentation-signoff.md: verdict "future work"), E2's real
    separation is AUC 0.601 ("heavy overlap", phase5-thickening-signoff.md)
    -- so both are shown with an explicit, un-hideable disclaimer next to
    the number/image, never presented as equivalent in confidence to the
    primary classifier's output. If either checkpoint/reference file is
    missing, that section falls back to the original "not available"
    placeholder rather than crashing or fabricating a result
    (PROJECT_BRIEF.md C7 / master-prompt Section 11).

Usage:
    python app.py
    python app.py --checkpoint artifacts/kaggle/<run>/artifacts/checkpoint_best.pt
"""

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import gradio as gr

import config
from evaluate import _softmax_positive
from train import MODEL_NAME, _eval_transform, build_model, load_checkpoint
from utils import preprocess_image
from segment import IMAGE_SIZE_SEG, build_seg_model, preprocess_pair as seg_preprocess_pair
from thickness import (
    ELEVATED_THRESHOLD,
    _isolate_retina_mask,
    compute_thickness_profile,
    find_foveal_column,
    find_peak_column,
    is_centre_involvement_indeterminate,
    windowed_mean_thickness,
)

# ---------------------------------------------------------------------------
# Judgment-call constants (see module docstring)
# ---------------------------------------------------------------------------
LOW_CONFIDENCE_MARGIN = 0.10
SEGMENTATION_FLUID_THRESHOLD = 0.5  # matches segment.py's own run_seg_epoch (torch.sigmoid(logits) > 0.5)

# Real distribution stats from the actual full held-out test-set run (not
# recomputed here -- read-only display constants, sourced verbatim from
# phase5-thickening-signoff.md / report/progress_report.md §3.8). Used only
# to give the thickening-index readout visual context ("is this scan's real
# value unusual relative to the real reported distributions"); never used to
# derive or alter the index itself, which always comes from this scan's own
# real boundary-detection run (see run_thickness).
THICKNESS_NORMAL_MEAN = 0.906
THICKNESS_NORMAL_STD = 0.419
THICKNESS_DME_MEAN = 1.049
THICKNESS_DME_STD = 0.458
THICKNESS_DIST_SCALE_MAX = 2.2  # covers DME mean + 2*std (1.965) with headroom

DEFAULT_CHECKPOINT = config.ARTIFACTS_DIR / "kaggle" / "20260917T035646Z" / "artifacts" / "checkpoint_best.pt"
# Real trained E1/E2 artifacts (see phase4-segmentation-signoff.md / phase5-thickening-signoff.md).
# Both are optional -- if a file is missing, the corresponding UI section degrades to the
# original "not available" placeholder rather than crashing or inventing a result.
DEFAULT_SEGMENTATION_CHECKPOINT = (
    config.ARTIFACTS_DIR / "kaggle" / "20260923T181925Z" / "artifacts" / "segmentation" / "checkpoint_best.pt"
)
DEFAULT_THICKNESS_SUMMARY = config.ARTIFACTS_DIR / "thickness" / "thickness_summary.json"

# ---------------------------------------------------------------------------
# Copy (verbatim per PROJECT_BRIEF.md Phase 6 -- "diagnosis" appears nowhere)
# ---------------------------------------------------------------------------
EMPTY_STATE_TEXT = "Upload an OCT B-scan to begin."
ERROR_STATE_TEXT = "That file isn't a readable image. Upload a JPEG or PNG B-scan."
FOOTER_TEXT = "Screening aid for triage. Not a diagnostic device. All findings require ophthalmologist review."

VIEW_CHOICES = ["original", "preprocessed", "heatmap", "fluid mask", "thickness map"]


# ---------------------------------------------------------------------------
# Model loading -- once at startup, threshold/temperature read from the
# checkpoint, never hardcoded.
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: Path):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[app] loading checkpoint: {checkpoint_path}")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Pass --checkpoint pointing at a checkpoint_best.pt that evaluate.py has already "
            "run on (so it carries a real threshold/temperature, not None)."
        )
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    model_name = ckpt.get("model_name", MODEL_NAME)
    model = build_model(model_name=model_name).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    threshold = ckpt.get("threshold")
    temperature = ckpt.get("temperature")
    if threshold is None or temperature is None:
        raise RuntimeError(
            f"{checkpoint_path} has no threshold/temperature (Phase 3c/3e not yet run on it). "
            "Refusing to invent a default operating point -- run evaluate.py on this checkpoint "
            "first, then relaunch the app."
        )

    gradcam_available = hasattr(model, "conv_head")
    print(
        f"[app] model={model_name} device={device} threshold={threshold:.4f} "
        f"temperature={temperature:.4f} gradcam_available={gradcam_available}"
    )
    return {
        "model": model,
        "device": device,
        "threshold": float(threshold),
        "temperature": float(temperature),
        "model_name": model_name,
        "gradcam_available": gradcam_available,
    }


def load_extended_models(
    device, segmentation_checkpoint: Path = DEFAULT_SEGMENTATION_CHECKPOINT, thickness_summary: Path = DEFAULT_THICKNESS_SUMMARY
) -> dict:
    """
    Loads the real E1 (segmentation) checkpoint and the real E2 (thickness)
    Normal reference, both optional -- a missing file degrades that section
    to the original "not available" placeholder (never a crash, never a
    fabricated result; see module docstring).

    The E2 reference is READ from the real full-corpus run's
    thickness_summary.json rather than recomputed here: that run took
    ~2 hours over 18,364 images (see phase5-thickening-signoff.md's compute
    cost section) and reference_px doesn't change per-scan, so recomputing
    it on every app launch (or worse, every upload) would be wasteful and
    would not produce a different number.
    """
    import torch

    segmentation_available = False
    seg_model = None
    # segmentation_status distinguishes *why* E1 is unavailable, since "not found"
    # and "found but failed to load" need different fixes (missing artifact vs a
    # real code/architecture bug) -- see render_extended_html, which surfaces
    # segmentation_status_detail instead of a single generic placeholder string.
    segmentation_status = "missing"
    segmentation_status_detail = None
    segmentation_checkpoint = Path(segmentation_checkpoint)
    if segmentation_checkpoint.exists():
        try:
            seg_ckpt = load_checkpoint(segmentation_checkpoint, map_location=device)
            seg_model = build_seg_model().to(device)
            seg_model.load_state_dict(seg_ckpt["model_state_dict"])
            seg_model.eval()
            segmentation_available = True
            segmentation_status = "ok"
            print(f"[app] E1 segmentation checkpoint loaded: {segmentation_checkpoint}")
        except Exception as exc:
            segmentation_status = "load_error"
            segmentation_status_detail = str(exc)
            print(f"[app] E1 segmentation checkpoint failed to load ({exc}) -- section will show as not available")
    else:
        segmentation_status_detail = str(segmentation_checkpoint)
        print(f"[app] E1 segmentation checkpoint not found at {segmentation_checkpoint} -- section will show as not available")

    thickness_reference_px = None
    thickness_status = "missing"
    thickness_status_detail = None
    thickness_summary = Path(thickness_summary)
    if thickness_summary.exists():
        try:
            summary = json.loads(thickness_summary.read_text())
            thickness_reference_px = float(summary["reference_px"])
            thickness_status = "ok"
            print(f"[app] E2 thickness reference loaded: {thickness_reference_px:.2f}px (from {thickness_summary})")
        except Exception as exc:
            thickness_status = "load_error"
            thickness_status_detail = str(exc)
            print(f"[app] E2 thickness summary failed to load ({exc}) -- section will show as not available")
    else:
        thickness_status_detail = str(thickness_summary)
        print(f"[app] E2 thickness summary not found at {thickness_summary} -- section will show as not available")

    return {
        "seg_model": seg_model,
        "segmentation_available": segmentation_available,
        "segmentation_status": segmentation_status,
        "segmentation_status_detail": segmentation_status_detail,
        "thickness_reference_px": thickness_reference_px,
        "thickness_available": thickness_reference_px is not None,
        "thickness_status": thickness_status,
        "thickness_status_detail": thickness_status_detail,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def draw_fluid_overlay_disclaimed(image_gray: np.ndarray, fluid_mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    A deliberately less-confident-looking variant of segment.draw_fluid_overlay,
    used only for this live interface. segment.py's own version stays untouched
    -- it also produces the already-signed-off offline pipeline figures, and
    changing its visual style would retroactively change what phase4-segmentation-
    signoff.md's figures show.

    This model's real, accepted test Dice is 0.0090 (see phase4-segmentation-
    signoff.md) -- essentially unusable. A solid, saturated red fill reads as a
    confident clinical finding regardless of the text disclaimer next to it, which
    visually contradicts what the number says. A 45-degree hatch pattern (instead
    of a solid fill) makes the overlay itself look provisional/flagged rather than
    diagnostic, so the honesty lives in the pixels, not only in the caption.
    """
    rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
    h, w = image_gray.shape[:2]
    ys, xs = np.indices((h, w))
    hatch = ((xs + ys) % 6) < 2  # 45-degree stripes, ~2px wide, 6px period
    red = np.zeros_like(rgb)
    red[:, :, 0] = 255
    mask3 = np.repeat((fluid_mask.astype(bool) & hatch)[:, :, None], 3, axis=2)
    out = rgb.copy()
    out[mask3] = (1 - alpha) * rgb[mask3] + alpha * red[mask3]
    return out.astype(np.uint8)


def describe_heatmap_location(heatmap: Optional[np.ndarray], content_bbox) -> Optional[str]:
    """
    A plain-language pointer to where the real Grad-CAM activation concentrates,
    from the intensity-weighted centroid of the actual heatmap pixels within the
    retina content region (content_bbox excludes letterbox padding, which carries
    no signal and would just pull the centroid toward the image centre).

    Deliberately geometric only ("upper third", "left region") -- never an
    anatomical/clinical claim such as nasal/temporal, since eye laterality and
    scan direction aren't recoverable from a single 2D B-scan and this pipeline
    doesn't have either. Returns None (never a fabricated location) if there's no
    heatmap, no content region, or the heatmap has no positive signal to weight.
    """
    if heatmap is None or content_bbox is None:
        return None
    y0, y1, x0, x1 = content_bbox
    region = np.asarray(heatmap)[y0:y1, x0:x1]
    if region.size == 0:
        return None
    region = np.clip(region, 0, None)
    total = float(region.sum())
    if total <= 0:
        return None
    ys, xs = np.indices(region.shape)
    cy = float((ys * region).sum() / total)
    cx = float((xs * region).sum() / total)
    h, w = region.shape
    col_frac = cx / max(w - 1, 1)
    row_frac = cy / max(h - 1, 1)
    col_label = "left" if col_frac < 0.4 else "right" if col_frac > 0.6 else "central"
    row_label = "upper" if row_frac < 0.4 else "lower" if row_frac > 0.6 else "mid"
    if col_label == "central" and row_label == "mid":
        return "centred on the middle of the scan"
    if col_label == "central":
        return f"concentrated in the {row_label} third of the scan, centred horizontally"
    if row_label == "mid":
        return f"concentrated toward the {col_label} third of the scan, centred vertically"
    return f"concentrated in the {row_label}-{col_label} region of the scan"


def render_thickness_distribution_html(index: float) -> str:
    """
    Visual context for the real thickening index: where it falls relative to the
    real reported Normal/DME distributions (THICKNESS_* constants above, sourced
    from phase5-thickening-signoff.md -- not recomputed, and this scan's own index
    is untouched by this function, purely a display overlay on top of it).

    Uses only --graphite at two opacities plus --ink for the marker, never
    --signal-high/--signal-none -- those two colours are reserved for referral
    state per PROJECT_BRIEF.md's Phase 6 design direction, and this band chart
    is not a state indicator.
    """
    scale_max = THICKNESS_DIST_SCALE_MAX

    def pct(value: float) -> float:
        return max(0.0, min(1.0, value / scale_max)) * 100

    normal_lo, normal_hi = THICKNESS_NORMAL_MEAN - THICKNESS_NORMAL_STD, THICKNESS_NORMAL_MEAN + THICKNESS_NORMAL_STD
    dme_lo, dme_hi = THICKNESS_DME_MEAN - THICKNESS_DME_STD, THICKNESS_DME_MEAN + THICKNESS_DME_STD
    marker_pct = pct(index)
    off_scale = index > scale_max
    return f'''
    <div class="thickness-dist">
      <div class="thickness-dist-row">
        <span class="thickness-dist-band-label">Normal</span>
        <div class="thickness-dist-track">
          <div class="thickness-dist-band normal" style="left:{pct(normal_lo):.2f}%; width:{pct(normal_hi) - pct(normal_lo):.2f}%"></div>
        </div>
      </div>
      <div class="thickness-dist-row">
        <span class="thickness-dist-band-label">DME</span>
        <div class="thickness-dist-track">
          <div class="thickness-dist-band dme" style="left:{pct(dme_lo):.2f}%; width:{pct(dme_hi) - pct(dme_lo):.2f}%"></div>
          <div class="thickness-dist-marker" style="left:{min(marker_pct, 100):.2f}%" title="This scan: {index:.2f}"></div>
        </div>
      </div>
      <div class="thickness-dist-caption">
        This scan ({index:.2f}) against the real reported held-out test-set bands (mean ± 1 SD).
        {"Off the shown scale." if off_scale else ""}
        Heavy overlap between bands is the real, accepted result (AUC 0.601) -- context, not a verdict.
      </div>
    </div>
    '''


def run_segmentation(ctx: dict, gray: np.ndarray) -> Optional[dict]:
    """
    E1 inference for one scan, through the real trained checkpoint. gray:
    single-channel uint8 (or convertible) image, same convention segment.py
    itself uses for a Duke B-scan. Returns None if the segmentation model
    isn't available (see load_extended_models) -- never a fabricated mask.
    """
    if not ctx.get("segmentation_available"):
        return None
    import torch

    dummy_mask = np.zeros_like(gray, dtype=np.uint8)
    seg_rgb, _ = seg_preprocess_pair(gray, dummy_mask, IMAGE_SIZE_SEG)
    gray_padded = seg_rgb[:, :, 0]  # preprocess_pair builds seg_rgb via GRAY2RGB -- R=G=B=the padded grayscale image

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    tfm = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
    tensor = tfm(image=seg_rgb)["image"].unsqueeze(0).to(ctx["device"])

    try:
        with torch.no_grad():
            logits = ctx["seg_model"](tensor)
            prob_mask = torch.sigmoid(logits)[0, 0].cpu().numpy()
    except Exception as exc:  # surfaced via None -> UI "not available", never a crash
        print(f"[app] E1 segmentation inference failed: {exc}")
        return None

    fluid_mask = (prob_mask > SEGMENTATION_FLUID_THRESHOLD).astype(np.uint8)
    overlay = draw_fluid_overlay_disclaimed(gray_padded, fluid_mask)
    fluid_area_frac = float(fluid_mask.mean())
    return {"overlay": overlay, "fluid_area_frac": fluid_area_frac}


def run_thickness(ctx: dict, preprocessed_rgb: np.ndarray, content_bbox) -> Optional[dict]:
    """
    E2 inference for one scan, using the real Normal reference from the full
    corpus run (see load_extended_models). Operates directly on the
    classifier's own preprocessed_rgb/content_bbox -- per thickness.py's
    module docstring, this is the same preprocessing pipeline E2 was
    designed and validated against. Returns {"valid": False, "reason": ...}
    (never a fabricated index) if boundary detection or the reference is
    unavailable for this scan.
    """
    if not ctx.get("thickness_available"):
        return None

    thickness = compute_thickness_profile(preprocessed_rgb, content_bbox)
    fcol = find_foveal_column(thickness)
    if fcol is None:
        return {"valid": False, "reason": "no retina tissue detected in the central third of this scan"}

    windowed_px, window_bounds = windowed_mean_thickness(thickness, fcol)
    if np.isnan(windowed_px):
        return {"valid": False, "reason": "windowed region around the foveal column had no valid columns"}

    pcol = find_peak_column(thickness)
    reference_px = ctx["thickness_reference_px"]
    index = windowed_px / reference_px
    indeterminate = is_centre_involvement_indeterminate(index, pcol, window_bounds, ELEVATED_THRESHOLD)

    overlay = preprocessed_rgb.copy()
    y0, y1, x0, x1 = content_bbox
    content_gray = cv2.cvtColor(preprocessed_rgb, cv2.COLOR_RGB2GRAY)[y0:y1, x0:x1]
    mask = _isolate_retina_mask(content_gray)
    for col in range(mask.shape[1]):
        rows = np.nonzero(mask[:, col])[0]
        if rows.size > 0:
            overlay[y0 + rows.min(), x0 + col] = (255, 255, 0)  # ILM proxy, yellow
            overlay[y0 + rows.max(), x0 + col] = (0, 255, 255)  # RPE proxy, cyan
    lo, hi = window_bounds
    cv2.line(overlay, (x0 + fcol, y0), (x0 + fcol, y1 - 1), (0, 255, 0), 1)
    cv2.rectangle(overlay, (x0 + lo, y0), (x0 + hi - 1, y1 - 1), (0, 255, 0), 1)
    if pcol is not None:
        cv2.line(overlay, (x0 + pcol, y0), (x0 + pcol, y1 - 1), (255, 0, 0), 1)

    return {
        "valid": True,
        "index": index,
        "windowed_px": windowed_px,
        "reference_px": reference_px,
        "indeterminate": indeterminate,
        "overlay": overlay,
    }


def run_screening(ctx: dict, bgr: np.ndarray) -> dict:
    """
    ctx: the dict returned by load_model() merged with load_extended_models().
    bgr: an image array as cv2.imread returns it (BGR if 3-channel), matching
    the convention every other script in this repo uses -- preprocess_image()
    assumes the same.
    """
    import torch

    preprocessed_rgb, content_bbox, fallback_triggered = preprocess_image(
        bgr, image_size=config.IMAGE_SIZE, flatten=config.RETINAL_FLATTENING_ENABLED
    )
    original_rgb = (
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    )

    transform = _eval_transform()
    tensor = transform(image=preprocessed_rgb)["image"].unsqueeze(0).to(ctx["device"])

    model = ctx["model"]
    with torch.no_grad():
        logits = model(tensor).cpu().numpy()

    # Calibrated probability vs the tuned threshold -- this is what would actually
    # ship (same convention as evaluate.py's "Test @ tuned" block; see its module
    # docstring for why calibration happens before thresholding, not after).
    prob = float(_softmax_positive(logits / ctx["temperature"])[0])
    prediction = "DME" if prob >= ctx["threshold"] else "Normal"

    heatmap = None
    if ctx["gradcam_available"]:
        from explain import generate_cam

        target_class = 1 if prediction == "DME" else 0
        try:
            heatmap = generate_cam(model, tensor, target_class)
        except Exception as exc:  # surfaced via heatmap=None -> UI message, never a crash
            print(f"[app] Grad-CAM generation failed: {exc}")
            heatmap = None

    heatmap_location = describe_heatmap_location(heatmap, content_bbox)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    # Start from the startup-time status (missing/load_error/ok, set once in
    # load_extended_models) and only override it if THIS scan's inference call
    # itself throws -- that's a third, distinct failure mode (a real artifact
    # loaded fine, but this particular image broke it) and needs its own label
    # rather than collapsing back into "no checkpoint found".
    segmentation_status = ctx.get("segmentation_status", "missing")
    segmentation_status_detail = ctx.get("segmentation_status_detail")
    try:
        segmentation = run_segmentation(ctx, gray)
    except Exception as exc:  # surfaced via segmentation=None -> UI "not available", never a crash
        print(f"[app] E1 segmentation failed: {exc}")
        segmentation = None
        segmentation_status = "runtime_error"
        segmentation_status_detail = str(exc)

    thickness_status = ctx.get("thickness_status", "missing")
    thickness_status_detail = ctx.get("thickness_status_detail")
    try:
        thickness = run_thickness(ctx, preprocessed_rgb, content_bbox)
    except Exception as exc:  # surfaced via thickness=None -> UI "not available", never a crash
        print(f"[app] E2 thickness failed: {exc}")
        thickness = None
        thickness_status = "runtime_error"
        thickness_status_detail = str(exc)

    return {
        "original_rgb": original_rgb,
        "preprocessed_rgb": preprocessed_rgb,
        "heatmap": heatmap,
        "heatmap_location": heatmap_location,
        "probability": prob,
        "prediction": prediction,
        "fallback_triggered": fallback_triggered,
        "segmentation": segmentation,
        "segmentation_status": segmentation_status,
        "segmentation_status_detail": segmentation_status_detail,
        "thickness": thickness,
        "thickness_status": thickness_status,
        "thickness_status_detail": thickness_status_detail,
    }


def render_view(result: dict, view: str, opacity: float):
    """Returns the numpy image to display for the selected view, or None (fluid mask -- not built)."""
    if result is None:
        return None
    if view == "original":
        return result["original_rgb"]
    if view == "preprocessed":
        return result["preprocessed_rgb"]
    if view == "heatmap":
        if result["heatmap"] is None:
            return result["preprocessed_rgb"]
        from pytorch_grad_cam.utils.image import show_cam_on_image

        # opacity: 0 = pure preprocessed image, 100 = pure heatmap. image_weight is
        # pytorch_grad_cam's own blend parameter (weight of the ORIGINAL image), so invert.
        image_weight = 1.0 - (float(opacity) / 100.0)
        return show_cam_on_image(
            result["preprocessed_rgb"].astype(np.float32) / 255.0,
            result["heatmap"],
            use_rgb=True,
            image_weight=image_weight,
        )
    if view == "fluid mask":
        seg = result.get("segmentation")
        return seg["overlay"] if seg is not None else None
    if view == "thickness map":
        thk = result.get("thickness")
        return thk["overlay"] if (thk is not None and thk.get("valid")) else None
    return result["original_rgb"]


# ---------------------------------------------------------------------------
# HTML rendering -- the confidence scale is a manually-rendered gr.HTML(),
# per the brief's Implementation section.
# ---------------------------------------------------------------------------


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _extended_unavailable_reason(kind: str, result: dict) -> str:
    """
    Builds the specific "why is this unavailable" message for an E1/E2 row,
    from the status/detail pair run_screening() attaches to `result` (sourced
    from load_extended_models()'s startup check at import, or overridden by a
    per-scan exception -- see run_screening). Distinguishes three genuinely
    different failure modes that used to collapse into one generic string:
    the artifact file was never found at startup, it was found but the load
    itself raised, or startup succeeded but this particular scan's inference
    raised. Each needs a different fix, so collapsing them hid the real
    problem behind a placeholder that always read "no checkpoint found" --
    the same fabrication risk C6 warns about, just applied to diagnosability
    instead of a number.
    """
    status = result.get(f"{kind}_status", "missing")
    detail = result.get(f"{kind}_status_detail")
    label = "checkpoint" if kind == "segmentation" else "reference file"
    if status == "load_error":
        msg = (detail or "unknown error")[:160]
        return f"not available ({label} found but failed to load: {msg})"
    if status == "runtime_error":
        msg = (detail or "unknown error")[:160]
        return f"not available for this scan (inference error: {msg})"
    return f"not available (no {label} found)"


def render_header_html(scan_id: Optional[str]) -> str:
    today = dt.date.today()
    date_str = f"{today.day} {today.strftime('%b %Y')}"
    meta = "Upload a scan to begin" if scan_id is None else f"Scan {scan_id}&nbsp;&nbsp;·&nbsp;&nbsp;{date_str}"
    return f'''
    <div id="app-header">
      <span class="title">DME screening</span>
      <span class="meta">{meta}</span>
    </div>
    '''


def render_label_html(prediction: str, prob: float) -> str:
    if prediction == "DME":
        text, color = "DME detected", "var(--signal-high)"
    else:
        text, color = "Within normal limits", "var(--signal-none)"
    return f'<div class="result-label" style="color:{color}">{text}</div>'


def render_confidence_html(prob: float, threshold: float) -> str:
    prob_pct = max(0.0, min(1.0, prob)) * 100
    thr_pct = max(0.0, min(1.0, threshold)) * 100
    marker_color = "var(--signal-high)" if prob >= threshold else "var(--signal-none)"
    return f'''
    <div class="confidence-block">
      <div class="confidence-row">
        <span class="confidence-label">Confidence</span>
        <span class="confidence-value">{_fmt(prob)}</span>
      </div>
      <div class="confidence-track">
        <div class="confidence-fill" style="width:{prob_pct:.2f}%"></div>
        <div class="confidence-threshold-tick" style="left:{thr_pct:.2f}%"></div>
        <div class="confidence-marker" style="left:{prob_pct:.2f}%; background:{marker_color}"></div>
      </div>
      <div class="confidence-scale-labels">
        <span class="scale-min">0</span>
        <span class="threshold-label" style="left:{thr_pct:.2f}%">{_fmt(threshold)}</span>
        <span class="scale-max">1.0</span>
      </div>
    </div>
    '''


def render_extended_html(result: dict) -> str:
    """
    Real E1/E2 output when available, each with an un-hideable reliability
    disclaimer (see module docstring) -- an explicit "not available"
    placeholder only when the underlying checkpoint/reference is missing or
    this particular scan failed boundary detection (never a fabricated
    mask, index, or flag; PROJECT_BRIEF.md C7).
    """
    thickness = result.get("thickness")
    if thickness is None:
        thickness_row = (
            '<div class="extended-row"><span class="extended-label">Thickening index</span>'
            f'<span class="extended-value placeholder">{_extended_unavailable_reason("thickness", result)}</span></div>'
            '<div class="extended-row"><span class="extended-label">Centre involvement</span>'
            f'<span class="extended-value placeholder">{_extended_unavailable_reason("thickness", result)}</span></div>'
        )
    elif not thickness.get("valid"):
        reason = thickness.get("reason", "boundary detection failed for this scan")
        thickness_row = (
            f'<div class="extended-row"><span class="extended-label">Thickening index</span>'
            f'<span class="extended-value placeholder">not available for this scan ({reason})</span></div>'
        )
    else:
        indeterminate_text = "indeterminate — volumetric imaging recommended" if thickness["indeterminate"] else "not indeterminate"
        thickness_row = (
            f'<div class="extended-row"><span class="extended-label">Thickening index (E2)</span>'
            f'<span class="extended-value">{thickness["index"]:.2f}</span></div>'
            f'{render_thickness_distribution_html(thickness["index"])}'
            f'<div class="extended-row"><span class="extended-label">Centre involvement</span>'
            f'<span class="extended-value">{indeterminate_text}</span></div>'
            '<div class="extended-note">Classical CV proxy, not a trained/validated layer segmentation. '
            'Real DME-vs-Normal separation on the held-out test set: AUC 0.601 ("heavy overlap") — weak on '
            'its own; informational only, does not drive the referral state above. '
            'See phase5-thickening-signoff.md.</div>'
        )

    segmentation = result.get("segmentation")
    if segmentation is None:
        seg_row = (
            '<div class="extended-row"><span class="extended-label">Fluid segmentation</span>'
            f'<span class="extended-value placeholder">{_extended_unavailable_reason("segmentation", result)}</span></div>'
        )
    else:
        seg_row = (
            f'<div class="extended-row"><span class="extended-label">Fluid area (E1)</span>'
            f'<span class="extended-value">{segmentation["fluid_area_frac"] * 100:.1f}% of scan</span></div>'
            '<div class="extended-note">Real model output, but under-trained — test Dice 0.0090 on held-out '
            'Duke scans. Not clinically reliable; treated as future work, not a usable result. '
            'See phase4-segmentation-signoff.md.</div>'
        )

    return (
        '<div class="section-label">Extended objectives — informational, never overrides the result above</div>'
        f'<div class="extended-block">{seg_row}{thickness_row}</div>'
    )


def render_referral_html(prediction: str, prob: float, threshold: float, margin: float) -> str:
    # Icons are distinct glyph shapes, not just colour -- greyscale/colour-blind legible
    # per the acceptance criterion.
    if prediction == "DME":
        state, icon, text = "dme", "&#10007;", "DME detected, centre involvement indeterminate &mdash; refer"
    elif prob >= threshold - margin:
        state, icon, text = "borderline", "!", "Below confidence threshold &mdash; refer for manual review"
    else:
        state, icon, text = "normal", "&#10003;", "Within normal limits &mdash; no referral"
    return f'''
    <div class="referral-state referral-{state}">
      <span class="referral-icon" aria-hidden="true">{icon}</span>
      <span class="referral-text">{text}</span>
    </div>
    '''


def render_results_html(result: dict, threshold: float) -> str:
    if result is None:
        return f'<div class="empty-state">{EMPTY_STATE_TEXT}</div>'
    label_html = render_label_html(result["prediction"], result["probability"])
    confidence_html = render_confidence_html(result["probability"], threshold)
    extended_html = render_extended_html(result)
    referral_html = render_referral_html(result["prediction"], result["probability"], threshold, LOW_CONFIDENCE_MARGIN)
    fallback_note = (
        '<div class="fallback-note">Retinal flattening fell back to the unflattened image for this scan '
        "(automatic quality guard, not an error).</div>"
        if result.get("fallback_triggered")
        else ""
    )
    return f'<div class="fade-in">{label_html}{confidence_html}{extended_html}{referral_html}{fallback_note}</div>'


# ---------------------------------------------------------------------------
# CSS -- OCT-report idiom per the brief: printed-white page, grayscale scan
# as the dark anchor, two semantic colours used only on state indicators.
# Explicitly NOT: dark dashboard with one bright accent, rounded-card grid,
# gradient washes, all-caps eyebrow labels.
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
:root {
  --paper: #FCFCFA;
  --ink: #16181C;
  --graphite: #5A6069;
  --rule: #DFE1E0;
  --scan-bg: #0A0C0F;
  --signal-high: #C0392B;
  --signal-none: #1B7F5A;
  --signal-defer: #B8860B;
}

/* Defensive reset -- gr.HTML() content sits inside Gradio's own base
   stylesheet, which can impose its own box-sizing/display defaults on bare
   divs. Pin these explicitly so the confidence bar's percentage-based
   width/left offsets are computed against a predictable box model. */
.confidence-track, .confidence-track * {
  box-sizing: border-box;
  display: block;
  flex: none;
}

.gradio-container {
  background: var(--paper) !important;
  color: var(--ink) !important;
  font-family: 'Inter', 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  font-variant-numeric: tabular-nums;
  max-width: 1280px !important;
}

#app-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  border-bottom: 1px solid var(--rule);
  padding-bottom: 12px;
  margin-bottom: 18px;
}
#app-header .title { font-size: 20px; font-weight: 600; color: var(--ink); }
#app-header .meta { font-size: 13px; color: var(--graphite); font-variant-numeric: tabular-nums; }

#scan-display { background: var(--scan-bg) !important; border-radius: 2px !important; }
#scan-display img { object-fit: contain !important; }

.result-label {
  font-size: 32px;
  font-weight: 600;
  line-height: 1.2;
  font-variant-numeric: tabular-nums;
  margin-bottom: 14px;
}

.confidence-block { margin-bottom: 18px; }
.confidence-row { display: flex; justify-content: space-between; font-size: 15px; margin-bottom: 8px; }
.confidence-label { color: var(--graphite); }
.confidence-value { font-variant-numeric: tabular-nums; font-weight: 600; color: var(--ink); }
.confidence-track {
  position: relative;
  height: 8px;
  background: var(--rule);
  border-radius: 2px;
  margin: 10px 2px 6px 2px;
}
.confidence-fill {
  position: absolute; left: 0; top: 0; height: 100%;
  background: var(--graphite);
  border-radius: 2px;
}
.confidence-threshold-tick {
  position: absolute; top: -5px; width: 2px; height: 18px;
  background: var(--ink);
  transform: translateX(-1px);
}
.confidence-marker {
  position: absolute; top: -6px; width: 10px; height: 20px;
  border-radius: 2px;
  transform: translateX(-5px);
  border: 2px solid var(--paper);
  box-shadow: 0 0 0 1px var(--ink);
}
.confidence-scale-labels {
  position: relative; height: 16px; font-size: 13px; color: var(--graphite);
  font-variant-numeric: tabular-nums;
}
.confidence-scale-labels .scale-min { position: absolute; left: 0; }
.confidence-scale-labels .scale-max { position: absolute; right: 0; }
.threshold-label { position: absolute; transform: translateX(-50%); font-weight: 600; color: var(--ink); }

.section-label {
  font-size: 13px; color: var(--graphite); margin: 22px 0 2px 0;
  padding-top: 14px; border-top: 1px solid var(--rule);
}

.extended-block { margin-top: 2px; }
.extended-row {
  display: flex; justify-content: space-between; align-items: baseline;
  font-size: 15px; padding: 8px 0; border-top: 1px solid var(--rule);
}
.extended-label { color: var(--graphite); }
.extended-value { font-variant-numeric: tabular-nums; }
.extended-value.placeholder { color: var(--graphite); font-style: italic; font-size: 13px; }
.extended-note { font-size: 12px; color: var(--graphite); padding: 4px 0 8px 0; line-height: 1.4; }

/* Grad-CAM colour legend, shown under the heatmap view -- explains the
   colour scale rather than leaving it to be inferred from the opacity
   slider alone. Approximates cv2.COLORMAP_JET, which show_cam_on_image
   uses internally. */
.heatmap-note { padding-top: 2px; }
.heatmap-legend {
  display: flex; align-items: center; gap: 8px; font-size: 11px;
  color: var(--graphite); margin-bottom: 6px;
}
.heatmap-legend-bar {
  flex: 1; height: 6px; border-radius: 2px;
  background: linear-gradient(to right, #00007F, #0000FF, #00CFFF, #4EFF4E, #FFFF00, #FF7F00, #C0392B);
}

/* Thickening-index distribution context: two neutral (--graphite-only, never
   the reserved --signal-* colours) bands showing the real reported Normal/DME
   ranges, plus a marker for this scan's real value. Not a state indicator, so
   it deliberately does not borrow the referral palette. */
.thickness-dist { margin: 2px 0 10px 0; }
.thickness-dist-row { display: flex; align-items: center; gap: 8px; margin-bottom: 3px; }
.thickness-dist-band-label { width: 46px; font-size: 12px; color: var(--graphite); flex-shrink: 0; }
.thickness-dist-track { position: relative; flex: 1; height: 10px; background: var(--rule); border-radius: 2px; }
.thickness-dist-band { position: absolute; top: 0; height: 100%; border-radius: 2px; }
.thickness-dist-band.normal { background: rgba(90, 96, 105, 0.28); }
.thickness-dist-band.dme { background: rgba(90, 96, 105, 0.5); }
.thickness-dist-marker {
  position: absolute; top: -4px; width: 2px; height: 18px;
  background: var(--ink); transform: translateX(-1px);
}
.thickness-dist-caption { font-size: 11px; color: var(--graphite); line-height: 1.4; margin-top: 2px; }

.referral-state {
  display: flex; align-items: center; gap: 10px;
  padding: 12px 14px;
  background: var(--paper);
  border: 1px solid var(--rule);
  border-left: 4px solid currentColor;
  margin: 18px 0 14px 0;
  font-size: 15px;
}
.referral-dme { color: var(--signal-high); }
.referral-borderline { color: var(--signal-defer); }
.referral-normal { color: var(--signal-none); }
.referral-icon { font-size: 16px; font-weight: 700; width: 18px; text-align: center; }
.referral-text { color: var(--ink); }

.fallback-note { font-size: 13px; color: var(--graphite); margin-top: 8px; }

.footer { font-size: 13px; color: var(--graphite); border-top: 1px solid var(--rule); padding-top: 12px; margin-top: 22px; }

.empty-state { font-size: 15px; color: var(--graphite); padding: 48px 0; text-align: center; }
.empty-state.error { color: var(--signal-high); }

@media (prefers-reduced-motion: no-preference) {
  .fade-in { animation: dme-fadein 200ms ease-out; }
}
@keyframes dme-fadein { from { opacity: 0; } to { opacity: 1; } }

button:focus-visible, input:focus-visible, [tabindex]:focus-visible {
  outline: 2px solid var(--ink) !important;
  outline-offset: 2px !important;
}
"""

# Left/right arrow cycles the view radio; space toggles heatmap<->original.
# Client-side only (no Python round trip needed to change the selection --
# clicking the underlying radio input fires Gradio's own change event, which
# re-renders the view exactly as a mouse click would).
KEYBOARD_NAV_JS = """
() => {
  function getRadios() {
    const container = document.getElementById('view-radio');
    if (!container) return [];
    return Array.from(container.querySelectorAll('input[type=radio]'));
  }
  document.addEventListener('keydown', (e) => {
    const tag = (e.target && e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea') return;
    const radios = getRadios();
    if (radios.length === 0) return;
    let idx = radios.findIndex((r) => r.checked);
    if (idx === -1) idx = 0;
    if (e.key === 'ArrowRight') {
      e.preventDefault();
      radios[(idx + 1) % radios.length].click();
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      radios[(idx - 1 + radios.length) % radios.length].click();
    } else if (e.key === ' ' || e.code === 'Space') {
      e.preventDefault();
      const current = radios[idx];
      const targetValue = current && current.value === 'heatmap' ? 'original' : 'heatmap';
      const target = radios.find((r) => r.value === targetValue);
      if (target) target.click();
    }
  });
}
"""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def render_view_note(result: Optional[dict], view: str) -> str:
    """HTML for the note shown under the view radio for the two extended-objective
    views -- real-result disclaimer, or an explicit not-available reason, mirroring
    render_extended_html()'s wording so the two never disagree with each other."""
    if result is None:
        return ""
    if view == "heatmap":
        if result.get("heatmap") is None:
            return '<div class="extended-value placeholder">Grad-CAM not available for this checkpoint architecture.</div>'
        location = result.get("heatmap_location")
        location_line = f" Activation is {location}." if location else ""
        return (
            '<div class="extended-note heatmap-note">'
            '<div class="heatmap-legend"><span>Low relevance</span>'
            '<div class="heatmap-legend-bar"></div><span>High relevance</span></div>'
            f"Colour marks which pixels most drove the model's decision (Grad-CAM on <code>conv_head</code>, "
            f"randomisation-sanity-checked — see report §2.6).{location_line}</div>"
        )
    if view == "fluid mask":
        seg = result.get("segmentation")
        if seg is None:
            reason = _extended_unavailable_reason("segmentation", result)
            return f'<div class="extended-value placeholder">Segmentation mask {reason}.</div>'
        return (
            '<div class="extended-note">Real E1 model output — under-trained (test Dice 0.0090), '
            "not clinically reliable. Hatched red = predicted fluid (deliberately not a solid fill, "
            "so the overlay itself doesn't look more confident than the result is). "
            "See phase4-segmentation-signoff.md.</div>"
        )
    if view == "thickness map":
        thk = result.get("thickness")
        if thk is None:
            reason = _extended_unavailable_reason("thickness", result)
            return f'<div class="extended-value placeholder">Thickness map {reason}.</div>'
        if not thk.get("valid"):
            return f'<div class="extended-value placeholder">Thickness map not available for this scan ({thk.get("reason", "boundary detection failed")}).</div>'
        return (
            '<div class="extended-note">Real E2 boundary proxy — yellow/cyan = detected ILM/RPE proxy, '
            "green = foveal column + window, red dashed = peak-thickness column. Classical CV, not a "
            "trained layer segmentation; weak DME-vs-Normal separation (AUC 0.601). See phase5-thickening-signoff.md.</div>"
        )
    return ""


def build_demo(ctx: dict) -> gr.Blocks:
    threshold = ctx["threshold"]

    def on_upload(path):
        if not path:
            return None, None, render_header_html(None), render_results_html(None, threshold), gr.update(visible=False, value="")
        bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            return (
                None,
                None,
                render_header_html(None),
                f'<div class="empty-state error">{ERROR_STATE_TEXT}</div>',
                gr.update(visible=False, value=""),
            )
        try:
            result = run_screening(ctx, bgr)
        except Exception as exc:
            print(f"[app] inference failed: {exc}")
            return (
                None,
                None,
                render_header_html(None),
                f'<div class="empty-state error">Could not process this image: {exc}</div>',
                gr.update(visible=False, value=""),
            )
        scan_id = Path(path).stem[:24]
        display = render_view(result, "original", 50)
        # Diagnostic: the confidence bar and the printed number below must always
        # agree, since render_results_html derives both from this same value --
        # logged here so a mismatch on screen is verifiable against the real
        # computed probability, not guessed at from a screenshot.
        print(
            f"[app] scan={scan_id} prediction={result['prediction']} "
            f"probability={result['probability']:.6f} threshold={threshold:.6f}"
        )
        return result, display, render_header_html(scan_id), render_results_html(result, threshold), gr.update(visible=False, value="")

    def on_view_change(result, view, opacity):
        img = render_view(result, view, opacity)
        note = render_view_note(result, view)
        show_note = view in ("heatmap", "fluid mask", "thickness map") and bool(note)
        return img, gr.update(visible=show_note, value=note)

    with gr.Blocks(title="DME screening") as demo:
        state = gr.State(None)

        header_html = gr.HTML(render_header_html(None))

        with gr.Row():
            with gr.Column(scale=6):
                image_input = gr.Image(
                    type="filepath", label="Upload OCT B-scan", sources=["upload"], elem_id="upload-image"
                )
                display_image = gr.Image(
                    type="numpy", show_label=False, interactive=False, elem_id="scan-display"
                )
                opacity_slider = gr.Slider(0, 100, value=50, step=1, label="Heatmap opacity")
                view_radio = gr.Radio(
                    choices=VIEW_CHOICES, value="original", label="View", elem_id="view-radio"
                )
                fluid_note = gr.HTML("", visible=False)
            with gr.Column(scale=5):
                results_html = gr.HTML(render_results_html(None, threshold))
                gr.HTML(f'<div class="footer">{FOOTER_TEXT}</div>')

        image_input.change(
            on_upload,
            inputs=[image_input],
            outputs=[state, display_image, header_html, results_html, fluid_note],
        )
        view_radio.change(
            on_view_change, inputs=[state, view_radio, opacity_slider], outputs=[display_image, fluid_note]
        )
        opacity_slider.change(
            on_view_change, inputs=[state, view_radio, opacity_slider], outputs=[display_image, fluid_note]
        )

        demo.load(fn=None, inputs=None, outputs=None, js=KEYBOARD_NAV_JS)

    return demo


def main():
    parser = argparse.ArgumentParser(description="Phase 6: DME screening interface.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="checkpoint_best.pt to serve (must already carry threshold/temperature from evaluate.py).")
    parser.add_argument("--segmentation-checkpoint", type=Path, default=DEFAULT_SEGMENTATION_CHECKPOINT, help="E1 checkpoint_best.pt (segment.py). Omit/missing -> that section shows as not available.")
    parser.add_argument("--thickness-summary", type=Path, default=DEFAULT_THICKNESS_SUMMARY, help="E2 thickness_summary.json (thickness.py), for the real Normal reference_px. Omit/missing -> that section shows as not available.")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    ctx = load_model(args.checkpoint)
    ctx.update(load_extended_models(ctx["device"], args.segmentation_checkpoint, args.thickness_summary))
    demo = build_demo(ctx)
    demo.launch(server_port=args.server_port, share=args.share, theme=gr.themes.Base(), css=CUSTOM_CSS)


if __name__ == "__main__":
    main()
