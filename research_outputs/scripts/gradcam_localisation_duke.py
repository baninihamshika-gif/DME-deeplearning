"""
Grad-CAM localisation benchmark on Duke DME 2015 manual fluid masks.

Why: app.py's heatmap (Grad-CAM on conv_head, 10x10 at 300px, explaining the
DME logit) often highlights retina beside the fluid rather than the fluid.
Rather than pick a replacement layer by eye, every candidate is scored
against manual fluid annotation the classifier never saw (it was trained on
Kermany only; Duke 2015 is external).

Protocol (fixed before running):
  * Scans: the 110 manually graded Duke 2015 B-scans; only scans with
    grader-1 fluid in the graded band are scored.
  * Each scan goes through utils.preprocess_image exactly as app.py does;
    the fluid mask and the graded band are moved by the same flattening
    shifts and letterbox (flatten_retina(source_valid=...)), so mask and
    heatmap share pixel coordinates.
  * CAM explains the DME class (all scans are DME); heatmaps are masked to
    real scan pixels as in app.py (mask_cam_to_image).
  * Metrics, on real scan pixels inside the graded band (fluid outside the
    band is unlabelled):
      - pixel AUROC: CAM value as a score for fluid vs non-fluid pixels;
      - pointing game: CAM peak within 10 px of grader-1 fluid;
      - mass in fluid: share of CAM mass on fluid pixels (fluid is ~2% of
        the band, so a uniform map would score ~0.02).
  * Selection: highest mean pixel AUROC on subjects 01-05 (pointing game
    breaks ties); subjects 06-10 are reported for confirmation only.
  * Sanity: explain.py's randomisation check (classifier re-initialised,
    mean Pearson correlation original vs randomised must be < 0.5) on the
    confirmation subjects for every variant.

Outputs: research_outputs/gradcam_localisation/{per_scan.csv, summary.json}.
Usage: python research_outputs/scripts/gradcam_localisation_duke.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import config  # noqa: E402
import segment as S  # noqa: E402
from app import load_model, mask_cam_to_image  # noqa: E402
from explain import cam_correlation, randomize_final_layer  # noqa: E402
from train import _eval_transform  # noqa: E402
from utils import _letterbox_resize, flatten_retina, preprocess_image, source_fill_mask  # noqa: E402

MAIN = Path("D:/Projects/DME-DEEPLEARN")
DUKE = MAIN / "data/raw/duke_dme/2015_BOE_Chiu"
CKPT = MAIN / "artifacts/kaggle/20260917T035646Z/artifacts/checkpoint_best.pt"
OUT = REPO / "research_outputs" / "gradcam_localisation"
SELECT_SUBJECTS = {"01", "02", "03", "04", "05"}
POINTING_TOL_PX = 10


class LogitDiff:
    """DME logit minus Normal logit: the quantity the 2-class decision depends on."""

    def __call__(self, out):
        return out[..., 1] - out[..., 0] if out.ndim == 2 else out[1] - out[0]


def variants(model):
    from pytorch_grad_cam import GradCAM, LayerCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    dme = ClassifierOutputTarget(1)
    diff = LogitDiff()
    return {
        "A GradCAM conv_head DME-logit (current app)": (GradCAM, lambda m: m.conv_head, dme),
        "B GradCAM conv_head DME-Normal": (GradCAM, lambda m: m.conv_head, diff),
        "C GradCAM blocks[5] (10x10) DME-Normal": (GradCAM, lambda m: m.blocks[5], diff),
        "D GradCAM blocks[4] (19x19) DME-Normal": (GradCAM, lambda m: m.blocks[4], diff),
        "E LayerCAM blocks[4] (19x19) DME-Normal": (LayerCAM, lambda m: m.blocks[4], diff),
        "F LayerCAM blocks[3] (19x19) DME-Normal": (LayerCAM, lambda m: m.blocks[3], diff),
        "G LayerCAM blocks[2] (38x38) DME-Normal": (LayerCAM, lambda m: m.blocks[2], diff),
    }


def compute_cam(model, spec, x):
    cam_cls, layer_fn, target = spec
    with cam_cls(model=model, target_layers=[layer_fn(model)]) as cam:
        return cam(input_tensor=x, targets=[target])[0]


def preprocess_with_masks(gray, fluid, band):
    """preprocess_image(gray) plus fluid / band masks in the same coordinates."""
    rgb, bbox, fallback, valid = preprocess_image(gray, image_size=config.IMAGE_SIZE, flatten=True, return_valid_mask=True)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    out = []
    for m in (fluid, band):
        _, fb, moved = flatten_retina(denoised, enabled=True, return_valid=True, source_valid=m.astype(bool) & ~source_fill_mask(gray))
        assert fb == fallback
        lb, _ = _letterbox_resize(moved * 255, config.IMAGE_SIZE)
        out.append(lb > 127)
    return rgb, valid, out[0], out[1]


def score(cam, valid, fluid, band):
    region = valid & band
    y = fluid[region]
    s = cam[region]
    auroc = float(roc_auc_score(y, s)) if 0 < y.sum() < y.size else float("nan")
    near = cv2.dilate(fluid.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * POINTING_TOL_PX + 1,) * 2)).astype(bool)
    peak = np.unravel_index(np.where(region, cam, -1).argmax(), cam.shape)
    return {
        "auroc": auroc,
        "pointing_hit": bool(near[peak]),
        "mass_in_fluid": float(cam[region & fluid].sum() / max(cam[region].sum(), 1e-9)),
        "fluid_frac_of_band": float(y.mean()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.SEED)
    model = load_model(CKPT)["model"].eval()
    random_model = randomize_final_layer(model).eval()
    tf = _eval_transform()
    specs = variants(model)

    rows, rand_corr = [], {k: [] for k in specs}
    for mat in S.find_subject_files(DUKE):
        sub = S.load_duke_subject(mat)
        for pos, scan_idx in enumerate(sub["scan_indices"]):
            fluid = S.binarize_fluid(sub["fluid1"][pos])
            band = S.graded_region(sub["fluid1"][pos], sub["layers1"][pos])
            if not (fluid & band).any():
                continue
            gray = sub["images"][pos]
            rgb, valid, fluid_p, band_p = preprocess_with_masks(gray, fluid & band, band)
            if not (fluid_p & valid & band_p).any():
                continue
            x = tf(image=rgb)["image"].unsqueeze(0)
            with torch.no_grad():
                p_dme = float(torch.softmax(model(x), 1)[0, 1])
            for name, spec in specs.items():
                cam = mask_cam_to_image(compute_cam(model, spec, x), valid)
                rows.append({"subject": sub["subject"], "scan_idx": scan_idx, "variant": name, "p_dme": p_dme, **score(cam, valid, fluid_p, band_p)})
                if sub["subject"] not in SELECT_SUBJECTS:
                    rcam = mask_cam_to_image(compute_cam(random_model, spec, x), valid)
                    rand_corr[name].append(cam_correlation(cam, rcam))
        print(f"[gradcam_localisation] Subject_{sub['subject']} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_scan.csv", index=False)
    df["set"] = np.where(df.subject.isin(SELECT_SUBJECTS), "selection_01-05", "confirmation_06-10")
    agg = (df.groupby(["set", "variant"])
             .agg(n_scans=("auroc", "size"), auroc=("auroc", "mean"), pointing=("pointing_hit", "mean"),
                  mass_in_fluid=("mass_in_fluid", "mean"), fluid_frac=("fluid_frac_of_band", "mean"))
             .reset_index())
    sel = agg[agg.set == "selection_01-05"].sort_values(["auroc", "pointing"], ascending=False)
    chosen = sel.iloc[0].variant
    summary = {
        "protocol": __doc__.split("Protocol (fixed before running):")[1].split("Outputs:")[0].strip(),
        "chosen_variant": chosen,
        "results": {s: {r.variant: {k: (float(getattr(r, k)) if k != "n_scans" else int(r.n_scans))
                                    for k in ("n_scans", "auroc", "pointing", "mass_in_fluid", "fluid_frac")}
                        for r in agg[agg.set == s].itertuples()} for s in agg.set.unique()},
        "randomisation_mean_corr_confirmation": {k: float(np.nanmean(v)) for k, v in rand_corr.items()},
        "randomisation_threshold": 0.5,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(agg.round(3).to_string(index=False))
    print("randomisation mean corr (06-10):", {k[:1]: round(v, 3) for k, v in summary["randomisation_mean_corr_confirmation"].items()})
    print("chosen on 01-05:", chosen)


if __name__ == "__main__":
    main()
