"""
Phase B4 -- independent re-evaluation of the existing U-Net checkpoint
(artifacts/kaggle/20260923T181925Z/artifacts/segmentation/checkpoint_best.pt)
on the Duke DME subjects held out by segment.py's own patient split.

No training. Re-uses segment.py's loader, split, preprocessing, model
constructor and Dice/IoU functions, then adds what the original evaluation
did not report:
  * predicted and ground-truth foreground fractions per image (detects
    "predict everything" / "predict nothing" collapse)
  * Dice/IoU with segment.py's convention (ungraded NaN -> background) AND
    restricted to the manually graded region only
  * pooled (pixel-aggregated) Dice in addition to the per-image mean
  * explicit counts of empty-ground-truth images and how they score
  * grader-1 vs grader-2 agreement on the same scans, as a human reference

Outputs: research_outputs/data/segmentation_per_image.csv,
segmentation_summary.json, and segmentation_examples.npz (images/masks for the
qualitative figure).

Usage: python research_outputs/scripts/05_segmentation_audit.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import segment as S  # noqa: E402

OUT = REPO / "research_outputs" / "data"
DUKE = REPO / "data" / "raw" / "duke_dme" / "2015_BOE_Chiu"
CKPT = REPO / "artifacts/kaggle/20260923T181925Z/artifacts/segmentation/checkpoint_best.pt"


def main():
    import albumentations as A
    import torch
    from albumentations.pytorch import ToTensorV2

    entries = S.build_duke_index(DUKE)
    train, val, test = S.build_splits_duke(entries)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = S.build_seg_model()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    norm = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])

    cache, rows, examples = {}, [], {}
    for split_name, split in (("val", val), ("test", test)):
        for e in split:
            if e["mat_path"] not in cache:
                cache[e["mat_path"]] = S.load_duke_subject(e["mat_path"])
            sub = cache[e["mat_path"]]; pos = e["pos"]
            img = sub["images"][pos]
            f1_raw, f2_raw = sub["fluid1"][pos], sub["fluid2"][pos]
            gt1 = S.binarize_fluid(f1_raw); gt2 = S.binarize_fluid(f2_raw)
            graded1 = (~np.isnan(f1_raw)).astype(np.uint8)
            rgb, gt1_lb = S.preprocess_pair(img, gt1)
            _, gt2_lb = S.preprocess_pair(img, gt2)
            _, graded_lb = S.preprocess_pair(img, graded1)
            with torch.no_grad():
                logit = model(norm(image=rgb)["image"].unsqueeze(0))[0, 0].numpy()
            prob = 1 / (1 + np.exp(-logit)); pred = (prob > 0.5).astype(np.uint8)
            g = graded_lb.astype(bool)
            def dice_iou(a, b, m=None):
                if m is not None:
                    a, b = a[m], b[m]
                return S.dice_coefficient(a, b), S.iou_score(a, b), int(np.logical_and(a, b).sum()), int(a.sum()), int(b.sum())
            d_all, i_all, inter_all, ps_all, gs_all = dice_iou(pred, gt1_lb)
            d_g, i_g, inter_g, ps_g, gs_g = dice_iou(pred, gt1_lb, g)
            d_h, i_h, *_ = dice_iou(gt2_lb, gt1_lb, g)
            content = rgb[..., 0] > 0
            rows.append({
                "split": split_name, "subject": e["subject"], "scan_idx": e["scan_idx"],
                "gt_fluid_pixels": int(gt1_lb.sum()), "gt_empty": bool(gt1_lb.sum() == 0),
                "graded_pixels": int(g.sum()), "frame_pixels": int(pred.size), "content_pixels": int(content.sum()),
                "pred_fluid_pixels": int(pred.sum()), "pred_frac_of_frame": float(pred.mean()), "pred_frac_of_graded": float(pred[g].mean()) if g.any() else np.nan,
                "gt_frac_of_graded": float(gt1_lb[g].mean()) if g.any() else np.nan,
                "mean_prob": float(prob.mean()), "mean_prob_graded": float(prob[g].mean()) if g.any() else np.nan,
                "dice_pipeline": d_all, "iou_pipeline": i_all, "inter_all": inter_all, "pred_sum_all": ps_all, "gt_sum_all": gs_all,
                "dice_graded_only": d_g, "iou_graded_only": i_g, "inter_graded": inter_g, "pred_sum_graded": ps_g, "gt_sum_graded": gs_g,
                "dice_grader2_vs_grader1_graded": d_h, "iou_grader2_vs_grader1_graded": i_h,
            })
            if split_name == "test":
                examples[f"{e['subject']}_{e['scan_idx']}"] = (rgb[..., 0], gt1_lb, pred, prob.astype(np.float32), graded_lb)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "segmentation_per_image.csv", index=False)
    summ = {"checkpoint": str(CKPT.relative_to(REPO).as_posix()), "checkpoint_epoch": ckpt.get("epoch"), "checkpoint_best_val_dice": ckpt.get("best_val_dice"),
            "target": "binary fluid mask = any labelled region in manualFluid1 (grader 1), Duke DME (Chiu et al. 2015)",
            "threshold": 0.5, "input": "320x320 letterbox, NLM denoise + CLAHE"}
    for s in ("val", "test"):
        d = df[df.split == s]
        summ[s] = {
            "subjects": sorted(d.subject.unique().tolist()), "n_images": len(d), "n_empty_gt": int(d.gt_empty.sum()),
            "mean_dice_pipeline_convention": d.dice_pipeline.mean(), "mean_iou_pipeline_convention": d.iou_pipeline.mean(),
            "pooled_dice_pipeline": 2 * d.inter_all.sum() / (d.pred_sum_all.sum() + d.gt_sum_all.sum()),
            "mean_dice_graded_only": d.dice_graded_only.mean(), "pooled_dice_graded_only": 2 * d.inter_graded.sum() / (d.pred_sum_graded.sum() + d.gt_sum_graded.sum()),
            "mean_dice_fluid_positive_images_only": d[~d.gt_empty].dice_pipeline.mean(),
            "mean_pred_frac_of_frame": d.pred_frac_of_frame.mean(), "min_pred_frac_of_frame": d.pred_frac_of_frame.min(),
            "mean_gt_frac_of_graded": d.gt_frac_of_graded.mean(), "mean_prob": d.mean_prob.mean(),
            "inter_grader_mean_dice_graded": d.dice_grader2_vs_grader1_graded.mean(),
            "inter_grader_mean_dice_graded_fluid_positive": d[~d.gt_empty].dice_grader2_vs_grader1_graded.mean(),
        }
    with open(OUT / "segmentation_summary.json", "w") as f:
        json.dump(summ, f, indent=2, default=float)
    np.savez_compressed(OUT.parent / ".cache" / "segmentation_examples.npz", **{f"{k}__{i}": a for k, v in examples.items() for i, a in enumerate(v)})
    print(json.dumps(summ, indent=2, default=float))


if __name__ == "__main__":
    main()
