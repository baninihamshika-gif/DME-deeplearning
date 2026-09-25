# Phase 3g — Ablation Table & Phase 3 Sign-off

## Ablation table

Per the brief's scope decision: only A1 (TTA) and A5 (label smoothing) are implemented and run. A2/A3/A4/A6 have no code support and were explicitly descoped for GPU-budget reasons — listed as "not run," not omitted.

| Configuration | Accuracy (default / tuned) | Sensitivity (default / tuned) | AUC | ECE after calibration |
|---|---|---|---|---|
| **Baseline** — B3, no technique | 99.79% / 99.17% | 99.59% / 98.35% | 1.0000 | 0.0027 (T=0.9171) |
| **+ A1 — TTA** | 99.79% (n/a — TTA evaluated at 0.5 only) | 99.59% | 1.0000 | not recomputed for TTA logits |
| **+ A5 — Label smoothing 0.05** (full retrain) | 99.79% / 99.59% | 99.59% / 99.17% | 1.0000 | 0.0035 (T=0.4792) |
| A2 — 5-fold ensemble | not implemented / not run | — | — | — |
| A3 — EMA of weights | not implemented / not run | — | — | — |
| A4 — Cosine annealing | not implemented / not run | — | — | — |
| A6 — Progressive resolution | not implemented / not run | — | — | — |

**Total gain across all run techniques, at the default (0.5) threshold: 0.00%.** Both A1 and A5 land on the exact same confusion matrix as the untouched baseline (241 TP / 1 FN / 242 TN / 0 FP) at that threshold. Per the brief's own instruction, this is stated plainly rather than talked up: the primary model was already at this dataset's ceiling, and neither technique moved it.

**A5 detail, reported honestly against the brief's prediction:** the brief expects label smoothing to trade slightly worse accuracy for meaningfully better calibration. The measured result only partly matches that. At the *tuned* threshold, A5 actually scored *higher* accuracy than the untouched baseline (99.59% vs 99.17%) — a different threshold was selected (0.89 vs 0.971), not a like-for-like comparison, so this isn't read as "label smoothing improved accuracy." On calibration, the raw (pre-temperature) confidence was substantially worse-calibrated under label smoothing (ECE 0.0306 vs baseline's 0.0028) — expected, since label smoothing flattens raw softmax confidence. After temperature scaling, though, A5's calibration (ECE 0.0035) ended up marginally *worse* than the baseline's (0.0027), not better. This is reported as a genuine negative-ish finding rather than forced to match the brief's generic prediction — real per-project results can and do diverge from general expectations, and that's exactly the kind of thing 3g asks to be stated plainly.

Checkpoint used: `artifacts/kaggle/20260922T160023Z/artifacts/checkpoint_best.pt` (B3, `label_smoothing=0.05`, early-stopped at global epoch 18/30, best epoch 13, best val_auc 0.9952 — vs the primary's best val_auc 0.9958 at epoch 20). Full evaluation output: `artifacts/ablation/label_smoothing_0.05/metrics.json`.

**One gap, disclosed rather than patched over:** the ablation run's `training_log.csv` didn't survive the fetch (a `RemoteDisconnected` mid-transfer), so `training_curves.png` wasn't regenerated for this variant — `evaluate.py` correctly skipped it rather than fabricating one. The real per-epoch numbers exist in the raw Kaggle execution log and are quoted above; only the plotted figure is missing.

## Phase 3 verification checklist

| Item | Status |
|---|---|
| Evaluation (3d) | ✅ Present — `metrics.json` for B3 primary, B0, ResNet-50, and the A5 ablation variant |
| Threshold tuning (3c) | ✅ Present — tuned threshold + val sensitivity/specificity in every `metrics.json` |
| Calibration (3e) | ✅ Present — temperature scaling, ECE before/after, reliability diagrams |
| Grad-CAM (3f) | ✅ Present — `gradcam_grid.png` (3×3 grid: DME / Normal / misclassified) |
| Grad-CAM randomisation sanity check (3f) | ✅ Present — `gradcam_randomization.json`: mean correlation −0.117 (threshold 0.5) → verdict **"changed substantially"**, confirming the heatmap carries real information |
| Baseline comparison (3g) | ✅ Present — `phase3g-baseline-comparison.md`, B0/ResNet-50/B3 under identical conditions |
| Ablation table (3g) | ✅ Present — this document |
| `metrics.json` contains real results | ✅ Verified — every number here traced to an actual console output or fetched file, not estimated |
| No fabricated results | ✅ Confirmed — B0's missing training curve and A5's missing `training_log.csv` were disclosed as gaps, not filled in |
| No test leakage | ✅ Confirmed — C8 patient-overlap exclusion (305 patients) ran identically and correctly in every single training and evaluation run's own log; `utils.py`'s `build_splits()` reviewed line-by-line earlier in this project |

All checks pass.

## PHASE 3 — COMPLETE

```
PHASE 3 COMPLETE
Tuned threshold: 0.971 (val sens 0.9720, val spec 0.9998)
Test @ 0.5:   acc 0.9979 sens 0.9959 spec 1.0000 AUC 1.0000
Test @ tuned: acc 0.9917 sens 0.9835 spec 1.0000 AUC 1.0000
ECE before/after: 0.0028 -> 0.0027, T = 0.9171
Grad-CAM randomisation: changed substantially
Baselines: B0 99.79%/1.0000, ResNet-50 99.79%/1.0000, B3 99.79%/1.0000
Ablation total gain: 0.00% (A1 TTA and A5 label-smoothing both matched baseline exactly at default threshold; A2/A3/A4/A6 not implemented, GPU-budget descoped)
Blocking questions: none
```

P1–P4 are complete. Per the brief, the project is shippable at this point; everything from here (Phase 6 UI onward) is queued, not blocking.
