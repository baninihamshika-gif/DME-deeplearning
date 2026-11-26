# Phase 4 — Segmentation (Extended Objective) — Sign-off

**Status: TIMEBOXED. Verdict: future work.** Decided with the user 2026-09-23 after reviewing the real numbers below — accepted as final, no further GPU spend on this phase. Real Val/Test Dice, IoU, and boundary-distance figures are reported here exactly as produced by the pipeline; none are invented (C6).

## What this phase covers

Extended objective (beyond the primary DME classifier): train a U-Net to segment intraretinal fluid in Duke DME OCT scans (Chiu et al. 2015), using the Phase 2 classifier's EfficientNet-B3 encoder as a pretrained initialization. Per the brief, this is explicitly time-boxed — a genuine "future work" outcome is an acceptable, honest result, not a failure to hide.

## Pipeline validation before the real run

- **Rasterization gate** (`verify_rasterization()`, mandatory per the brief): run against real staged `.mat` data for two subjects (Subject_01, Subject_05). Fluid mask and all 8 layer boundaries land correctly on the visible cystoid pockets and retinal layers.
- **Structural consistency across all 10 real Duke subjects** confirmed directly: 11 annotated scans each, identical `(496,768)`/`(8,768)` shapes, monotonic boundary ordering. 4 of 10 subjects use different annotated scan indices than the other 6 — confirming the per-subject index detection is load-bearing, not defensive. Total: 110 annotated scans, matching the brief.
- **Two real local bugs found and fixed** before any GPU time was spent: `IMAGE_SIZE_SEG=320` (U-Net encoder needs input divisible by 32; the classifier's `IMAGE_SIZE=300` isn't), and `scipy.io.loadmat(..., variable_names=[...])` (was loading ~192MB/subject of unused arrays, causing a local OOM).
- **Kaggle automation extended** from the proven Phase 2.5 pattern: 3 new private datasets (source+segment.py, Duke, classifier checkpoint), a self-contained kernel bootstrap (`entry_segment.py`), and `run_segment.py`'s push/poll/fetch driver, sharing the account-wide GPU quota log.
- **Two real live-infrastructure bugs found and fixed** during the push process itself: a `ModuleNotFoundError` from a `kaggle kernels push` platform constraint (a script kernel only uploads its single `code_file`, not sibling `.py` files — fixed by making `entry_segment.py` self-contained), and a `503 Service Unavailable` from Kaggle's own API on a later push (fixed by adding an opt-in infra-retry to `_run_kaggle_cli()`, verified with 4 new unit tests, 123/123 passing).
- A **smoke run** (1 epoch, full 110-scan Duke set, no subsampling) completed successfully on a Tesla T4 before the full run was attempted, confirming the entire pipeline end-to-end on live infrastructure.

## The real full run (2026-09-23, kernel `dme-oct-segment` v9)

- **Data**: 110 manually-annotated Duke scans, 10 subjects. Patient-grouped split (seed=42, zero patient overlap by construction): train 6 patients / 66 scans (`02,03,04,05,09,10`), val 2 patients / 22 scans (`01,08`), test 2 patients / 22 scans (`06,07`).
- **Encoder transfer**: 572/572 keys matched from the Phase 2 primary B3 classifier checkpoint.
- **Training**: U-Net (smp), Adam lr=1e-4, loss = 0.5·DiceLoss + 0.5·BCEWithLogitsLoss, batch size 16, up to 60 epochs with early stopping (patience 5). Ran on a real Tesla T4 GPU, 4.89 real GPU minutes recorded in `quota_log.json`.
- **Early stopping**: triggered after epoch 6 (best epoch was 1; no val_dice improvement for 5 epochs after that).

### Results (real, from the fetched Kaggle log)

| Split | Dice | IoU | Boundary distance (px) |
|---|---|---|---|
| Val | 0.0139 | 0.0071 | 16.01 |
| Test | 0.0090 | 0.0045 | 14.26 |

**Verdict (from the pipeline's own sign-off block): future work. Blocking questions: none.**

## Diagnostic read — why the numbers are this low

A Dice around 0.01 is close to what a model predicting almost no foreground would score, so before accepting "future work" at face value this was checked for a hidden bug rather than just reported — the same scrutiny this project applied (in the opposite direction) to the suspicious AUC=1.0 classifier result in Phase 3g:

- **Metric code is correct**: `dice_coefficient()` and `iou_score()` are standard set-based implementations (`2·|A∩B| / (|A|+|B|)` and `|A∩B|/|A∪B|`), reviewed directly — no bug found.
- **Mask pipeline is correct**: `binarize_fluid()` correctly thresholds the integer region labels; the letterbox resize (`_letterbox_pair()`) uses `cv2.INTER_NEAREST` for the mask specifically (not linear interpolation, which would have introduced fractional mask values) — reviewed directly, no bug found.
- **Not total collapse**: boundary distance is only defined (non-NaN) when both the predicted and true masks are non-empty. Getting finite boundary-distance averages (16.01px val, 14.26px test) confirms the model *is* predicting some fluid on most images — just poorly localized, not literally empty output everywhere.
- **Most likely real cause**: severe class imbalance (fluid is ~2.8% of pixels, confirmed by the rasterization check: 10519/380928) combined with no imbalance-aware loss term (`BCEWithLogitsLoss` has no `pos_weight`), a decoder trained from scratch on only 66 images, and very few epochs before early stopping killed the run (`train_dice` stayed flat at 0.0096–0.0102 across all 6 epochs — the model essentially never moved off its initial near-empty prediction). This reads as genuine under-training on a small, imbalanced dataset within the time-boxed budget, not a code defect.

**Plausible next steps, if this phase is revisited later** (not pursued now — accepted as future work per the user's decision): a class-imbalance-aware loss (`pos_weight` on the BCE term, or a Tversky/Focal loss), a longer patience or minimum-epoch floor before early stopping can trigger, and post-hoc threshold calibration on the val set (the classifier phase tuned its operating point in Phase 3c; segmentation currently uses a fixed 0.5 threshold with no equivalent step).

## Decision

Reviewed with the user 2026-09-23: accept this as the final Phase 4 result. No further GPU spend on segmentation. Phase 4 is closed out as **TIMEBOXED / future work**, honestly reported per C6, and the project proceeds to Phase 5.

## Artifacts

`artifacts/kaggle/20260923T181925Z/`: `checkpoint_best.pt`, `checkpoint_last.pt`, `segmentation_rasterization_check.png`, `segmentation_training_log.csv`, plus copies of the source files used for this run.

## GPU quota (segmentation pipeline total)

| Run | Minutes |
|---|---|
| Smoke attempt #1 (errored at import — `ModuleNotFoundError`) | ~0.1 |
| Smoke attempt #2 (succeeded — 1 epoch, Tesla T4) | 3.16 |
| Full-run attempt #1 (503 on kernel push, no kernel ran) | 0 |
| Full-run attempt #2 (succeeded — 6 epochs, Tesla T4) | 4.89 |
| **Segmentation total** | **8.15 min** |
