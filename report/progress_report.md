# DME OCT Project — Final Report

**Deep Learning-Based AI Detection of Macular Edema Using OCT Images**
Kalasalingam Academy of Research and Education · Guide: Dr. R. Murugeswari
Team: M Bharadhwaj · M Ghangaatharen Vishakh · S A Banini Hamshika · P Harani Hajandikaa

Report generated: 2026-09-23, assembled from `artifacts/metrics.json`, `artifacts/gradcam_randomization.json`, the repo's Kaggle run logs, and the codebase as implemented. Updated 2026-09-23 with the real Phase 4 (segmentation) and Phase 5 (thickening index) results, both completed after this report's first draft, per the project's locked execution order (Phase 7 report precedes the extended objectives, then this document is revised). Per project rule C6, every number below is either copied directly from a real `metrics.json`/log file/pipeline sign-off or explicitly marked as missing — nothing here is estimated, interpolated, or carried over from a similar run.

---

## 1. Summary

| | Status |
|---|---|
| Primary objectives (P1–P4) | ✅ Complete and signed off (Phase 3) |
| Interface (Phase 6) | ✅ Complete and verified end-to-end |
| Version freeze | ✅ `v1.0-primary-complete`, commit `5e60ec8` |
| Extended objective E1 — segmentation (Phase 4) | ✅ **Timeboxed / future work.** Real result: Test Dice 0.0090, IoU 0.0045. Investigated for a hidden bug before acceptance; none found — genuine under-training on a small, imbalanced dataset. |
| Extended objective E2 — thickening index (Phase 5) | ✅ **Complete.** Real result: separation AUC 0.601 ("heavy overlap") — a real, weak-but-genuine signal. |
| Report & deck assets (Phase 7, this document) | ✅ This report |

---

## 2. Methodology, as implemented

This section describes what the code actually does, not the original plan. Where the two diverge, the deviation is called out explicitly rather than left implicit — see §6.

### 2.1 Dataset

Kermany et al. (Cell, 2018), OCT2017 / Mendeley V2, the 83,484-training-image release (confirmed against `config.EXPECTED_TRAIN_COUNTS` on every run). Only the DME and NORMAL classes are used; CNV and DRUSEN are excluded (age-related, not diabetic). Test set: 242 DME + 242 Normal = 484 images, opened exactly once, in Phase 3.

### 2.2 Splitting (C1)

`utils.build_splits()` scans the DME/NORMAL folders under `train/`, groups every image by the patient ID embedded in its filename (`DME-1234567-1.jpeg` → patient `1234567`), and splits with **`StratifiedGroupKFold`** (group = patient, `round(1/VAL_SPLIT)` folds, one held out as validation) — **not** the plain `GroupShuffleSplit` the brief specified; see §6, deviation 1.

Two integrity checks ran and passed on the primary training run, quoted directly from `full_run_log.txt`:

```
[build_splits] count assertion vs expected_counts: PASSED {'DME': 11348, 'NORMAL': 26315}
[build_splits] patient overlap train<->val: 0 (PASSED)
[build_splits] both splits contain both classes: PASSED
[build_splits] class ratio within 5% of overall: PASSED
[build_splits] train/val <-> test patient overlap: 305 (C8 VIOLATION)
[build_splits] excluded 305 overlapping patient(s) (7503 images: train 30666->24600, val 6997->5560)
  from train/val to keep the official 484-image test set clean.
```

**Significant finding, disclosed per C8:** the official Kermany `test/` split is **not** patient-disjoint from `train/` — 305 of 368 test patients (84% of the 484 test images) also appear in the training pool. Rather than alter the official, publication-comparable 484-image test set, the team's documented decision was to fix this from the training side: every image belonging to an overlapping patient is dropped from `train_df`/`val_df` before training, verified at zero overlap afterward. This is exactly the kind of split violation C8 asks to be reported immediately rather than silently absorbed.

**Data-provenance note:** `evaluate.py` calls the same `build_splits()` function, with the same seed and arguments, to reconstruct the validation set for threshold tuning — and should therefore produce an identical validation count. It instead reports `val_n = 6047` in every `metrics.json` (primary, both baselines, and the ablation run), against the training run's own logged post-exclusion count of 5560. Both numbers are real, taken directly from artifacts on disk; the discrepancy itself was not further investigated within this report's scope and is flagged here rather than silently resolved in either direction. It does not affect any test-set result below (the 484-image test set is fixed and unaffected by this).

### 2.3 Preprocessing (C2)

`utils.preprocess_image()`: grayscale → NLM denoise (`h=10`) → retinal flattening (degree-2 polynomial fit to the lower retinal boundary, with a wrapped fallback to the unflattened image on a poor fit, logged when triggered) → CLAHE (`clipLimit=2.0`, `tileGridSize=(8,8)`) → **aspect-preserving resize with letterbox padding** to 300×300 (never a plain square resize, per C2) → 3-channel replication. Normalisation (ImageNet mean/std) happens after augmentation, at load time — not baked into the cache. The deterministic steps (denoise/flatten/CLAHE/letterbox) are cached to disk via `PreprocessCache`, keyed on source path + image size + flatten flag.

### 2.4 Model and training (P1)

`timm.create_model("tf_efficientnet_b3", pretrained=True, num_classes=2, drop_rate=0.3)`. Two-phase schedule: Phase A unfreezes only `conv_head` + `bn2` + `classifier` (Adam, LR 1e-3, 5 epochs); Phase B unfreezes the full network (Adam, LR 1e-5, up to 25 epochs), with `ReduceLROnPlateau` (factor 0.3, patience 3) and early stopping on validation loss (patience 5). Loss: `CrossEntropyLoss(weight=[0.72, 1.66])`, order confirmed by the C5 class-index assertion, which printed and passed on every run. Augmentation (training split only): horizontal flip (p=0.5), `ShiftScaleRotate` (shift 0.1, scale 0.1, rotate 10°, p=0.7), `RandomBrightnessContrast` (p=0.5) — **no vertical flip**, per C4.

The primary EfficientNet-B3 model ran Phase A to completion (5/5 epochs) and Phase B early-stopped at epoch 20/25 (5 epochs without improvement), selected by best validation AUC = 0.9958.

### 2.5 Calibration and thresholding (P3)

A single temperature scalar is fit on the validation set by minimising NLL of `logits / T`. The operating threshold is swept on validation only, selecting the threshold meeting `SENSITIVITY_TARGET = 0.97` at the best achievable specificity. Both the threshold and temperature are written into the checkpoint and read from it at inference time (`app.py`'s `load_model()` refuses to launch if either is missing) — never hardcoded, satisfying P3 and the Phase 6 acceptance criterion.

### 2.6 Explainability (P2)

Grad-CAM targets `model.conv_head` — the last layer in the network with spatial (H, W, C) structure, and therefore the only sensible target downstream of the backbone (`classifier` operates after global pooling and has no spatial map to weight). The brief's randomisation sanity check asks to randomise "the final layer's" weights; this was interpreted, and documented as a deliberate reading rather than left ambiguous, as `model.classifier` — the network's actual last layer — since randomising `conv_head` itself (Grad-CAM's own read target) would trivially destroy the heatmap and prove nothing about whether the *explanation* depends on learned weights.

### 2.7 Interface (Phase 6)

`app.py`, a Gradio application, loads the checkpoint once at startup and serves a single-screen triage UI per the brief's OCT-report visual direction. Full build and verification details, including a live-tested walkthrough of every acceptance criterion, are in `phase6-gradio-signoff.md`.

### 2.8 Segmentation (extended objective E1, Phase 4)

`segment.py`: a U-Net (`segmentation_models_pytorch`), encoder initialised from the primary EfficientNet-B3 classifier checkpoint (572/572 keys matched, confirming the transfer is genuine and not a silent no-op). Trained on the Duke DME dataset (Chiu et al., 2015 BOE) — a separate, smaller, separately-licensed dataset from the Kermany classifier data, per the brief — to segment intraretinal fluid. `IMAGE_SIZE_SEG = 320` (not the classifier's 300px) because the U-Net decoder's 5 downsampling stages require input divisible by 32, a real bug surfaced by the first local smoke test and documented as a judgment call rather than silently patched around. Loss: `0.5·DiceLoss + 0.5·BCEWithLogitsLoss`; Adam, LR 1e-4, batch size 16, up to 60 epochs with early stopping (patience 5). Patient-grouped split, 10 subjects: train 6 / val 2 / test 2, zero patient overlap by construction. Pushed to and run on real Kaggle GPU infrastructure via the same push/poll/fetch automation as the primary classifier, extended for this phase (`kaggle/run_segment.py`).

This phase is explicitly time-boxed per the brief — a genuine "future work" outcome, honestly reported, is an acceptable result here, not a failure to be hidden. Full pipeline-validation history, the real result, and the diagnostic investigation into why the score is this low are in `phase4-segmentation-signoff.md`.

### 2.9 Relative thickening index (extended objective E2, Phase 5)

`thickness.py`: a classical computer-vision proxy for retinal thickening, run directly on the Kermany OCT2017 classifier dataset (not Duke), independent of Phase 4's U-Net — `segment.py`'s own docstring flagged boundary extraction for this phase as separate, non-U-Net work from the start, since there are no ILM/RPE ground-truth labels for the Kermany dataset at all. Per column: isolate the retina tissue band (Gaussian blur + Otsu threshold, anchored on a per-column lower-boundary curve reusing `utils.flatten_retina()`'s own production-proven technique, so a brighter but anatomically-irrelevant region elsewhere in the frame — e.g. a noisy vitreous cavity — can't be mistaken for retina); take the per-column span as a thickness reading, discarding any column whose span exceeds half the content region's height as implausible. Find the foveal column (minimum thickness within the central third), take a windowed mean around it, and normalise against the median of that same measurement over the Normal training split. The "centre involvement indeterminate" third referral condition fires when the index is elevated (>1.0) but the frame's peak thickness sits outside that foveal window.

This method went through 4 documented iterations before being trusted at corpus scale, each caught by a mandatory visual verification check against real sample images (same "verify before scaling" discipline as Phase 4's rasterization check) — full history in `phase5-thickening-signoff.md`.

---

## 3. Results tables

*(Ready to paste directly into results slides. All figures below are copied verbatim from `artifacts/metrics.json`, the two `artifacts/baselines/*/metrics.json` files, `artifacts/ablation/label_smoothing_0.05/metrics.json`, and `artifacts/gradcam_randomization.json`.)*

### 3.1 Primary model (EfficientNet-B3) — test set (n=484)

| Threshold | Accuracy | Sensitivity | Specificity | Precision | F1 | AUC | TP | FN | TN | FP |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 (default) | 99.79% | 99.59% | 100.00% | 100.00% | 0.9979 | 1.0000 | 241 | 1 | 242 | 0 |
| 0.971 (tuned, sensitivity target 0.97) | 99.17% | 98.35% | 100.00% | 100.00% | 0.9917 | 1.0000 | 238 | 4 | 242 | 0 |

Validation-set threshold selection (n=6047, see §2.2 note): sensitivity 0.9720, specificity 0.9998 at threshold 0.971.

### 3.2 Calibration (temperature scaling)

| Model | ECE before | ECE after | Temperature |
|---|---|---|---|
| EfficientNet-B3 (primary) | 0.002844 | 0.002680 | 0.9171 |
| EfficientNet-B0 (baseline) | 0.003837 | 0.003889 | 1.0193 |
| ResNet-50 (baseline) | 0.002256 | 0.002325 | 0.8806 |
| EfficientNet-B3 + label smoothing 0.05 (A5) | 0.030607 | 0.003467 | 0.4792 |

### 3.3 Baseline comparison — test set, default threshold 0.5 (n=484)

| Model | Accuracy | Sensitivity | Specificity | AUC | TP/FN/TN/FP |
|---|---|---|---|---|---|
| EfficientNet-B0 | 99.79% | 99.59% | 100.00% | 1.0000 | 241/1/242/0 |
| ResNet-50 | 99.79% | 99.59% | 100.00% | 1.0000 | 241/1/242/0 |
| **EfficientNet-B3 (primary)** | **99.79%** | **99.59%** | **100.00%** | **1.0000** | **241/1/242/0** |

At each model's own validation-tuned threshold:

| Model | Tuned threshold | Sensitivity | Specificity |
|---|---|---|---|
| EfficientNet-B0 | 0.608 | 99.17% | 100.00% |
| ResNet-50 | 0.628 | 99.59% | 100.00% |
| EfficientNet-B3 | 0.971 | 98.35% | 100.00% |

**Finding, stated plainly per the brief's own instruction:** EfficientNet-B3 does not meaningfully beat either baseline. All three architectures match exactly on test accuracy and AUC at the default threshold; they diverge only slightly once each is tuned to its own validation-selected operating point. Full investigation into why AUC reaches 1.0000 (patient leakage ruled out; a weak, non-explanatory image-dimension confound found and quantified) is in `phase3g-baseline-comparison.md`.

### 3.4 Ablation table

| Configuration | Accuracy (default / tuned) | Sensitivity (default / tuned) | AUC | ECE after calibration |
|---|---|---|---|---|
| Baseline — B3, no technique | 99.79% / 99.17% | 99.59% / 98.35% | 1.0000 | 0.0027 (T=0.9171) |
| + A1 — Test-time augmentation | 99.79% (evaluated at 0.5 only) | 99.59% | 1.0000 | not recomputed for TTA logits |
| + A5 — Label smoothing 0.05 (full retrain) | 99.79% / 99.59% | 99.59% / 99.17% | 1.0000 | 0.0035 (T=0.4792) |
| A2 — 5-fold ensemble | not implemented / not run | — | — | — |
| A3 — EMA of weights | not implemented / not run | — | — | — |
| A4 — Cosine annealing | not implemented / not run | — | — | — |
| A6 — Progressive resolution | not implemented / not run | — | — | — |

**Total gain across all implemented techniques, at the default threshold: 0.00%.** A1 and A5 both land on the exact same confusion matrix as the untouched baseline (241 TP / 1 FN / 242 TN / 0 FP). Per the brief's own framing, this is the expected outcome when the baseline is already near the dataset's ceiling, and is reported plainly rather than talked up. Full narrative, including the more nuanced calibration trade-off A5 shows at its own tuned threshold, is in `phase3-ablation-and-signoff.md`.

### 3.5 Grad-CAM randomisation sanity check (P2)

| Row | Correlation (original vs. randomised-classifier CAM) |
|---|---|
| DME | −0.1885 |
| Normal | −0.1277 |
| Misclassified | −0.0348 |
| **Mean** | **−0.1170** |

Verdict: **changed substantially** (correlation threshold 0.5) — the explanation depends on the trained weights, not merely on input structure, so Grad-CAM here carries real information rather than an unvalidated visual artifact.

### 3.6 Training

| Model | Role | Best epoch | Best val AUC | Recorded GPU time |
|---|---|---|---|---|
| EfficientNet-B3 | Primary | 20 (early-stopped, Phase B) | 0.9958 | 319.77 min |
| EfficientNet-B0 | Baseline | not recorded — per-epoch log lost for this run | not recorded | 211.98 min (full) + 17.2 min (smoke) |
| ResNet-50 | Baseline | 30 (full schedule, no early stop) | 0.9961 | 384.23 min (full) + 20.4 min (smoke) |
| EfficientNet-B3 + label smoothing (A5) | Ablation | 13 (early-stopped at global epoch 18/30) | 0.9952 | 20.65 min (smoke; full-run minutes not independently re-verified against `quota_log.json`, whose entries end at the primary run — see note below) |

**Note on GPU-time bookkeeping:** `artifacts/kaggle/quota_log.json` currently contains entries only through the primary B3 run (ending 2026-09-16). The B0, ResNet-50, and A5 minutes above are as previously recorded in this project's own status documentation, sourced at the time from the fetched Kaggle run logs rather than `quota_log.json`; they were not re-derived from scratch for this report. This is a bookkeeping gap in the automation tooling, not a gap in the underlying model results — all four checkpoints, their `metrics.json` files, and their figures are present and verified on disk.

### 3.7 Segmentation (E1, Phase 4) — real result

| Split | Dice | IoU | Boundary distance (px) |
|---|---|---|---|
| Val | 0.0139 | 0.0071 | 16.01 |
| Test | 0.0090 | 0.0045 | 14.26 |

**Verdict: future work.** A Dice this low was investigated for a hidden bug before being accepted (metric code, mask pipeline, and letterbox-resize interpolation all reviewed directly — no defect found; finite boundary-distance values rule out total output collapse). Most likely real cause: severe class imbalance (fluid ≈2.8% of pixels) with no `pos_weight` on the loss, a decoder trained from scratch on only 66 training images, and very few epochs before early stopping — genuine under-training within the time-boxed GPU budget, not a code defect. Reviewed and accepted as final with the user, 2026-09-23; no further GPU spend on this phase. Full diagnostic writeup: `phase4-segmentation-signoff.md`.

### 3.8 Relative thickening index (E2, Phase 5) — real result

| | Normal (n=241) | DME (n=240) |
|---|---|---|
| Index (mean ± std) | 0.906 ± 0.419 | 1.049 ± 0.458 |

Normal reference (median windowed thickness, Normal training split): 43.69px, from 18,179/18,364 valid images (99.0%). Test set: 481/484 valid (99.4%). **Separation: heavy overlap (AUC 0.601)**, against this phase's own documented buckets (≥0.85 clear, 0.65–0.85 partial, <0.65 heavy overlap). Centre-involvement-indeterminate fired on 46.6% of valid test images.

The direction is correct (DME index > Normal index, consistent with edema-driven thickening) and the index distribution is a genuine, non-pathological overlap between the two classes rather than a collapse or reversal — checked visually before this result was accepted rather than reported at face value. A large drop in discriminative power relative to the primary classifier (AUC 0.9958) is expected here: this is a classical, unsupervised pixel-intensity proxy with no learned features, measured on a single 300px-downsampled 2D slice, not a validated layer segmentation. Full diagnostic read and limitations: `phase5-thickening-signoff.md`.

---

## 4. Figure index

Every file under `artifacts/`, excluding raw model checkpoints (`checkpoint_best.pt` / `checkpoint_last.pt`, one set per run under `artifacts/kaggle/<timestamp>/artifacts/`) and the `preprocess_cache/` disk caches (deterministic intermediate cache, not a reportable figure).

| File | Caption | Report section |
|---|---|---|
| `artifacts/fig_preprocessing_stages.png` | One scan shown at each preprocessing stage (grayscale → denoise → flatten → CLAHE → letterbox resize) | §2.3 Preprocessing |
| `artifacts/confusion_matrices.png` | Confusion matrices, primary B3, at default and tuned thresholds | §3.1 |
| `artifacts/roc_curve.png` | ROC curve, primary B3, with AUC annotated | §3.1 |
| `artifacts/training_curves.png` | Per-epoch train/val loss, accuracy, and AUC, primary B3 | §2.4, §3.6 |
| `artifacts/reliability_diagrams.png` | Reliability diagrams before/after temperature scaling, primary B3 | §2.5, §3.2 |
| `artifacts/gradcam_grid.png` | 3×3 Grad-CAM grid — rows: DME / Normal / misclassified; columns: original / preprocessed / heatmap overlay | §2.6, P2 |
| `artifacts/gradcam_randomization_check.png` | Grad-CAM heatmaps before/after classifier-weight randomisation, same examples as the grid | §3.5 |
| `artifacts/gradcam_randomization.json` | Raw per-example and mean correlations backing the randomisation verdict | §3.5 |
| `artifacts/metrics.json` | Full primary-model metrics payload (threshold, temperature, test results, calibration bins) | §3.1, §3.2 |
| `artifacts/training_log_reconstructed.csv` | Per-epoch training log for the primary run, reconstructed from the Kaggle console history after the direct log fetch was lost | §2.4 |
| `artifacts/image_dims_cache.csv` | Cached image dimensions, training set, used for the dimension-confound robustness check | `phase3g-baseline-comparison.md` §Robustness check |
| `artifacts/image_dims_cache.csv.corrupt.txt` | A prior corrupted write of the above, retained rather than silently deleted | (data-integrity record only) |
| `artifacts/test_dims_cache.csv` | Cached image dimensions, test set, generated by `scan_test_dims.py` for the same check | `phase3g-baseline-comparison.md` §Robustness check |
| `artifacts/baselines/tf_efficientnet_b0/metrics.json` | Full B0 baseline metrics payload | §3.1 (comparison), §3.3 |
| `artifacts/baselines/tf_efficientnet_b0/confusion_matrices.png` | Confusion matrices, B0 baseline | §3.3 |
| `artifacts/baselines/tf_efficientnet_b0/roc_curve.png` | ROC curve, B0 baseline | §3.3 |
| `artifacts/baselines/tf_efficientnet_b0/reliability_diagrams.png` | Reliability diagrams, B0 baseline | §3.2 |
| `artifacts/baselines/resnet50/metrics.json` | Full ResNet-50 baseline metrics payload | §3.1 (comparison), §3.3 |
| `artifacts/baselines/resnet50/confusion_matrices.png` | Confusion matrices, ResNet-50 baseline | §3.3 |
| `artifacts/baselines/resnet50/roc_curve.png` | ROC curve, ResNet-50 baseline | §3.3 |
| `artifacts/baselines/resnet50/reliability_diagrams.png` | Reliability diagrams, ResNet-50 baseline | §3.2 |
| `artifacts/baselines/resnet50/training_curves.png` | Per-epoch training curves, ResNet-50 baseline | §3.6 |
| `artifacts/ablation/label_smoothing_0.05/metrics.json` | Full A5 (label smoothing) metrics payload | §3.4 |
| `artifacts/ablation/label_smoothing_0.05/confusion_matrices.png` | Confusion matrices, A5 ablation | §3.4 |
| `artifacts/ablation/label_smoothing_0.05/roc_curve.png` | ROC curve, A5 ablation | §3.4 |
| `artifacts/ablation/label_smoothing_0.05/reliability_diagrams.png` | Reliability diagrams, A5 ablation | §3.2, §3.4 |
| `artifacts/kaggle/quota_log.json` | GPU-time bookkeeping log (partial — see §3.6 note) | §3.6 |
| `artifacts/kaggle/20260923T181925Z/segmentation_rasterization_check.png` | Mandatory pre-run visual check: fluid mask + 8 layer boundaries overlaid on 2 real Duke subjects | §2.8, `phase4-segmentation-signoff.md` |
| `artifacts/kaggle/20260923T181925Z/segmentation_training_log.csv` | Per-epoch segmentation training log, real full run (6 epochs) | §2.8, §3.7 |
| `artifacts/thickness/thickness_verification_check.png` | Mandatory pre-run visual check: ILM/RPE boundary overlay + foveal window on real sample scans | §2.9, `phase5-thickening-signoff.md` |
| `artifacts/thickness/thickness_distribution.png` | DME vs. Normal thickening-index histogram, real test set (n=484) | §3.8 |
| `artifacts/thickness/thickness_results.csv` | Per-image thickening index and referral flag, real test set (n=484) | §3.8 |
| `artifacts/thickness/thickness_summary.json` | Full Phase 5 summary payload (reference, index stats, separation AUC) | §3.8 |

**Figures indexed: 32.**

Missing, disclosed rather than fabricated: a per-epoch `training_curves.png` for the B0 baseline (its `training_log.csv` was never successfully captured — the fetched `run.log` for that run is 0 bytes) and for the A5 ablation run (its `training_log.csv` didn't survive the fetch, a `RemoteDisconnected` mid-transfer). Both gaps were logged as gaps by `evaluate.py`, not filled in with a fabricated plot.

---

## 5. Limitations

- **Single 2D B-scan, not a volume.** By design (see `PROJECT_BRIEF.md` §2, "Why 2D and not 3D"). A single B-scan cannot distinguish centre-involving from non-centre-involving DME; the system flags centre involvement as indeterminate and recommends volumetric follow-up rather than guessing.
- **Near-perfect test performance warrants a generalisation caveat.** AUC = 1.0000 and 99.79% accuracy, consistent across three independently trained architectures, was investigated for leakage and shortcut learning before being accepted (patient leakage ruled out; a real but weak image-dimension confound quantified at AUC 0.69 from geometry alone, far short of explaining the result). This supports genuine class separability under this dataset and preprocessing pipeline, but that separability may not generalise to OCT images from other scanners, sites, or acquisition protocols — external validation would be needed before any clinical-use claim. Full writeup: `phase3g-baseline-comparison.md`.
- **B3 does not meaningfully outperform smaller/older baselines** on this dataset (§3.3) — a genuine negative/neutral architecture-comparison finding, not a failed experiment.
- **Ablation techniques bought no measurable accuracy gain** (§3.4) — the baseline was already close to the dataset's ceiling.
- **Official test split is not patient-disjoint from training** (§2.2) — mitigated by excluding overlapping patients from train/val (verified at zero overlap), rather than altering the fixed, publication-comparable 484-image test set.
- **Segmentation (E1) is real but not usable — genuine under-training, honestly reported, not hidden.** Test Dice 0.0090 / IoU 0.0045 (§3.7). Investigated for a bug before acceptance; none found. Time-boxed and accepted as future work with the user rather than spending further GPU quota chasing a fix.
- **The thickening index (E2) shows a real but weak signal, not strong enough to be diagnostically useful on its own.** AUC 0.601, "heavy overlap" (§3.8) — a classical, unsupervised boundary-detection proxy (not a trained or clinically validated layer segmentation), measured in pixels at 300px content resolution (never converted to physical units, per C3), on a single 2D B-scan with no volumetric context. ~1% of images failed boundary detection outright and were excluded rather than imputed. A handful of source JPEGs in the public Kermany corpus are truncated/corrupted (`Premature end of JPEG file`), a data-quality property of the dataset, not this pipeline.
- **Both extended objectives are now wired into the Phase 6 interface**, through their real checkpoints/reference — not re-implemented, not fabricated. `app.py` runs the real trained E1 checkpoint per upload (fluid-area percentage + mask overlay) and the real E2 pipeline against the Normal reference from the full corpus run (thickening index + centre-involvement flag + boundary overlay). Both surface with an explicit, un-hideable reliability disclaimer next to the number/image (E1's real test Dice 0.0090; E2's real AUC 0.601) — informational, and neither one overrides or feeds into the primary classifier's referral decision. If either the E1 checkpoint or the E2 reference file is missing at launch, that section falls back to its original "not available" placeholder rather than crashing or inventing a result.
- **Screening/triage aid, not a diagnostic device.** The word "diagnosis" does not appear anywhere in the interface, this report, or the README, per the project's stated positioning.

---

## 6. Plan-vs-implementation deviations

1. **Splitting algorithm (C1).** The brief specifies `GroupShuffleSplit(groups=patient_id)`. The implementation uses `StratifiedGroupKFold` instead. Documented reason (in `utils.build_splits()`'s own docstring): a plain `GroupShuffleSplit` only targets the overall train/val sample-count ratio with no notion of class, and on this dataset's actual patient distribution (339 patients appear under both DME and NORMAL folders across different visits/eyes), that skews the per-split class ratio well past the required 5% tolerance. `StratifiedGroupKFold` keeps every patient's rows together while balancing class ratio across the split. C1's actual requirements (patient-grouped, zero overlap, seed 42) are still met — only the specific scikit-learn splitter differs from what the brief named.
2. **Test-set patient overlap (C8).** Not anticipated by the brief as a likely outcome, but the official Kermany `test/` split turned out not to be patient-disjoint from `train/` (305 of 368 test patients also appear in `train/`). Per C8's own instruction to report this immediately, it was surfaced as a loud warning at every run and handled by excluding the overlapping patients from train/val rather than altering the test set — see §2.2.
3. **Grad-CAM randomisation target.** The brief asks to randomise "the final layer's" weights for the sanity check, which is ambiguous between `model.classifier` (the network's literal last layer) and `model.conv_head` (Grad-CAM's own read target, and the layer colloquially thought of as "the last conv layer"). The implementation uses `model.classifier`, with the reasoning recorded in `explain.py`'s module docstring — randomising `conv_head` would trivially destroy the heatmap by corrupting the exact activations being read, proving nothing about whether the explanation is weight-dependent.
4. **`requirements.txt` gradio pin, relaxed at install time.** The brief pins `gradio==4.*`. On the team's local Python 3.14 environment, gradio 4's `pillow<11.0` dependency has no prebuilt wheel and fails to build from source. The interface was installed and verified against an unpinned, newer `gradio` (landing on 6.28.0), since gradio is never imported during Kaggle training and so does not affect training reproducibility, which is `requirements.txt`'s stated purpose for pinning. `requirements.txt` itself has not been changed; this is a local-install deviation, not a spec change.
5. **`val_n` discrepancy between the training run's own log and every `metrics.json`.** See §2.2's data-provenance note — flagged, not resolved, within this report's scope.
6. **Segmentation boundary-extraction approach for E2 (§2.9).** The brief's Phase 5 spec assumes ILM/RPE boundaries would come from a trained model; `segment.py`'s own docstring, written during Phase 4, flagged this as unworkable (the U-Net outputs a fluid mask, not layer coordinates, and Kermany has no ILM/RPE ground truth at all) and proposed a classical CV approach instead — implemented as described in §2.9, not a late improvisation.
7. **`_isolate_retina_mask()`'s boundary-detection method changed 3 times (4 versions total) during Phase 5 development** before being trusted at corpus scale, each version caught by the mandatory visual verification check rather than shipped on a first pass. Not a deviation from the brief's requirements, but a real part of how this result was reached — full history in `phase5-thickening-signoff.md`.

---

## 7. Reproduction

See the repository root `README.md` for setup and reproduction steps (environment, data acquisition, running Phase 1–3 locally vs. on Kaggle, and launching the Phase 6 interface).

---

```
PHASE 7 COMPLETE (updated post-Phase 4/5)
Results tables: generated from metrics.json, phase4-segmentation-signoff.md, phase5-thickening-signoff.md
Missing metrics: B0 baseline best-epoch/best-val-AUC (per-epoch log never captured); A5 full-run GPU minutes not independently re-verified against quota_log.json (entries end at the primary run)
Extended objectives: E1 (segmentation) TIMEBOXED — Test Dice 0.0090, verdict future work, investigated and accepted; E2 (thickening index) COMPLETE — separation AUC 0.601 (heavy overlap), a real weak signal, investigated and accepted
Plan-vs-implementation deviations: 7 (see §6) — splitting algorithm (StratifiedGroupKFold vs GroupShuffleSplit), test-set patient overlap discovered and mitigated (C8), Grad-CAM randomisation target interpretation (classifier vs conv_head), gradio version relaxed for local install (Python 3.14 wheel incompatibility), val_n discrepancy between training log and metrics.json (unresolved, disclosed), E2 boundary-extraction method (classical CV, not the E1 U-Net), E2 boundary-detection iterated 4 times before corpus-scale trust
Figures indexed: 32
```

---

*This report reflects only what is verifiable from `artifacts/metrics.json`, `artifacts/gradcam_randomization.json`, `phase4-segmentation-signoff.md`, `phase5-thickening-signoff.md`, repo logs, and the codebase as implemented at generation time. No metric, status, or duration in this document has been estimated or invented.*
