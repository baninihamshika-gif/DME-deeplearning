# Phase 3g — Baseline Comparison (EfficientNet-B3 vs. B0 vs. ResNet-50)

**Status:** Complete. All three models evaluated on the same held-out test set (484 images, 242 DME / 242 Normal), same preprocessing, same patient-grouped splits, same evaluation script (`evaluate.py`).

## Results table

| Model | Role | Test Accuracy | Sensitivity | Specificity | AUC | Val AUC (best epoch) |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | Baseline | 99.79% | 99.59% | 100.0% | 1.0000 | — |
| ResNet-50 | Baseline | 99.79% | 99.59% | 100.0% | 1.0000 | 0.9961 |
| EfficientNet-B3 | Primary | 99.79% | 99.59% | 100.0% | 1.0000 | — |

Figures are at the default 0.5 decision threshold (n=484 test set: tp=241, fn=1, tn=242, fp=0 — identical confusion matrix for all three models). At their individual sensitivity-tuned thresholds (validation-set only, C-rule compliant) they diverge slightly:

| Model | Tuned threshold | Tuned sensitivity | Tuned specificity |
|---|---|---|---|
| EfficientNet-B0 | 0.608 | 99.17% | 100.0% |
| ResNet-50 | 0.628 | 99.59% | 100.0% |
| EfficientNet-B3 | 0.971 | 98.35% | 100.0% |

Checkpoints:
- B0: `artifacts/kaggle/20260920T092347Z/artifacts/checkpoint_best.pt`
- ResNet-50: `artifacts/kaggle/20260920T110729Z/artifacts/checkpoint_best.pt`
- B3 (primary): `artifacts/kaggle/20260917T035646Z/artifacts/checkpoint_best.pt`

## Finding: B3 does not meaningfully beat the baselines

Per the brief's own instruction ("if B3 doesn't meaningfully beat B0, say so plainly; that's a finding"): it doesn't. EfficientNet-B3's added size and complexity bought no measurable improvement over the much smaller EfficientNet-B0 or the older ResNet-50 architecture — all three match exactly on test accuracy and AUC at the default threshold. This should be reported as-is rather than downplayed; a negative/neutral architecture-comparison result is a legitimate finding, not a failed experiment.

## Robustness check: why is AUC = 1.0000 across three independent models?

A perfect AUC on first sight should be treated as a probable bug (per project rule C7), not celebrated, so this was investigated before being accepted:

1. **Patient leakage — ruled out.** `build_splits()` in `utils.py` was reviewed line by line. It correctly identifies and excludes the 305 patients whose images appear in both the official test split and the train/val pool (`StratifiedGroupKFold`, patient-grouped), confirmed by this run's own log: `train/val <-> test patient overlap: 305 (C8 VIOLATION)` → `excluded 305 overlapping patient(s) ... to keep the official 484-image test set clean`. Zero overlap after exclusion, every run.

2. **Image-dimension/border shortcut — present but insufficient to explain the result.** DME and Normal images differ somewhat in raw dimensions (e.g. in train, 89.6% of DME images are width=512 vs. 68.5% of Normal), which the aspect-ratio-preserving letterbox resize (C2) could in principle turn into a shortcut (models learning to read padding/border patterns instead of retinal content). A logistic regression trained only on letterbox-derived geometry features (width, height, aspect ratio, resized dims, padding) scored **AUC = 0.6902** on the real test set — a real but modest signal, far short of explaining a perfect result. A naive "predict DME if width==512" rule (direction chosen from train) actually scored **30.99%** test accuracy — worse than chance — because the width/class relationship *reverses* between train and test (DME test-set images are wider on average, not narrower). This confirms the dimension confound is real but weak and inconsistent, not a dominant shortcut.

3. **Cross-architecture agreement.** Three independently trained models (different architectures, different training runs, different checkpoints) converge on the identical test-set confusion matrix at the default threshold. If the perfect score were an artifact of one model exploiting a bug or shortcut, independent architectures would be expected to disagree at least slightly on which images they get wrong. Instead they agree exactly, which is more consistent with genuine near-complete class separability under this preprocessing pipeline than with a shared latent bug.

**Conclusion for the report's limitations section:**

> The near-perfect test performance (AUC = 1.0000, accuracy = 99.79%) observed consistently across three independently trained architectures (EfficientNet-B0, ResNet-50, EfficientNet-B3) was investigated for potential leakage or shortcut learning before being accepted, per project protocol. Patient-level leakage was ruled out by construction and verified in every run (zero train/val–test patient overlap after exclusion). A modest image-dimension confound was identified — geometric features alone (derived from the aspect-ratio-preserving resize) achieve AUC = 0.69 on the test set — but this is far too weak to account for the observed near-perfect scores, and a naive rule based on the dominant dimension pattern in training data actually performs worse than chance on the test set, since the pattern does not hold in the same direction on unseen data. The consistency of the result across three architecturally distinct models trained independently further supports genuine class separability, under this dataset and preprocessing pipeline, over a shared measurement artifact. This should nonetheless be disclosed as a limitation: the dataset's apparent separability may not generalize to OCT images from other scanners, sites, or acquisition protocols, and external validation would be needed before any clinical-use claim.

## Reproducibility notes

- Dimension-confound diagnostic used `artifacts/image_dims_cache.csv` (train, pre-existing) and `artifacts/test_dims_cache.csv` (test, generated via `scan_test_dims.py`, a standalone read-only PIL header scanner).
- All three baseline runs used identical `evaluate.py` invocation pattern (train-dir, test-dir, checkpoint, training-log-csv), auto-namespaced output to `artifacts/baselines/<model_name>/` for non-primary models.
