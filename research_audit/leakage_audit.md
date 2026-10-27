# Phase B3 — Leakage and Shortcut Audit

**Question:** is the reported test performance (accuracy 99.79%, AUROC 1.0000 for all three architectures) an artifact of leakage, a shortcut, or an evaluation bug?

**Short answer:** No train↔test leakage, duplication, label-encoding shortcut or evaluation bug was found that could explain the test result. The perfect AUROC is real for these checkpoints on this test set. Two genuine problems were found, though:

1. The original tuned thresholds and temperatures were fitted on a validation set that overlaps the training data (84.3% of its images were training images). They have now been refit on the rebuilt training-time validation split (§5, §5.1).
2. The official 484-image test set is **systematically different from, and apparently easier than,** the training distribution (§4, §6). The test result therefore probably overstates performance on unselected Kermany images, let alone on external data.

Evidence files: `research_outputs/data/leakage_summary.json`, `image_hashes.csv`, `near_duplicate_pairs.csv`, `dimension_by_class.csv`, `border_artifact_by_class.csv`, `test_predictions_*.csv`, `supplementary_excluded_pool_performance.csv`. Scripts: `research_outputs/scripts/02_*`, `03_*`, `06_*`.

## 1. Checklist of the 14 candidate causes

| # | Candidate | Finding | Evidence |
|---|---|---|---|
| 1 | Train/test leakage | **Not found.** 305 of the 368 test patients also appear in `train/`, and 114 test images are byte-identical copies of `train/` images from those patients. Every image of these patients is removed from train and validation before training. After removal, patient overlap = **0** and exact-duplicate pairs test↔training pool = **0**. | `leakage_summary.json`; `full_run_log.txt:83`; ResNet-50 and A5 run logs |
| 2 | Patient-level overlap | **0 after exclusion.** This is verified from logs for B3, ResNet-50 and A5. For B0 it is implied by code (`kaggle/entry.py` always passes `--test-dir`), but the B0 log is empty. Eye-level identity is not available. | as above |
| 3 | Duplicate or near-duplicate images | MD5: 0 test↔pool pairs, 0 within test. dHash (64-bit, stored as exact hex, Hamming ≤ 4): **5 test↔pool candidate pairs, 0 within test, 0 with conflicting labels**, all Normal–Normal. **Visual inspection: all 5 are visibly different scans** (different foveal contour, tilt, vessel shadows and noise; generic retinal shape gives false hash matches). Without eye IDs, two scans of one eye filed under different patient numbers cannot be excluded. All 37,663 + 484 images are hashed; 25 truncated JPEGs that `cv2.imdecode` rejects are decoded with `cv2.imread`, the decoder the training pipeline uses. Within the training pool there are 365 exact-duplicate pairs, 9 of them with **conflicting labels** (minor label noise in training). *Correction (2026-10-27): an earlier version of this audit reported 10 candidate pairs (3 cross-label). That count came from hashes stored as float64, which cannot hold 64 bits exactly, so Hamming distances were wrong and changed between reruns. Fixed in `02_leakage_scan.py`; the rerun is deterministic.* | `near_duplicate_pairs.csv`; visual contact sheets |
| 4 | Same OCT volume split across sets | Volume IDs do not exist in Kermany. Patient-level grouping is the strongest grouping available, and it is enforced. | filenames |
| 5 | Labels encoded in filenames, directories, borders or overlays | The class name is in the filename and folder, but the model sees pixels only; `OCTDataset` reads labels from the dataframe, not from the image. White wedge-shaped corner fill (rotation artifact) is present but **balanced by class in the training pool** (Normal 59.6%, DME 58.4%; AUROC of the flag alone 0.49), so it cannot be a learned shortcut. No text overlays seen. | `border_artifact_by_class.csv` |
| 6 | Device or dimension shortcut | DME and Normal differ in dimensions in training (e.g. 74.6% of DME vs 24.7% of Normal are 512×512). A logistic model on geometry alone (width, height, aspect, letterbox size and padding), trained on the pool, scores **AUROC 0.72 on unseen excluded-patient images but 0.31 on the official test set** (below chance, because the relationship reverses). The official test set contains **no 512×512 images at all**. A dimension shortcut cannot produce a perfect score on a test set where the cue points the wrong way. | `dimension_by_class.csv`; `leakage_summary.json` |
| 7 | Augmented copies crossing splits | Augmentation is applied on the fly to training images only; nothing augmented is written to disk. | `train.py::OCTDataset` |
| 8 | Preprocessing using test labels | Preprocessing is deterministic and per-image; no label input. The cache key is path + size + flag only. | `utils.PreprocessCache` |
| 9 | Wrong class-index mapping | Not found. Index 1 = DME is asserted at train time and checked again here. A flipped mapping would give AUROC ≈ 0. | `metric_verification.md` B1 |
| 10 | Evaluated on train or val instead of test | Not found. `load_class_folder_df(test_dir)` asserts 242/242. The 484 re-inferred files are exactly `test/`. | `01_reinfer_classifiers.py` |
| 11 | Checkpoint selection on test | Not found. Selection uses validation AUROC; the test set is not touched in `train.py`. | `train.py::run_phase` |
| 12 | Labels used as predictions or similar bug | Not found. Predictions come from a fresh forward pass, and scores are continuous and graded (P(DME) for DME ranges 0.03–1.0). | `test_predictions_*.csv` |
| 13 | Threshold selected on test | Not found on test. The original threshold was selected on a contaminated validation set (§5); refit on the training-time validation split (§5.1). | `evaluate.py`; script 12 |
| 14 | Calibration or model selection using test | Not on test. The original calibration used the same contaminated validation set (§5); refit on the training-time validation split (§5.1). | `evaluate.py`; script 12 |

## 2. Test-set composition

242 DME / 242 Normal from 368 patients (median 1 image per patient, maximum 13). 6 patients have images in both classes. The test set is artificially balanced; prevalence in a screening population would be far lower, so the measured PPV (1.000) **does not transfer** to clinical prevalence.

## 3. Cross-architecture agreement

The phase-3g report treats identical confusion matrices across B3, B0 and ResNet-50 as evidence against a shared artifact. At the image level, **B3 misses a different image** (DME-8366524-1) from B0 and ResNet-50 (DME-6151003-1). And because all three models share the same data, preprocessing and split, agreement between them is not evidence against a dataset-level artifact in any case.

## 4. The official test set is drawn from a different distribution than the training data

| Property | Training pool DME | Training pool Normal | Official test DME | Official test Normal |
|---|---|---|---|---|
| 512×512 | 74.6% | 24.7% | **0%** | **0%** |
| 512×496 | 14.3% | 43.0% | 55.8% | 93.8% |
| 768×496 | 7.0% | 23.6% | 43.8% | 6.2% |
| White-corner artifact | 58.4% | 59.6% | 58.3% | **82.6%** |

(`dimension_by_class.csv`, `border_artifact_by_class.csv`)

The 305 excluded patients' own `train/` images look like the training pool (79% of their DME images are 512×512), not like their test images. The test images are therefore a **separately selected subset**, not a random sample of the same pool. This matches the known history of the Kermany test split, which the dataset authors assembled separately. That history is from prior knowledge and is not re-verified in this audit.

The models' validation AUROC on patient-disjoint validation data (0.995–0.996, from the training logs) is lower than their test AUROC (1.000). This is consistent with an easier test set.

## 5. Validation-set contamination at evaluation time (new finding)

- Training (Kaggle, scikit-learn **1.4.2**): `StratifiedGroupKFold` gives val 6,997 → **5,560** after exclusion.
- Evaluation (local, scikit-learn 1.9.0, same seed and code): val 7,535 → **6,047**.

The two partitions differ. `evaluate.py` fits the temperature and chooses the "tuned" threshold on the 6,047-image local partition.

**Measured overlap (after rebuilding the Kaggle split, see below):** 5,096 of the 6,047 local "validation" images (**84.3%**) and 553 of its 676 patients were in the training-time **training** folds; only 951 images were genuinely held out (`research_outputs/data/val_split_contamination.json`).

Consequences:

- The original tuned thresholds (B3 0.971, B0 0.608, ResNet-50 0.628, B3+LS 0.890) and their "validation sensitivity ≥ 0.97" are biased and are **superseded** (kept in Table 2 for reconciliation only).
- The original temperatures were fitted on mostly-seen data, where the models are over-confident, so all four were < 1 or ≈ 1 (0.88–1.02; LS 0.48).
- Test results at 0.5 and test AUROC are **unaffected** (the test set is not involved).

### 5.1 Fix: training-time split rebuilt, threshold and temperature refit (no retraining)

- `research_outputs/scripts/11_rebuild_training_split.py` runs the project's unmodified `utils.build_splits()` under Python 3.12 + scikit-learn 1.4.2 (isolated uv environment). It reproduces the Kaggle log exactly: 305 patients / 7,503 images excluded, train 30,666 → **24,600**, val 6,997 → **5,560** (1,254 DME / 4,306 Normal; 672 patients). Manifest: `research_outputs/data/training_split_manifest.csv`.
- Cross-check: val AUROC recomputed on the rebuilt split matches each checkpoint's stored training-time `val_auc` to within 7×10⁻⁵ (B3 0.99575 vs 0.99582; B0 0.99499 vs 0.99501; ResNet-50 0.99607 vs 0.99610; LS 0.99514 vs 0.99520). The small residual is plausibly CPU vs GPU numerics (not investigated further); the image counts match exactly.
- `research_outputs/scripts/12_refit_calibration_threshold.py` scores those 5,560 images with the read-only checkpoints and calls `evaluate.fit_temperature` and `evaluate.sweep_threshold` (target sensitivity 0.97) unchanged. Nothing is written back to any checkpoint. Results: `research_outputs/data/refit_calibration_threshold.csv`.

| Model | T old → refit | Threshold old → refit | Val sens / spec at refit | Test TP / FP / TN / FN at refit |
|---|---|---|---|---|
| EfficientNet-B3 | 0.917 → 1.549 | 0.971 → 0.181 | 0.9705 / 0.9666 | 242 / 0 / 242 / 0 |
| EfficientNet-B0 | 1.019 → 1.557 | 0.608 → 0.121 | 0.9705 / 0.9626 | 242 / 0 / 242 / 0 |
| ResNet-50 | 0.881 → 1.121 | 0.628 → 0.219 | 0.9705 / 0.9821 | 241 / 0 / 242 / 1 |
| B3 + LS 0.05 | 0.479 → 0.625 | 0.890 → 0.096 | 0.9705 / 0.9561 | 242 / 0 / 242 / 0 |

Interpretation: on genuinely held-out validation patients the models are less confident, so the refit temperatures are larger and the thresholds needed for 97% sensitivity are much lower. The refit operating points are valid (fitted without test data, applied to the test set once), but the perfect test counts mainly restate that the test set is easier than the validation data (§4, §6). The validation specificity at 97% sensitivity (0.956–0.982) is the more realistic estimate of this operating point. The one remaining ResNet-50 miss is DME-6151003-1 (refit P(DME) 0.043).

## 6. Supplementary evaluation on images never used by any model (new analysis)

These are the 7,503 `train/` images belonging to the 305 patients who also appear in the official test set. `build_splits()` removed them from both training and validation, so no model was fitted or selected on them (verified from logs for B3 and ResNet-50; implied by code for B0). They come from the same patients as the test set, but they are **different images** (114 are byte-identical copies of test images). They are reported **separately** with **patient-clustered** CIs and are never pooled with the official test set. All 7,503 images were readable (4,276 DME / 3,227 Normal; 305 patients). Threshold 0.5, raw probabilities. CIs: patient-cluster percentile bootstrap (B = 2,000, seed 42). Source: `research_outputs/data/supplementary_excluded_pool_performance.csv`.

| Model | TP / FP / TN / FN | Accuracy (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | F1 (95% CI) | AUROC (95% CI) |
|---|---|---|---|---|---|---|
| EfficientNet-B3 | 4124 / 55 / 3172 / 152 | 0.9724 (0.9636–0.9801) | 0.9645 (0.9503–0.9770) | 0.9830 (0.9756–0.9896) | 0.9755 (0.9674–0.9823) | 0.9964 (0.9946–0.9978) |
| EfficientNet-B0 | 4094 / 34 / 3193 / 182 | 0.9712 (0.9624–0.9794) | 0.9574 (0.9437–0.9702) | 0.9895 (0.9837–0.9945) | 0.9743 (0.9664–0.9813) | 0.9966 (0.9949–0.9978) |
| ResNet-50 | 4128 / 38 / 3189 / 148 | 0.9752 (0.9664–0.9831) | 0.9654 (0.9516–0.9785) | 0.9882 (0.9828–0.9932) | 0.9780 (0.9705–0.9847) | 0.9970 (0.9954–0.9982) |

On unseen images from the same dataset, all three models **make errors at a rate of about 2.5–2.9%**, and AUROC is **0.996–0.997, not 1.000**. This agrees with the training-time patient-disjoint validation AUROC (0.995–0.996), and it supports §4: the official 484-image test set is easier than typical Kermany images. The perfect test score should be presented together with this more conservative internal estimate. **Neither figure is an external validation.** The architectures again overlap completely within their CIs.

## 7. What would be needed to settle generalisation

1. **External test data** from another site or device (e.g. a public DME dataset not derived from Kermany), with patient-level labels.
2. The Kaggle-time split manifest (see §5), to make thresholds and calibration valid.
3. Prospective or consecutive (unselected) images, to estimate performance at realistic prevalence.

Nothing in this audit retrained a model or changed any checkpoint.
