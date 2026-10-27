# Phase A — Repository Inventory

Audit branch: `research-audit` (created from `main` @ `a2ea4a5`; working tree was clean, no user changes at risk).
Audit date: 2026-10-27. Auditor: Claude Code (automated), at the project team's request.

Note: `artifacts/` and `data/` are git-ignored. Everything below about checkpoints, logs, metrics and data refers to the **local working copy** at `D:\Projects\DME-DEEPLEARN`, not the GitHub repository. A reader with only the GitHub repository cannot reproduce any number in this audit without the checkpoints and the Kermany/Duke data.

SHA-256 checksums of every checkpoint and `metrics.json` were recorded before any analysis (`research_audit/checksums_before.sha256`) and re-checked at the end (see `research_outputs/final_audit_checklist.md`).

## 1. Files inspected

| Area | Files |
|---|---|
| Documentation | `README.md`, `PROJECT_BRIEF.md`, `report/progress_report.md`, `report/phase3-ablation-and-signoff.md`, `report/phase3g-baseline-comparison.md`, `report/phase4-segmentation-signoff.md`, `report/phase5-thickening-signoff.md`, `report/phase6-gradio-signoff.md` |
| Configuration | `config.py`, `requirements.txt`, `.gitignore`, `kaggle/kernel-metadata.json`, every `artifacts/kaggle/<run>/TRAIN_ARGS.json` and `SMOKE_MODE` |
| Data / splitting / preprocessing | `utils.py`, `prepare_data.py`, `scan_test_dims.py` |
| Classification training | `train.py`, `kaggle/entry.py`, `kaggle/run.py` |
| Evaluation / calibration / threshold | `evaluate.py` |
| Explainability | `explain.py`, `artifacts/gradcam_randomization.json`, `artifacts/gradcam_grid.png`, `artifacts/gradcam_randomization_check.png` |
| Segmentation | `segment.py`, `kaggle/entry_segment.py`, `kaggle/run_segment.py`, `artifacts/kaggle/20260923T181925Z/**` |
| Thickness index | `thickness.py`, `artifacts/thickness/*` |
| Interface | `app.py` (skimmed; not part of the evaluation) |
| Logs | `full_run_log.txt`, `artifacts/kaggle/*/run.log`, `artifacts/kaggle/*/dme-oct-*.log`, `reconstruct_training_log.py` |
| Tests | `tests/*.py` (10 files) |
| Git history | 2 commits (`5e60ec8`, `a2ea4a5`). The history is too short to trace how the methodology changed over time. |

## 2. Files relevant to classification

| Role | File | Authoritative for |
|---|---|---|
| Architecture | `train.py::build_model`, `MODEL_CONFIGS` | `timm.create_model(<name>, pretrained=True, num_classes=2, drop_rate=0.3)` for `tf_efficientnet_b3`, `tf_efficientnet_b0`, `resnet50` |
| Class mapping | `config.CLASS_NAMES = ["Normal", "DME"]` | index 0 = Normal, index 1 = DME (positive class) |
| Split | `utils.build_splits` | StratifiedGroupKFold(5, shuffle, seed 42) grouped by patient ID parsed from filename; train/val patients that also appear in the official test set are excluded |
| Preprocessing | `utils.preprocess_image` | NLM denoise → flattening → CLAHE → aspect-preserving letterbox to 300×300 → 3-channel copy; ImageNet normalisation |
| Augmentation | `train.py::_train_augmentation` | horizontal flip, ShiftScaleRotate, brightness/contrast (training only) |
| Checkpoint selection | `train.py::run_phase` | best **validation AUC** (early stopping uses validation loss) |
| Threshold / temperature | `evaluate.py` | temperature fitted by NLL on "validation"; threshold swept on calibrated "validation" probabilities for sensitivity ≥ 0.97 |
| Primary checkpoint (B3) | `artifacts/kaggle/20260917T035646Z/artifacts/checkpoint_best.pt` | referenced by `artifacts/metrics.json` |
| B0 checkpoint | `artifacts/kaggle/20260920T092347Z/artifacts/checkpoint_best.pt` | referenced by `artifacts/baselines/tf_efficientnet_b0/metrics.json` |
| ResNet-50 checkpoint | `artifacts/kaggle/20260920T110729Z/artifacts/checkpoint_best.pt` | referenced by `artifacts/baselines/resnet50/metrics.json` |
| B3 + label smoothing (ablation) | `artifacts/kaggle/20260922T160023Z/artifacts/checkpoint_best.pt` | referenced by `artifacts/ablation/label_smoothing_0.05/metrics.json` |
| Test data | `data/raw/OCT2017/test/{DME,NORMAL}` | 242 + 242 images (counted on disk) |
| Training data | `data/raw/OCT2017/train/{DME,NORMAL}` | 11,348 + 26,315 images (counted on disk) |

## 3. Files relevant to segmentation

`segment.py` (U-Net, `smp.Unet(encoder="timm-efficientnet-b3")`, encoder initialised from the B3 classifier), `data/raw/duke_dme/2015_BOE_Chiu/Subject_01..10.mat` (Duke DME, Chiu et al. 2015), checkpoint `artifacts/kaggle/20260923T181925Z/artifacts/segmentation/checkpoint_best.pt`, log `.../segmentation_training_log.csv`, and `report/phase4-segmentation-signoff.md`. The repository saved **no predicted masks**.

## 4. Files containing predictions or evaluation results

| File | Content | Per-image predictions? |
|---|---|---|
| `artifacts/metrics.json` | B3 aggregate test metrics at 0.5 and 0.971, temperature, ECE and bins, TTA | **No** |
| `artifacts/baselines/{tf_efficientnet_b0,resnet50}/metrics.json` | same structure | **No** |
| `artifacts/ablation/label_smoothing_0.05/metrics.json` | same structure | **No** |
| `artifacts/thickness/thickness_results.csv` | per-image thickening index (E2) | Yes (E2 only) |
| `artifacts/gradcam_randomization.json` | Grad-CAM selection indices and correlations | n/a |
| PNG figures under `artifacts/` | rendered at 150 dpi from the above | n/a |

**No per-image logits, probabilities, or labels were ever saved for any classifier.** `evaluate.py` computes them in memory and writes only aggregates. Independent reconstruction therefore requires re-running inference with the saved checkpoints (Phase B does this without retraining).

## 5. Files containing training logs

| Model | Log | Status |
|---|---|---|
| B3 (primary) | `full_run_log.txt` (Kaggle console stream); `artifacts/training_log_reconstructed.csv` (parsed from it by `reconstruct_training_log.py`) | Authentic per-epoch train/val loss and val AUC. **No accuracy** was logged in this run's console output, and the direct `training_log.csv` fetch was lost. |
| ResNet-50 | `artifacts/kaggle/20260920T110729Z/artifacts/training_log.csv`, `run.log` | Complete: train/val loss, train/val accuracy, val AUC, LR, 30 epochs |
| B0 | `artifacts/kaggle/20260920T092347Z/run.log` is **0 bytes**; no `training_log.csv` | **Missing.** Best epoch, val AUC and the test-patient exclusion cannot be confirmed from a log. |
| B3 + LS 0.05 | `artifacts/kaggle/20260922T160023Z/run.log` | Console log present; `training_log.csv` lost |
| U-Net | `.../segmentation/segmentation_training_log.csv`, `dme-oct-segment.log` | 6 epochs, present |

## 6. Files containing research references

None. The repository has no bibliography, `.bib` file, or literature review. `README.md` cites Kermany et al. (Cell 2018) and Chiu et al. (BOE 2015) by name only. Neither of the two comparator papers named in the task (Diabetes Care 2021, BMJ Open 2025) is referenced anywhere in the repository.

## 7. Missing artifacts required for rigorous evaluation

1. **Per-image test predictions** for every classifier. Recovered in Phase B by re-running inference; these are new outputs, not saved originals.
2. **The exact training-time validation split** (no manifest saved). It cannot be regenerated locally; see conflict C1.
3. **A saved split manifest** for train/val/test (`data/splits/` exists but is empty).
4. **The B0 training log** (0-byte `run.log`).
5. **The B3 `training_log.csv`.** Only a console-parsed reconstruction exists, without accuracy.
6. **Predicted segmentation masks.** These are regenerated in Phase B from the checkpoint.
7. **Any external test set.** The only test set is the official Kermany OCT2017 test split.
8. **Eye identifiers, device metadata and acquisition dates** for Kermany. The filename gives only a patient ID and an image index.

## 8. Conflicts between documentation, logs, metrics and code

| # | Conflict | Evidence | Resolution in this audit |
|---|---|---|---|
| C1 | **Validation set size: 5,560 (training log) vs 6,047 (every `metrics.json`).** | `full_run_log.txt` line 83 and the ResNet-50 and A5 run logs give "val 6997->5560". `metrics.json` gives `val_n: 6047`. The project's own report flags this as unresolved (§2.2, §6.5). | **Explained.** The Kaggle runs used scikit-learn **1.4.2** (`full_run_log.txt` line 30 pip messages). Running the identical `build_splits()` locally (scikit-learn 1.9.0) gives pre-exclusion val **7,535** (vs 6,997 on Kaggle) and post-exclusion val **6,047**, exactly the `metrics.json` value. `StratifiedGroupKFold` therefore produced a **different patient partition** locally. The "validation" set used in `evaluate.py` to fit the temperature and choose the tuned threshold is not the training-time validation set, and almost certainly contains patients the models were trained on. Consequence: the **tuned thresholds (0.971, 0.608, 0.628, 0.89) and temperatures are fitted on data that overlaps training**. Test-set results at the fixed 0.5 threshold, and the AUROC, are unaffected. See `leakage_audit.md` §5. |
| C2 | README claims "patient leakage ruled out". | The training-time exclusion is real and logged for B3, ResNet-50 and A5. The official test split has 305/368 patients shared with `train/`. | Train↔test patient leakage is excluded by construction for the logged runs. For B0 it is **not verifiable from logs** (0-byte log), though `kaggle/entry.py` always passes `--test-dir`, so the code path implies exclusion. Validation↔training contamination at evaluation time (C1) was not previously known. |
| C3 | README: "Calibrated via temperature scaling (ECE 0.0028 → 0.0027)" is presented as a calibration improvement. | The change is 0.0002 on n = 484, the temperature was fitted on contaminated validation data (C1), and for B0/ResNet-50 ECE got slightly worse. | Report as "no material change in calibration". |
| C4 | Report §3.3: the three architectures are "independent" and "converge on the identical confusion matrix". | All three have 241/1/242/0 at 0.5. | Confirmed in Phase B (same single false negative image). Agreement across architectures does not by itself rule out a shared data artifact, because all three use identical data, preprocessing and split. |
| C5 | Phase-4 sign-off: "Dice ~0.01 is close to what a model predicting almost no foreground would score". | Under `dice_coefficient` with eps = 1e-7, an empty prediction scores **0** on fluid-positive images and **1** on empty-GT images. A Dice near 0.01 is the signature of predicting **almost everything as foreground**. | Investigated in Phase B4 by regenerating masks. |
| C6 | `binarize_fluid()` sets NaN (ungraded) pixels to background. | `segment.py:211`. In Duke, manual grading covers only part of each B-scan. | Ungraded regions are scored as true negatives or false positives. This is a label-definition defect that affects Dice/IoU. Documented in Phase B4. |
| C7 | README results table lists AUC "1.0000" for both thresholds. | AUROC is threshold-independent. Only one AUROC exists per model. | Clarified in the tables. |
| C8 | `requirements.txt` pins torch 2.3 / sklearn 1.4 / numpy 1.26. The local environment has torch 2.14 (CPU), sklearn 1.9, numpy 2.4 and Python 3.14. | `pip` versions printed in Phase B. | Re-inference is run in the local environment. Small floating-point differences from the original Kaggle/local evaluation are possible and are checked against `metrics.json`. |

## 9. Secrets

`config.py` contains the Kaggle **username** (not a credential). No API keys, tokens or passwords were found in tracked files. `.gitignore` excludes `kaggle.json` and `access_token`. None were printed into these reports.
