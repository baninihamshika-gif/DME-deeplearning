# Per-value ledger of every project result (script 17)

Every result value of this project not already covered by script 16: **3,185 values**, one row each in `research_audit/value_ledger_project.csv`.

- **RECOMPUTED**: recalculated independently from per-image predictions, the image hash table or per-image segmentation rows, and equal.
- **TRACED**: equal to its source; recalculating would repeat a seeded bootstrap or model inference (bootstrap CIs: the recomputed point estimate lies inside), or a report number matched to a data value.
- **SOURCE ONLY**: no saved inputs to recompute it from (listed below).
- **FAIL** / **UNTRACED**: mismatch, or a report number with no data value.

| Artifact | RECOMPUTED | SOURCE ONLY | TRACED | total |
|---|---|---|---|---|
| Results: Duke 2014 post hoc (central B-scan) | 52 | 0 | 0 | 52 |
| Results: Duke 2015 | 96 | 0 | 0 | 96 |
| Results: Grad-CAM localisation (Duke 2015) | 78 | 7 | 0 | 85 |
| Results: OCTDL patient level | 100 | 0 | 16 | 116 |
| Results: calibration | 36 | 8 | 8 | 52 |
| Results: dataset and leakage | 35 | 4 | 0 | 39 |
| Results: external validation | 536 | 0 | 176 | 712 |
| Results: official test (Table 2 data) | 300 | 0 | 48 | 348 |
| Results: paired comparisons | 15 | 0 | 0 | 15 |
| Results: segmentation audit (original U-Net) | 36 | 2 | 0 | 38 |
| Results: segmentation retrain v1, superseded | 103 | 35 | 0 | 138 |
| Results: segmentation retrain v2, current | 115 | 39 | 0 | 154 |
| Results: threshold/temperature refit | 92 | 0 | 8 | 100 |
| Results: unseen pool | 51 | 0 | 42 | 93 |
| Table 1 | 12 | 1 | 0 | 13 |
| final_audit_checklist.md (counts, %) | 0 | 2 | 51 | 53 |
| leakage_audit.md (counts, %) | 0 | 3 | 218 | 221 |
| metric_verification.md (counts, %) | 0 | 2 | 260 | 262 |
| repository_inventory.md (counts, %) | 0 | 6 | 41 | 47 |
| research_results_report.md (counts, %) | 2 | 5 | 359 | 366 |
| segmentation_retrain_report.md (counts, %) | 0 | 1 | 163 | 164 |
| value_validation.md (counts, %) | 0 | 2 | 19 | 21 |
| **All** | 1,659 | 117 | 1,409 | 3,185 |

## Values that exist only in their own output (SOURCE ONLY)

| Artifact | Why it cannot be recomputed | Values |
|---|---|---|
| Results: Grad-CAM localisation (Duke 2015) | needs the heatmaps of the re-initialised classifier (not saved) | 7 |
| Results: calibration | original evaluation output (metrics.json / run log) | 8 |
| Results: dataset and leakage | fixed setting (script 02) | 1 |
| Results: dataset and leakage | geometry-only logistic regression (script 02); no saved inputs to re-fit | 3 |
| Results: segmentation audit (original U-Net) | stored in the original checkpoint | 2 |
| Results: segmentation retrain v1, superseded | boundary distance needs the masks (not saved per image) | 8 |
| Results: segmentation retrain v1, superseded | needs probability maps (not saved per image) | 26 |
| Results: segmentation retrain v1, superseded | training log | 1 |
| Results: segmentation retrain v2, current | 320x320 app frame fractions need the masks | 4 |
| Results: segmentation retrain v2, current | boundary distance needs the masks (not saved per image) | 8 |
| Results: segmentation retrain v2, current | needs probability maps (not saved per image) | 26 |
| Results: segmentation retrain v2, current | training log | 1 |
| Table 1 | Table 1 row (other source) | 1 |
| final_audit_checklist.md (counts, %) | fixed sample size of the brute-force AUROC check (10_final_checks.py) | 1 |
| final_audit_checklist.md (counts, %) | pytest run output (tests/); re-run at each commit | 1 |
| leakage_audit.md (counts, %) | pre-exclusion validation size in the Kaggle training log (full_run_log.txt) | 2 |
| leakage_audit.md (counts, %) | pre-exclusion validation size of the local scikit-learn 1.9.0 split (evaluate.py run output) | 1 |
| metric_verification.md (counts, %) | pre-exclusion validation size in the Kaggle training log (full_run_log.txt) | 1 |
| metric_verification.md (counts, %) | pre-exclusion validation size of the local scikit-learn 1.9.0 split (evaluate.py run output) | 1 |
| repository_inventory.md (counts, %) | figure resolution setting (dpi) of the original scripts | 1 |
| repository_inventory.md (counts, %) | line number in full_run_log.txt | 1 |
| repository_inventory.md (counts, %) | numpy version pinned in requirements.txt | 1 |
| repository_inventory.md (counts, %) | pre-exclusion validation size in the Kaggle training log (full_run_log.txt) | 2 |
| repository_inventory.md (counts, %) | pre-exclusion validation size of the local scikit-learn 1.9.0 split (evaluate.py run output) | 1 |
| research_results_report.md (counts, %) | download size of 2014_BOE_Srinivasan.zip (MB), external file | 1 |
| research_results_report.md (counts, %) | size of the full OCT2017 release (4 classes; only DME and NORMAL are on disk), config.py | 2 |
| research_results_report.md (counts, %) | test images per class stated in Kermany et al. 2018 (published figure) | 1 |
| research_results_report.md (counts, %) | training-image count stated in Kermany et al. 2018 (published figure) | 1 |
| segmentation_retrain_report.md (counts, %) | v1 whole-frame pooled Dice from the diagnostic full_frame_check run (not saved in v1's summary) | 1 |
| value_validation.md (counts, %) | number of distinct §8.1 values counted by script 15 (its own output) | 1 |
| value_validation.md (counts, %) | pytest run output (tests/); re-run at each commit | 1 |

## Values needing attention

None: every value is RECOMPUTED, TRACED or SOURCE ONLY.
