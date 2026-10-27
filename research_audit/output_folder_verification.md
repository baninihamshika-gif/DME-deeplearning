# Verification of research_outputs/ and research_audit/ (script 18)

111 files checked (the preprocessing cache `research_outputs/.cache/` is excluded). Per-value checks: scripts 15–17.

| Check | Result | Detail |
|---|---|---|
| Every file opens and parses | **PASS** | 111 / 111 |
| Every file is cited in a document or written by a script | **PASS** | 111 / 111 |
| Every repository path cited in the Markdown files exists | **PASS** | 0 missing |
| Every DOI in literature_extraction.csv is in references.bib | **PASS** | 5 DOIs; missing from .bib: none; extra in .bib: 10.1016/j.cell.2018.02.010, 10.1109/lsp.2014.2337313, 10.1364/boe.6.001172, 10.2307/2531595 |
| Every Markdown table shows exactly the cells of its CSV | **PASS** | 5 / 5 |
| Files hashed before the audit are unchanged | **PASS** | 9 / 9 |
| No secret-looking strings | **PASS** | none found |

## Every file

| File | Parses | Detail | Cited or produced |
|---|---|---|---|
| `research_outputs/.gitignore` | PASS | UTF-8 text | yes |
| `research_outputs/data/border_artifact_by_class.csv` | PASS | 6 rows × 5 columns | yes |
| `research_outputs/data/border_artifact_summary.json` | PASS | dict with 3 entries | yes |
| `research_outputs/data/calibration_summary.csv` | PASS | 4 rows × 14 columns | yes |
| `research_outputs/data/checkpoint_metadata.json` | PASS | dict with 5 entries | yes |
| `research_outputs/data/dimension_by_class.csv` | PASS | 25 rows × 5 columns | yes |
| `research_outputs/data/excluded_pool_predictions.csv` | PASS | 7,503 rows × 14 columns | yes |
| `research_outputs/data/excluded_pool_unreadable.json` | PASS | dict with 0 entries | yes |
| `research_outputs/data/external_predictions_duke2014.csv` | PASS | 2,508 rows × 11 columns | yes |
| `research_outputs/data/external_predictions_duke2015.csv` | PASS | 110 rows × 12 columns | yes |
| `research_outputs/data/external_predictions_octdl.csv` | PASS | 479 rows × 13 columns | yes |
| `research_outputs/data/external_validation_duke2014_posthoc.csv` | PASS | 4 rows × 15 columns | yes |
| `research_outputs/data/external_validation_duke2015.csv` | PASS | 8 rows × 17 columns | yes |
| `research_outputs/data/external_validation_patient_level.csv` | PASS | 4 rows × 38 columns | yes |
| `research_outputs/data/external_validation_performance.csv` | PASS | 24 rows × 40 columns | yes |
| `research_outputs/data/final_checks.json` | PASS | dict with 11 entries | yes |
| `research_outputs/data/image_hashes.csv` | PASS | 38,147 rows × 10 columns | yes |
| `research_outputs/data/leakage_summary.json` | PASS | dict with 31 entries | yes |
| `research_outputs/data/near_duplicate_pairs.csv` | PASS | 5 rows × 6 columns | yes |
| `research_outputs/data/paired_model_comparisons.csv` | PASS | 3 rows × 10 columns | yes |
| `research_outputs/data/per_paper_comparison.csv` | PASS | 49 rows × 19 columns | yes |
| `research_outputs/data/reconstructed_vs_reported.csv` | PASS | 80 rows × 9 columns | yes |
| `research_outputs/data/refit_calibration_threshold.csv` | PASS | 4 rows × 27 columns | yes |
| `research_outputs/data/reliability_bins.csv` | PASS | 120 rows × 8 columns | yes |
| `research_outputs/data/roc_coordinates_excluded_pool.csv` | PASS | 2,739 rows × 4 columns | yes |
| `research_outputs/data/roc_coordinates_test.csv` | PASS | 109 rows × 4 columns | yes |
| `research_outputs/data/segmentation_per_image.csv` | PASS | 44 rows × 31 columns | yes |
| `research_outputs/data/segmentation_summary.json` | PASS | dict with 8 entries | yes |
| `research_outputs/data/supplementary_excluded_pool_performance.csv` | PASS | 3 rows × 39 columns | yes |
| `research_outputs/data/table2_within_project_performance.csv` | PASS | 12 rows × 37 columns | yes |
| `research_outputs/data/test_misclassified_any_model.csv` | PASS | 2 rows × 6 columns | yes |
| `research_outputs/data/test_predictions_EfficientNet-B0.csv` | PASS | 484 rows × 8 columns | yes |
| `research_outputs/data/test_predictions_EfficientNet-B3.csv` | PASS | 484 rows × 8 columns | yes |
| `research_outputs/data/test_predictions_EfficientNet-B3_LS005.csv` | PASS | 484 rows × 8 columns | yes |
| `research_outputs/data/test_predictions_ResNet-50.csv` | PASS | 484 rows × 8 columns | yes |
| `research_outputs/data/training_split_manifest.csv` | PASS | 30,160 rows × 4 columns | yes |
| `research_outputs/data/val_predictions_true_split.csv` | PASS | 5,560 rows × 11 columns | yes |
| `research_outputs/data/val_split_contamination.json` | PASS | dict with 7 entries | yes |
| `research_outputs/figures/figure1_within_project_performance.pdf` | PASS | 1 page(s) | yes |
| `research_outputs/figures/figure1_within_project_performance.png` | PASS | 4351×2052 px | yes |
| `research_outputs/figures/figure1_within_project_performance_plotted_values.csv` | PASS | 203 rows × 8 columns | yes |
| `research_outputs/figures/figure2_roc_curves.pdf` | PASS | 1 page(s) | yes |
| `research_outputs/figures/figure2_roc_curves.png` | PASS | 3770×2374 px | yes |
| `research_outputs/figures/figure2_roc_curves_plotted_values.csv` | PASS | 2,882 rows × 6 columns | yes |
| `research_outputs/figures/figure3_literature_context.pdf` | PASS | 1 page(s) | yes |
| `research_outputs/figures/figure3_literature_context.png` | PASS | 3413×2597 px | yes |
| `research_outputs/figures/figure3_literature_context_plotted_values.csv` | PASS | 18 rows × 5 columns | yes |
| `research_outputs/figures/figure4_learning_curves.pdf` | PASS | 1 page(s) | yes |
| `research_outputs/figures/figure4_learning_curves.png` | PASS | 4006×2304 px | yes |
| `research_outputs/figures/figure4_learning_curves_plotted_values.csv` | PASS | 97 rows × 7 columns | yes |
| `research_outputs/figures/figure5_reliability.pdf` | PASS | 1 page(s) | yes |
| `research_outputs/figures/figure5_reliability.png` | PASS | 3858×1335 px | yes |
| `research_outputs/figures/figure5_reliability_plotted_values.csv` | PASS | 35 rows × 6 columns | yes |
| `research_outputs/figures/figureA1_segmentation_audit.pdf` | PASS | 1 page(s) | yes |
| `research_outputs/figures/figureA1_segmentation_audit.png` | PASS | 4335×2004 px | yes |
| `research_outputs/figures/figureA1_segmentation_audit_plotted_values.csv` | PASS | 46 rows × 8 columns | yes |
| `research_outputs/final_audit_checklist.md` | PASS | UTF-8 text | yes |
| `research_outputs/gradcam_localisation/per_scan.csv` | PASS | 546 rows × 8 columns | yes |
| `research_outputs/gradcam_localisation/summary.json` | PASS | dict with 5 entries | yes |
| `research_outputs/research_results_report.md` | PASS | UTF-8 text | yes |
| `research_outputs/scripts/01_reinfer_classifiers.py` | PASS | compiles | yes |
| `research_outputs/scripts/02_leakage_scan.py` | PASS | compiles | yes |
| `research_outputs/scripts/03_reinfer_excluded_pool.py` | PASS | compiles | yes |
| `research_outputs/scripts/04_compute_metrics.py` | PASS | compiles | yes |
| `research_outputs/scripts/05_segmentation_audit.py` | PASS | compiles | yes |
| `research_outputs/scripts/06_border_artifact_scan.py` | PASS | compiles | yes |
| `research_outputs/scripts/07_build_literature_tables.py` | PASS | compiles | yes |
| `research_outputs/scripts/08_make_figures.py` | PASS | compiles | yes |
| `research_outputs/scripts/09_make_tables.py` | PASS | compiles | yes |
| `research_outputs/scripts/10_final_checks.py` | PASS | compiles | yes |
| `research_outputs/scripts/11_rebuild_training_split.py` | PASS | compiles | yes |
| `research_outputs/scripts/12_refit_calibration_threshold.py` | PASS | compiles | yes |
| `research_outputs/scripts/13_external_validation.py` | PASS | compiles | yes |
| `research_outputs/scripts/14_per_paper_comparison.py` | PASS | compiles | yes |
| `research_outputs/scripts/15_validate_values.py` | PASS | compiles | yes |
| `research_outputs/scripts/16_value_ledger.py` | PASS | compiles | yes |
| `research_outputs/scripts/17_project_value_ledger.py` | PASS | compiles | yes |
| `research_outputs/scripts/18_verify_output_folders.py` | PASS | compiles | yes |
| `research_outputs/scripts/gradcam_localisation_duke.py` | PASS | compiles | yes |
| `research_outputs/segmentation_retrain/segmentation_per_image.csv` | PASS | 44 rows × 12 columns | yes |
| `research_outputs/segmentation_retrain/segmentation_summary.json` | PASS | dict with 15 entries | yes |
| `research_outputs/segmentation_retrain/segmentation_training_log.csv` | PASS | 32 rows × 11 columns | yes |
| `research_outputs/segmentation_retrain/train_log.txt` | PASS | UTF-8 text | yes |
| `research_outputs/segmentation_retrain_report.md` | PASS | UTF-8 text | yes |
| `research_outputs/segmentation_retrain_v2/segmentation_per_image.csv` | PASS | 44 rows × 13 columns | yes |
| `research_outputs/segmentation_retrain_v2/segmentation_summary.json` | PASS | dict with 17 entries | yes |
| `research_outputs/segmentation_retrain_v2/segmentation_training_log.csv` | PASS | 31 rows × 11 columns | yes |
| `research_outputs/segmentation_retrain_v2/train_log.txt` | PASS | UTF-8 text | yes |
| `research_outputs/tables/table1_dataset_setup.csv` | PASS | 4 rows × 14 columns | yes |
| `research_outputs/tables/table1_dataset_setup.md` | PASS | UTF-8 text | yes |
| `research_outputs/tables/table2_within_project_performance.csv` | PASS | 15 rows × 17 columns | yes |
| `research_outputs/tables/table2_within_project_performance.md` | PASS | UTF-8 text | yes |
| `research_outputs/tables/table3_literature_comparison.csv` | PASS | 43 rows × 16 columns | yes |
| `research_outputs/tables/table3_literature_comparison.md` | PASS | UTF-8 text | yes |
| `research_outputs/tables/table4_segmentation_NOT_PRODUCED.md` | PASS | UTF-8 text | yes |
| `research_outputs/tables/table5_reproducibility_limitations.csv` | PASS | 15 rows × 2 columns | yes |
| `research_outputs/tables/table5_reproducibility_limitations.md` | PASS | UTF-8 text | yes |
| `research_outputs/tables/table6_per_paper_comparison.csv` | PASS | 21 rows × 7 columns | yes |
| `research_outputs/tables/table6_per_paper_comparison.md` | PASS | UTF-8 text | yes |
| `research_audit/checksums_before.sha256` | PASS | UTF-8 text | yes |
| `research_audit/comparability_matrix.csv` | PASS | 12 rows × 15 columns | yes |
| `research_audit/leakage_audit.md` | PASS | UTF-8 text | yes |
| `research_audit/literature_extraction.csv` | PASS | 43 rows × 23 columns | yes |
| `research_audit/metric_verification.md` | PASS | UTF-8 text | yes |
| `research_audit/references.bib` | PASS | UTF-8 text | yes |
| `research_audit/repository_inventory.md` | PASS | UTF-8 text | yes |
| `research_audit/value_ledger.csv` | PASS | 7,962 rows × 7 columns | yes |
| `research_audit/value_ledger_project.csv` | PASS | 3,185 rows × 7 columns | yes |
| `research_audit/value_ledger_project_summary.md` | PASS | UTF-8 text | yes |
| `research_audit/value_ledger_summary.md` | PASS | UTF-8 text | yes |
| `research_audit/value_validation.md` | PASS | UTF-8 text | yes |
