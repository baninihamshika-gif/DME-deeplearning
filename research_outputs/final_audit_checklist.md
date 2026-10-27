# Final Audit Checklist

Status values: **PASS** (checked, and the check succeeded) · **FAIL** (checked, and the requirement is not met) · **NOT VERIFIABLE** (cannot be checked with the available artifacts) · **NOT APPLICABLE**.
Automated results: `research_outputs/data/final_checks.json` (from `scripts/10_final_checks.py`; 11/11 automated checks passed on the last run).

## Final automated checks (task list items 1–15)

| # | Requirement | Status | How it was checked |
|---|---|---|---|
| 1 | All reported metrics verified against underlying outputs | **PASS** | 80/80 values in `metrics.json` reproduced from re-inferred per-image predictions (`reconstructed_vs_reported.csv`). Original per-image outputs never existed; predictions were regenerated from the unmodified checkpoints. |
| 2 | Plotted values match the CSV data | **PASS** | Check 2: Figures 1, 2, 3 and 5 plotted-value CSVs diffed against the source CSVs, 0 mismatches. Figure 4 plots the training-log CSVs and parsed console logs directly. |
| 3 | Table values match figures and evaluation report | **PASS** | Check 3: Table 2 Markdown parsed back and compared with `table2_within_project_performance.csv`, 0 mismatches. Report §5 values were hand-compared with Table 2. |
| 4 | CIs use the stated methods | **PASS** | Check 4: Wilson CIs recomputed with SciPy. DeLong recomputed with an independent brute-force implementation (degenerate cases confirmed; a non-degenerate case confirmed on 3,000 pool images). The bootstrap uses a fixed seed (42, B = 2,000). |
| 5 | Positive-class definitions | **PASS** | Check 5: label 1 ⇔ `DME-` filename for all rows. `CLASS_NAMES` index 1 = DME (asserted at train time). |
| 6 | Duplicate images and patient overlap | **PASS** | Check 6: MD5 test↔pool 0, within test 0; patient overlap 0. 5 dHash candidates (exact hex hashes; earlier lossy count of 10 corrected) were inspected visually and are visibly different scans. Eye-level overlap is **NOT VERIFIABLE** (no eye IDs). |
| 7 | ROC curves use continuous scores | **PASS** | Check 7: 396–484 unique probabilities per model. ROC is computed from `p_dme_raw`. |
| 8 | Published values have traceable citations | **PASS** | Check 8: 40/40 published literature rows carry a DOI and a table location. Values were transcribed from the PMC full texts, the figshare, BMJ and Sci Rep supplements, and the article PDFs of Li 2019, Rodríguez-Miguel 2024 and Singh & Gorantla 2020 (inconsistencies flagged in notes). The three later references were Crossref-checked. |
| 9 | Captions free of misleading head-to-head claims | **PASS** | Check 9 (automated wording scan of the report, tables and figure source), plus a manual read of every caption. Figure 3 states "Not a head-to-head comparison". |
| 10 | Every PNG/PDF reopened and readable | **PASS (PNG)** / **NOT VERIFIABLE (PDF visual)** | All 6 PNGs were opened and inspected visually; layout defects were fixed and re-inspected. All 6 PDFs pass a structural check (header and EOF). They were not rendered visually because no PDF rasteriser is installed; they come from the same Matplotlib figure objects as the PNGs. |
| 11 | Project tests run | **PASS** | `python -m pytest tests/`: 249 passed. |
| 12 | No checkpoints or source results overwritten | **PASS** | Check 10: SHA-256 of 5 checkpoints and 4 `metrics.json` files identical to `research_audit/checksums_before.sha256`. `git status` shows only the new `research_audit/` and `research_outputs/` directories. |
| 13 | All created/modified files reported | **PASS** | See "Files created" below. No existing tracked file was modified. |
| 14 | Reproduction commands documented | **PASS** | `research_results_report.md` §16. |
| 15 | Remaining blockers documented | **PASS** | See "Unresolved issues" below and report §15/§17. |

## Phase requirements

| Phase | Requirement | Status | Note |
|---|---|---|---|
| Setup | Isolated branch/directory; user work preserved | **PASS** | Branch `research-audit` from clean `main`; outputs in new directories only |
| A | Repository inventory with conflicts | **PASS** | `research_audit/repository_inventory.md` (8 conflicts documented) |
| B1 | Implementation established from source | **PASS** | `metric_verification.md` B1 |
| B1 | Patient-level split independence | **PASS** (all four models) | B0: audited weights bitwise identical to run `20260920T095427Z`, whose log records the exclusion (`metric_verification.md` B1) |
| B1 | Threshold/calibration fitted without test data | **PASS** (test not used). Original fit used a contaminated validation set (84.3% training images) and is superseded; **refit** on the rebuilt training-time validation split (n = 5,560, counts match the Kaggle log exactly) | `leakage_audit.md` §5, §5.1; `refit_calibration_threshold.csv` |
| B2 | Metrics reconstructed with CIs | **PASS** | Table 2 |
| B2 | Undefined metrics handled explicitly | **PASS** | NaN logic in `04_compute_metrics.py`; none occurred |
| B3 | 14-point leakage audit | **PASS** | `leakage_audit.md` §1 |
| B3 | Metadata/dimension-only shortcut classifier | **PASS** | Test AUROC 0.31; unseen-pool AUROC 0.72 |
| B4 | Segmentation independently re-evaluated | **PASS** (reconstructed; graded-region mask corrected) · original checkpoint **FAIL** (collapsed) · retrained model **PASS** (test pooled Dice 0.706 vs inter-grader 0.694; aggregates re-checked from per-image rows) | Dice 0.0090 reproduced; all-foreground collapse documented. Graded-only metrics now use the `manualLayers` column band (500–542 of 768 columns; 0 NaN fluid pixels and 0 fluid pixels outside the band on all 44 val/test scans): test Dice 0.0196 (previously 0.0138 on a whole-frame mask); inter-grader Dice unchanged (0.758) |
| B5 | Calibration audited | **PASS** | Raw vs calibrated reported separately; no material change claimed |
| B5 | Grad-CAM audited | **PASS** (protocol reviewed) | Not regenerated; no clinical-reasoning claim made |
| C | 2021 paper read in full incl. supplement | **PASS** | PMC HTML + figshare supplementary tables and methods |
| C | 2025 paper read in full | **PASS** (main text and online supplement) | PMC XML and publisher PDF agree; supplement (`bmjopen-15-5-s001.docx`, supplied by the user) read in full; two printing errors flagged |
| C | Li F 2019, Rodríguez-Miguel 2024, Singh & Gorantla 2020 read in full | **PASS** | Article PDFs (supplied by the user) plus the Sci Rep Supplementary Material 1; 15 rows; inconsistencies flagged in notes (report §8) |
| C | `literature_extraction.csv`, `references.bib` | **PASS** | Crossref-verified bibliographic data for all five papers (BOE end page from the PDF) |
| D | Comparability matrix before plotting | **PASS** | `research_audit/comparability_matrix.csv` (A/B/C/D) |
| D | Paired tests, multiple-comparison handling | **PASS** | McNemar + Holm; paired DeLong undefined (both AUROC = 1), stated |
| D | No cross-study significance tests or invented CIs | **PASS** | None performed; published CIs copied as printed |
| E | Figure 1 (within-project) | **PASS** | 3 genuinely evaluated models, full 0–1 axis |
| E | Figure 2 (ROC) | **PASS** | Raw probabilities stated; no literature curves overlaid |
| E | Figure 3 (literature) | **PASS** | Grouped by category, explicit non-comparability caption, plus Table 3 |
| E | Figure 4 (learning curves) | **PASS** (all four models) | No epochs invented. B0 and A5 parsed from their console logs (`20260920T095427Z`, `20260922T160023Z`); selected epochs match `checkpoint_metadata.json` |
| E | Figure 5 (reliability) | **PASS** | Fixed 10-bin scheme, counts shown |
| E | Figure 6 (segmentation performance) | **NOT APPLICABLE** | Replaced by Audit Figure A1 because the evaluation is invalid |
| E | 300-dpi PNG + vector PDF | **PASS** | Check 11 (dpi ≥ 300) |
| E2 | Tables 1, 2, 3, 5 | **PASS** | `research_outputs/tables/` |
| E2 | Table 4 | **NOT APPLICABLE** (original checkpoint) | Original evaluation invalid; `table4_segmentation_NOT_PRODUCED.md` points to the retrained model's results (`segmentation_retrain_report.md`, report §10.2) |
| F | Results report | **PASS** | `research_outputs/research_results_report.md` |
| F | External validation claimed only if performed | **PASS** | External evaluation performed on OCTDL, Duke 2014 and Duke 2015 (report §7.4) and described with its limits; no clinical validation claimed |

## Files created

Audit outputs were added in new directories. The only existing files modified are `requirements.txt` (scikit-learn pinned to 1.4.2), `train.py` (writes `split_manifest.csv`) and `evaluate.py` (writes `predictions.csv`); model logic, checkpoints and source results are unchanged (check 10).

- `research_audit/`: `repository_inventory.md`, `metric_verification.md`, `leakage_audit.md`, `literature_extraction.csv`, `comparability_matrix.csv`, `references.bib`, `checksums_before.sha256`
- `research_outputs/research_results_report.md`, `research_outputs/final_audit_checklist.md`, `research_outputs/.gitignore` (ignores `.cache/`)
- `research_outputs/scripts/01_…13_*.py` (13 scripts; 11 runs in an isolated Python 3.12 / scikit-learn 1.4.2 environment; 13 needs OCTDL and Duke 2014 in git-ignored `data/external/`)
- `research_outputs/data/`: test predictions (4 models), excluded-pool predictions, checkpoint metadata, metric/CI tables, ROC coordinates, reliability bins, calibration summary, paired comparisons, leakage/dimension/border scans, image hashes, segmentation per-image and summary, `training_split_manifest.csv`, `val_predictions_true_split.csv`, `refit_calibration_threshold.csv`, `val_split_contamination.json`, `external_predictions_{octdl,duke2014,duke2015}.csv`, `external_validation_{performance,patient_level,duke2015,duke2014_posthoc}.csv`, `final_checks.json`
- `research_outputs/figures/`: figures 1–5 and A1 (PNG + PDF + `_plotted_values.csv`)
- `research_outputs/tables/`: Tables 1, 2, 3, 5 (CSV + MD), Table 4 note
- Local cache only (git-ignored): `research_outputs/.cache/` (preprocessing cache, logs, segmentation example arrays)

## Unresolved issues

1. ~~Tuned thresholds and temperatures are invalid~~ **Resolved:** refit on the rebuilt training-time validation split (scripts 11–12). Residual caveat: on the easier test set the refit operating points give perfect or near-perfect counts, and refit scaling worsens test ECE for B3 and B0; validation specificity at 97% sensitivity (0.956–0.982) is the more realistic figure.
2. **External evidence is limited.** OCTDL (Optovue; AUROC 0.993–0.999, sensitivity at 0.5 0.93–0.96), Duke 2014 (volume AUROC 1.000, but the pre-specified max rule flags 4–9 of 15 normal volumes from off-centre slices) and Duke 2015 (DME only; sensitivity 0.95–0.96) are small public sets, not a clinical cohort. The official test set also appears unrepresentative (easier). The unseen-pool estimate (AUROC 0.996–0.997, accuracy 97.1–97.5%) is the more conservative internal figure.
3. ~~B0 test-patient exclusion cannot be confirmed from a log~~ **Resolved:** verified via run `20260920T095427Z` (bitwise-identical weights; exclusion logged).
4. ~~**The segmentation model is non-functional**~~ **Resolved by retraining (report §10.2):** the original checkpoint remains collapsed, but the retrained U-Net (graded-only loss and metrics, val-only epoch/threshold selection) reaches test pooled Dice 0.706 (inter-grader 0.694; baselines 0.020 / 0). Remaining limits: only 2 test subjects / 22 scans, worse boundary distance than between graders, no external segmentation test. Original issue: its ground-truth handling of ungraded pixels was flawed (ungraded margins are stored as 0, so the old `segment.py` scored them as background). ~~The audit's own graded-only re-evaluation used a whole-frame mask~~ **Resolved:** script 05 now uses the `manualLayers` column band as the graded region. Graded-only test Dice is 0.0196 / pooled 0.0204 (val 0.0300 / 0.0308), up from the whole-frame 0.0138; `metric_verification.md` B4, inventory C6 and report §10 are corrected. The collapse finding, the pipeline-convention Dice (0.0090) and the inter-grader Dice (0.758 test / 0.522 val) are unchanged.
5. ~~Zhu 2025 online supplementary tables not retrieved~~ **Resolved:** supplement read in full. Confusion matrices recompute every metric; per-device and reader-comparison rows were added to `literature_extraction.csv`.
6. ~~Kermany device and grading details from prior knowledge~~ **Resolved:** verified against the published STAR Methods (archived Cell full text): Spectralis OCT; five centres; July 2013 – March 2017; three-tier grading (students → four ophthalmologists → two senior retinal specialists).
