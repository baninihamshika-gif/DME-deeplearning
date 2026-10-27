# Final Audit Checklist

Status values: **PASS** (checked, and the check succeeded) · **FAIL** (checked, and the requirement is not met) · **NOT VERIFIABLE** (cannot be checked with the available artifacts) · **NOT APPLICABLE**.
Automated results: `research_outputs/data/final_checks.json` (from `scripts/10_final_checks.py`; 11/11 automated checks passed on the last run).

## Final automated checks (task list items 1–15)

| # | Requirement | Status | How it was checked |
|---|---|---|---|
| 1 | All reported metrics verified against underlying outputs | **PASS** | 80/80 values in `metrics.json` reproduced from re-inferred per-image predictions (`reconstructed_vs_reported.csv`). Original per-image outputs never existed; predictions were regenerated from the unmodified checkpoints. |
| 2 | Plotted values match the CSV data | **PASS** | Check 2: Figures 1, 2, 3 and 5 plotted-value CSVs diffed against the source CSVs, 0 mismatches. Figure 4 plots the training-log CSVs directly. |
| 3 | Table values match figures and evaluation report | **PASS** | Check 3: Table 2 Markdown parsed back and compared with `table2_within_project_performance.csv`, 0 mismatches. Report §5 values were hand-compared with Table 2. |
| 4 | CIs use the stated methods | **PASS** | Check 4: Wilson CIs recomputed with SciPy. DeLong recomputed with an independent brute-force implementation (degenerate cases confirmed; a non-degenerate case confirmed on 3,000 pool images). The bootstrap uses a fixed seed (42, B = 2,000). |
| 5 | Positive-class definitions | **PASS** | Check 5: label 1 ⇔ `DME-` filename for all rows. `CLASS_NAMES` index 1 = DME (asserted at train time). |
| 6 | Duplicate images and patient overlap | **PASS** | Check 6: MD5 test↔pool 0, within test 0; patient overlap 0. 10 dHash candidates were inspected visually and are all different eyes. Eye-level overlap is **NOT VERIFIABLE** (no eye IDs). |
| 7 | ROC curves use continuous scores | **PASS** | Check 7: 396–484 unique probabilities per model. ROC is computed from `p_dme_raw`. |
| 8 | Published values have traceable citations | **PASS** | Check 8: 22/22 literature rows carry a DOI and a table location. Values were transcribed from the PMC full texts and the figshare supplement. |
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
| B1 | Patient-level split independence | **PASS** for B3, ResNet-50, A5 · **NOT VERIFIABLE** for B0 | B0 run log is 0 bytes; exclusion implied by `kaggle/entry.py` only |
| B1 | Threshold/calibration fitted without test data | **PASS** (test not used) · **FAIL** (validation set used ≠ training-time validation; overlaps training) | `leakage_audit.md` §5 |
| B2 | Metrics reconstructed with CIs | **PASS** | Table 2 |
| B2 | Undefined metrics handled explicitly | **PASS** | NaN logic in `04_compute_metrics.py`; none occurred |
| B3 | 14-point leakage audit | **PASS** | `leakage_audit.md` §1 |
| B3 | Metadata/dimension-only shortcut classifier | **PASS** | Test AUROC 0.31; unseen-pool AUROC 0.72 |
| B4 | Segmentation independently re-evaluated | **PASS** (reconstructed) · segmentation model itself **FAIL** (collapsed) | Dice 0.0090 reproduced; all-foreground collapse documented |
| B5 | Calibration audited | **PASS** | Raw vs calibrated reported separately; no material change claimed |
| B5 | Grad-CAM audited | **PASS** (protocol reviewed) | Not regenerated; no clinical-reasoning claim made |
| C | 2021 paper read in full incl. supplement | **PASS** | PMC HTML + figshare supplementary tables and methods |
| C | 2025 paper read in full | **PASS** (main text) · **NOT VERIFIABLE** (online supplementary tables) | BMJ site returned HTTP 403; PMC XML has no supplementary tables |
| C | `literature_extraction.csv`, `references.bib` | **PASS** | Crossref-verified bibliographic data |
| D | Comparability matrix before plotting | **PASS** | `research_audit/comparability_matrix.csv` (A/B/C/D) |
| D | Paired tests, multiple-comparison handling | **PASS** | McNemar + Holm; paired DeLong undefined (both AUROC = 1), stated |
| D | No cross-study significance tests or invented CIs | **PASS** | None performed; published CIs copied as printed |
| E | Figure 1 (within-project) | **PASS** | 3 genuinely evaluated models, full 0–1 axis |
| E | Figure 2 (ROC) | **PASS** | Raw probabilities stated; no literature curves overlaid |
| E | Figure 3 (literature) | **PASS** | Grouped by category, explicit non-comparability caption, plus Table 3 |
| E | Figure 4 (learning curves) | **PASS** (B3, ResNet-50) · **NOT VERIFIABLE** (B0, A5: logs missing) | No epochs invented |
| E | Figure 5 (reliability) | **PASS** | Fixed 10-bin scheme, counts shown |
| E | Figure 6 (segmentation performance) | **NOT APPLICABLE** | Replaced by Audit Figure A1 because the evaluation is invalid |
| E | 300-dpi PNG + vector PDF | **PASS** | Check 11 (dpi ≥ 300) |
| E2 | Tables 1, 2, 3, 5 | **PASS** | `research_outputs/tables/` |
| E2 | Table 4 | **NOT APPLICABLE** | Segmentation invalid; `table4_segmentation_NOT_PRODUCED.md` |
| F | Results report | **PASS** | `research_outputs/research_results_report.md` |
| F | External validation claimed only if performed | **PASS** | Explicitly stated as absent |

## Files created (no existing files modified)

- `research_audit/`: `repository_inventory.md`, `metric_verification.md`, `leakage_audit.md`, `literature_extraction.csv`, `comparability_matrix.csv`, `references.bib`, `checksums_before.sha256`
- `research_outputs/research_results_report.md`, `research_outputs/final_audit_checklist.md`, `research_outputs/.gitignore` (ignores `.cache/`)
- `research_outputs/scripts/01_…10_*.py` (10 scripts)
- `research_outputs/data/`: test predictions (4 models), excluded-pool predictions, checkpoint metadata, metric/CI tables, ROC coordinates, reliability bins, calibration summary, paired comparisons, leakage/dimension/border scans, image hashes, segmentation per-image and summary, `final_checks.json`
- `research_outputs/figures/`: figures 1–5 and A1 (PNG + PDF + `_plotted_values.csv`)
- `research_outputs/tables/`: Tables 1, 2, 3, 5 (CSV + MD), Table 4 note
- Local cache only (git-ignored): `research_outputs/.cache/` (preprocessing cache, logs, segmentation example arrays)

## Unresolved issues

1. **Tuned thresholds and temperatures are invalid** (contaminated validation partition). Fixing this needs scikit-learn 1.4.2 to rebuild the Kaggle split; it does not need retraining.
2. **No external validation**, and the official test set appears unrepresentative (easier). The unseen-pool estimate (AUROC 0.996–0.997, accuracy 97.1–97.5%) is the more conservative internal figure.
3. **B0 test-patient exclusion** cannot be confirmed from a log.
4. **The segmentation model is non-functional**, and its ground-truth handling of ungraded pixels is flawed.
5. **Zhu 2025 online supplementary tables** were not retrieved (HTTP 403).
6. **Kermany device and grading details** are cited from prior knowledge and were not re-verified.
