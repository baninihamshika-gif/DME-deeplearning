# Value validation (script 15)

Expected values recomputed from per-image predictions and from `literature_extraction.csv`, independently of scripts 08, 09 and 14; compared with every plotted, tabulated and reported comparison value.

| Area | Check | Result | Detail |
|---|---|---|---|
| Project | 12 project rows recomputed from per-image predictions (threshold 0.5, Duke max rule) | **PASS** | 12 rows x 5 metrics + n |
| Papers | Every DME-presence row of the five papers carried over unchanged | **PASS** | 37 rows |
| Papers | Headline row per paper matches the independently re-applied rule | **PASS** | Tang 2021: Supplementary Table S1 (B-scan, Zhu 2025: Table 3, Li F 2019: Table 6 (3 ensembling binary c, Rodríguez-Miguel 2024: Table 2 and Results text (AURO, Singh & Gorantla 2020: Table 3 and Fig 6A (Gamma Ense |
| Figure 1 | Bars (3 models + 5 headlines) and every dot equal the expected values | **PASS** | 40 bars, 163 dots |
| Figure 2 | ROC curves (AUC of plotted coordinates) and all operating points equal the expected values | **PASS** | 38 points, 6 curves |
| Figure 3 | Every AUROC equals a recomputed project AUROC or a published AUROC; all five papers present | **PASS** |  papers present: 5/5, rows 18 |
| Table 6 | Headline cells, min-max ranges with k, and unit-matched project values | **PASS** | 6 columns |
| Report §8.1 | Comparison table rows equal expected values | **PASS** | 6 rows |
| Report §8.1 | Every 3-decimal number in §8.1 traces to a recomputed or extracted value | **PASS** | 46 distinct numbers |
| Paper text | Tang 2021: printed values found in the full text | **PASS** | 64 found, 0 not found |
| Paper text | Zhu 2025: printed values found in the full text | **PASS** | 35 found, 0 not found |
| Paper text | Li F 2019: printed values found in the full text | **PASS** | 10 found, 0 not found |
| Paper text | Rodríguez-Miguel 2024: printed values found in the full text | **PASS** | 15 found, 0 not found |
| Paper text | Singh & Gorantla 2020: printed values found in the full text | **PASS** | 15 found, 0 not found |

## Scope and limits

- Paper-text rows search each printed value in the paper's full text and online supplement (Tang: main text + Supplementary Table S1; Zhu: PDF + supplement docx; Li, Rodríguez-Miguel, Singh & Gorantla: article PDFs), as 0.xxx, .xxx or xx.x%. A hit shows the number is printed in the paper; it does not prove it came from the cited table cell, which was checked by hand during extraction (Phase C).
- Also run on this commit: repository test suite (`pytest tests`, 249 passed), `10_final_checks.py` (11/11 PASS), and a scan tracing every 3–4-decimal number in `research_results_report.md` (192) and `final_audit_checklist.md` (18) to a value in `research_outputs/data`, the tables or `literature_extraction.csv` (0 untraced).
- Fixed by this validation: Figure 1 drew a duplicate dot on the EfficientNet-B0 and ResNet-50 bars (their own official-test value); its y-axis floor hid ResNet-50's Duke 2014 specificity (0.400) and was lowered to 0.35.
