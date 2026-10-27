# Per-value validation ledger (script 16)

Every number shown in the figures, Tables 2, 3 and 6, the report and the checklist, and every published value of the five papers: **7,936 values**. One row per value in `research_audit/value_ledger.csv` (value shown, expected value, its source, status).

- **RECOMPUTED**: recalculated here from per-image predictions or console logs, independently of the pipeline, and equal.
- **TRACED**: equal to the stated source file; re-running would only repeat that step (seeded bootstraps, segmentation inference, values extracted from a paper).
- **FOUND IN TEXT**: the published value appears in the paper's full text or supplement.
- **UNTRACED**: a report/checklist number with no matching data value (manual review). **FAIL**: mismatch.

| Artifact | RECOMPUTED | TRACED | FOUND IN TEXT | total |
|---|---|---|---|---|
| Figure 1 | 78 | 155 | 0 | 233 |
| Figure 2 | 5,696 | 68 | 0 | 5,764 |
| Figure 3 | 4 | 50 | 0 | 54 |
| Figure 4 | 351 | 0 | 0 | 351 |
| Figure 5 | 121 | 0 | 0 | 121 |
| Figure A1 | 0 | 94 | 0 | 94 |
| Figure A1 / report §10 | 4 | 0 | 0 | 4 |
| Table 2 | 276 | 27 | 0 | 303 |
| Table 3 | 15 | 240 | 0 | 255 |
| Table 6 | 120 | 33 | 0 | 153 |
| Report §8.1 table | 12 | 25 | 0 | 37 |
| research_results_report.md | 0 | 404 | 0 | 404 |
| final_audit_checklist.md | 0 | 24 | 0 | 24 |
| Published value | 0 | 0 | 139 | 139 |
| **All** | **6,677** | **1,120** | **139** | **7,936** |

## Values needing attention

None: every value is RECOMPUTED, TRACED or FOUND IN TEXT.
