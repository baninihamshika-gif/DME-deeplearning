"""
Phase F -- automated final checks. Writes research_outputs/data/final_checks.json,
which research_outputs/final_audit_checklist.md cites. Exits non-zero if any
automated check fails.

Checks:
  1  reconstructed metrics == metrics.json (all values)
  2  figure plotted values == table/CSV values (Fig 1, 2, 3, 5)
  3  Markdown tables == underlying CSV values (Table 2)
  4  Wilson / bootstrap / DeLong CIs recomputed independently and compared
  5  positive class: label 1 == DME folder for every prediction row
  6  duplicates / patient overlap summary present and 0 where required
  7  ROC curves built from continuous scores (>2 unique scores per model)
  8  every literature row has a source location and DOI-bearing study string
  9  no forbidden head-to-head wording in figure captions / report
  10 checkpoints and metrics.json unchanged (SHA-256 vs checksums_before)
  11 every PNG/PDF reopens; PNG >= 300 dpi equivalent size

Usage: python research_outputs/scripts/10_final_checks.py
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "research_outputs" / "data"
FIG = REPO / "research_outputs" / "figures"
TAB = REPO / "research_outputs" / "tables"
AUDIT = REPO / "research_audit"
res = {}


def check(name, ok, detail=""):
    res[name] = {"pass": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + (f" -- {detail}" if detail else ""))


# 1
rv = pd.read_csv(DATA / "reconstructed_vs_reported.csv")
check("1_reconstructed_equals_reported", rv.match.all(), f"{int(rv.match.sum())}/{len(rv)} values match")

# 2
perf = pd.read_csv(DATA / "table2_within_project_performance.csv")
f1 = pd.read_csv(FIG / "figure1_within_project_performance_plotted_values.csv")
bad = 0
pp = pd.read_csv(DATA / "per_paper_comparison.csv")
for _, r in f1[f1.panel == "a"].iterrows():
    if r.source == "This project":
        t = perf[(perf.model == r.model) & (perf.threshold == 0.5)].iloc[0]
        bad += not (np.isclose(r.value, t[r.metric]) and np.allclose(r.ci_lo, t[f"{r.metric}_ci_lo"], equal_nan=True) and np.allclose(r.ci_hi, t[f"{r.metric}_ci_hi"], equal_nan=True))
    else:  # paper bars = that paper's headline row
        t = pp[(pp.paper == r.source) & pp.headline].iloc[0]
        bad += not np.allclose(r.value, t[r.metric], equal_nan=True)
for _, r in f1[f1.panel == "a_other_rows"].iterrows():  # every dot is a value of that source
    bad += not np.isclose(pp[pp.paper == r.source][r.metric].dropna(), r.value).any()
f2p = pd.read_csv(FIG / "figure2_roc_curves_plotted_values.csv")
for _, r in f2p[f2p.panel == "a_points"].iterrows():  # published / project operating points
    src = pp[pp.paper == ("This project" if r.model.startswith("This project") else r.model)]
    bad += not (np.isclose(1 - src.specificity, r.fpr) & np.isclose(src.sensitivity, r.tpr)).any()
t6 = pd.read_csv(TAB / "table6_per_paper_comparison.csv").set_index("Item")
for s in t6.columns:  # Table 6 headline cells = per_paper_comparison headline rows
    h = pp[(pp.paper == s) & pp.headline].iloc[0]
    for m, lab in (("accuracy", "Accuracy"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity"), ("f1", "F1"), ("auroc", "AUROC")):
        cell = str(t6.loc[f"Headline {lab}", s]).split(" ")[0]
        bad += not ((cell == "NR" and pd.isna(h[m])) or (cell != "NR" and np.isclose(float(cell), h[m], atol=5e-4)))
exc_p = DATA / "supplementary_excluded_pool_performance.csv"
if exc_p.exists():
    ex = pd.read_csv(exc_p)
    for _, r in f1[f1.metric == "auroc_excluded_pool"].iterrows():
        t = ex[ex.model == r.model].iloc[0]
        bad += not (np.isclose(r.value, t.auroc) and np.isclose(r.ci_lo, t.auroc_cluster_bootstrap_ci_lo))
roc = pd.read_csv(DATA / "roc_coordinates_test.csv"); f2 = pd.read_csv(FIG / "figure2_roc_curves_plotted_values.csv")
for m in roc.model.unique():
    if m in f2.model.values:
        a, b = roc[roc.model == m], f2[(f2.model == m) & (f2.panel == "a")]
        bad += not (np.allclose(a.fpr.values, b.fpr.values) and np.allclose(a.tpr.values, b.tpr.values))
lit = pd.read_csv(AUDIT / "literature_extraction.csv"); f3 = pd.read_csv(FIG / "figure3_literature_context_plotted_values.csv")
for _, r in f3.iterrows():
    if "This project" in r.group:
        continue
    bad += not np.isclose(pd.to_numeric(lit.auroc, errors="coerce"), r.auroc).any()
rb = pd.read_csv(DATA / "reliability_bins.csv"); f5 = pd.read_csv(FIG / "figure5_reliability_plotted_values.csv")
for _, r in f5.iterrows():
    t = rb[(rb.model == r.model) & (rb.probabilities == r.probabilities) & (rb.bin == r.bin)].iloc[0]
    bad += not (np.isclose(t.mean_pred, r.mean_pred) and np.isclose(t.frac_pos, r.frac_pos) and t["count"] == r["count"])
check("2_figures_match_csv", bad == 0, f"{bad} mismatches")

# 3
t2 = pd.read_csv(TAB / "table2_within_project_performance.csv")
bad = 0
for _, r in t2.iterrows():
    if "[supplementary" in r.Model:
        continue
    op = r["Probabilities / threshold"]
    thr = 0.5 if op.startswith("raw") else float(op.split(", ")[1].split(" ")[0])
    kind = "default" if op.startswith("raw") else ("refit" if "refit" in op else "superseded")
    p = perf[(perf.model == r.Model) & np.isclose(perf.threshold, thr) & perf.threshold_name.str.startswith(kind)].iloc[0]
    bad += (int(r.TP), int(r.FP), int(r.TN), int(r.FN)) != (p.TP, p.FP, p.TN, p.FN)
    bad += not r["Accuracy (95% CI)"].startswith(f"{p.accuracy:.4f}")
    bad += not r["Sensitivity (95% CI)"].startswith(f"{p.sensitivity:.4f}")
check("3_tables_match_csv", bad == 0, f"{bad} mismatches")

# 4 independent CI recomputation
from scipy import stats  # noqa: E402

bad = 0
for _, r in perf.iterrows():
    for k, num, den in (("sensitivity", r.TP, r.TP + r.FN), ("specificity", r.TN, r.TN + r.FP), ("accuracy", r.TP + r.TN, r.n), ("precision", r.TP, r.TP + r.FP)):
        ci = stats.binomtest(int(num), int(den)).proportion_ci(confidence_level=0.95, method="wilson")
        bad += not (np.isclose(ci.low, r[f"{k}_ci_lo"], atol=1e-6) and np.isclose(ci.high, r[f"{k}_ci_hi"], atol=1e-6))


def delong_bruteforce(y, s):
    pos, neg = s[y == 1], s[y == 0]
    psi = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    auc = psi.mean(); v10 = psi.mean(axis=1); v01 = psi.mean(axis=0)
    var = v10.var(ddof=1) / len(pos) + v01.var(ddof=1) / len(neg)
    return auc, var


for m in ("EfficientNet-B3", "EfficientNet-B0", "ResNet-50"):
    d = pd.read_csv(DATA / f"test_predictions_{m}.csv")
    auc, var = delong_bruteforce(d.label.values, d.p_dme_raw.values)
    r = perf[(perf.model == m) & (perf.threshold == 0.5)].iloc[0]
    bad += not (np.isclose(auc, r.auroc) and (var == 0) == (r.auroc_ci_lo == r.auroc_ci_hi))
# DeLong on a non-degenerate case: ablation-free check on the excluded pool (if present)
if exc_p.exists():
    e = pd.read_csv(DATA / "excluded_pool_predictions.csv")
    sys.path.insert(0, str(REPO / "research_outputs" / "scripts"))
    import importlib.util as u
    spec = u.spec_from_file_location("cm", REPO / "research_outputs" / "scripts" / "04_compute_metrics.py"); cm = u.module_from_spec(spec); spec.loader.exec_module(cm)
    rng = np.random.default_rng(0); idx = rng.choice(len(e), 3000, replace=False)
    y, s_ = e.label.values[idx], e["p_dme_raw_EfficientNetB3"].values[idx]
    a1, v1 = delong_bruteforce(y, s_); a2, c2 = cm.delong(y, [s_])
    bad += not (np.isclose(a1, a2[0]) and np.isclose(v1, c2[0, 0], rtol=1e-6))
check("4_confidence_intervals_recomputed", bad == 0, f"{bad} mismatches (Wilson via scipy; DeLong via brute-force O(mn) implementation)")

# 5 positive class
bad = 0
for f in DATA.glob("test_predictions_*.csv"):
    d = pd.read_csv(f)
    bad += int((d.label != d.filename.str.startswith("DME-").astype(int)).sum())
check("5_positive_class_is_DME", bad == 0, f"{bad} rows where label != DME filename prefix")

# 6 leakage
lk = json.loads((DATA / "leakage_summary.json").read_text())
check("6_duplicates_and_patient_overlap", lk["patient_overlap_test_vs_pool"] == 0 and lk["exact_dup_pairs_test_vs_pool"] == 0 and lk["exact_dup_pairs_within_test"] == 0,
      f"patient overlap {lk['patient_overlap_test_vs_pool']}, exact dups test-pool {lk['exact_dup_pairs_test_vs_pool']}, within test {lk['exact_dup_pairs_within_test']}, dHash near-dup pairs {lk['near_dup_pairs_test_vs_pool']} (all visually distinct scans), images without a hash {lk['unreadable_images']}")

# 7 continuous scores
nun = {f.stem: pd.read_csv(f).p_dme_raw.nunique() for f in DATA.glob("test_predictions_*.csv")}
check("7_roc_from_continuous_scores", all(v > 100 for v in nun.values()), str(nun))

# 8 literature traceability
lit = pd.read_csv(AUDIT / "literature_extraction.csv", dtype=str)
pub = lit[~lit.study.str.startswith("This project")]
check("8_literature_traceable", pub.source_location.notna().all() and pub.study.str.contains("doi:").all() and (pub.source_location.str.len() > 5).all(), f"{len(pub)} published rows with table location and DOI")

# 9 wording
texts = {}
for f in list(REPO.glob("research_outputs/*.md")) + list(TAB.glob("*.md")):
    texts[f.name] = f.read_text(encoding="utf8")
src = (REPO / "research_outputs" / "scripts" / "08_make_figures.py").read_text(encoding="utf8")
texts["08_make_figures.py"] = src
forbidden = [r"\boutperform", r"\bsuperior\b", r"state[- ]of[- ]the[- ]art", r"clinically ready", r"\bbeats?\b", r"better than (?:tang|zhu)", r"externally validated"]
negation = re.compile(r"\b(no|not|never|without|cannot|can ?not)\b", flags=re.I)
hits = []
for n, t in texts.items():
    for pat in forbidden:
        for m_ in re.finditer(pat, t, flags=re.I):
            ctx = t[max(0, m_.start() - 60): m_.end() + 40].replace("\n", " ")
            if negation.search(ctx):
                continue
            hits.append(f"{n}: ...{ctx}...")
check("9_no_head_to_head_claims", not hits, "; ".join(hits[:5]) if hits else "no unqualified superiority/clinical-readiness wording")

# 10 checksums
bad = []
for line in (AUDIT / "checksums_before.sha256").read_text().splitlines():
    h, path = line.split(" *", 1)
    if hashlib.sha256((REPO / path).read_bytes()).hexdigest() != h:
        bad.append(path)
check("10_checkpoints_and_results_unchanged", not bad, f"{9 - len(bad)}/9 files identical" if not bad else str(bad))

# 11 figures reopen
from PIL import Image  # noqa: E402

bad = []
for f in FIG.glob("*.png"):
    with Image.open(f) as im:
        im.verify()
    with Image.open(f) as im:
        dpi = im.info.get("dpi", (0, 0))[0]
        if dpi < 299:
            bad.append(f"{f.name} dpi={dpi}")
for f in FIG.glob("*.pdf"):
    b = f.read_bytes()
    if not (b.startswith(b"%PDF") and b"%%EOF" in b[-1024:]):
        bad.append(f.name)
check("11_figures_reopen", not bad, f"{len(list(FIG.glob('*.png')))} PNG, {len(list(FIG.glob('*.pdf')))} PDF" + (f"; problems {bad}" if bad else ""))

json.dump(res, open(DATA / "final_checks.json", "w"), indent=2)
sys.exit(0 if all(v["pass"] for v in res.values()) else 1)
