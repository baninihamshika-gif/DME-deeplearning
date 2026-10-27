"""
Independent value validation for the comparison outputs (figures 1-3, Table 6,
report section 8.1). Expected values are recomputed from first sources, not
read from the pipeline's intermediate tables:

  project   per-image predictions (test_predictions_*.csv, excluded_pool_predictions.csv,
            external_predictions_octdl.csv, external_predictions_duke2014.csv),
            thresholded at 0.5 and scored with scikit-learn here
  papers    research_audit/literature_extraction.csv, with the headline rule
            re-implemented from its description; optionally each printed value is
            searched for in the paper's full text (--text PAPER=path.txt)

Every plotted / tabulated / reported number is compared with these expected
values. Output: research_audit/value_validation.md (+ exit code 1 on any FAIL).

Usage: python research_outputs/scripts/15_validate_values.py [--text "Li F 2019=path.txt" ...]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "research_outputs" / "data"
FIG = REPO / "research_outputs" / "figures"
TAB = REPO / "research_outputs" / "tables"
AUDIT = REPO / "research_audit"
MODELS = ["EfficientNet-B3", "EfficientNet-B0", "ResNet-50"]
PAPERS = ["Tang 2021", "Zhu 2025", "Li F 2019", "Rodríguez-Miguel 2024", "Singh & Gorantla 2020"]
METRICS = ["accuracy", "sensitivity", "specificity", "f1", "auroc"]
LAB = {"accuracy": "Accuracy", "sensitivity": "Sensitivity", "specificity": "Specificity", "f1": "F1", "auroc": "AUROC"}

results = []


def check(section, name, ok, detail=""):
    results.append((section, name, "PASS" if ok else "FAIL", detail))


def close(a, b, tol=5e-4):
    if pd.isna(a) and pd.isna(b):
        return True
    return not (pd.isna(a) or pd.isna(b)) and abs(float(a) - float(b)) <= tol


def scores(y, p):
    y = np.asarray(y).astype(int); pred = (np.asarray(p) >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    prec = tp / (tp + fp) if tp + fp else np.nan
    return {"n": len(y), "accuracy": (tp + tn) / len(y), "sensitivity": sens, "specificity": spec,
            "f1": 2 * prec * sens / (prec + sens) if tp else 0.0, "auroc": roc_auc_score(y, p)}


# ------------------------------------------------------------------ expected project values, from predictions
def project_expected():
    exp = {}
    for m in MODELS:
        d = pd.read_csv(DATA / f"test_predictions_{m}.csv")
        exp[(m, "official_test")] = scores(d.label, d.p_dme_raw)
    pool = pd.read_csv(DATA / "excluded_pool_predictions.csv")
    for m in MODELS:
        exp[(m, "unseen_pool")] = scores(pool.label, pool[f"p_dme_raw_{m.replace('-', '')}"])
    oc = pd.read_csv(DATA / "external_predictions_octdl.csv")
    du = pd.read_csv(DATA / "external_predictions_duke2014.csv")
    for m in MODELS:
        exp[(m, "octdl")] = scores(oc.label, oc[f"p_dme_raw__{m}"])
        v = du.groupby("volume").agg(label=("label", "first"), p=(f"p_dme_raw__{m}", "max"))  # max rule
        exp[(m, "duke2014")] = scores(v.label, v.p)
    return exp


# ------------------------------------------------------------------ expected published values, from literature_extraction.csv
def published_expected():
    lit = pd.read_csv(AUDIT / "literature_extraction.csv", dtype=str)
    lit["paper"] = [next((p for p in PAPERS if s.startswith(p)), None) for s in lit.study]
    pub = lit[lit.paper.notna() & lit.positive_class.str.match(r"^(DME|any DME)")].copy()
    for m in METRICS + ["auroc_ci_lo", "auroc_ci_hi"]:
        pub[m] = pd.to_numeric(pub[m], errors="coerce")
    # headline rule, re-implemented from its description: internal image-level row if the paper has one, else main-text primary
    head = {}
    for p in PAPERS:
        d = pub[pub.paper == p]
        img_int = d[d.evaluation_unit.str.contains("B-scan|fundus image", regex=True) & d.cohort_role.str.contains("internal|primary|cross-validation", regex=True)
                    & ~d.cohort_role.str.contains("6 sequential", regex=False)]
        if p == "Tang 2021":
            img_int = img_int[img_int.device.str.contains("SPECTRALIS")]  # S1 lists Spectralis first
        if p == "Singh & Gorantla 2020":
            img_int = img_int[img_int.source_location.str.startswith("Table 3")]  # pooled test result; Tables 4-6 are per-dataset / 3-grade
        if len(img_int):
            head[p] = img_int.iloc[0]
        elif p == "Zhu 2025":
            head[p] = d[(d.source_location == "Table 3") & (d.dataset == "Testing")].iloc[0]
        else:  # Rodriguez-Miguel: main-text Table 2, Model 2
            head[p] = d[d.source_location.str.startswith("Table 2") & d.model.str.contains("Model 2")].iloc[0]
    return pub, head


def number_set(text):
    return {float(x) for x in re.findall(r"(?<![\d.])(\d\.\d{3})(?!\d)", text)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--text", action="append", default=[], help='"Paper name=path to full text"')
    args = ap.parse_args()
    proj = project_expected(); pub, head = published_expected()
    pp = pd.read_csv(DATA / "per_paper_comparison.csv")

    # 1. project values in per_paper_comparison.csv vs recomputation
    bad = []
    for (m, s), e in proj.items():
        r = pp[(pp.paper == "This project") & (pp.model == m) & (pp.setting_key == s)]
        if len(r) != 1:
            bad.append(f"{m}/{s}: {len(r)} rows"); continue
        r = r.iloc[0]
        bad += [f"{m}/{s}/{k}: {r[k]:.4f} vs {e[k]:.4f}" for k in METRICS if not close(r[k], e[k])]
        if int(r.n) != e["n"]:
            bad.append(f"{m}/{s}/n {r.n} vs {e['n']}")
    check("Project", "12 project rows recomputed from per-image predictions (threshold 0.5, Duke max rule)", not bad, "; ".join(bad) or "12 rows x 5 metrics + n")

    # 2. published rows complete and equal to the extraction
    bad = []
    for p in PAPERS:
        a, b = pub[pub.paper == p], pp[pp.paper == p]
        if len(a) != len(b):
            bad.append(f"{p}: {len(b)} rows vs {len(a)}"); continue
        for m in METRICS:
            if not np.allclose(a[m].values, b[m].values, equal_nan=True):
                bad.append(f"{p}/{m}")
    check("Papers", "Every DME-presence row of the five papers carried over unchanged", not bad, "; ".join(bad) or f"{len(pub)} rows")
    bad = [p for p in PAPERS if not all(close(head[p][m], pp[(pp.paper == p) & pp.headline].iloc[0][m]) for m in METRICS) or pp[(pp.paper == p)].headline.sum() != 1]
    check("Papers", "Headline row per paper matches the independently re-applied rule", not bad, ", ".join(bad) or ", ".join(f"{p}: {head[p].source_location[:30]}" for p in PAPERS))

    # 3. Figure 1
    f1 = pd.read_csv(FIG / "figure1_within_project_performance_plotted_values.csv")
    bad = []
    for m in MODELS:
        for k in METRICS:
            r = f1[(f1.panel == "a") & (f1.source == "This project") & (f1.model == m) & (f1.metric == k)]
            if len(r) != 1 or not close(r.value.iloc[0], proj[(m, "official_test")][k]):
                bad.append(f"bar {m}/{k}")
        for k in METRICS:  # dots: the same model's other three settings
            want = sorted(round(proj[(m, s)][k], 6) for s in ("unseen_pool", "octdl", "duke2014"))
            got = sorted(round(v, 6) for v in f1[(f1.panel == "a_other_rows") & (f1.model == m) & (f1.metric == k)].value)
            if len(want) != len(got) or not np.allclose(want, got, atol=5e-4):
                bad.append(f"dots {m}/{k}")
    for p in PAPERS:
        d = pub[pub.paper == p]
        for k in METRICS:
            r = f1[(f1.panel == "a") & (f1.source == p) & (f1.metric == k)]
            if len(r) != 1 or not close(r.value.iloc[0], head[p][k]):
                bad.append(f"bar {p}/{k}")
            got = sorted(f1[(f1.panel == "a_other_rows") & (f1.source == p) & (f1.metric == k)].value)
            want = sorted(d[d.index != head[p].name][k].dropna())
            if len(got) != len(want) or not np.allclose(got, want):
                bad.append(f"dots {p}/{k} ({len(got)} vs {len(want)})")
    check("Figure 1", "Bars (3 models + 5 headlines) and every dot equal the expected values", not bad, "; ".join(bad) or f"{(f1.panel == 'a').sum()} bars, {(f1.panel == 'a_other_rows').sum()} dots")

    # 4. Figure 2
    f2 = pd.read_csv(FIG / "figure2_roc_curves_plotted_values.csv")
    roc = pd.read_csv(DATA / "test_predictions_EfficientNet-B3.csv")
    bad = []
    from sklearn.metrics import auc
    for m in MODELS:
        c = f2[(f2.panel == "a") & (f2.model == m)]
        d = pd.read_csv(DATA / f"test_predictions_{m}.csv")
        if not close(auc(c.fpr, c.tpr), roc_auc_score(d.label, d.p_dme_raw), 1e-6):
            bad.append(f"official curve {m}")
        c = f2[(f2.panel == "a_pool") & (f2.model == m)]
        if not close(auc(c.fpr, c.tpr), proj[(m, "unseen_pool")]["auroc"], 1e-6):
            bad.append(f"pool curve {m}")
    pts = f2[f2.panel == "a_points"]
    for s in ("official_test", "unseen_pool", "octdl", "duke2014"):
        r = pts[(pts.model == "This project EfficientNet-B3") & (pts.setting == s)]
        e = proj[("EfficientNet-B3", s)]
        if len(r) != 1 or not (close(r.fpr.iloc[0], 1 - e["specificity"]) and close(r.tpr.iloc[0], e["sensitivity"])):
            bad.append(f"project point {s}")
    for p in PAPERS:
        d = pub[(pub.paper == p) & pub.sensitivity.notna() & pub.specificity.notna()]
        got = pts[pts.model == p]
        want = sorted(zip(1 - d.specificity, d.sensitivity)); have = sorted(zip(got.fpr, got.tpr))
        if len(want) != len(have) or not np.allclose(want, have):
            bad.append(f"{p} points ({len(have)} vs {len(want)})")
        if p != "Rodríguez-Miguel 2024" and got.headline.sum() != 1:
            bad.append(f"{p} headline marker")
    check("Figure 2", "ROC curves (AUC of plotted coordinates) and all operating points equal the expected values", not bad, "; ".join(bad) or f"{len(pts)} points, 6 curves")

    # 5. Figure 3
    f3 = pd.read_csv(FIG / "figure3_literature_context_plotted_values.csv")
    bad = []
    proj_aucs = {round(v["auroc"], 6) for (m, s), v in proj.items() if m == "EfficientNet-B3"}
    for _, r in f3.iterrows():
        if r.group.startswith("This project"):
            ok = any(abs(r.auroc - v) < 5e-4 for v in proj_aucs)
        else:
            ok = np.isclose(pub.auroc.dropna(), r.auroc).any()
        bad += [] if ok else [r.label]
    papers_in = {p for p in PAPERS if f3.label.str.contains(p.split(" ")[0]).any() or f3.group.str.contains(p.split(" ")[0]).any()}
    check("Figure 3", "Every AUROC equals a recomputed project AUROC or a published AUROC; all five papers present", not bad and len(papers_in) == 5,
          "; ".join(bad) + f" papers present: {len(papers_in)}/5, rows {len(f3)}")

    # 6. Table 6
    t6 = pd.read_csv(TAB / "table6_per_paper_comparison.csv").set_index("Item")
    bad = []
    num = lambda cell: np.nan if str(cell).startswith("NR") else float(str(cell).split(" ")[0])
    for s in ["This project"] + PAPERS:
        e = proj[("EfficientNet-B3", "official_test")] if s == "This project" else head[s]
        for k in METRICS:
            if not close(num(t6.loc[f"Headline {LAB[k]}", s]), e[k]):
                bad.append(f"{s} headline {k}")
        allv = (pd.Series([proj[(m, st)][k] for (m, st) in proj]) if s == "This project" else pub[pub.paper == s][k]).dropna()
        for k in METRICS:
            allv = (pd.Series([proj[key][k] for key in proj]) if s == "This project" else pub[pub.paper == s][k]).dropna()
            cell = str(t6.loc[f"Range {LAB[k]} (k)", s])
            if allv.empty:
                bad += [] if cell == "NR" else [f"{s} range {k}"]
                continue
            mm = re.match(r"([\d.]+)–([\d.]+) \((\d+)\)", cell)
            if not mm or not (close(float(mm[1]), allv.min()) and close(float(mm[2]), allv.max()) and int(mm[3]) == len(allv)):
                bad.append(f"{s} range {k}: {cell} vs {allv.min():.3f}–{allv.max():.3f} ({len(allv)})")
        if s != "This project":
            key = "duke2014" if head[s].evaluation_unit.lower().startswith(("oct", "volume")) else "official_test"
            e = proj[("EfficientNet-B3", key)]
            cell = t6.loc["Unit-matched project value (EfficientNet-B3, 0.5)", s]
            want = f"Sens {e['sensitivity']:.3f}, Spec {e['specificity']:.3f}, AUROC {e['auroc']:.3f}"
            bad += [] if want in cell else [f"{s} matched value: {cell[-45:]}"]
    check("Table 6", "Headline cells, min-max ranges with k, and unit-matched project values", not bad, "; ".join(bad) or "6 columns")

    # 7. Report section 8.1
    rep = (REPO / "research_outputs" / "research_results_report.md").read_text(encoding="utf8")
    sec = rep[rep.index("### 8.1"):rep.index("## 9.")]
    bad = []
    rows = {"**This project**, EfficientNet-B3, official test": proj[("EfficientNet-B3", "official_test")]}
    rows.update({f"**{p}** headline": head[p] for p in PAPERS})
    for key, e in rows.items():
        line = next(l for l in sec.splitlines() if l.startswith("| " + key))
        cells = [c.strip().rstrip("*") for c in line.strip("|").split("|")][2:]
        for k, c in zip(METRICS, cells):
            if not close(num(c), e[k]):
                bad.append(f"{key[:25]} {k}: {c}")
    known = set()
    for e in proj.values():
        known |= {round(e[k], 3) for k in METRICS}
    for _, r in pub.iterrows():
        known |= {round(r[k], 3) for k in METRICS + ["auroc_ci_lo", "auroc_ci_hi"] if not pd.isna(r[k])}
    for p in PAPERS:
        for k in METRICS:
            v = pub[pub.paper == p][k].dropna()
            if len(v):
                known |= {round(v.min(), 3), round(v.max(), 3)}
    sub = pub[pub.paper == "Tang 2021"]
    for grp in (sub[sub.evaluation_unit == "B-scan"], sub[sub.evaluation_unit != "B-scan"]):
        known |= {round(grp.auroc.min(), 3), round(grp.auroc.max(), 3)}
    pool_ci = pd.read_csv(DATA / "supplementary_excluded_pool_performance.csv").set_index("model").loc["EfficientNet-B3"]
    known |= {round(pool_ci.auroc_cluster_bootstrap_ci_lo, 3), round(pool_ci.auroc_cluster_bootstrap_ci_hi, 3)}
    du_mean = pd.read_csv(DATA / "external_validation_performance.csv")
    du_mean = du_mean[du_mean.dataset.str.startswith("Duke 2014") & (du_mean.model == "EfficientNet-B3") & du_mean.threshold_name.str.contains("mean over") & (du_mean.threshold == 0.5)].iloc[0]
    known |= {round(du_mean.sensitivity, 3), round(du_mean.specificity, 3)}
    unexplained = sorted(v for v in number_set(sec) if round(v, 3) not in known)
    check("Report §8.1", "Comparison table rows equal expected values", not bad, "; ".join(bad) or f"{len(rows)} rows")
    check("Report §8.1", "Every 3-decimal number in §8.1 traces to a recomputed or extracted value", not unexplained,
          ("untraced: " + ", ".join(f"{v:.3f}" for v in unexplained)) if unexplained else f"{len(number_set(sec))} distinct numbers")

    # 8. optional: printed values found in the papers' full text
    for spec in args.text:
        p, path = spec.split("=", 1)
        text = Path(path).read_text(encoding="utf8", errors="ignore")
        d = pub[pub.paper == p]; found = missing = 0; miss = []
        for _, r in d.iterrows():
            for k in METRICS:
                v = r[k]
                if pd.isna(v):
                    continue
                forms = {f"{v:.3f}", f"{v:.4f}", f"{v:.2f}", f"{v * 100:.1f}", f"{v * 100:.2f}", f"{v * 100:.0f}", f"{v:.3f}"[1:], f"{v:.2f}"[1:]}
                if any(re.search(r"(?<![\d.])" + re.escape(x) + r"(?!\d)", text) for x in forms):
                    found += 1
                else:
                    missing += 1; miss.append(f"{k} {v:g} [{r.source_location[:25]}]")
        check("Paper text", f"{p}: printed values found in the full text", missing == 0, f"{found} found, {missing} not found" + (": " + "; ".join(miss[:12]) if miss else ""))

    out = ["# Value validation (script 15)", "", "Expected values recomputed from per-image predictions and from `literature_extraction.csv`, "
           "independently of scripts 08, 09 and 14; compared with every plotted, tabulated and reported comparison value.", "",
           "| Area | Check | Result | Detail |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | **{c}** | {d.replace('|', '/')} |" for a, b, c, d in results]
    (AUDIT / "value_validation.md").write_text("\n".join(out) + "\n", encoding="utf8")
    for a, b, c, d in results:
        print(f"{c} [{a}] {b} -- {d}")
    sys.exit(any(c == "FAIL" for _, _, c, _ in results))


if __name__ == "__main__":
    main()
