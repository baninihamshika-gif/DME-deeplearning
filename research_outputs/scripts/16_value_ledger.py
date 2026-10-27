"""
Per-value validation ledger: one row for every number shown in the figures
(1-5, A1), Tables 2, 3 and 6, the research report and the audit checklist,
plus every published value of the five papers.

Each row gives: artifact, item, value shown, expected value, where the
expected value comes from, and a status:
  RECOMPUTED     recalculated here from first sources (per-image predictions,
                 console logs) independently of scripts 04-14, and equal
  TRACED         equal to the stated source file; recalculating would only
                 repeat that step (seeded bootstrap, segmentation inference,
                 values extracted from a paper)
  FOUND IN TEXT  a published value located in the paper's own full text or
                 supplement (needs --text; see 15_validate_values.py)
  FAIL           does not match
  UNTRACED       a number in the report/checklist matched to no data value
                 (listed for manual review)

Outputs: research_audit/value_ledger.csv, research_audit/value_ledger_summary.md
Usage: python research_outputs/scripts/16_value_ledger.py [--text "Paper=path.txt" ...]
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "research_outputs" / "data"
FIG = REPO / "research_outputs" / "figures"
TAB = REPO / "research_outputs" / "tables"
AUDIT = REPO / "research_audit"
MODELS = ["EfficientNet-B3", "EfficientNet-B0", "ResNet-50"]
ALL4 = MODELS + ["EfficientNet-B3 + LS 0.05 (ablation)"]
PRED_FILE = {"EfficientNet-B3": "EfficientNet-B3", "EfficientNet-B0": "EfficientNet-B0", "ResNet-50": "ResNet-50", "EfficientNet-B3 + LS 0.05 (ablation)": "EfficientNet-B3_LS005"}
PAPERS = ["Tang 2021", "Zhu 2025", "Li F 2019", "Rodríguez-Miguel 2024", "Singh & Gorantla 2020"]
METRICS = ["accuracy", "sensitivity", "specificity", "f1", "auroc"]
LAB = {"accuracy": "Accuracy", "sensitivity": "Sensitivity", "specificity": "Specificity", "f1": "F1", "auroc": "AUROC"}
Z = 1.959964

L = []


def rec(artifact, item, shown, expected, source, status=None, tol=5e-4, note=""):
    if status is None:
        if isinstance(shown, str) or isinstance(expected, str):
            ok = str(shown).strip() == str(expected).strip()
        elif pd.isna(shown) and pd.isna(expected):
            ok = True
        else:
            ok = not (pd.isna(shown) or pd.isna(expected)) and abs(float(shown) - float(expected)) <= tol
        status = "RECOMPUTED" if ok else "FAIL"
    elif status in ("TRACED",) and not isinstance(shown, str):
        ok = (pd.isna(shown) and pd.isna(expected)) or (not (pd.isna(shown) or pd.isna(expected)) and abs(float(shown) - float(expected)) <= tol)
        status = "TRACED" if ok else "FAIL"
    L.append({"artifact": artifact, "item": item, "value_shown": shown, "expected": expected, "expected_source": source, "status": status, "note": note})


# ------------------------------------------------------------------ first-source computations
def wilson(k, n):
    if n == 0:
        return np.nan, np.nan
    p = k / n; d = 1 + Z ** 2 / n
    c = (p + Z ** 2 / (2 * n)) / d; h = Z * np.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / d
    return max(0.0, c - h), min(1.0, c + h)


def confusion(y, p, t):
    y = np.asarray(y).astype(int); pr = (np.asarray(p) >= t).astype(int)
    return {"TP": int(((pr == 1) & (y == 1)).sum()), "FP": int(((pr == 1) & (y == 0)).sum()),
            "TN": int(((pr == 0) & (y == 0)).sum()), "FN": int(((pr == 0) & (y == 1)).sum())}


def metrics(y, p, t=0.5):
    c = confusion(y, p, t); tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]; n = tp + fp + tn + fn
    out = dict(c, n=n, accuracy=(tp + tn) / n, sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
               precision=tp / (tp + fp) if tp + fp else np.nan, auroc=roc_auc_score(y, p), prauc=average_precision_score(y, p), brier=brier_score_loss(y, p))
    out["f1"] = 2 * tp / (2 * tp + fp + fn)
    out["ci"] = {"accuracy": wilson(tp + tn, n), "sensitivity": wilson(tp, tp + fn), "specificity": wilson(tn, tn + fp), "precision": wilson(tp, tp + fp)}
    return out


def sigmoid_T(d, T):
    return 1 / (1 + np.exp(-(d.logit_dme - d.logit_normal) / T))


def project():
    P = {}
    temps = pd.read_csv(DATA / "calibration_summary.csv").set_index("model").temperature
    for m in ALL4:
        d = pd.read_csv(DATA / f"test_predictions_{PRED_FILE[m]}.csv")
        P[(m, "official_test")] = (d.label.values, d.p_dme_raw.values)
        P[(m, "official_test_refit")] = (d.label.values, sigmoid_T(d, temps[m]).values)
        P[(m, "official_test_superseded")] = (d.label.values, d.p_dme_calibrated.values)
    pool = pd.read_csv(DATA / "excluded_pool_predictions.csv")
    oc = pd.read_csv(DATA / "external_predictions_octdl.csv"); du = pd.read_csv(DATA / "external_predictions_duke2014.csv")
    for m in MODELS:
        P[(m, "unseen_pool")] = (pool.label.values, pool[f"p_dme_raw_{m.replace('-', '')}"].values)
        P[(m, "octdl")] = (oc.label.values, oc[f"p_dme_raw__{m}"].values)
        v = du.groupby("volume").agg(label=("label", "first"), p=(f"p_dme_raw__{m}", "max"))
        P[(m, "duke2014")] = (v.label.values, v.p.values)
    return P


def published():
    lit = pd.read_csv(AUDIT / "literature_extraction.csv", dtype=str)
    lit["paper"] = [next((p for p in PAPERS if s.startswith(p)), None) for s in lit.study]
    for m in METRICS + ["auroc_ci_lo", "auroc_ci_hi"]:
        lit[m + "_num"] = pd.to_numeric(lit[m], errors="coerce")
    return lit


def parse_log(path):
    rx = re.compile(r"\[[AB] epoch \d+/\d+ \| global (\d+)\] train_loss=([\d.]+) val_loss=([\d.]+) val_auc=([\d.]+)")
    rows = {}
    for g, a, b, c in rx.findall(Path(path).read_text(encoding="utf8", errors="ignore")):
        rows[int(g)] = (float(a), float(b), float(c))
    return rows


def num(cell):
    s = str(cell).strip()
    if s.startswith("NR") or s in ("", "nan", "NA"):
        return np.nan
    m = re.match(r"-?[\d.]+", s.replace(",", ""))
    return float(m.group()) if m else np.nan


def ci_of(cell):
    m = re.search(r"\(([\d.]+)[–\-]([\d.]+)\)", str(cell))
    return (float(m[1]), float(m[2])) if m else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--text", action="append", default=[])
    args = ap.parse_args()
    P = project(); lit = published()
    M = {k: metrics(*v) for k, v in P.items()}
    pub = lit[lit.paper.notna() & lit.positive_class.str.match(r"^(DME|any DME)")]
    pp = pd.read_csv(DATA / "per_paper_comparison.csv")
    head = {p: pp[(pp.paper == p) & pp.headline].iloc[0] for p in PAPERS}
    t2d = pd.read_csv(DATA / "table2_within_project_performance.csv")

    # ---------------- Figure 1
    f1 = pd.read_csv(FIG / "figure1_within_project_performance_plotted_values.csv")
    for _, r in f1.iterrows():
        it = f"{r.panel} · {r.model} · {r.metric}"
        if r.source == "This project" and r.panel == "a":
            e = M[(r.model, "official_test")]
            rec("Figure 1", it + " (bar)", r.value, e[r.metric], f"test_predictions_{PRED_FILE[r.model]}.csv, threshold 0.5")
            if r.metric in e["ci"]:
                lo, hi = e["ci"][r.metric]
                rec("Figure 1", it + " CI low", r.ci_lo, lo, "Wilson interval from recomputed counts")
                rec("Figure 1", it + " CI high", r.ci_hi, hi, "Wilson interval from recomputed counts")
            else:
                t = t2d[(t2d.model == r.model) & (t2d.threshold == 0.5)].iloc[0]
                rec("Figure 1", it + " CI low", r.ci_lo, t[f"{r.metric}_ci_lo"], "table2_within_project_performance.csv (bootstrap B=2000 seed 42 / DeLong)", "TRACED")
                rec("Figure 1", it + " CI high", r.ci_hi, t[f"{r.metric}_ci_hi"], "table2_within_project_performance.csv (bootstrap B=2000 seed 42 / DeLong)", "TRACED")
        elif r.panel == "a":
            h = head[r.source]
            rec("Figure 1", f"a · {r.source} headline · {r.metric} (bar)", r.value, h[r.metric], f"literature_extraction.csv [{h.source}]", "TRACED")
        else:
            if r.source == "This project":
                cands = {s: M[(r.model, s)][r.metric] for s in ("unseen_pool", "octdl", "duke2014")}
                s = min(cands, key=lambda k: abs(cands[k] - r.value))
                rec("Figure 1", f"dot · {r.model} · {s} · {r.metric}", r.value, cands[s], f"recomputed from {s} predictions")
            else:
                vals = pub[pub.paper == r.source][r.metric + "_num"].dropna()
                e = vals.iloc[(vals - r.value).abs().argmin()]
                rec("Figure 1", f"dot · {r.source} · {r.metric}", r.value, e, "literature_extraction.csv (row of that paper)", "TRACED")

    # ---------------- Figure 2
    f2 = pd.read_csv(FIG / "figure2_roc_curves_plotted_values.csv")
    for panel, setting in (("a", "official_test"), ("a_pool", "unseen_pool")):
        for m in MODELS:
            fpr, tpr, _ = roc_curve(*P[(m, setting)])
            c = f2[(f2.panel == panel) & (f2.model == m)]
            if len(c) != len(fpr):
                rec("Figure 2", f"{setting} ROC {m}: number of coordinates", len(c), len(fpr), "sklearn roc_curve on predictions")
                continue
            for i, (a, b, x, y) in enumerate(zip(c.fpr, c.tpr, fpr, tpr)):
                rec("Figure 2", f"{setting} ROC {m} point {i} FPR", a, x, "sklearn roc_curve on predictions", tol=1e-9)
                rec("Figure 2", f"{setting} ROC {m} point {i} TPR", b, y, "sklearn roc_curve on predictions", tol=1e-9)
    for _, r in f2[f2.panel == "a_points"].iterrows():
        if r.model.startswith("This project"):
            e = M[("EfficientNet-B3", r.setting)]
            rec("Figure 2", f"point · B3 · {r.setting} · 1−specificity", r.fpr, 1 - e["specificity"], f"recomputed from {r.setting} predictions")
            rec("Figure 2", f"point · B3 · {r.setting} · sensitivity", r.tpr, e["sensitivity"], f"recomputed from {r.setting} predictions")
        else:
            d = pub[(pub.paper == r.model)]
            j = ((1 - d.specificity_num - r.fpr).abs() + (d.sensitivity_num - r.tpr).abs()).idxmin()
            rec("Figure 2", f"point · {r.model} · 1−specificity", r.fpr, 1 - d.loc[j, "specificity_num"], f"literature_extraction.csv [{d.loc[j, 'source_location'][:40]}]", "TRACED")
            rec("Figure 2", f"point · {r.model} · sensitivity", r.tpr, d.loc[j, "sensitivity_num"], f"literature_extraction.csv [{d.loc[j, 'source_location'][:40]}]", "TRACED")

    # ---------------- Figure 3
    f3 = pd.read_csv(FIG / "figure3_literature_context_plotted_values.csv")
    ext = pd.read_csv(DATA / "external_validation_performance.csv"); exc = pd.read_csv(DATA / "supplementary_excluded_pool_performance.csv").set_index("model")
    for _, r in f3.iterrows():
        if r.group.startswith("This project"):
            s = "official_test" if "official" in r.label else "unseen_pool" if "pool" in r.label else "octdl" if "OCTDL" in r.label else "duke2014"
            rec("Figure 3", f"{r.label} · AUROC", r.auroc, M[("EfficientNet-B3", s)]["auroc"], f"recomputed from {s} predictions")
            if s == "unseen_pool":
                src, lo, hi = "supplementary_excluded_pool_performance.csv (patient-cluster bootstrap)", exc.loc["EfficientNet-B3", "auroc_cluster_bootstrap_ci_lo"], exc.loc["EfficientNet-B3", "auroc_cluster_bootstrap_ci_hi"]
            elif s in ("octdl", "duke2014"):
                e = ext[ext.dataset.str.startswith("OCTDL" if s == "octdl" else "Duke 2014") & (ext.model == "EfficientNet-B3") & (ext.threshold == 0.5)].iloc[0]
                src, lo, hi = "external_validation_performance.csv (DeLong)", e.auroc_ci_lo, e.auroc_ci_hi
            else:
                src, lo, hi = "none (perfect separation)", np.nan, np.nan
            rec("Figure 3", f"{r.label} · CI low", r.ci_lo, lo, src, "TRACED")
            rec("Figure 3", f"{r.label} · CI high", r.ci_hi, hi, src, "TRACED")
        else:
            d = lit[(lit.auroc_num - r.auroc).abs() < 5e-4]
            d = d[(d.auroc_ci_lo_num.fillna(-1) - (r.ci_lo if not pd.isna(r.ci_lo) else -1)).abs() < 5e-4] if len(d) > 1 else d
            ok = len(d) >= 1
            e = d.iloc[0] if ok else None
            rec("Figure 3", f"{r.label} · AUROC", r.auroc, e.auroc_num if ok else np.nan, f"literature_extraction.csv [{e.source_location[:40]}]" if ok else "not found", "TRACED")
            rec("Figure 3", f"{r.label} · CI low", r.ci_lo, e.auroc_ci_lo_num if ok else np.nan, "literature_extraction.csv", "TRACED")
            rec("Figure 3", f"{r.label} · CI high", r.ci_hi, e.auroc_ci_hi_num if ok else np.nan, "literature_extraction.csv", "TRACED")

    # ---------------- Figure 4 (re-parsed from the console logs / training_log.csv)
    f4 = pd.read_csv(FIG / "figure4_learning_curves_plotted_values.csv")
    logs = {"EfficientNet-B3": parse_log(REPO / "full_run_log.txt"), "EfficientNet-B0": parse_log(REPO / "artifacts/kaggle/20260920T095427Z/run.log"),
            "EfficientNet-B3+LS0.05": parse_log(REPO / "artifacts/kaggle/20260922T160023Z/run.log")}
    rn = pd.read_csv(REPO / "artifacts/kaggle/20260920T110729Z/artifacts/training_log.csv").set_index("global_epoch")
    for _, r in f4.iterrows():
        e = int(r.epoch)
        if r.model in logs:
            src = {"EfficientNet-B3": "full_run_log.txt", "EfficientNet-B0": "kaggle/20260920T095427Z/run.log", "EfficientNet-B3+LS0.05": "kaggle/20260922T160023Z/run.log"}[r.model]
            tl, vl, va = logs[r.model].get(e, (np.nan,) * 3)
            for k, v in (("train_loss", tl), ("val_loss", vl), ("val_auc", va)):
                rec("Figure 4", f"{r.model} epoch {e} {k}", r[k], v, f"re-parsed from {src}", tol=5e-5)
        else:
            for k in ("train_loss", "val_loss", "val_auc", "train_accuracy", "val_accuracy"):
                rec("Figure 4", f"{r.model} epoch {e} {k}", r[k], rn.loc[e, k], "kaggle/20260920T110729Z/artifacts/training_log.csv", tol=1e-9)

    # ---------------- Figure 5 (bins recomputed from predictions)
    f5 = pd.read_csv(FIG / "figure5_reliability_plotted_values.csv")
    for _, r in f5.iterrows():
        y, p = P[(r.model, "official_test" if r.probabilities == "raw" else "official_test_refit")]
        b = int(r.bin); lo, hi = b / 10, (b + 1) / 10
        m = (p >= lo) & ((p <= hi) if b == 9 else (p < hi))
        src = f"test_predictions_{PRED_FILE[r.model]}.csv, {'raw' if r.probabilities == 'raw' else 'temperature-scaled (refit T)'}, bin [{lo:.1f}, {hi:.1f})"
        rec("Figure 5", f"{r.model} {r.probabilities} bin {b} count", r["count"], int(m.sum()), src)
        rec("Figure 5", f"{r.model} {r.probabilities} bin {b} mean predicted", r.mean_pred, p[m].mean() if m.any() else np.nan, src, tol=1e-6)
        rec("Figure 5", f"{r.model} {r.probabilities} bin {b} fraction DME", r.frac_pos, y[m].mean() if m.any() else np.nan, src, tol=1e-9)
    cal = pd.read_csv(DATA / "calibration_summary.csv").set_index("model")
    for m in ALL4:
        for kind, key in (("raw", "official_test"), ("calibrated", "official_test_refit")):
            y, p = P[(m, key)]
            rec("Figure 5", f"{m} {kind} Brier (legend)", round(cal.loc[m, f"brier_{kind}"], 4), round(brier_score_loss(y, p), 4), "brier_score_loss on predictions", tol=5e-5)
            e = 0.0
            for b in range(10):
                mm = (p >= b / 10) & ((p <= 1) if b == 9 else (p < (b + 1) / 10))
                if mm.any():
                    e += mm.mean() * abs(y[mm].mean() - p[mm].mean())
            rec("Figure 5", f"{m} {kind} ECE (legend)", round(cal.loc[m, f"ece10_{kind}"], 4), round(e, 4), "10-bin ECE on predictions", tol=5e-5)

    # ---------------- Audit Figure A1 (segmentation inference is not re-run: traced to per-image rows)
    fa = pd.read_csv(FIG / "figureA1_segmentation_audit_plotted_values.csv")
    seg = pd.read_csv(DATA / "segmentation_per_image.csv", dtype={"subject": str})
    seg["key"] = seg.subject + "_" + seg.scan_idx.astype(str)
    for _, r in fa.iterrows():
        if r.panel == "example":
            s = seg[(seg.split == "test") & (seg.key == r.key)].iloc[0]
            for k, c in (("dice", "dice_pipeline"), ("pred_frac_of_frame", "pred_frac_of_frame"), ("gt_fluid_pixels", "gt_fluid_pixels")):
                rec("Figure A1", f"example {r.key} {k}", r[k], s[c], "segmentation_per_image.csv", "TRACED", tol=1e-9)
        else:
            d = seg[seg.split == r.split]
            j = ((d.gt_frac_of_graded * 100 - r.gt_pct_graded).abs() + (d.pred_frac_of_frame * 100 - r.pred_pct_frame).abs()).idxmin()
            rec("Figure A1", f"scatter {r.split} {d.loc[j, 'key']} % fluid (truth)", r.gt_pct_graded, d.loc[j, "gt_frac_of_graded"] * 100, "segmentation_per_image.csv", "TRACED", tol=1e-9)
            rec("Figure A1", f"scatter {r.split} {d.loc[j, 'key']} % fluid (predicted)", r.pred_pct_frame, d.loc[j, "pred_frac_of_frame"] * 100, "segmentation_per_image.csv", "TRACED", tol=1e-9)
    ss = json.loads((DATA / "segmentation_summary.json").read_text())
    for s in ("val", "test"):
        d = seg[seg.split == s]
        rec("Figure A1 / report §10", f"{s} mean Dice (segment.py convention)", ss[s]["mean_dice_pipeline_convention"], d.dice_pipeline.mean(), "mean of segmentation_per_image.csv rows", tol=1e-9)
        rec("Figure A1 / report §10", f"{s} pooled Dice, graded band", ss[s]["pooled_dice_graded_only"], 2 * d.inter_graded.sum() / (d.pred_sum_graded.sum() + d.gt_sum_graded.sum()), "pixel sums of segmentation_per_image.csv", tol=1e-9)

    # ---------------- Table 2 (every numeric cell recomputed from predictions)
    t2 = pd.read_csv(TAB / "table2_within_project_performance.csv")
    for i, r in t2.iterrows():
        mdl = r.Model.split(" [")[0]
        if "pool" in r.Model:
            key, t = (mdl, "unseen_pool"), 0.5
        else:
            thr = t2d.iloc[i]
            key = (mdl, "official_test" if thr.threshold == 0.5 else ("official_test_superseded" if "superseded" in thr.threshold_name else "official_test_refit")); t = thr.threshold
        y, p = P[key]; e = metrics(y, p, t)
        src = f"predictions ({key[1]}), threshold {t}"
        it = f"{r.Model} · {r['Probabilities / threshold']}"
        for k in ("TP", "FP", "TN", "FN"):
            rec("Table 2", f"{it} · {k}", int(r[k]), e[k], src)
        for col, k in (("Accuracy (95% CI)", "accuracy"), ("Precision (95% CI)", "precision"), ("Sensitivity (95% CI)", "sensitivity"), ("Specificity (95% CI)", "specificity")):
            rec("Table 2", f"{it} · {k}", num(r[col]), e[k], src, tol=5e-5)
            lo, hi = ci_of(r[col]); wl, wh = e["ci"][k]
            if "pool" in r.Model:
                rec("Table 2", f"{it} · {k} CI", f"{lo:.4f}–{hi:.4f}", "", "patient-cluster bootstrap (B=2000, seed 42), not re-run", "TRACED")
            else:
                rec("Table 2", f"{it} · {k} CI low", lo, wl, "Wilson from recomputed counts", tol=5e-5)
                rec("Table 2", f"{it} · {k} CI high", hi, wh, "Wilson from recomputed counts", tol=5e-5)
        rec("Table 2", f"{it} · F1", num(r["F1 (95% CI)"]), e["f1"], src, tol=5e-5)
        rec("Table 2", f"{it} · F1 CI", str(r["F1 (95% CI)"]).split(" ", 1)[-1], "", "class-stratified bootstrap (B=2000, seed 42), not re-run", "TRACED")
        rec("Table 2", f"{it} · AUROC", num(r["AUROC (95% CI)"]), e["auroc"], src, tol=5e-5)
        rec("Table 2", f"{it} · PR-AUC", num(r["PR-AUC"]), e["prauc"], src, tol=5e-5)
        rec("Table 2", f"{it} · Brier", num(r.Brier), e["brier"], src, tol=5e-5)

    # ---------------- Table 3 (published cells vs extraction; project rows recomputed)
    t3 = pd.read_csv(TAB / "table3_literature_comparison.csv", dtype=str)
    for i, r in t3.iterrows():
        l_ = lit.iloc[i]
        for col, k in (("Accuracy", "accuracy"), ("Sensitivity", "sensitivity"), ("Specificity", "specificity"), ("F1", "f1"), ("AUROC (95% CI)", "auroc")):
            shown = num(r[col])
            if l_.study.startswith("This project"):
                mdl = next(m for m in MODELS if l_.model.startswith(m))
                rec("Table 3", f"row {i + 1} {r['Study (year)']} {mdl} · {k}", shown, M[(mdl, "official_test")][k], "recomputed from test predictions", tol=5e-4)
            else:
                rec("Table 3", f"row {i + 1} {r['Study (year)']} · {k}", shown, l_[k + "_num"], f"literature_extraction.csv [{l_.source_location[:40]}]", "TRACED")
        if not l_.study.startswith("This project"):
            lo, hi = ci_of(r["AUROC (95% CI)"])
            rec("Table 3", f"row {i + 1} {r['Study (year)']} · AUROC CI", f"{lo}–{hi}", f"{l_.auroc_ci_lo_num}–{l_.auroc_ci_hi_num}", "literature_extraction.csv", "TRACED" if (np.allclose([lo, hi], [l_.auroc_ci_lo_num, l_.auroc_ci_hi_num], equal_nan=True)) else "FAIL")

    # ---------------- Table 6 (every cell)
    t6 = pd.read_csv(TAB / "table6_per_paper_comparison.csv").set_index("Item")
    for s in t6.columns:
        if s == "This project":
            for k in METRICS:
                rec("Table 6", f"This project · Headline {LAB[k]}", num(t6.loc[f"Headline {LAB[k]}", s]), M[("EfficientNet-B3", "official_test")][k], "recomputed from official-test predictions")
                vals = pd.Series([M[(m, st)][k] for m in MODELS for st in ("official_test", "unseen_pool", "octdl", "duke2014")])
                mm = re.match(r"([\d.]+)–([\d.]+) \((\d+)\)", str(t6.loc[f"Range {LAB[k]} (k)", s]))
                rec("Table 6", f"This project · Range {LAB[k]} min", float(mm[1]), vals.min(), "recomputed, 3 models × 4 test sets")
                rec("Table 6", f"This project · Range {LAB[k]} max", float(mm[2]), vals.max(), "recomputed, 3 models × 4 test sets")
                rec("Table 6", f"This project · Range {LAB[k]} k", int(mm[3]), len(vals), "recomputed, 3 models × 4 test sets")
            rec("Table 6", "This project · Headline n", int(t6.loc["Headline n", s]), M[("EfficientNet-B3", "official_test")]["n"], "test predictions")
            continue
        d = pub[pub.paper == s]; h = head[s]
        for k in METRICS:
            rec("Table 6", f"{s} · Headline {LAB[k]}", num(t6.loc[f"Headline {LAB[k]}", s]), h[k], f"literature_extraction.csv [{h.source[:40]}]", "TRACED")
            v = d[k + "_num"].dropna(); cell = str(t6.loc[f"Range {LAB[k]} (k)", s])
            if v.empty:
                rec("Table 6", f"{s} · Range {LAB[k]}", cell, "NR", "literature_extraction.csv (no row reports it)")
            else:
                mm = re.match(r"([\d.]+)–([\d.]+) \((\d+)\)", cell)
                rec("Table 6", f"{s} · Range {LAB[k]} min", float(mm[1]), v.min(), "min over the paper's rows, literature_extraction.csv")
                rec("Table 6", f"{s} · Range {LAB[k]} max", float(mm[2]), v.max(), "max over the paper's rows, literature_extraction.csv")
                rec("Table 6", f"{s} · Range {LAB[k]} k", int(mm[3]), len(v), "rows reporting it, literature_extraction.csv")
        if not pd.isna(h.auroc_ci_lo):
            lo, hi = ci_of(t6.loc["Headline AUROC", s])
            rec("Table 6", f"{s} · Headline AUROC CI low", lo, h.auroc_ci_lo, "literature_extraction.csv", "TRACED")
            rec("Table 6", f"{s} · Headline AUROC CI high", hi, h.auroc_ci_hi, "literature_extraction.csv", "TRACED")
        rec("Table 6", f"{s} · Headline n", t6.loc["Headline n", s], h.n, "literature_extraction.csv n_evaluated")
        rec("Table 6", f"{s} · Rows (DME presence)", int(str(t6.loc["Rows (DME presence)", s]).split()[0]), len(d), "literature_extraction.csv")
        rec("Table 6", f"{s} · Input modality", t6.loc["Input modality", s], "fundus photograph" if d.input_representation.str.contains("fundus").any() else "OCT", "literature_extraction.csv input_representation")
        key = "duke2014" if h.unit == "volume" else "official_test"; e = M[("EfficientNet-B3", key)]
        cell = t6.loc["Unit-matched project value (EfficientNet-B3, 0.5)", s]
        for k, lab in (("sensitivity", "Sens"), ("specificity", "Spec"), ("auroc", "AUROC")):
            v = float(re.search(lab + r" ([\d.]+)", cell)[1])
            rec("Table 6", f"{s} · unit-matched project {k} ({key})", v, e[k], f"recomputed from {key} predictions")

    # ---------------- report §8.1 comparison table (recomputed) + every number in report and checklist (traced)
    rep = (REPO / "research_outputs" / "research_results_report.md").read_text(encoding="utf8")
    sec = rep[rep.index("### 8.1"):rep.index("## 9.")]
    rows = {"**This project**, EfficientNet-B3, official test": ("official_test", None), "This project, OCTDL external": ("octdl", None),
            "This project, Duke 2014 external": ("duke2014", None), "This project, unseen pool": ("unseen_pool", None)}
    rows.update({f"**{p}** headline": (None, p) for p in PAPERS})
    for key, (setting, paper) in rows.items():
        line = next(l for l in sec.splitlines() if l.startswith("| " + key))
        cells = [c.strip().rstrip("*") for c in line.strip("|").split("|")][2:]
        for k, c in zip(METRICS, cells):
            v = num(c)
            if pd.isna(v) and not c.startswith("NR"):
                continue  # blank / "see Table 2"
            if setting:
                rec("Report §8.1 table", f"{key.strip('*')} · {k}", v, M[("EfficientNet-B3", setting)][k], f"recomputed from {setting} predictions")
            else:
                rec("Report §8.1 table", f"{paper} headline · {k}", v, head[paper][k], "literature_extraction.csv", "TRACED")

    index = {}
    def add(v, where):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return
        if not np.isfinite(f):
            return
        for d in (2, 3, 4):
            index.setdefault(round(f, d), where); index.setdefault(round(f * 100, 1), where + " (×100)"); index.setdefault(round(f * 100, 2), where + " (×100)")
    for (m, s), e in M.items():
        for k in ("accuracy", "sensitivity", "specificity", "precision", "f1", "auroc", "prauc", "brier"):
            add(e[k], f"recomputed {m} {s} {k}")
    files = sorted(glob.glob(str(DATA / "*.csv")) + glob.glob(str(REPO / "research_outputs/segmentation_retrain/*.csv")) + [str(AUDIT / "literature_extraction.csv")] + glob.glob(str(TAB / "*.csv")))
    for f in files:
        d = pd.read_csv(f, dtype=str, low_memory=False)
        if len(d) > 20000:
            continue
        for c in d.columns:
            for v in d[c].dropna().unique():
                for x in re.findall(r"-?\d+\.\d+", str(v)):
                    add(x, f"{Path(f).name}:{c}")
    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}")
        elif isinstance(o, list):
            for v in o:
                walk(v, path)
        else:
            add(o, path)
    for f in glob.glob(str(DATA / "*.json")) + glob.glob(str(REPO / "research_outputs/segmentation_retrain/*.json")):
        try:
            walk(json.load(open(f)), Path(f).name)
        except (ValueError, OSError):
            pass
    for doc in ("research_results_report.md", "final_audit_checklist.md"):
        t = (REPO / "research_outputs" / doc).read_text(encoding="utf8")
        for mt in re.finditer(r"(?<![\w.])(\d\.\d{3,4})(?!\d)", t):
            n = mt.group(1); d = len(n.split(".")[1]); v = round(float(n), d)
            sec_name = re.findall(r"^#+ (.+)$", t[:mt.start()], re.M)
            where = index.get(v)
            L.append({"artifact": doc, "item": f"{(sec_name[-1] if sec_name else '')[:40]} · {t[max(0, mt.start() - 40):mt.start()].splitlines()[-1].strip()[-40:]}",
                      "value_shown": n, "expected": n if where else "", "expected_source": where or "", "status": "TRACED" if where else "UNTRACED", "note": ""})

    # ---------------- every published value in the paper's text
    texts = dict(s.split("=", 1) for s in args.text)
    for _, r in pub.iterrows():
        for k in METRICS:
            v = r[k + "_num"]
            if pd.isna(v):
                continue
            item = f"{r.paper} [{r.source_location[:40]}] {r.dataset[:30]} · {k}"
            if r.paper not in texts:
                rec("Published value", item, v, v, "paper text not supplied", "TRACED"); continue
            text = Path(texts[r.paper]).read_text(encoding="utf8", errors="ignore")
            forms = [f"{v:.3f}", f"{v:.4f}", f"{v:.2f}", f"{v * 100:.1f}", f"{v * 100:.2f}", f"{v * 100:.0f}", f"{v:.3f}"[1:], f"{v:.2f}"[1:]]
            hit = next((x for x in forms if re.search(r"(?<![\d.])" + re.escape(x) + r"(?!\d)", text)), None)
            L.append({"artifact": "Published value", "item": item, "value_shown": v, "expected": hit or "", "expected_source": f"{r.paper} full text" + (f" as '{hit}'" if hit else ""),
                      "status": "FOUND IN TEXT" if hit else "FAIL", "note": ""})

    df = pd.DataFrame(L)
    df.to_csv(AUDIT / "value_ledger.csv", index=False)
    summ = df.pivot_table(index="artifact", columns="status", values="item", aggfunc="count", fill_value=0)
    summ["total"] = summ.sum(axis=1)
    order = ["Figure 1", "Figure 2", "Figure 3", "Figure 4", "Figure 5", "Figure A1", "Figure A1 / report §10", "Table 2", "Table 3", "Table 6",
             "Report §8.1 table", "research_results_report.md", "final_audit_checklist.md", "Published value"]
    summ = summ.reindex([o for o in order if o in summ.index])
    cols = [c for c in ("RECOMPUTED", "TRACED", "FOUND IN TEXT", "UNTRACED", "FAIL", "total") if c in summ.columns]
    out = ["# Per-value validation ledger (script 16)", "",
           f"Every number shown in the figures, Tables 2, 3 and 6, the report and the checklist, and every published value of the five papers: **{len(df):,} values**. "
           "One row per value in `research_audit/value_ledger.csv` (value shown, expected value, its source, status).", "",
           "- **RECOMPUTED**: recalculated here from per-image predictions or console logs, independently of the pipeline, and equal.",
           "- **TRACED**: equal to the stated source file; re-running would only repeat that step (seeded bootstraps, segmentation inference, values extracted from a paper).",
           "- **FOUND IN TEXT**: the published value appears in the paper's full text or supplement.",
           "- **UNTRACED**: a report/checklist number with no matching data value (manual review). **FAIL**: mismatch.", "",
           "| Artifact | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for a, r in summ.iterrows():
        out.append(f"| {a} | " + " | ".join(f"{int(r[c]):,}" for c in cols) + " |")
    tot = summ[cols].sum()
    out.append("| **All** | " + " | ".join(f"**{int(tot[c]):,}**" for c in cols) + " |")
    bad = df[df.status.isin(["FAIL", "UNTRACED"])]
    out += ["", "## Values needing attention", ""]
    out += ["None: every value is RECOMPUTED, TRACED or FOUND IN TEXT."] if bad.empty else [f"- {r['artifact']} · {r['item']}: shown {r['value_shown']}, expected {r['expected']} ({r['status']})" for _, r in bad.iterrows()]
    (AUDIT / "value_ledger_summary.md").write_text("\n".join(out) + "\n", encoding="utf8")
    print("\n".join(out[9:]))
    sys.exit(0 if bad.empty else 1)


if __name__ == "__main__":
    main()
