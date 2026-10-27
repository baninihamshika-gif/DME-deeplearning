"""
Phase B2/D -- independent metric reconstruction and statistics from the
per-image prediction CSVs written by 01_reinfer_classifiers.py (and, if
present, 03_reinfer_excluded_pool.py). Every number in research_outputs/tables
and every figure is generated from the CSVs this script writes.

Positive class: DME (label 1). Normal = 0. Scores: softmax P(DME).

Metrics (per model, per threshold):
  confusion matrix computed two ways (manual counts and
  sklearn.metrics.confusion_matrix) and asserted equal; accuracy, precision,
  sensitivity, specificity, F1 (undefined values reported as NaN with a note,
  never 0); ROC-AUC (sklearn) cross-checked against an independent
  Mann-Whitney/DeLong implementation; PR-AUC (average precision); Brier;
  log loss; ECE (10 equal-width bins, same binning as evaluate.py).

Uncertainty:
  * proportions: Wilson score 95% CI (closed form, no resampling)
  * F1, and all metrics on the clustered excluded pool: percentile bootstrap,
    B=2000, seed 42 -- stratified by class on the test set; resampling
    PATIENTS (cluster bootstrap) on the excluded pool, where patients
    contribute many images
  * AUROC: DeLong (1988) variance, fast midrank algorithm (Sun & Xu 2014);
    when AUROC = 1 the DeLong variance is exactly 0 and the interval is
    degenerate -- this is reported as such, not presented as certainty
Paired model comparisons (same 484 test images):
  * exact McNemar test on discordant classifications at 0.5
  * paired DeLong test on AUROC (undefined when both AUROCs are 1)
  * Holm correction across the 3 pairwise comparisons per test family

Usage: python research_outputs/scripts/04_compute_metrics.py
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "research_outputs" / "data"
SEED = 42
B = 2000

MODELS = {  # display name -> (prediction csv slug, metrics.json with the originally reported values)
    "EfficientNet-B3": ("EfficientNet-B3", "artifacts/metrics.json"),
    "EfficientNet-B0": ("EfficientNet-B0", "artifacts/baselines/tf_efficientnet_b0/metrics.json"),
    "ResNet-50": ("ResNet-50", "artifacts/baselines/resnet50/metrics.json"),
    "EfficientNet-B3 + LS 0.05 (ablation)": ("EfficientNet-B3_LS005", "artifacts/ablation/label_smoothing_0.05/metrics.json"),
}
PRIMARY_COMPARISON = ["EfficientNet-B3", "EfficientNet-B0", "ResNet-50"]


# --------------------------------------------------------------------------- helpers
def counts(y, yhat):
    tp = int(((yhat == 1) & (y == 1)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    tn2, fp2, fn2, tp2 = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    assert (tp, fn, tn, fp) == (tp2, fn2, tn2, fp2), "manual vs sklearn confusion matrix mismatch"
    return tp, fp, tn, fn


def safe_div(a, b):
    return a / b if b > 0 else float("nan")


def point_metrics(tp, fp, tn, fn):
    sens = safe_div(tp, tp + fn); spec = safe_div(tn, tn + fp); prec = safe_div(tp, tp + fp)
    f1 = safe_div(2 * prec * sens, prec + sens) if not (np.isnan(prec) or np.isnan(sens)) else float("nan")
    return {"accuracy": safe_div(tp + tn, tp + tn + fp + fn), "precision": prec, "sensitivity": sens, "specificity": spec, "f1": f1}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x); t = np.zeros(n); i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1; i = k
    out = np.empty(n); out[j] = t
    return out


def delong(y, scores_list):
    """Fast DeLong. Returns (aucs, covariance matrix)."""
    y = np.asarray(y); pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    m, n = len(pos), len(neg); k = len(scores_list)
    v01 = np.zeros((k, m)); v10 = np.zeros((k, n)); aucs = np.zeros(k)
    for r, s in enumerate(scores_list):
        s = np.asarray(s, float); x, yy = s[pos], s[neg]
        tx, ty, tz = midrank(x), midrank(yy), midrank(np.r_[x, yy])
        aucs[r] = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        v01[r] = (tz[:m] - tx) / n
        v10[r] = 1.0 - (tz[m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_ci(auc, var, z=1.959964):
    if var <= 0:
        return (auc, auc), True
    se = np.sqrt(var)
    return (max(0.0, auc - z * se), min(1.0, auc + z * se)), False


def ece(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1); rows = []; e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & ((p <= hi) if i == n_bins - 1 else (p < hi))
        if m.sum() == 0:
            rows.append({"bin": i, "lo": lo, "hi": hi, "count": 0, "mean_pred": np.nan, "frac_pos": np.nan}); continue
        rows.append({"bin": i, "lo": lo, "hi": hi, "count": int(m.sum()), "mean_pred": float(p[m].mean()), "frac_pos": float(y[m].mean())})
        e += m.sum() / len(y) * abs(y[m].mean() - p[m].mean())
    return float(e), rows


def bootstrap(y, p, thr, groups=None, stratified=True):
    rng = np.random.default_rng(SEED); out = {k: [] for k in ("accuracy", "precision", "sensitivity", "specificity", "f1", "auroc")}
    if groups is not None:
        ug = np.unique(groups); idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(B):
        if groups is not None:
            gs = rng.choice(ug, size=len(ug), replace=True); idx = np.concatenate([idx_by_g[g] for g in gs])
        elif stratified:
            idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        else:
            idx = rng.integers(0, len(y), len(y))
        yy, pp = y[idx], p[idx]
        pm = point_metrics(*counts(yy, (pp >= thr).astype(int)))
        for k in pm:
            out[k].append(pm[k])
        out["auroc"].append(roc_auc_score(yy, pp) if len(np.unique(yy)) == 2 else np.nan)
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in out.items()}


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i])); adj[i] = running
    return adj


# --------------------------------------------------------------------------- main
def evaluate_set(name, y, p, thresholds, groups=None, source="", t_note=""):
    rows = []
    aucs, cov = delong(y, [p])
    auc_sk = roc_auc_score(y, p)
    assert abs(aucs[0] - auc_sk) < 1e-9, "DeLong AUC != sklearn AUC"
    ci_auc, degenerate = delong_ci(aucs[0], cov[0, 0])
    for thr_name, thr in thresholds:
        tp, fp, tn, fn = counts(y, (p >= thr).astype(int))
        pm = point_metrics(tp, fp, tn, fn)
        bs = bootstrap(y, p, thr, groups=groups)
        r = {"model": name, "threshold_name": thr_name, "threshold": thr, "n": len(y), "n_dme": int(y.sum()), "n_normal": int((y == 0).sum()),
             "TP": tp, "FP": fp, "TN": tn, "FN": fn, **pm}
        if groups is None:
            for k, (num, den) in {"accuracy": (tp + tn, len(y)), "precision": (tp, tp + fp), "sensitivity": (tp, tp + fn), "specificity": (tn, tn + fp)}.items():
                r[f"{k}_ci_lo"], r[f"{k}_ci_hi"] = wilson(num, den)
            r["proportion_ci_method"] = "Wilson score"
        else:
            for k in ("accuracy", "precision", "sensitivity", "specificity"):
                r[f"{k}_ci_lo"], r[f"{k}_ci_hi"] = bs[k]
            r["proportion_ci_method"] = f"patient-cluster percentile bootstrap B={B} seed={SEED}"
        r["f1_ci_lo"], r["f1_ci_hi"] = bs["f1"]
        r["f1_ci_method"] = f"{'patient-cluster' if groups is not None else 'class-stratified'} percentile bootstrap B={B} seed={SEED}"
        r["auroc"] = float(aucs[0]); r["auroc_ci_lo"], r["auroc_ci_hi"] = ci_auc
        r["auroc_ci_method"] = "DeLong" + (" (degenerate: variance = 0 because classes are perfectly separated; interval does not represent true uncertainty)" if degenerate else "")
        if groups is not None:
            r["auroc_cluster_bootstrap_ci_lo"], r["auroc_cluster_bootstrap_ci_hi"] = bs["auroc"]
        r["pr_auc_average_precision"] = float(average_precision_score(y, p))
        r["brier"] = float(brier_score_loss(y, p)); r["log_loss"] = float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]))
        r["ece_10bin"] = ece(y, p)[0]
        r["prediction_source"] = source; r["notes"] = t_note
        rows.append(r)
    return rows


def main():
    rows, recon_vs_reported, calib_rows, rel_bins = [], [], [], []
    preds = {}
    refit_path = DATA / "refit_calibration_threshold.csv"
    refit_params = {}
    if refit_path.exists():
        rf = pd.read_csv(refit_path)
        refit_params = {r.model: (r.temperature_refit, r.threshold_refit) for _, r in rf.iterrows()}
    for name, (slug, mpath) in MODELS.items():
        d = pd.read_csv(DATA / f"test_predictions_{slug}.csv")
        preds[name] = d
        y = d.label.to_numpy(); p_raw = d.p_dme_raw.to_numpy(); p_cal = d.p_dme_calibrated.to_numpy()
        rep = json.loads((REPO / mpath).read_text())
        thr_t = rep["tuned_threshold"]; src = f"research_outputs/data/test_predictions_{slug}.csv"
        rows += evaluate_set(name, y, p_raw, [("default 0.5 (raw probabilities)", 0.5)], source=src)
        rows += evaluate_set(name, y, p_cal, [(f"superseded tuned {thr_t} (threshold and temperature fitted on a contaminated validation set, see leakage_audit.md §5)", thr_t)], source=src,
                             t_note="tuned operating point not methodologically valid; reported only to reconcile with existing metrics.json")
        # reconstructed vs reported
        for key, thr, pp in (("test_at_default_threshold", 0.5, p_raw), ("test_at_tuned_threshold", thr_t, p_cal)):
            tp, fp, tn, fn = counts(y, (pp >= thr).astype(int)); pm = point_metrics(tp, fp, tn, fn)
            r = rep[key]
            for metric, recon in [("TP", tp), ("FP", fp), ("TN", tn), ("FN", fn), ("accuracy", pm["accuracy"]), ("sensitivity", pm["sensitivity"]),
                                  ("specificity", pm["specificity"]), ("precision", pm["precision"]), ("f1", pm["f1"]), ("auc", roc_auc_score(y, pp))]:
                rv = r[metric.lower()] if metric.lower() in r else r[metric]
                recon_vs_reported.append({"model": name, "setting": key, "threshold": thr, "metric": metric, "reported_metrics_json": rv, "reconstructed": recon,
                                          "abs_diff": abs(float(rv) - float(recon)), "match": abs(float(rv) - float(recon)) < 1e-9, "reported_source": mpath})
        # Valid operating point: temperature and threshold refit on the training-time validation set (script 12).
        refit = refit_params.get(name)
        if refit is not None:
            t_ref, thr_ref = refit
            z = d[["logit_normal", "logit_dme"]].to_numpy() / t_ref
            z = z - z.max(axis=1, keepdims=True)
            p_ref = np.exp(z)[:, 1] / np.exp(z).sum(axis=1)
            rows += evaluate_set(name, y, p_ref, [(f"refit tuned {thr_ref} (temperature {t_ref:.4f} and threshold fitted on the training-time validation set, n=5,560)", thr_ref)], source=src,
                                 t_note="valid tuned operating point: fitted on the scikit-learn 1.4.2 training-time validation split (research_outputs/scripts/12_refit_calibration_threshold.py)")
        else:
            t_ref, p_ref = rep["temperature"], p_cal
        e_raw, bins_raw = ece(y, p_raw); e_cal, bins_cal = ece(y, p_ref); e_old, bins_old = ece(y, p_cal)
        calib_rows.append({"model": name, "temperature": t_ref,
                           "temperature_fit_set": "training-time validation split, n=5,560 (scikit-learn 1.4.2)" if refit is not None else "local re-split 'validation' n=6047 (contaminated)",
                           "brier_raw": brier_score_loss(y, p_raw), "brier_calibrated": brier_score_loss(y, p_ref),
                           "log_loss_raw": log_loss(y, np.clip(p_raw, 1e-15, 1 - 1e-15)), "log_loss_calibrated": log_loss(y, np.clip(p_ref, 1e-15, 1 - 1e-15)),
                           "ece10_raw": e_raw, "ece10_calibrated": e_cal,
                           "temperature_contaminated": rep["temperature"], "brier_calibrated_contaminated": brier_score_loss(y, p_cal), "ece10_calibrated_contaminated": e_old,
                           "ece10_raw_reported": rep["calibration"]["ece_before"], "ece10_calibrated_contaminated_reported": rep["calibration"]["ece_after"]})
        for kind, bins in (("raw", bins_raw), ("calibrated", bins_cal), ("calibrated_contaminated", bins_old)):
            for b in bins:
                rel_bins.append({"model": name, "probabilities": kind, **b})

    perf = pd.DataFrame(rows)
    perf.to_csv(DATA / "table2_within_project_performance.csv", index=False)
    pd.DataFrame(recon_vs_reported).to_csv(DATA / "reconstructed_vs_reported.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(DATA / "calibration_summary.csv", index=False)
    pd.DataFrame(rel_bins).to_csv(DATA / "reliability_bins.csv", index=False)

    # ROC curve coordinates (raw probabilities)
    from sklearn.metrics import roc_curve
    roc_rows = []
    for name in MODELS:
        d = preds[name]; fpr, tpr, thr = roc_curve(d.label, d.p_dme_raw)
        roc_rows += [{"model": name, "fpr": a, "tpr": b, "threshold": c} for a, b, c in zip(fpr, tpr, thr)]
    pd.DataFrame(roc_rows).to_csv(DATA / "roc_coordinates_test.csv", index=False)

    # Paired comparisons among the three architectures (same 484 images, threshold 0.5, raw)
    y = preds[PRIMARY_COMPARISON[0]].label.to_numpy()
    for n in PRIMARY_COMPARISON[1:]:
        assert (preds[n].filename.values == preds[PRIMARY_COMPARISON[0]].filename.values).all(), "test rows not aligned"
    pairs = []
    for a, b in itertools.combinations(PRIMARY_COMPARISON, 2):
        ca = (preds[a].p_dme_raw.to_numpy() >= 0.5).astype(int) == y; cb = (preds[b].p_dme_raw.to_numpy() >= 0.5).astype(int) == y
        n01 = int((ca & ~cb).sum()); n10 = int((~ca & cb).sum())
        p_mc = float(stats.binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 > 0 else 1.0
        aucs, cov = delong(y, [preds[a].p_dme_raw, preds[b].p_dme_raw])
        var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
        p_dl = float(2 * stats.norm.sf(abs(aucs[0] - aucs[1]) / np.sqrt(var))) if var > 0 else float("nan")
        pairs.append({"model_a": a, "model_b": b, "a_correct_b_wrong": n01, "a_wrong_b_correct": n10, "mcnemar_exact_p": p_mc,
                      "auroc_a": aucs[0], "auroc_b": aucs[1], "delong_paired_p": p_dl,
                      "delong_note": "undefined: both AUROC = 1, zero variance" if not var > 0 else ""})
    pairs = pd.DataFrame(pairs)
    pairs["mcnemar_holm_p"] = holm(pairs.mcnemar_exact_p.to_numpy())
    pairs.to_csv(DATA / "paired_model_comparisons.csv", index=False)

    # Per-image disagreement on the test set
    dis = preds[PRIMARY_COMPARISON[0]][["filename", "label"]].copy()
    for n in MODELS:
        dis[f"p_{n}"] = preds[n].p_dme_raw.values
    dis[(dis.filter(like="p_") < 0.5).any(axis=1) & (dis.label == 1) | (dis.filter(like="p_") >= 0.5).any(axis=1) & (dis.label == 0)].to_csv(DATA / "test_misclassified_any_model.csv", index=False)

    # Supplementary: excluded-patient pool
    exc_path = DATA / "excluded_pool_predictions.csv"
    if exc_path.exists():
        e = pd.read_csv(exc_path, dtype={"patient": str}); erows = []
        for name in PRIMARY_COMPARISON:
            slug = name.replace("-", "")
            erows += evaluate_set(name, e.label.to_numpy(), e[f"p_dme_raw_{slug}"].to_numpy(), [("default 0.5 (raw probabilities)", 0.5)],
                                  groups=e.patient.to_numpy(), source="research_outputs/data/excluded_pool_predictions.csv",
                                  t_note="NEW analysis; images of the 305 test-set patients excluded from training/validation; clustered by patient; not independent of the official test set")
        pd.DataFrame(erows).to_csv(DATA / "supplementary_excluded_pool_performance.csv", index=False)
        from sklearn.metrics import roc_curve
        rr = []
        for name in PRIMARY_COMPARISON:
            slug = name.replace("-", "")
            fpr, tpr, thr = roc_curve(e.label, e[f"p_dme_raw_{slug}"])
            rr += [{"model": name, "fpr": a, "tpr": b, "threshold": c} for a, b, c in zip(fpr, tpr, thr)]
        pd.DataFrame(rr).to_csv(DATA / "roc_coordinates_excluded_pool.csv", index=False)

    print(perf[["model", "threshold", "TP", "FP", "TN", "FN", "accuracy", "sensitivity", "specificity", "f1", "auroc", "auroc_ci_lo", "auroc_ci_hi"]].to_string())
    print(pairs.to_string())
    print("all reconstructed == reported:", pd.DataFrame(recon_vs_reported).match.all())


if __name__ == "__main__":
    main()
