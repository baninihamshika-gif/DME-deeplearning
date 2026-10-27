"""
Per-value validation of every result value of this project that script 16
does not already cover: every numeric cell of the project's result files and
the dataset counts, recomputed from first sources where the inputs exist.

  table2_within_project_performance.csv     official test, all 12 operating points
  supplementary_excluded_pool_performance.csv
  external_validation_performance.csv        OCTDL image level, Duke 2014 max/mean rules, raw + refit
  external_validation_patient_level.csv      OCTDL patient = max over images
  external_validation_duke2015.csv           sensitivity on DME-only eyes
  external_validation_duke2014_posthoc.csv   central B-scan per volume
  calibration_summary.csv                    Brier, log loss, ECE (raw, refit, superseded)
  paired_model_comparisons.csv               McNemar exact p, Holm
  refit_calibration_threshold.csv            temperature + threshold refit on the validation split
  leakage_summary.json                       dataset counts, overlap, duplicates, image widths
  segmentation_summary.json                  original U-Net audit (from per-image rows)
  segmentation_retrain/segmentation_summary.json     retrained U-Net v1, superseded (from per-image rows)
  segmentation_retrain_v2/segmentation_summary.json  retrained U-Net v2, current (from per-image rows)
  gradcam_localisation/summary.json          Grad-CAM localisation on Duke 2015 (from per-scan rows)
  tables/table1_dataset_setup.csv            dataset counts
  research_results_report.md                 every remaining number (counts, percentages, 1-2 decimals)

Statuses: RECOMPUTED (independent recalculation, equal), TRACED (equal to its
source; recalculating would repeat a seeded bootstrap or model inference),
SOURCE ONLY (exists only in its own output, no saved inputs to recompute it
from), FAIL, UNTRACED (report number with no data value; listed for review).

Output: research_audit/value_ledger_project.csv (+ summary appended to
research_audit/value_ledger_summary.md by the page builder).
Usage: python research_outputs/scripts/17_project_value_ledger.py
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "research_outputs" / "data"
TAB = REPO / "research_outputs" / "tables"
AUDIT = REPO / "research_audit"
RT = REPO / "research_outputs" / "segmentation_retrain"      # v1, superseded
RT2 = REPO / "research_outputs" / "segmentation_retrain_v2"  # v2, current (app.py)
spec = importlib.util.spec_from_file_location("ledger16", Path(__file__).with_name("16_value_ledger.py"))
L16 = importlib.util.module_from_spec(spec); spec.loader.exec_module(L16)
MODELS, ALL4, PRED_FILE, wilson = L16.MODELS, L16.ALL4, L16.PRED_FILE, L16.wilson

L = []


def rec(artifact, item, shown, expected, source, status="RECOMPUTED", tol=1e-6):
    if status in ("RECOMPUTED", "TRACED"):
        if isinstance(shown, str) or isinstance(expected, str):
            ok = str(shown) == str(expected)
        else:
            ok = (pd.isna(shown) and pd.isna(expected)) or (not (pd.isna(shown) or pd.isna(expected)) and abs(float(shown) - float(expected)) <= tol * max(1, abs(float(expected))))
        status = status if ok else "FAIL"
    L.append({"artifact": artifact, "item": item, "value_shown": shown, "expected": expected, "expected_source": source, "status": status, "note": ""})


def ece10(y, p):
    e = 0.0
    for b in range(10):
        m = (p >= b / 10) & ((p <= 1) if b == 9 else (p < (b + 1) / 10))
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def full_metrics(y, p, t):
    y = np.asarray(y).astype(int); p = np.asarray(p, float); pr = (p >= t).astype(int)
    tp = int(((pr == 1) & (y == 1)).sum()); fp = int(((pr == 1) & (y == 0)).sum()); tn = int(((pr == 0) & (y == 0)).sum()); fn = int(((pr == 0) & (y == 1)).sum())
    n = len(y)
    out = {"n": n, "n_dme": int(y.sum()), "n_normal": int((y == 0).sum()), "TP": tp, "FP": fp, "TN": tn, "FN": fn,
           "accuracy": (tp + tn) / n, "precision": tp / (tp + fp) if tp + fp else np.nan, "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
           "specificity": tn / (tn + fp) if tn + fp else np.nan, "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}
    for k, (a, b) in (("accuracy", (tp + tn, n)), ("precision", (tp, tp + fp)), ("sensitivity", (tp, tp + fn)), ("specificity", (tn, tn + fp))):
        out[f"{k}_ci_lo"], out[f"{k}_ci_hi"] = wilson(a, b)
    if 0 < y.sum() < n:
        out.update(auroc=roc_auc_score(y, p), pr_auc_average_precision=average_precision_score(y, p))
    out.update(brier=brier_score_loss(y, p), log_loss=log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]), ece_10bin=ece10(y, p))
    return out


RECOMP = ["n", "n_dme", "n_normal", "TP", "FP", "TN", "FN", "accuracy", "precision", "sensitivity", "specificity", "f1",
          "accuracy_ci_lo", "accuracy_ci_hi", "precision_ci_lo", "precision_ci_hi", "sensitivity_ci_lo", "sensitivity_ci_hi", "specificity_ci_lo", "specificity_ci_hi",
          "auroc", "pr_auc_average_precision", "brier", "log_loss", "ece_10bin"]
BOOT = ["f1_ci_lo", "f1_ci_hi", "auroc_ci_lo", "auroc_ci_hi", "auroc_cluster_bootstrap_ci_lo", "auroc_cluster_bootstrap_ci_hi"]


def check_perf_row(artifact, label, row, y, p, t, src):
    e = full_metrics(y, p, t)
    boot_prop = "bootstrap" in str(row.get("proportion_ci_method", ""))
    for k in RECOMP:
        if k in row and not (pd.isna(row[k]) and k not in e):
            if boot_prop and k.endswith(("_ci_lo", "_ci_hi")):
                base = k.rsplit("_ci_", 1)[0]; inside = row[f"{base}_ci_lo"] - 1e-9 <= e[base] <= row[f"{base}_ci_hi"] + 1e-9
                rec(artifact, f"{label} · {k}", row[k], row[k] if inside else np.nan,
                    f"{row['proportion_ci_method']} (seeded; not re-run); recomputed point estimate lies inside", "TRACED")
            else:
                rec(artifact, f"{label} · {k}", row[k], e.get(k, np.nan), src)
    for k in BOOT:
        if k in row and not pd.isna(row[k]):
            meth = row.get("f1_ci_method" if k.startswith("f1") else ("auroc_ci_method" if k.startswith("auroc_ci") else "auroc_cluster_bootstrap_ci_lo"), "")
            rec(artifact, f"{label} · {k}", row[k], row[k], f"{meth or 'patient-cluster bootstrap'} (seeded; not re-run)", "TRACED")


def main():
    T = pd.read_csv(DATA / "calibration_summary.csv").set_index("model").temperature
    sig = lambda d, a, b, t: 1 / (1 + np.exp(-(d[b] - d[a]) / t))

    # ---------------- official test: all 12 operating points
    t2 = pd.read_csv(DATA / "table2_within_project_performance.csv")
    for _, r in t2.iterrows():
        d = pd.read_csv(DATA / f"test_predictions_{PRED_FILE[r.model]}.csv")
        if r.threshold == 0.5:
            p, kind = d.p_dme_raw, "raw softmax"
        elif "superseded" in r.threshold_name:
            p, kind = d.p_dme_calibrated, "superseded temperature (stored in checkpoint)"
        else:
            p, kind = sig(d, "logit_normal", "logit_dme", T[r.model]), f"refit temperature {T[r.model]:.4f}"
        check_perf_row("Results: official test (Table 2 data)", f"{r.model} @ {r.threshold} ({kind})", r, d.label, p, r.threshold, f"test_predictions_{PRED_FILE[r.model]}.csv, {kind}")

    # ---------------- unseen pool
    pool = pd.read_csv(DATA / "excluded_pool_predictions.csv")
    for _, r in pd.read_csv(DATA / "supplementary_excluded_pool_performance.csv").iterrows():
        check_perf_row("Results: unseen pool", f"{r.model} @ 0.5", r, pool.label, pool[f"p_dme_raw_{r.model.replace('-', '')}"], 0.5, "excluded_pool_predictions.csv")

    # ---------------- external validation (image / volume level)
    oc = pd.read_csv(DATA / "external_predictions_octdl.csv"); du = pd.read_csv(DATA / "external_predictions_duke2014.csv")
    for _, r in pd.read_csv(DATA / "external_validation_performance.csv").iterrows():
        col = f"p_dme_{'raw' if r.threshold_name.startswith('default') else 'refit'}__{r.model}"
        if r.dataset.startswith("OCTDL"):
            y, p, src = oc.label, oc[col], "external_predictions_octdl.csv"
        else:
            agg = "max" if "max over" in r.threshold_name else "mean"
            v = du.groupby("volume").agg(y=("label", "first"), p=(col, agg)); y, p = v.y, v.p
            src = f"external_predictions_duke2014.csv, volume = {agg} over B-scans"
        check_perf_row("Results: external validation", f"{r.dataset.split(',')[0]} · {r.model} · {r.threshold_name[:40]}", r, y, p, r.threshold, src + f", {col.split('__')[0][6:]}")

    # ---------------- OCTDL patient level
    for _, r in pd.read_csv(DATA / "external_validation_patient_level.csv").iterrows():
        v = oc.groupby("patient").agg(y=("label", "max"), p=(f"p_dme_raw__{r.model}", "max"))
        check_perf_row("Results: OCTDL patient level", f"{r.model} @ 0.5", r, v.y, v.p, 0.5, "external_predictions_octdl.csv, patient = max over images")

    # ---------------- Duke 2015 (DME eyes only)
    dk = pd.read_csv(DATA / "external_predictions_duke2015.csv", dtype={"subject": str})
    for _, r in pd.read_csv(DATA / "external_validation_duke2015.csv").iterrows():
        p = dk[f"p_dme_{'raw' if r.threshold_name.startswith('default') else 'refit'}__{r.model}"]; th = r.threshold
        pos = dk.fluid_present_grader1 == 1
        k, n = int((p[pos] >= th).sum()), int(pos.sum()); subj = p.groupby(dk.subject).max(); ks, ns = int((subj >= th).sum()), len(subj)
        exp = {"fluid_bscans_n": n, "fluid_bscans_called_dme": k, "sensitivity_fluid_bscans": k / n, "sensitivity_ci_lo": wilson(k, n)[0], "sensitivity_ci_hi": wilson(k, n)[1],
               "subjects_n": ns, "subjects_called_dme_max_rule": ks, "subject_sensitivity": ks / ns, "subject_sensitivity_ci_lo": wilson(ks, ns)[0],
               "subject_sensitivity_ci_hi": wilson(ks, ns)[1], "fluid_free_bscans_n": int((~pos).sum()), "fluid_free_bscans_called_dme_fraction": float((p[~pos] >= th).mean())}
        for c, v in exp.items():
            rec("Results: Duke 2015", f"{r.model} @ {th} · {c}", r[c], v, "external_predictions_duke2015.csv")

    # ---------------- Duke 2014 post hoc, central B-scan
    d = du.copy(); d["idx"] = d.file_name.str.extract(r"/(\d+)\.tif", expand=False).astype(int)
    d["nmax"] = d.groupby("volume").idx.transform("max"); c = d[d.idx == (d.nmax + 1) // 2]
    for _, r in pd.read_csv(DATA / "external_validation_duke2014_posthoc.csv").iterrows():
        col = f"p_dme_raw__{r.model}"; y, p = c.label.values, c[col].values
        tp, fp = int(((p >= .5) & (y == 1)).sum()), int(((p >= .5) & (y == 0)).sum()); npos, nneg = int(y.sum()), int((y == 0).sum())
        exp = {"TP": tp, "FN": npos - tp, "FP": fp, "TN": nneg - fp, "sensitivity": tp / npos, "specificity": (nneg - fp) / nneg,
               "sensitivity_ci_lo": wilson(tp, npos)[0], "sensitivity_ci_hi": wilson(tp, npos)[1], "specificity_ci_lo": wilson(nneg - fp, nneg)[0],
               "specificity_ci_hi": wilson(nneg - fp, nneg)[1], "auroc": roc_auc_score(y, p),
               "all_bscans_normal_called_dme_fraction": float((d.loc[d.label == 0, col] >= .5).mean()),
               "all_bscans_dme_volume_called_dme_fraction": float((d.loc[d.label == 1, col] >= .5).mean())}
        for k, v in exp.items():
            rec("Results: Duke 2014 post hoc (central B-scan)", f"{r.model} · {k}", r[k], v, "external_predictions_duke2014.csv, central B-scan (n+1)//2 per volume")

    # ---------------- calibration summary
    for _, r in pd.read_csv(DATA / "calibration_summary.csv").iterrows():
        dd = pd.read_csv(DATA / f"test_predictions_{PRED_FILE[r.model]}.csv"); y = dd.label.values
        raw, ref, old = dd.p_dme_raw.values, sig(dd, "logit_normal", "logit_dme", r.temperature).values, dd.p_dme_calibrated.values
        old2 = sig(dd, "logit_normal", "logit_dme", r.temperature_contaminated).values
        for k, v, src in (("brier_raw", brier_score_loss(y, raw), "raw"), ("brier_calibrated", brier_score_loss(y, ref), "refit T"),
                          ("log_loss_raw", log_loss(y, np.clip(raw, 1e-15, 1 - 1e-15)), "raw"), ("log_loss_calibrated", log_loss(y, np.clip(ref, 1e-15, 1 - 1e-15)), "refit T"),
                          ("ece10_raw", ece10(y, raw), "raw"), ("ece10_calibrated", ece10(y, ref), "refit T"),
                          ("brier_calibrated_contaminated", brier_score_loss(y, old), "stored (superseded) calibrated probabilities"),
                          ("ece10_calibrated_contaminated", ece10(y, old), "stored (superseded) calibrated probabilities")):
            rec("Results: calibration", f"{r.model} · {k}", r[k], v, f"test predictions, {src}")
        rec("Results: calibration", f"{r.model} · superseded calibrated probabilities = softmax(logits / stored T)", float(np.abs(old - old2).max()), 0.0, "max abs difference", tol=1e-5)
        rec("Results: calibration", f"{r.model} · temperature (refit)", r.temperature, pd.read_csv(DATA / "refit_calibration_threshold.csv").set_index("model").loc[r.model, "temperature_refit"], "refit_calibration_threshold.csv", "TRACED")
        meta = json.loads((DATA / "checkpoint_metadata.json").read_text())
        key = r.model if r.model in meta else "EfficientNet-B3+LS0.05"
        rec("Results: calibration", f"{r.model} · temperature (superseded)", r.temperature_contaminated, meta[key]["stored_temperature"], "checkpoint_metadata.json (stored in checkpoint)", "TRACED")
        for k in ("ece10_raw_reported", "ece10_calibrated_contaminated_reported"):
            rec("Results: calibration", f"{r.model} · {k}", r[k], r[k], "original evaluation output (metrics.json / run log)", "SOURCE ONLY")

    # ---------------- refit temperature + threshold on the validation split
    val = pd.read_csv(DATA / "val_predictions_true_split.csv")
    for _, r in pd.read_csv(DATA / "refit_calibration_threshold.csv").iterrows():
        sfx = PRED_FILE[r.model].replace("EfficientNet-B3_LS005", "EfficientNet-B3_LS005")
        a, b = val[f"logit_normal_{sfx}"].values, val[f"logit_dme_{sfx}"].values; y = val.label.values
        def nll(t):
            z = np.stack([a, b], 1) / t; z = z - z.max(1, keepdims=True); lp = z - np.log(np.exp(z).sum(1, keepdims=True))
            return -lp[np.arange(len(y)), y].mean()
        t_hat = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x
        pv = 1 / (1 + np.exp(-(b - a) / r.temperature_refit))
        best = None
        for t in np.linspace(0, 1, 1001):
            pr = pv >= t; se = (pr & (y == 1)).sum() / (y == 1).sum(); sp = (~pr & (y == 0)).sum() / (y == 0).sum()
            if se >= 0.97 and (best is None or sp > best[2]):
                best = (t, se, sp)
        rec("Results: threshold/temperature refit", f"{r.model} · val_n", r.val_n, len(y), "val_predictions_true_split.csv")
        rec("Results: threshold/temperature refit", f"{r.model} · val_n_dme", r.val_n_dme, int(y.sum()), "val_predictions_true_split.csv")
        rec("Results: threshold/temperature refit", f"{r.model} · val_auroc_recomputed", r.val_auroc_recomputed, roc_auc_score(y, b - a), "val logits")
        rec("Results: threshold/temperature refit", f"{r.model} · temperature_refit", r.temperature_refit, t_hat, "own NLL fit on val logits (bounded 0.05–10)", tol=2e-3)
        rec("Results: threshold/temperature refit", f"{r.model} · threshold_refit", r.threshold_refit, round(best[0], 3), "own sweep: highest specificity with val sensitivity ≥ 0.97", tol=1e-9)
        rec("Results: threshold/temperature refit", f"{r.model} · val_sensitivity_at_refit", r.val_sensitivity_at_refit, best[1], "own sweep")
        rec("Results: threshold/temperature refit", f"{r.model} · val_specificity_at_refit", r.val_specificity_at_refit, best[2], "own sweep")
        rec("Results: threshold/temperature refit", f"{r.model} · val_ece_raw", r.val_ece_raw, ece10(y, 1 / (1 + np.exp(-(b - a)))), "val logits, raw")
        rec("Results: threshold/temperature refit", f"{r.model} · val_ece_calibrated", r.val_ece_calibrated, ece10(y, pv), "val logits, refit T")
        meta = json.loads((DATA / "checkpoint_metadata.json").read_text()); key = r.model if r.model in meta else "EfficientNet-B3+LS0.05"
        rec("Results: threshold/temperature refit", f"{r.model} · val_auroc_stored_in_checkpoint", r.val_auroc_stored_in_checkpoint, meta[key]["val_auc_at_selection"], "checkpoint_metadata.json", "TRACED")
        rec("Results: threshold/temperature refit", f"{r.model} · threshold_old_contaminated", r.threshold_old_contaminated, meta[key]["stored_threshold"], "checkpoint_metadata.json", "TRACED")
        td = pd.read_csv(DATA / f"test_predictions_{PRED_FILE[r.model]}.csv"); e = full_metrics(td.label, sig(td, "logit_normal", "logit_dme", r.temperature_refit), r.threshold_refit)
        for k in ("TP", "FP", "TN", "FN", "accuracy", "sensitivity", "specificity", "precision", "f1", "auroc"):
            rec("Results: threshold/temperature refit", f"{r.model} · test_{k}", r[f"test_{k}"], e[k], "test predictions, refit T and threshold")
        for k, v in (("test_ece_raw", ece10(td.label.values, td.p_dme_raw.values)), ("test_ece_refit_calibrated", e["ece_10bin"]),
                     ("test_brier_raw", brier_score_loss(td.label, td.p_dme_raw)), ("test_brier_refit_calibrated", e["brier"])):
            rec("Results: threshold/temperature refit", f"{r.model} · {k}", r[k], v, "test predictions", tol=1e-5)

    # ---------------- paired model comparisons
    pc = pd.read_csv(DATA / "paired_model_comparisons.csv")
    correct = {m: (pd.read_csv(DATA / f"test_predictions_{PRED_FILE[m]}.csv").sort_values("path").pipe(lambda d: (d.p_dme_raw >= .5).astype(int) == d.label)).values for m in MODELS}
    ps = []
    for _, r in pc.iterrows():
        n01 = int((correct[r.model_a] & ~correct[r.model_b]).sum()); n10 = int((~correct[r.model_a] & correct[r.model_b]).sum())
        p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0; ps.append(p)
        rec("Results: paired comparisons", f"{r.model_a} vs {r.model_b} · a correct, b wrong", r.a_correct_b_wrong, n01, "test predictions at 0.5")
        rec("Results: paired comparisons", f"{r.model_a} vs {r.model_b} · a wrong, b correct", r.a_wrong_b_correct, n10, "test predictions at 0.5")
        rec("Results: paired comparisons", f"{r.model_a} vs {r.model_b} · McNemar exact p", r.mcnemar_exact_p, p, "binomial test on discordant pairs")
        rec("Results: paired comparisons", f"{r.model_a} vs {r.model_b} · AUROC a / b", f"{r.auroc_a:.6f} / {r.auroc_b:.6f}", f"{roc_auc_score(*L16.project()[(r.model_a, 'official_test')]):.6f} / {roc_auc_score(*L16.project()[(r.model_b, 'official_test')]):.6f}", "test predictions")
    order = np.argsort(ps); m = len(ps); holm = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, min(1, (m - rank) * ps[i])); holm[i] = run
    for (_, r), h in zip(pc.iterrows(), holm):
        rec("Results: paired comparisons", f"{r.model_a} vs {r.model_b} · Holm-adjusted p", r.mcnemar_holm_p, h, "Holm step-down on the three p-values")

    # ---------------- dataset counts / leakage summary (from the image hash table)
    h = pd.read_csv(DATA / "image_hashes.csv", dtype={"patient": str})
    tr, te = h[h.split == "train"], h[h.split == "test"]; tp_ = set(te.patient)
    pool_, exc = tr[~tr.patient.isin(tp_)], tr[tr.patient.isin(tp_)]
    ls = json.loads((DATA / "leakage_summary.json").read_text())
    per = te.groupby("patient").size().describe()
    md5 = lambda a, b: int(a.merge(b, on="md5").shape[0])
    within = lambda a: int(sum(k * (k - 1) // 2 for k in a.md5.value_counts()))
    cross = int(sum(1 for _, g in pool_.groupby("md5") if g.label.nunique() > 1 for _ in [0]))
    xs = lambda g: int(sum((g.label == 0).sum() * (g.label == 1).sum() for _ in [0]))
    cross_pairs = int(sum(xs(g) for _, g in pool_.groupby("md5") if len(g) > 1))
    def ham(a, b):
        A = np.array([int(x, 16) for x in a.dhash], dtype=np.uint64); B = np.array([int(x, 16) for x in b.dhash], dtype=np.uint64)
        out = []
        for i, x in enumerate(A):
            v = np.bitwise_xor(B, x); cnt = np.zeros(len(B), dtype=np.int64)
            for s in range(64):
                cnt += ((v >> np.uint64(s)) & np.uint64(1)).astype(np.int64)
            out.append(cnt)
        return np.array(out)
    H = ham(te, pool_); near = H <= ls["near_dup_threshold_hamming"]
    Ht = ham(te, te); np.fill_diagonal(Ht, 99)
    near_lab = sum(int(te.label.iloc[i] != pool_.label.iloc[j]) for i, j in zip(*np.where(near)))
    exp = {"n_train_folder": len(tr), "n_test": len(te), "n_training_pool_after_exclusion": len(pool_), "n_excluded_test_patient_images": len(exc),
           "n_test_patients": te.patient.nunique(), "n_test_patients_also_in_train_folder": len(tp_ & set(tr.patient)),
           "patient_overlap_test_vs_pool": len(tp_ & set(pool_.patient)), "test_class_counts.DME": int((te.label == 1).sum()), "test_class_counts.Normal": int((te.label == 0).sum()),
           **{f"test_images_per_patient.{k}": per[k] for k in per.index},
           "test_patients_with_both_labels": int((te.groupby("patient").label.nunique() > 1).sum()),
           "exact_dup_pairs_within_test": within(te), "exact_dup_pairs_test_vs_pool": md5(te[["md5"]], pool_[["md5"]]), "exact_dup_pairs_test_vs_excluded": md5(te[["md5"]], exc[["md5"]]),
           "exact_dup_pairs_within_pool": within(pool_), "exact_dup_pairs_within_pool_cross_label": cross_pairs,
           "near_dup_pairs_test_vs_pool": int(near.sum()), "near_dup_test_images_with_pool_match": int(near.any(1).sum()),
           "near_dup_pairs_within_test": int((Ht <= ls["near_dup_threshold_hamming"]).sum() // 2), "near_dup_pairs_cross_label": near_lab,
           "decoded_only_by_imread_fallback": int((h.decoded_by != "imdecode").sum())}
    for nm, g in (("pool", pool_), ("test", te), ("excluded", exc)):
        for lab, v in (("DME", 1), ("Normal", 0)):
            exp[f"frac_width512_{nm}_{lab}"] = float((g[g.label == v].width == 512).mean())
    flat = {}
    for k, v in ls.items():
        if isinstance(v, dict):
            flat.update({f"{k}.{kk}": vv for kk, vv in v.items()})
        else:
            flat[k] = v
    for k, v in flat.items():
        if k in exp:
            rec("Results: dataset and leakage", k, v, exp[k], "image_hashes.csv (md5, dHash, patient, width)")
        elif k == "near_dup_threshold_hamming":
            rec("Results: dataset and leakage", k, v, v, "fixed setting (script 02)", "SOURCE ONLY")
        elif k == "unreadable_images":
            rec("Results: dataset and leakage", k, v, 0 if h.md5.notna().all() else int(h.md5.isna().sum()), "image_hashes.csv: every image has a hash")
        else:
            rec("Results: dataset and leakage", k, v, v, "geometry-only logistic regression (script 02); no saved inputs to re-fit", "SOURCE ONLY")

    # ---------------- Table 1 counts
    t1 = pd.read_csv(TAB / "table1_dataset_setup.csv", dtype=str)
    num = lambda s: float(str(s).replace(",", "")) if re.fullmatch(r"[\d,]+", str(s).strip()) else np.nan
    subsets = {0: pool_, 1: te, 2: exc}
    for i, r in t1.iterrows():
        g = subsets.get(i)
        if g is None:
            for c in ("Images", "Unique patients", "DME-positive", "DME-negative (Normal)"):
                if not pd.isna(num(r[c])):
                    rec("Table 1", f"row {i + 1} {r['Subset'][:40]} · {c}", num(r[c]), num(r[c]), "Table 1 row (other source)", "SOURCE ONLY")
            continue
        for c, v in (("Images", len(g)), ("Unique patients", g.patient.nunique()), ("DME-positive", int(g.label.sum())), ("DME-negative (Normal)", int((g.label == 0).sum()))):
            rec("Table 1", f"row {i + 1} {r['Subset'][:40]} · {c}", num(r[c]), v, "image_hashes.csv")

    # ---------------- original segmentation audit summary (from per-image rows)
    seg = pd.read_csv(DATA / "segmentation_per_image.csv", dtype={"subject": str}); ss = json.loads((DATA / "segmentation_summary.json").read_text())
    for s in ("val", "test"):
        d = seg[seg.split == s]; o = ss[s]
        exp = {"n_images": len(d), "n_empty_gt": int(d.gt_empty.sum()), "mean_dice_pipeline_convention": d.dice_pipeline.mean(), "mean_iou_pipeline_convention": d.iou_pipeline.mean(),
               "pooled_dice_pipeline": 2 * d.inter_all.sum() / (d.pred_sum_all.sum() + d.gt_sum_all.sum()), "mean_dice_graded_only": d.dice_graded_only.mean(),
               "pooled_dice_graded_only": 2 * d.inter_graded.sum() / (d.pred_sum_graded.sum() + d.gt_sum_graded.sum()), "mean_dice_fluid_positive_images_only": d[~d.gt_empty].dice_pipeline.mean(),
               "mean_pred_frac_of_frame": d.pred_frac_of_frame.mean(), "min_pred_frac_of_frame": d.pred_frac_of_frame.min(), "mean_gt_frac_of_graded": d.gt_frac_of_graded.mean(),
               "mean_prob": d.mean_prob.mean(), "inter_grader_mean_dice_graded": d.dice_grader2_vs_grader1_graded.mean(),
               "inter_grader_mean_dice_graded_fluid_positive": d[~d.gt_empty].dice_grader2_vs_grader1_graded.mean(),
               "nan_fluid_pixels_total": int(d.nan_fluid_pixels.sum()), "fluid_pixels_outside_band_total": int(d.gt1_fluid_outside_band.sum() + d.gt2_fluid_outside_band.sum())}
        for k, v in exp.items():
            rec("Results: segmentation audit (original U-Net)", f"{s} · {k}", o[k], v, "segmentation_per_image.csv (per-image inference rows)")
        rec("Results: segmentation audit (original U-Net)", f"{s} · graded columns min", o["graded_columns_min_max"][0], int(d.graded_columns.min()), "segmentation_per_image.csv")
        rec("Results: segmentation audit (original U-Net)", f"{s} · graded columns max", o["graded_columns_min_max"][1], int(d.graded_columns.max()), "segmentation_per_image.csv")
    for k in ("checkpoint_epoch", "checkpoint_best_val_dice"):
        rec("Results: segmentation audit (original U-Net)", k, ss[k], ss[k], "stored in the original checkpoint", "SOURCE ONLY")

    # ---------------- retrained U-Nets: v2 (current, whole-frame loss) and v1 (superseded), from their per-image rows
    for rdir, tag in ((RT2, "v2, current"), (RT, "v1, superseded")):
        rp = check_retrain(rdir, f"Results: segmentation retrain {tag}")
    rp = pd.read_csv(RT2 / "segmentation_per_image.csv", dtype={"subject": str})  # report counts below refer to v2
    check_gradcam()

    # ---------------- every count, percentage and 1-2 decimal number in the report and checklist
    report_numbers(tr, du, dk, oc, rp)

    df = pd.DataFrame(L)
    df.to_csv(AUDIT / "value_ledger_project.csv", index=False)
    summ = df.pivot_table(index="artifact", columns="status", values="item", aggfunc="count", fill_value=0)
    summ["total"] = summ.sum(axis=1); summ.loc["**All**"] = summ.sum()
    so = df[df.status == "SOURCE ONLY"].groupby(["artifact", "expected_source"]).size().reset_index(name="n")
    (AUDIT / "value_ledger_project_summary.md").write_text("\n".join([
        "# Per-value ledger of every project result (script 17)", "",
        f"Every result value of this project not already covered by script 16: **{len(df):,} values**, one row each in `research_audit/value_ledger_project.csv`.", "",
        "- **RECOMPUTED**: recalculated independently from per-image predictions, the image hash table or per-image segmentation rows, and equal.",
        "- **TRACED**: equal to its source; recalculating would repeat a seeded bootstrap or model inference (bootstrap CIs: the recomputed point estimate lies inside), or a report number matched to a data value.",
        "- **SOURCE ONLY**: no saved inputs to recompute it from (listed below).",
        "- **FAIL** / **UNTRACED**: mismatch, or a report number with no data value.", "",
        md_table(summ.reset_index().rename(columns={"artifact": "Artifact"})), "",
        "## Values that exist only in their own output (SOURCE ONLY)", "",
        md_table(so.rename(columns={"artifact": "Artifact", "expected_source": "Why it cannot be recomputed", "n": "Values"})), "",
        "## Values needing attention", "",
        "None: every value is RECOMPUTED, TRACED or SOURCE ONLY." if not df.status.isin(["FAIL", "UNTRACED"]).any() else "See FAIL/UNTRACED rows in the CSV.", ""]), encoding="utf8")
    print(df.groupby(["artifact", "status"]).size().unstack(fill_value=0).to_string())
    bad = df[df.status.isin(["FAIL", "UNTRACED"])]
    for _, r in bad.iterrows():
        print("FAIL", r["artifact"], "|", r["item"], "| shown", r["value_shown"], "| expected", r["expected"])
    sys.exit(1 if len(bad) else 0)


def md_table(df):
    cell = lambda v: f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)] + ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)])


def check_gradcam():
    """Grad-CAM localisation on Duke 2015 fluid masks: group means and the selection rule, from per-scan rows."""
    art, gd = "Results: Grad-CAM localisation (Duke 2015)", REPO / "research_outputs" / "gradcam_localisation"
    ps = pd.read_csv(gd / "per_scan.csv", dtype={"subject": str}); gs = json.loads((gd / "summary.json").read_text())
    grp = {"selection_01-05": ps.subject.astype(int) <= 5, "confirmation_06-10": ps.subject.astype(int) >= 6}
    src = "gradcam_localisation/per_scan.csv (mean over scans with grader-1 fluid)"
    for g, mask in grp.items():
        for v, kv in gs["results"][g].items():
            d = ps[mask & (ps.variant == v)]
            exp = {"n_scans": len(d), "auroc": d.auroc.mean(), "pointing": d.pointing_hit.astype(str).eq("True").mean(),
                   "mass_in_fluid": d.mass_in_fluid.mean(), "fluid_frac": d.fluid_frac_of_band.mean()}
            for k, val in kv.items():
                rec(art, f"{g} · {v} · {k}", val, exp[k], src, tol=1e-9)
    sel = gs["results"]["selection_01-05"]
    best = max(sel, key=lambda v: (sel[v]["auroc"], sel[v]["pointing"]))
    rec(art, "chosen variant = highest mean pixel AUROC on subjects 01-05", gs["chosen_variant"], best, "selection rule applied to recomputed means")
    for v, c in gs["randomisation_mean_corr_confirmation"].items():
        rec(art, f"randomisation check · {v} · mean correlation", c, c, "needs the heatmaps of the re-initialised classifier (not saved)", "SOURCE ONLY")
        rec(art, f"randomisation check · {v} · passes (< {gs['randomisation_threshold']})", float(c < gs["randomisation_threshold"]), 1.0, "summary.json threshold")


def check_retrain(rdir, art):
    src = f"{rdir.name}/segmentation_per_image.csv (per-image Dice and pixel counts)"
    rp = pd.read_csv(rdir / "segmentation_per_image.csv", dtype={"subject": str}); rs = json.loads((rdir / "segmentation_summary.json").read_text())
    for s in ("val", "test"):
        d = rp[rp.split == s].copy(); G, A = d.gt1_fluid_pixels.astype(float), d.graded_pixels.astype(float); pos = G > 0
        # per block: predicted pixels and per-image Dice (empty-vs-empty = 1, as in the smoothed Dice)
        blocks = {"model": (d.pred_fluid_pixels.astype(float), d.dice_model_vs_g1),
                  "inter_grader_g2_vs_g1": (d.gt2_fluid_pixels.astype(float), d.dice_g2_vs_g1),
                  "baseline_all_foreground": (A, 2 * G / (A + G)),
                  "baseline_all_background": (A * 0, (~pos).astype(float))}
        for blk, (P, dc) in blocks.items():
            I = dc * (P + G) / 2; iou = dc / (2 - dc)
            exp = {"n_images": len(d), "n_fluid_positive": int(pos.sum()), "n_empty_gt": int((~pos).sum()), "n_empty_gt_with_any_predicted_fluid": int(((~pos) & (P > 0)).sum()),
                   "mean_dice": dc.mean(), "mean_iou": iou.mean(), "mean_dice_fluid_positive": dc[pos].mean(), "mean_iou_fluid_positive": iou[pos].mean(),
                   "pooled_dice": 2 * I.sum() / (P.sum() + G.sum()), "pooled_iou": I.sum() / (P.sum() + G.sum() - I.sum()),
                   "mean_pred_frac_of_graded": (P / A).mean(), "gt_frac_of_graded_pooled": G.sum() / A.sum()}
            for k, v in rs[s][blk].items():
                if k in exp:
                    rec(art, f"{s} · {blk} · {k}", v, exp[k], src, tol=1e-6)
                elif not isinstance(v, (dict, list)):
                    rec(art, f"{s} · {blk} · {k}", v, v, "boundary distance needs the masks (not saved per image)", "SOURCE ONLY")
        # per-subject pooled Dice (quoted in the report even where the JSON does not store it)
        stored = rs[s].get("model_per_subject", {})
        for subj, dd in d.groupby("subject"):
            Ii = dd.dice_model_vs_g1 * (dd.pred_fluid_pixels + dd.gt1_fluid_pixels) / 2; v = 2 * Ii.sum() / (dd.pred_fluid_pixels.sum() + dd.gt1_fluid_pixels.sum())
            if subj in stored:
                kv = stored[subj]; rec(art, f"{s} · subject {subj} pooled Dice", kv["pooled_dice"] if isinstance(kv, dict) else kv, v, src)
            else:
                rec(art, f"{s} · subject {subj} pooled Dice (not stored; recomputed)", round(v, 3), round(v, 3), src)
        for k, v in rs.get(f"{s}_at_0.5", {}).items():
            if not isinstance(v, (dict, list)):
                rec(art, f"{s} at threshold 0.5 (not selected) · {k}", v, v, "needs probability maps (not saved per image)", "SOURCE ONLY")
        ff = rs.get(f"{s}_full_frame_check")
        if ff:  # whole-frame behaviour (v2): no saved masks, but outside-band pixels and band totals are per image
            P = d.pred_fluid_pixels.astype(float); out = d.pred_fluid_pixels_outside_band.astype(float); I = d.dice_model_vs_g1 * (P + G) / 2
            exp = {"pred_frac_graded_band": P.sum() / A.sum(), "full_frame_pooled_dice": 2 * I.sum() / (P.sum() + out.sum() + G.sum()),
                   "n_empty_gt": int((~pos).sum()), "n_empty_gt_with_any_predicted_fluid_full_frame": int(((~pos) & ((P + out) > 0)).sum())}
            for k, v in ff.items():
                if k in exp:
                    rec(art, f"{s} · whole-frame check · {k}", v, exp[k], src + " (grader-1 fluid lies inside the band)", tol=1e-6)
                elif k in ("pred_frac_ungraded_margins", "pred_frac_letterbox_padding"):
                    rec(art, f"{s} · whole-frame check · {k} (zero iff no predicted pixel outside the band)", float(v == 0), float(out.sum() == 0), src + ": pred_fluid_pixels_outside_band")
                else:
                    rec(art, f"{s} · whole-frame check · {k}", v, v, "320x320 app frame fractions need the masks", "SOURCE ONLY")
    sw = rs["val_threshold_sweep_pooled_dice"]
    rec(art, "chosen threshold = best val pooled Dice in sweep", rs["chosen_threshold"], float(max(sw, key=sw.get)), "val_threshold_sweep_pooled_dice")
    rec(art, "val pooled Dice at chosen threshold", rs["val"]["model"]["pooled_dice"], sw[str(rs["chosen_threshold"])], "val_threshold_sweep_pooled_dice", tol=1e-6)
    rec(art, "best_epoch", rs["best_epoch"], rs["best_epoch"], "training log", "SOURCE ONLY")
    tl = pd.read_csv(rdir / "segmentation_training_log.csv")
    rec(art, "last_epoch = rows in training log", rs["last_epoch"], int(tl.epoch.max()) if "epoch" in tl else len(tl), f"{rdir.name}/segmentation_training_log.csv")
    return rp


def report_numbers(tr, du, dk, oc, rp):
    """Every count, percentage and 1-2 decimal number in the report and checklist (3-4 decimal
    numbers are traced by script 16). Bibliographic numbers in §19 References are out of scope."""
    import glob
    # three separate indexes so a count only matches a count, a percentage only a proportion x 100
    ints, pcts, decs = {}, {}, {}
    def add(x, where):
        try:
            f = float(str(x).replace(",", ""))
        except (ValueError, TypeError):
            return
        if f != f or abs(f) == float("inf"):
            return
        f = abs(f)  # documents print the sign as a separate "−" character, which the number regex does not capture
        if f == int(f):
            ints.setdefault(int(f), where)
        for d in (0, 1, 2, 3):
            pcts.setdefault(round(f * 100, d), where + " (×100)")
        for d in (1, 2, 3, 4):
            decs.setdefault(round(f, d), where)
    cnt = {"train folder DME images (image_hashes.csv)": int((tr.label == 1).sum()), "train folder Normal images (image_hashes.csv)": int((tr.label == 0).sum()),
           "B-scans in external_predictions_duke2014.csv": len(du), "B-scans in external_predictions_duke2015.csv": len(dk), "images in external_predictions_octdl.csv": len(oc)}
    man = pd.read_csv(DATA / "training_split_manifest.csv")
    for (sp, lab), n in man.groupby(["split", "label"]).size().items():
        cnt[f"training_split_manifest.csv {sp} {lab}"] = n
    for sp, n in man.groupby("split").patient.nunique().items():
        cnt[f"training_split_manifest.csv {sp} patients"] = n
    cnt["Duke 2015 annotated scans in training subjects (110 - val - test rows)"] = 110 - len(rp)
    cnt["train folder images - 6,997 Kaggle-logged validation images (pre-exclusion training folds)"] = len(tr) - 6997
    if (AUDIT / "value_ledger.csv").exists():
        vl0 = pd.read_csv(AUDIT / "value_ledger.csv")
        for a, n in vl0.artifact.value_counts().items():
            cnt[f"value_ledger.csv rows for {a}"] = n
        for pap, n in vl0[vl0.status == "FOUND IN TEXT"].item.str.split(" \\[").str[0].value_counts().items():
            cnt[f"value_ledger.csv published values found in text: {pap}"] = n
    ppc = pd.read_csv(DATA / "per_paper_comparison.csv")
    cnt["per_paper_comparison.csv rows of the five papers"] = int((ppc.paper != "This project").sum())
    f1v = pd.read_csv(REPO / "research_outputs/figures/figure1_within_project_performance_plotted_values.csv")
    for hl, n in f1v.headline.value_counts().items():
        cnt[f"Figure 1 plotted {'bars' if hl else 'dots'}"] = n
    t2 = rp[rp.split == "test"]
    add(t2[t2.gt1_fluid_pixels > 0].dice_model_vs_g2.mean(), "v2 test: mean Dice vs grader 2 on fluid-positive scans (per-image rows)")
    T = pd.read_csv(DATA / "calibration_summary.csv").set_index("model")
    for m in ALL4:
        tp_ = pd.read_csv(DATA / f"test_predictions_{PRED_FILE[m]}.csv"); wrong = tp_[(tp_.p_dme_raw >= 0.5).astype(int) != tp_.label]
        for _, w in wrong.iterrows():
            add(1 / (1 + np.exp(-(w.logit_dme - w.logit_normal) / T.loc[m, "temperature"])), f"{m} refit P(DME) of misclassified {w.filename}")
        add(T.loc[m, "ece10_raw_reported"] - T.loc[m, "ece10_calibrated_contaminated_reported"], f"{m} reported ECE change (raw - superseded calibrated)")
    for sp, n in man.split.value_counts().items():
        cnt[f"training_split_manifest.csv {sp}"] = n
    if (AUDIT / "value_ledger.csv").exists():
        vl = pd.read_csv(AUDIT / "value_ledger.csv")
        cnt["rows in research_audit/value_ledger.csv"] = len(vl)
        for st, n in vl.status.value_counts().items():
            cnt[f"value_ledger.csv rows with status {st}"] = n
    cnt["rows in reconstructed_vs_reported.csv"] = len(pd.read_csv(DATA / "reconstructed_vs_reported.csv"))
    cnt["scans in segmentation_retrain_v2/segmentation_per_image.csv (val + test)"] = len(rp)
    lit = pd.read_csv(AUDIT / "literature_extraction.csv", dtype=str)
    for i, (st, ne) in enumerate(zip(lit.study, lit.n_evaluated.fillna(""))):
        tp_fn = re.search(r"FN (\d+), TP (\d+)", ne)
        if tp_fn:
            cnt[f"{st.split(',')[0]} row {i + 1}: TP + FN (literature_extraction.csv n_evaluated)"] = int(tp_fn.group(1)) + int(tp_fn.group(2))
    rt = rp[rp.split == "test"]
    for v in rt[(rt.gt1_fluid_pixels > 0) & (rt.dice_model_vs_g1 < 1e-6)].gt1_fluid_pixels:
        cnt["test scan with Dice 0: grader-1 fluid pixels (retrain per-image rows)"] = int(v)
    for v in rt[(rt.gt1_fluid_pixels == 0) & (rt.pred_fluid_pixels > 0)].pred_fluid_pixels:
        cnt["fluid-free test scan with a false positive: predicted pixels (retrain per-image rows)"] = int(v)
    for m in ALL4:
        tdp = pd.read_csv(DATA / f"test_predictions_{PRED_FILE[m]}.csv").p_dme_raw
        cnt[f"{m}: test predictions in the two outermost ECE bins"] = int(((tdp < 0.1) | (tdp >= 0.9)).sum())
        cnt[f"{m}: unique test probabilities"] = int(tdp.nunique())
    for _, r in pd.read_csv(DATA / "supplementary_excluded_pool_performance.csv").iterrows():
        add(1 - r.accuracy, f"unseen pool {r.model} error rate (1 - accuracy)")
    for _, r in pd.read_csv(DATA / "table2_within_project_performance.csv").iterrows():
        add(1 - r.accuracy, f"official test {r.model} @ {r.threshold} error rate (1 - accuracy)")
    for k, v in cnt.items():
        ints.setdefault(int(v), k)
    cp = stats.binomtest(116, 126).proportion_ci(0.95, method="exact")  # report §8: exact interval for Rodriguez-Miguel 116/126
    rec("research_results_report.md (counts, %)", "8. Literature comparison · exact (Clopper-Pearson) CI of 116/126, low %", 85.9, round(cp.low * 100, 1), "scipy binomtest exact interval", tol=1e-9)
    rec("research_results_report.md (counts, %)", "8. Literature comparison · exact (Clopper-Pearson) CI of 116/126, high %", 96.1, round(cp.high * 100, 1), "scipy binomtest exact interval", tol=1e-9)
    for r in L:
        add(r["expected"], f"recomputed: {r['artifact']} · {r['item']}")
    files = sorted(glob.glob(str(DATA / "*.csv")) + glob.glob(str(RT / "*.csv")) + glob.glob(str(RT2 / "*.csv")) + glob.glob(str(TAB / "*.csv")) + [str(AUDIT / "literature_extraction.csv")])
    for f in files:
        if re.search(r"predictions|per_image|roc_coordinates|hashes|manifest|near_duplicate|reliability|training_log", Path(f).name):
            continue  # per-row files: any count would match some pixel count or index by chance
        d = pd.read_csv(f, dtype=str, low_memory=False)
        for c in d.columns:
            for v in d[c].dropna().unique():
                for x in re.findall(r"\d[\d,]*\.?\d*", str(v)):  # no sign: "0.956-0.961" is a range
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
    for f in glob.glob(str(DATA / "*.json")) + glob.glob(str(RT / "*.json")) + glob.glob(str(RT2 / "*.json")) + glob.glob(str(REPO / "research_outputs/gradcam_localisation/*.json")) + [str(REPO / "artifacts/gradcam_randomization.json")]:
        try:
            walk(json.loads(Path(f).read_text()), Path(f).name)
        except (ValueError, OSError):
            pass
    cited = {"83,484": "size of the full OCT2017 release (4 classes; only DME and NORMAL are on disk), config.py",
             "108,312": "training-image count stated in Kermany et al. 2018 (published figure)",
             "250": "test images per class stated in Kermany et al. 2018 (published figure)",
             "626": "download size of 2014_BOE_Srinivasan.zip (MB), external file",
             "3,000": "fixed sample size of the brute-force AUROC check (10_final_checks.py)",
             "254": "pytest run output (tests/); re-run at each commit"}
    cited_doc = {  # numbers that exist only in an unsaved run output, a log line, a version pin or a setting
        ("segmentation_retrain_report.md", "0.13"): "v1 whole-frame pooled Dice from the diagnostic full_frame_check run (not saved in v1's summary)",
        ("leakage_audit.md", "6,997"): "pre-exclusion validation size in the Kaggle training log (full_run_log.txt)",
        ("metric_verification.md", "6,997"): "pre-exclusion validation size in the Kaggle training log (full_run_log.txt)",
        ("repository_inventory.md", "6997"): "pre-exclusion validation size in the Kaggle training log (full_run_log.txt)",
        ("repository_inventory.md", "6,997"): "pre-exclusion validation size in the Kaggle training log (full_run_log.txt)",
        ("leakage_audit.md", "7,535"): "pre-exclusion validation size of the local scikit-learn 1.9.0 split (evaluate.py run output)",
        ("metric_verification.md", "7,535"): "pre-exclusion validation size of the local scikit-learn 1.9.0 split (evaluate.py run output)",
        ("repository_inventory.md", "7,535"): "pre-exclusion validation size of the local scikit-learn 1.9.0 split (evaluate.py run output)",
        ("repository_inventory.md", "150"): "figure resolution setting (dpi) of the original scripts",
        ("repository_inventory.md", "83"): "line number in full_run_log.txt",
        ("repository_inventory.md", "1.26"): "numpy version pinned in requirements.txt",
        ("value_validation.md", "46"): "number of distinct §8.1 values counted by script 15 (its own output)"}
    SKIP =r"(?:§|Figure |Fig\. |Table |Phase |\[|B=|B = |seed |seed=|page |p\. |pp\. |doi|DOI|v|Section |S|step |Step |#+ |epoch |Check |check |Python |python |==)$"
    # report and checklist: 1-2 decimals here (3-4 decimals in script 16); every other document: up to 4 decimals here
    docs = [("research_results_report.md", REPO / "research_outputs/research_results_report.md", 2), ("final_audit_checklist.md", REPO / "research_outputs/final_audit_checklist.md", 2),
            ("segmentation_retrain_report.md", REPO / "research_outputs/segmentation_retrain_report.md", 4)]
    docs += [(p.name, p, 4) for p in sorted(AUDIT.glob("*.md")) if not p.name.startswith(("value_ledger", "output_folder_verification"))]
    for doc, path, maxdp in docs:
        t = path.read_text(encoding="utf8")
        t = t.split("\n## 19. References")[0]
        for mt in re.finditer(r"(?<![\w.\-/:])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d{1,%d})?)(%%?)(?![\w.]\d|\d|[\w/])" % maxdp, t):
            n, pct = mt.group(1), mt.group(2); f = float(n.replace(",", ""))
            pct = pct or ("%" if re.match(r"[–-]\d+(?:\.\d+)?%", t[mt.end():]) else "")  # "58–60%": both ends are percentages
            if re.match(r"\(\d+\)", t[mt.end():]) or re.match(r"\.\*", t[mt.end():]):
                continue  # journal volume(issue), version wildcard
            if re.search(SKIP, t[max(0, mt.start() - 12):mt.start()]) or re.fullmatch(r"(19|20)\d\d", n) or ("," not in n and "." not in n and f < 10 and not pct):
                continue
            sec = re.findall(r"^#+ (.+)$", t[:mt.start()], re.M); ctx = t[max(0, mt.start() - 40):mt.start()].splitlines()[-1].strip()[-40:]
            item = f"{(sec[-1] if sec else '')[:40]} · {ctx}"
            why = cited_doc.get((doc, n)) or cited.get(n)
            if why:
                L.append({"artifact": doc + " (counts, %)", "item": item, "value_shown": n + pct, "expected": n, "expected_source": why, "status": "SOURCE ONLY", "note": ""})
                continue
            dp = len(n.split(".")[1]) if "." in n else 0
            where = pcts.get(round(f, dp)) if pct else (decs.get(round(f, dp)) if dp else ints.get(int(f)))
            if pct and not where:
                where = decs.get(round(f, dp)) if dp else ints.get(int(f))  # e.g. "97%" stated as a sensitivity threshold
            L.append({"artifact": doc + " (counts, %)", "item": item, "value_shown": n + pct, "expected": n if where else "", "expected_source": where or "",
                      "status": "TRACED" if where else "UNTRACED", "note": ""})


if __name__ == "__main__":
    main()
