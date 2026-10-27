"""
Phase E2 -- publication tables (CSV + Markdown) generated from the same CSVs
the figures use. Missing values: "NR" (not reported) / "NA" (not applicable).

Table 1  dataset and experimental setup
Table 2  verified within-project performance (raw @0.5 primary; tuned rows flagged)
Table 3  literature comparison (with comparability category)
Table 4  NOT produced (segmentation evaluation invalid) -- an audit note is written instead
Table 5  reproducibility and limitations

Usage: python research_outputs/scripts/09_make_tables.py
"""

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "research_outputs" / "data"
TAB = REPO / "research_outputs" / "tables"
AUDIT = REPO / "research_audit"
TAB.mkdir(parents=True, exist_ok=True)


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]).replace("|", "/").replace("\n", " ") for c in cols) + " |")
    return "\n".join(out)


def write(df, name, title, caption):
    df.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.md").write_text(f"**{title}**\n\n{md(df)}\n\n*{caption}*\n", encoding="utf8")


def ci(v, lo, hi, d=3):
    return f"{v:.{d}f} ({lo:.{d}f}–{hi:.{d}f})" if not any(pd.isna([v, lo, hi])) else ("NR" if pd.isna(v) else f"{v:.{d}f}")


def main():
    leak = json.loads((DATA / "leakage_summary.json").read_text())
    h = pd.read_csv(DATA / "image_hashes.csv", dtype={"patient": str})
    tp = set(h[h.split == "test"].patient)
    pool = h[(h.split == "train") & ~h.patient.isin(tp)]; exc = h[(h.split == "train") & h.patient.isin(tp)]; test = h[h.split == "test"]
    seg = json.loads((DATA / "segmentation_summary.json").read_text())

    def r1(src, subset, d, split, unit, indep, notes, n_override=None):
        return {"Dataset/source": src, "Subset": subset, "Images": n_override or (f"{len(d):,}" if d is not None else "NR"),
                "Unique patients": f"{d.patient.nunique():,}" if d is not None else "NR", "Eyes / volumes": "NR (not encoded in Kermany filenames)",
                "DME-positive": f"{int(d.label.sum()):,}" if d is not None else "NR", "DME-negative (Normal)": f"{int((d.label == 0).sum()):,}" if d is not None else "NR",
                "Split": split, "Unit of splitting": unit, "Dimensionality": "2D B-scan", "OCT device": "Heidelberg Spectralis according to the dataset paper (prior knowledge, not re-verified in this audit; not recorded per image)",
                "Label source": "Kermany et al. 2018 class folders (grading protocol not re-verified in this audit)", "Independent of training": indep, "Notes": notes}

    t1 = [
        r1("Kermany OCT2017 (Cell 2018), DME+NORMAL", "train/ folder, training pool after exclusion", pool, "train + validation (StratifiedGroupKFold 5-fold, 1 fold val)", "patient",
           "NA", "Kaggle runs: train 24,600 / val 5,560 (logged). Per-class split of these two subsets not logged; the local re-run of build_splits gives a different partition (scikit-learn version)."),
        r1("Kermany OCT2017", "test/ folder (official test split)", test, "held-out test", "patient (fixed by dataset authors)", "Yes, at patient level after exclusion of 305 shared patients from train/val",
           "368 patients; 6 patients have both DME and Normal images; 0 exact duplicates with the training pool; distribution differs from training (no 512x512 images)."),
        r1("Kermany OCT2017", "train/ images of the 305 test-set patients (excluded from training)", exc, "supplementary unseen evaluation pool (NEW in this audit)", "patient",
           "Not used for training/validation (verified for B3, ResNet-50; implied for B0); NOT independent of the test set (same patients; 114 byte-identical copies of test images)",
           f"All {len(exc):,} images decoded with cv2.imread during re-inference. Across train/+test/, {leak['decoded_only_by_imread_fallback']} truncated JPEGs are rejected by cv2.imdecode but decode with cv2.imread (with a 'Premature end of JPEG' warning); {leak['unreadable_images']} are unreadable."),
        {"Dataset/source": "Duke DME (Chiu et al. BOE 2015)", "Subset": "segmentation (U-Net)", "Images": "110 annotated B-scans", "Unique patients": "10",
         "Eyes / volumes": "10 volumes (61 B-scans each; 11 annotated)", "DME-positive": "NA (all DME subjects)", "DME-negative (Normal)": "NA",
         "Split": f"train 6 / val 2 ({', '.join(seg['val']['subjects'])}) / test 2 ({', '.join(seg['test']['subjects'])}) subjects", "Unit of splitting": "patient",
         "Dimensionality": "2D B-scan", "OCT device": "Heidelberg Spectralis according to the dataset paper (prior knowledge, not re-verified)", "Label source": "manual fluid masks, grader 1 (manualFluid1)",
         "Independent of training": "Yes (disjoint subjects)", "Notes": "Ungraded (NaN) pixels treated as background by segment.py."},
    ]
    write(pd.DataFrame(t1), "table1_dataset_setup", "Table 1. Dataset and experimental setup",
          "Counts computed from files on disk (research_outputs/data/image_hashes.csv). Patient = ID parsed from the Kermany filename; eye and volume identity are not available. No patient counts are inferred from image counts.")

    p = pd.read_csv(DATA / "table2_within_project_performance.csv")
    t2 = []
    for _, r in p.iterrows():
        raw = r.threshold == 0.5
        t2.append({"Model": r.model, "Probabilities / threshold": "raw softmax, 0.5" if raw else f"temperature-scaled, {r.threshold} (tuned)",
                   "Test n (DME/Normal)": f"{r.n} ({r.n_dme}/{r.n_normal})", "TP": r.TP, "FP": r.FP, "TN": r.TN, "FN": r.FN,
                   "Accuracy (95% CI)": ci(r.accuracy, r.accuracy_ci_lo, r.accuracy_ci_hi, 4), "Precision (95% CI)": ci(r.precision, r.precision_ci_lo, r.precision_ci_hi, 4),
                   "Sensitivity (95% CI)": ci(r.sensitivity, r.sensitivity_ci_lo, r.sensitivity_ci_hi, 4), "Specificity (95% CI)": ci(r.specificity, r.specificity_ci_lo, r.specificity_ci_hi, 4),
                   "F1 (95% CI)": ci(r.f1, r.f1_ci_lo, r.f1_ci_hi, 4), "AUROC (95% CI)": f"{r.auroc:.4f} (DeLong CI degenerate)" if r.auroc_ci_lo == r.auroc_ci_hi else ci(r.auroc, r.auroc_ci_lo, r.auroc_ci_hi, 4),
                   "PR-AUC": f"{r.pr_auc_average_precision:.4f}", "Brier": f"{r.brier:.4f}", "Prediction source": r.prediction_source,
                   "Verification status": "independently reconstructed from checkpoint; matches metrics.json" + ("" if raw else "; operating point NOT valid (threshold/temperature fitted on contaminated validation set)")})
    exc_p = DATA / "supplementary_excluded_pool_performance.csv"
    if exc_p.exists():
        for _, r in pd.read_csv(exc_p).iterrows():
            t2.append({"Model": r.model + " [supplementary unseen pool]", "Probabilities / threshold": "raw softmax, 0.5",
                       "Test n (DME/Normal)": f"{r.n} ({r.n_dme}/{r.n_normal}); {r.notes.split(';')[0]}", "TP": r.TP, "FP": r.FP, "TN": r.TN, "FN": r.FN,
                       "Accuracy (95% CI)": ci(r.accuracy, r.accuracy_ci_lo, r.accuracy_ci_hi, 4), "Precision (95% CI)": ci(r.precision, r.precision_ci_lo, r.precision_ci_hi, 4),
                       "Sensitivity (95% CI)": ci(r.sensitivity, r.sensitivity_ci_lo, r.sensitivity_ci_hi, 4), "Specificity (95% CI)": ci(r.specificity, r.specificity_ci_lo, r.specificity_ci_hi, 4),
                       "F1 (95% CI)": ci(r.f1, r.f1_ci_lo, r.f1_ci_hi, 4), "AUROC (95% CI)": ci(r.auroc, r.auroc_cluster_bootstrap_ci_lo, r.auroc_cluster_bootstrap_ci_hi, 4) + " (cluster bootstrap)",
                       "PR-AUC": f"{r.pr_auc_average_precision:.4f}", "Brier": f"{r.brier:.4f}", "Prediction source": r.prediction_source,
                       "Verification status": "NEW analysis (this audit); patient-cluster bootstrap CIs; not independent of official test set"})
    write(pd.DataFrame(t2), "table2_within_project_performance", "Table 2. Verified within-project model performance (positive class = DME)",
          "Official test rows: Wilson 95% CIs for proportions, class-stratified bootstrap (B=2000, seed 42) for F1, DeLong for AUROC (degenerate when AUROC = 1). "
          "Supplementary rows: patient-cluster bootstrap (B=2000, seed 42). Only the raw/0.5 rows are methodologically valid operating points.")

    lit = pd.read_csv(AUDIT / "literature_extraction.csv", dtype=str)
    cat = {"Supplementary Table S1": "B", "Main text Table 2": "C", "Table 3": "C"}
    t3 = []
    for _, r in lit.iterrows():
        c = "A" if r.study.startswith("This project") else next((v for k, v in cat.items() if r.source_location.startswith(k)), "C")
        if r.study.startswith("Zhu") and r.task.startswith("non-CI-DME"):
            c = "D"
        f = lambda x: x if x in ("NR", "NA") else f"{float(x):.3f}"
        t3.append({"Study (year)": r.study.split(",")[0], "Architecture": r.model, "Input": r.input_representation, "Task": r.task, "Dataset / cohort": f"{r.device} · {r.dataset}",
                   "Evaluation unit": r.evaluation_unit, "Internal / external": r.cohort_role, "n": r.n_evaluated, "Accuracy": f(r.accuracy), "Sensitivity": f(r.sensitivity),
                   "Specificity": f(r.specificity), "F1": f(r.f1), "AUROC (95% CI)": f"{f(r.auroc)} ({f(r.auroc_ci_lo)}–{f(r.auroc_ci_hi)})" if r.auroc_ci_lo not in ("NR", "nan") else f(r.auroc),
                   "Metric source": r.source_location, "Comparability category": c, "Limitations / notes": r.notes})
    write(pd.DataFrame(t3), "table3_literature_comparison", "Table 3. Reported performance of published DME classifiers and the present model (not a head-to-head comparison)",
          "A = within-project; B = similar task and unit (2D B-scan) but different dataset/population/label definition; C = different evaluation unit or input (volume-level/3D); D = different task, not plotted. "
          "Published values copied as printed (two internally inconsistent CIs in Tang 2021 flagged in notes). NR = not reported. This project's AUROC CI is degenerate (perfect separation).")

    (TAB / "table4_segmentation_NOT_PRODUCED.md").write_text(
        "**Table 4 (segmentation) was not produced.**\n\nThe U-Net checkpoint labels ~99.99% of every frame as fluid (research_outputs/data/segmentation_summary.json). "
        f"Its test Dice ({seg['test']['mean_dice_pipeline_convention']:.4f}) and IoU ({seg['test']['mean_iou_pipeline_convention']:.4f}) are reproduced, but they describe a collapsed model, "
        "and the ground-truth definition counts ungraded pixels as background. A results table would imply a valid evaluation; see Audit Figure A1 and research_audit/metric_verification.md §B4 instead.\n", encoding="utf8")

    import sklearn, scipy, torch, timm, matplotlib
    t5 = [
        ("Software (this audit)", f"Python {platform.python_version()}, torch {torch.__version__}, timm {timm.__version__}, scikit-learn {sklearn.__version__}, scipy {scipy.__version__}, numpy {np.__version__}, pandas {pd.__version__}, matplotlib {matplotlib.__version__}"),
        ("Software (original training)", "Kaggle image; requirements.txt pins torch 2.3 / timm 0.9 / scikit-learn 1.4 (1.4.2 installed per full_run_log.txt)"),
        ("Hardware", "Training: Kaggle GPU, Tesla T4 (named in full_run_log.txt and the ResNet-50 run.log). This audit: CPU-only Windows 11 workstation, 12 logical cores"),
        ("Random seeds", "SEED=42 for splits, torch, numpy (utils.set_seed); bootstrap seed 42 in this audit"),
        ("Preprocessing", "NLM denoise h=10 -> polynomial retinal flattening -> CLAHE (2.0, 8x8) -> aspect-preserving letterbox 300x300 -> ImageNet normalisation"),
        ("Split strategy", "Patient-grouped StratifiedGroupKFold(5) on Kermany train/; one fold = validation; all images of 305 patients shared with test/ removed"),
        ("Checkpoint selection", "Best validation AUROC; early stop on validation loss (patience 5)"),
        ("Threshold selection", "0.5 (primary, pre-specified by argmax). 'Tuned' thresholds chosen for sensitivity >= 0.97 on a validation set that does not match the training-time split -> not valid"),
        ("Calibration protocol", "Single temperature, NLM fit on the same mismatched validation set; test set not used"),
        ("Evaluation metrics", "Confusion-matrix metrics (manual + sklearn cross-check), AUROC (sklearn + DeLong cross-check), PR-AUC, Brier, log loss, ECE (10 bins)"),
        ("Statistical methods", "Wilson CIs; stratified / patient-cluster percentile bootstrap (B=2000); DeLong AUROC CI; exact McNemar + paired DeLong with Holm correction"),
        ("Missing data / predictions", "No per-image predictions were saved originally (regenerated); B0 training log empty; B3 training_log.csv lost (console-parsed version used); training-time split manifest not saved"),
        ("Leakage concerns and evidence", "Train/val<->test patient overlap 0 after exclusion; 0 exact or visually confirmed near duplicates test<->pool; validation-set mismatch at evaluation time contaminates threshold/temperature; test set distribution differs from training (dimensions, border artifact)"),
        ("External validation", "None. All results are internal to the Kermany dataset"),
        ("Deviation from original methodology (this audit)", "No retraining. New: re-inference of saved checkpoints on test set and on the readable images of the 305 excluded patients (count in Table 1 / Table 2); leakage scans; statistics"),
    ]
    write(pd.DataFrame(t5, columns=["Item", "Detail"]), "table5_reproducibility_limitations", "Table 5. Reproducibility and limitations", "Compiled from source code, run logs, checkpoints and this audit's scripts.")
    print("tables written")


if __name__ == "__main__":
    main()
