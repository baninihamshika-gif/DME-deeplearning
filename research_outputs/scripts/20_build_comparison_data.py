"""
Phase C -- data behind the comparison figures.

Builds research_outputs/data/literature_comparison.csv, the ONLY file the figure
script (21_plot_comparison_figures.py) reads. Nothing is typed into a plotting call.

  * This project's rows are READ from research_outputs/data/table2_within_project_performance.csv
    (and checked against the per-image test predictions). The Wilson 95% CIs are recomputed
    here and asserted equal to the ones already stored in table2.
  * Comparator values (P1-P4) are entered ONCE, below, in COMPARATORS, each with a
    source_citation naming where the value comes from and a verification_status stating
    whether it was checked against a primary source.
  * Wilson 95% CIs for comparators are computed from the source's stated n. Where n is not
    stated the CI cells are the literal string "null" (never imputed). Counts k are
    round(rate * n) where only a rate is published; that rounding is recorded in notes.

Pre-plot test-set identity check is computed here from files on disk and written into the
test_set_identity column. Cause of any discrepancy is NOT inferred.

Usage: python research_outputs/scripts/20_build_comparison_data.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "research_outputs" / "data"
OUT = DATA / "literature_comparison.csv"
Z = 1.959964  # same z as the rest of the audit (04_compute_metrics.py)


def wilson(k, n):
    k, n = float(k), float(n)
    p = k / n
    d = 1 + Z ** 2 / n
    c = (p + Z ** 2 / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ------------------------------------------------------------------ pre-plot check
pred = pd.read_csv(DATA / "test_predictions_EfficientNet-B3.csv")
leak = json.load(open(DATA / "leakage_summary.json"))
n_dme_pred = int((pred.label == 1).sum())
n_norm_pred = int((pred.label == 0).sum())
n_patients = int(pred.patient.nunique())
assert (n_dme_pred, n_norm_pred, len(pred)) == (242, 242, 484)
assert n_patients == leak["n_test_patients"] == 368

test_root = REPO / "data" / "raw" / "OCT2017" / "test"
folder_files = {}
for cls in ["CNV", "DME", "DRUSEN", "NORMAL"]:
    d = test_root / cls
    folder_files[cls] = sorted(p.name for p in d.iterdir()) if d.is_dir() else None
on_disk = {c: (len(v) if v is not None else None) for c, v in folder_files.items()}
pred_names = {"DME": set(pred.loc[pred.label == 1, "filename"]), "NORMAL": set(pred.loc[pred.label == 0, "filename"])}
exact_subset = all(folder_files[c] is not None and set(folder_files[c]) == pred_names[c] for c in ["DME", "NORMAL"])
PER_CLASS = 242
total_968 = 968
arith_968 = (total_968 == 4 * PER_CLASS)
P3_N, P3_NDME = 1000, 250
p3_delta_dme = P3_NDME - PER_CLASS
p3_delta_total = P3_N - total_968

ID_PROJECT = (
    f"Official Kermany test split (83,484-image train release): DME {n_dme_pred} + Normal {n_norm_pred} = {len(pred)} images from "
    f"{n_patients} patients. All {len(pred)} per-image prediction files match test/DME ({on_disk['DME']} files) and test/NORMAL "
    f"({on_disk['NORMAL']} files) exactly (set equality = {exact_subset}). test/CNV and test/DRUSEN are not on this machine "
    f"(on-disk counts: CNV={on_disk['CNV']}, DRUSEN={on_disk['DRUSEN']}); the project uses DME and Normal only."
)
ID_P2 = (
    f"Official 968-image test set; 968 = 4 x {PER_CLASS} is {arith_968} (242 per class per Chen 2021 BMC Bioinf and Wen 2022 ReLaX, fetched "
    f"2026-10-31; DME and Normal folders verified here at {on_disk['DME']}/{on_disk['NORMAL']}; CNV/DRUSEN counts not verifiable on disk). "
    f"This project's {len(pred)} images are exactly the DME + Normal folders (set equality = {exact_subset}), i.e. a strict subset of the 968. "
    f"P2 DME recall and this project's DME sensitivity are therefore over the SAME {PER_CLASS} DME images. P2 text (Materials and methods, read from the "
    f"PDF 2026-10-31): 'We follow the original data division strategy'; 108,312 train / 32 validation / 968 test; '8 and 242 OCT images were included in each "
    f"category' in the validation and test sets."
)
ID_P3 = (
    f"n = {P3_N} ({P3_NDME} DME / {P3_N - P3_NDME} non-DME). DISCREPANCY: {P3_NDME} DME vs {PER_CLASS} per class in the 968-image official test "
    f"set (difference {p3_delta_dme:+d}); total {P3_N} vs {total_968} (difference {p3_delta_total:+d}). P3 (PDF read 2026-11-01) names the dataset 'Kermany', states 109,312 images "
    f"(CNV 37,455 / DME 11,559 / DRUSEN 8,866 / NORMAL 51,432) and '1,000 data points used for testing', but does not state its split or release. Cause NOT established. "
    f"Arithmetic from the primary sources only (NOT a stated cause for P3): He 2023 states 32 validation (8 per class) + 968 test (242 per class) images in a 109,312-image release, "
    f"i.e. 250 per class / 1,000 in total, which equals the 1,000-image test set (250 per category, 633 patients) stated in Kermany 2018 and in Ai 2022."
)
ID_P1 = (
    "NOT STATED in the paper for Table 5. Full PDF read 2026-10-31. Section 4.4: 'DME_VS_NORMAL represents that 1,000 DME and normal random images are "
    "selected from the training set for model training and prediction'; the test set used for Table 5 is not stated. The external Beijing Chao-Yang (Cirrus) set "
    "(277 images: CNV 60, DME 107, drusen 27, normal 83) is used for the 4-class external evaluation (abstract: accuracy 92%, AUC 94.5%), not described as the source "
    "of the DME_VS_NORMAL values. The task brief's statement that the 1.000 values are on the external Beijing set is NOT supported by the paper text. Internal Kermany test: 1,000 images, 250 per class."
)
ID_P4 = (
    "Cell paper read 2026-10-31: independent test set of 1,000 images (250 per category) from 633 patients not in the training set (4-class/referral evaluation); "
    "the test-set size used for the DME-vs-Normal binary classifier is NOT stated (Results: 'Binary classifiers were also implemented ... using the same datasets'). n therefore left null."
)

# ------------------------------------------------------------------ project rows
t2 = pd.read_csv(DATA / "table2_within_project_performance.csv")


def pick(model, prefix):
    m = t2[(t2.model == model) & (t2.threshold_name.str.startswith(prefix))]
    assert len(m) == 1, (model, prefix, len(m))
    return m.iloc[0]


REP_PROJECT = dict(
    patient_split=1, external=1, ci=1, leakage=1, calibration=1, explain_validated=1, peer=0,
    evidence=json.dumps({
        "patient_split": "StratifiedGroupKFold grouped by patient; 0 train/test patient overlap after exclusion (metric_verification.md B1; leakage_audit.md 1-2)",
        "external": "OCTDL n=479 (Optovue) and Duke 2014 (30 volumes), Duke 2015 (external_validation_*.csv)",
        "ci": "Wilson 95% CIs (table2_within_project_performance.csv)",
        "leakage": "14-candidate leakage audit (leakage_audit.md)",
        "calibration": "Temperature scaling, ECE, reliability bins (calibration_summary.csv)",
        "explain_validated": "Grad-CAM weight-randomisation check (progress_report 3.5) and localisation check against Duke fluid masks (gradcam_localisation_duke.py)",
        "peer": "Student project report; not peer reviewed"}),
    read_status="full (own repository)")


def project_row(row_id, label, r, op):
    n, nd = int(r.n), int(r.n_dme)
    nn = n - nd
    cis = {
        "acc": wilson(r.TP + r.TN, n), "sens": wilson(r.TP, nd), "spec": wilson(r.TN, nn)}
    for key, col in [("acc", "accuracy"), ("sens", "sensitivity"), ("spec", "specificity")]:
        assert abs(cis[key][0] - r[f"{col}_ci_lo"]) < 1e-6 and abs(cis[key][1] - r[f"{col}_ci_hi"]) < 1e-6, (row_id, key)
    return dict(
        row_id=row_id, source=label, task_definition="DME vs Normal (binary), single 2D B-scan",
        negative_class_composition="Normal only", test_set_identity=ID_PROJECT, n_test=n, n_dme=nd,
        split_unit="patient", patient_split_stated="yes", external_validation="yes (OCTDL, Duke 2014/2015; separate rows in external_validation_performance.csv)",
        accuracy=r.accuracy, sensitivity=r.sensitivity, specificity=r.specificity, auroc=r.auroc,
        acc_ci_lo=cis["acc"][0], acc_ci_hi=cis["acc"][1], sens_ci_lo=cis["sens"][0], sens_ci_hi=cis["sens"][1],
        spec_ci_lo=cis["spec"][0], spec_ci_hi=cis["spec"][1], comparability_category="A",
        verification_status="verified: read from repository CSV; Wilson CIs recomputed and equal to stored values",
        source_citation=f"research_outputs/data/table2_within_project_performance.csv, row '{r.model} / {r.threshold_name[:40]}'",
        operating_point=op, notes="AUROC CI is degenerate (classes perfectly separated) and is not plotted", **{f"rep_{k}": v for k, v in REP_PROJECT.items()})


rows = [
    project_row("proj_b3_050", "This project: EfficientNet-B3 @ 0.5", pick("EfficientNet-B3", "default 0.5"), "threshold 0.5, raw probabilities"),
    project_row("proj_b3_refit", "This project: EfficientNet-B3 @ 0.181 (refit)", pick("EfficientNet-B3", "refit tuned"), "threshold 0.181, temperature 1.5485 refit on training-time validation split (n=5,560)"),
    project_row("proj_b0_050", "This project: EfficientNet-B0 @ 0.5", pick("EfficientNet-B0", "default 0.5"), "threshold 0.5, raw probabilities"),
    project_row("proj_r50_050", "This project: ResNet-50 @ 0.5", pick("ResNet-50", "default 0.5"), "threshold 0.5, raw probabilities"),
]

# discordance between this project's architectures at threshold 0.5 (read from per-image predictions and the paired-comparison table)
_preds = {m: pd.read_csv(DATA / f"test_predictions_{m}.csv") for m in ["EfficientNet-B3", "EfficientNet-B0", "ResNet-50"]}
for _m, _d in _preds.items():
    assert (_d.filename.values == _preds["EfficientNet-B3"].filename.values).all()
_P = np.vstack([(_d.p_dme_raw.values >= 0.5) for _d in _preds.values()])
DISC_ANY = int((_P.min(axis=0) != _P.max(axis=0)).sum())
_pair = pd.read_csv(DATA / "paired_model_comparisons.csv")
DISC_PAIRWISE = int((_pair.a_correct_b_wrong + _pair.a_wrong_b_correct).sum())
assert DISC_ANY == 2, DISC_ANY
for _r in rows:
    _r["discordant_images_any_pair"] = DISC_ANY
    _r["discordant_pairwise_total"] = DISC_PAIRWISE

# ------------------------------------------------------------------ comparators (entered ONCE)
NULLREP = dict(patient_split=0, external=0, ci=0, leakage=0, calibration=0, explain_validated=0, peer=0)

COMPARATORS = [
    dict(row_id="P1", source="P1: Ai 2022 (FN-OCT)", task_definition="DME vs Normal (binary), 2D B-scan",
         negative_class_composition="Normal only (1,000 DME and normal images drawn from the training set for this experiment)", test_set_identity=ID_P1,
         n_test=None, n_dme=None, split_unit="not stated", patient_split_stated="not stated",
         external_validation="reported for the 4-class model only (Beijing Chao-Yang Cirrus set, 277 images); NOT stated for the DME_VS_NORMAL values",
         accuracy=1.000, sensitivity=1.000, specificity=1.000, auroc=1.000, comparability_category="C",
         verification_status="verified against primary source: Table 5 (p. 11), DME_VS_NORMAL rows ACC/Recall/Specificity/AUC print '1' for FN-F1-OCT, FN-Weight-OCT and FN-Auto-OCT; test set for this table not stated; the task brief's 'external Beijing test set' is not supported by the paper",
         source_citation="Ai Z et al. 2022, Front Neuroinform 16:876927, DOI 10.3389/fninf.2022.876927, Table 5 (Comparison of three fusion strategies in binary classifiers), p. 11; experiment definition in section 4.4.",
         operating_point="not stated", notes="n not stated, so no CI; values are printed as the integer 1 (three fusion variants identical); same table's 'Kermany' column (0.982/0.968/0.996/0.998) is the second-hand source of P4",
         rep=dict(NULLREP, external=1, explain_validated=1, peer=1),
         evidence={"patient_split": "not stated (text searched)", "external": "Beijing Chao-Yang external set (277 images), 4-class, abstract 92% accuracy / 94.5% AUC", "ci": "not reported (text searched for confidence interval/95% CI/bootstrap)", "leakage": "not reported (text searched)",
                   "calibration": "not reported (text searched)", "explain_validated": "Section 4.6: Grad-CAM localization maps checked by professional clinicians (qualitative; no quantitative result)", "peer": "Frontiers in Neuroinformatics 16:876927"},
         read_status="full (PDF read 2026-10-31)"),
    dict(row_id="P1_int_comp", source="P1 Ai 2022: internal Kermany test (composition only)", task_definition="4-class (composition only)",
         negative_class_composition="n/a", test_set_identity="1,000 images, 250 per class (text retrieved 2026-10-31)", n_test=1000, n_dme=250,
         split_unit="not stated", patient_split_stated="not stated (text read)", external_validation="no (internal)", comparability_category="C",
         verification_status="verified against primary source: full PDF read 2026-10-31", source_citation="Ai 2022, Front Neuroinform 16:876927, dataset/test-set description",
         operating_point="composition only", notes="no metrics taken from this row", rep=None),
    dict(row_id="P1_ext", source="P1 Ai 2022: external Beijing set (4-class model, DME one-vs-rest)", task_definition="4-class model; DME class one-vs-rest on external set, 2D B-scan",
         negative_class_composition="CNV + drusen + Normal (4-class decision)", test_set_identity="External Beijing Chao-Yang Hospital set, Cirrus HD-OCT (Carl Zeiss): 277 images (CNV 60, DME 107, drusen 27, normal 83); retrospective adults 2019-2021 (PDF read 2026-11-01)",
         n_test=277, n_dme=107, split_unit="not stated", patient_split_stated="not stated (text read)", external_validation="yes (external, different device and hospital)",
         accuracy=None, sensitivity=0.785, specificity=None, auroc=0.807, comparability_category="C",
         verification_status="verified against primary source: Table 8 (p. 13), FN-F1-OCT, Complete model, DME row: ACC 0.785, Recall 0.785, F1 0.7636, AUC 0.807",
         source_citation="Ai Z et al. 2022, Front Neuroinform 16:876927, Table 8 (Comparison of different fusion strategies on external test datasets), FN-F1-OCT 'Complete model' column, DME rows; set description in section 4.2.",
         operating_point="4-class argmax", notes="per-class ACC equals Recall in Table 8 (one-vs-rest), so overall accuracy and specificity are left null; sensitivity count k=round(0.785*107)=84; FN-F1-OCT complete model chosen because it is trained on the full training set (the limited model gives DME recall 0.7663 / AUC 0.883)",
         rep=None),
    dict(row_id="P2", source="P2: He 2023 (Swin-Poly)", task_definition="4-class (CNV/DME/drusen/Normal); DME recall reported per class",
         negative_class_composition="CNV + drusen + Normal (4-class decision; no DME-vs-Normal specificity available)", test_set_identity=ID_P2,
         n_test=968, n_dme=242, split_unit="not stated", patient_split_stated="not stated", external_validation="not reported (OCT-C8 is a second, separately evaluated dataset, not external validation of the OCT2017 model)",
         accuracy=None, sensitivity=0.996, specificity=None, auroc=0.9996, comparability_category="B",
         verification_status="verified against primary source: Table 1 (p. 9), OCT2017, method 'Ours', class DME: Recall 0.9960, AUC 0.9996 (per-class Accuracy 0.9960 also printed); test set 968 / 242 per category in Materials and methods",
         source_citation="He J et al. 2023, Sci Rep 13:3637, DOI 10.1038/s41598-023-30853-z, Table 1 (Experimental results on OCT image classification), p. 9; dataset description in Materials and methods.",
         operating_point="4-class argmax", notes="sensitivity count k=round(0.996*242)=241; AUC is the DME one-vs-rest AUC printed in Table 1; per-class 'Accuracy' in Table 1 is one-vs-rest, not DME-vs-Normal, so accuracy and specificity are left null; text states 109,312 images with 108,312 train, while the listed class counts sum to 83,484",
         rep=dict(NULLREP, peer=1),
         evidence={"patient_split": "not stated (text searched)", "external": "not reported (OCT-C8 separate dataset)", "ci": "not reported (text searched)", "leakage": "not reported (text searched)", "calibration": "not reported (text searched)",
                   "explain_validated": "Score-CAM confidence score maps shown (Fig. 5); no validation check reported", "peer": "Scientific Reports 13:3637"},
         read_status="full (PDF read 2026-10-31)"),
    dict(row_id="P3", source="P3: Takinami 2025 (medRxiv preprint)", task_definition="DME vs non-DME (binary), 2D B-scan",
         negative_class_composition="CNV + drusen + Normal pooled (97,753 negatives in training pool)", test_set_identity=ID_P3, n_test=P3_N, n_dme=P3_NDME,
         split_unit="not stated", patient_split_stated="not stated (text searched)", external_validation="not reported (limitations: single OCT device)",
         accuracy=0.985, sensitivity=0.996, specificity=0.981, auroc=0.999, comparability_category="C",
         verification_status="verified against primary source: preprint PDF read 2026-10-31 (abstract, Results table, Figure 3 caption); authors Takinami, Morikawa, Oshika",
         source_citation="Takinami S, Morikawa S, Oshika T. Pre-trained Vision Transformer With Masked Autoencoder for Automated Diabetic Macular Edema Detection from Optical Coherence Tomography Images. medRxiv DOI 10.1101/2025.09.29.25336139 (v1, posted 2025-10-07; not peer reviewed); abstract, Results table (MAE_ViT) and Figure 3 caption (n=1,000: 250 DME, 750 non-DME). Paper-stated CIs: AUROC 0.999 (0.999-1.000), accuracy 0.985 (0.977-0.992), sens 0.996 (0.987-1.000), spec 0.981 (0.972-0.991).",
         operating_point="not stated", notes="Wilson CIs from rounded counts: acc k=985/1000, sens k=249/250, spec k=736/750; NOT peer reviewed",
         rep=dict(NULLREP, ci=1, peer=0),
         evidence={"patient_split": "not stated (text searched)", "external": "not reported; limitations note a single OCT device", "ci": "95% bootstrap CIs (1,000 resamples) reported for AUROC, accuracy, sensitivity, specificity", "leakage": "not reported (text searched)",
                   "calibration": "not reported (text searched)", "explain_validated": "not reported (text searched)", "peer": "medRxiv preprint, not certified by peer review"},
         read_status="full (PDF read 2026-10-31)"),
    dict(row_id="P4_comp", source="P4 Kermany 2018: independent test set (composition only)", task_definition="4-class test set (composition only)",
         negative_class_composition="n/a", test_set_identity="Independent test set of 1,000 images, 250 per category, from 633 patients not in the training set (Cell Results, 'Patient and Image Characteristics'); the n used for the binary DME-vs-Normal classifier is not stated",
         n_test=1000, n_dme=250, split_unit="patient", patient_split_stated="yes (test partition from patients independent of the training patients)", external_validation="no (internal test set)",
         comparability_category="B", verification_status="verified against primary source: Cell PDF read 2026-11-01 (p. 1124)",
         source_citation="Kermany DS et al. 2018, Cell 172:1122-1131, Results 'Patient and Image Characteristics' (p. 1124) and STAR Methods 'Quantification and statistical analysis'.",
         operating_point="composition only", notes="no metrics taken from this row; P4's own metric row keeps n null because the binary test n is not stated", rep=None),
    dict(row_id="P4", source="P4: Kermany 2018 (Cell)", task_definition="DME vs Normal (binary), 2D B-scan",
         negative_class_composition="Normal only", test_set_identity=ID_P4, n_test=None, n_dme=None, split_unit="patient (test partition from patients independent of training patients)", patient_split_stated="yes (Methods: test partition 'containing images from patients independent of the patients represented in the training partition')",
         external_validation="not reported for OCT (the chest X-ray experiment is a different task)", accuracy=0.982, sensitivity=0.968, specificity=0.996, auroc=0.9987, comparability_category="B",
         verification_status="verified against primary source: Cell 172:1122-1131, Results p. 1126: DME vs normal 'accuracy of 98.2%, sensitivity of 96.8% and specificity of 99.6%. The area under the ROC curve was 99.87%' (Figure S2B)",
         source_citation="Kermany DS et al. 2018, Cell 172:1122-1131, DOI 10.1016/j.cell.2018.02.010, Results (binary classifiers, p. 1126) and Figure S2B caption. AUROC 99.87% = 0.9987; the earlier 0.998 came from Ai 2022 Table 5 'Kermany' column and is a truncation.",
         operating_point="not stated", notes="n for the binary test not stated, so no CI",
         rep=dict(NULLREP, patient_split=1, ci=1, explain_validated=1, peer=1),
         evidence={"patient_split": "Methods: test partition from patients independent of the training patients; 633 test patients not in training set", "external": "not reported for OCT", "ci": "95% CIs reported for likelihood ratios vs human experts (Fig. S3); none for accuracy/sensitivity/specificity", "leakage": "not reported as an investigation (patient independence only)", "calibration": "not reported (text searched)",
                   "explain_validated": "occlusion-test regions scored (Table S3) and verified by human experts", "peer": "Cell 172:1122-1131"},
         read_status="full (PDF read 2026-10-31)"),
]

for c in COMPARATORS:
    n, nd = c["n_test"], c["n_dme"]
    nn = (n - nd) if (n is not None and nd is not None) else None
    for key, rate, den in [("acc", c.get("accuracy"), n), ("sens", c.get("sensitivity"), nd), ("spec", c.get("specificity"), nn)]:
        if rate is not None and den is not None and not (c["row_id"] == "P2" and key != "sens"):
            lo, hi = wilson(round(rate * den), den)
            c[f"{key}_ci_lo"], c[f"{key}_ci_hi"] = lo, hi
        else:
            c[f"{key}_ci_lo"] = c[f"{key}_ci_hi"] = None
    rep = c.pop("rep")
    ev = c.pop("evidence", None)
    rs = c.pop("read_status", "n/a (composition row)")
    if rep is None:
        rep = {k: None for k in NULLREP}
        ev = {}
    for k, v in rep.items():
        c[f"rep_{k}"] = v
    c["rep_evidence"] = json.dumps(ev)
    c["rep_read_status"] = rs
    rows.append(c)

df = pd.DataFrame(rows)
COLS = ["source", "task_definition", "negative_class_composition", "test_set_identity", "n_test", "n_dme", "split_unit", "patient_split_stated",
        "external_validation", "accuracy", "sensitivity", "specificity", "auroc", "acc_ci_lo", "acc_ci_hi", "sens_ci_lo", "sens_ci_hi", "spec_ci_lo",
        "spec_ci_hi", "comparability_category", "verification_status", "source_citation",
        "row_id", "operating_point", "notes", "rep_patient_split", "rep_external", "rep_ci", "rep_leakage", "rep_calibration", "rep_explain_validated",
        "rep_peer", "rep_evidence", "rep_read_status", "discordant_images_any_pair", "discordant_pairwise_total"]
proj_ev = REP_PROJECT["evidence"]
df.loc[df.row_id.str.startswith("proj_"), "rep_evidence"] = proj_ev
df.loc[df.row_id.str.startswith("proj_"), "rep_read_status"] = REP_PROJECT["read_status"]
for col in COLS:
    if col not in df.columns:
        df[col] = None
df = df[COLS]
DATA.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False, na_rep="null", float_format="%.10g")
print(f"wrote {OUT.relative_to(REPO)}  ({len(df)} rows, {len(COLS)} columns)")
print(df[["row_id", "n_test", "n_dme", "accuracy", "sensitivity", "specificity", "auroc", "comparability_category"]].to_string(index=False))
print("\nPRE-PLOT CHECK")
print(f"  968 == 4 x 242: {arith_968}")
print(f"  484 test files == test/DME + test/NORMAL folders exactly (strict subset of 968): {exact_subset}")
print(f"  on-disk test class counts: {on_disk}")
print(f"  P3: {P3_NDME} DME vs {PER_CLASS} per class; {P3_N} vs {total_968} total -> discrepancy recorded, cause not established")
