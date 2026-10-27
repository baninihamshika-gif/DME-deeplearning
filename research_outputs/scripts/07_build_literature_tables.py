"""
Phase C/D -- write research_audit/literature_extraction.csv and
research_audit/comparability_matrix.csv.

Every literature number below was transcribed by hand from the primary source
on 2026-10-27 and carries its exact table location. Sources read:
  Tang et al. 2021 (Diabetes Care 44(9):2078-2088): PMC8740924 full-text HTML,
    and figshare 10.2337/figshare.14710284 (Supplementary_Tables_Clean_14April2021.pdf,
    Supplementary_1_Clean_13April2021.pdf).
  Zhu et al. 2025 (BMJ Open 15(5):e099167): PMC12128415 full-text XML. Its
    online supplementary tables (per-device results, human comparison) were
    NOT retrievable (publisher site returned HTTP 403) -- marked pending.
Published values are copied as printed, including two internally inconsistent
CIs in Tang 2021 (flagged in the notes column, not corrected).
"NR" = not reported; "NA" = not applicable.

This project's own rows are read from research_outputs/data/ (never typed in).

Usage: python research_outputs/scripts/07_build_literature_tables.py
"""

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / "research_audit"
DATA = REPO / "research_outputs" / "data"

T21 = "Tang 2021, Diabetes Care, doi:10.2337/dc20-3064"
Z25 = "Zhu 2025, BMJ Open, doi:10.1136/bmjopen-2025-099167"

COLS = ["study", "source_location", "model", "input_representation", "task", "positive_class", "device", "dataset", "cohort_role",
        "evaluation_unit", "n_evaluated", "accuracy", "sensitivity", "specificity", "f1", "auroc", "auroc_ci_lo", "auroc_ci_hi",
        "sensitivity_ci", "specificity_ci", "accuracy_ci", "notes"]


def row(**k):
    return {c: k.get(c, "NR") for c in COLS}


L = []
# ---------------- Tang 2021, main Table 2 (volume level), presence vs absence of DME
t2 = [
    ("CIRRUS HD-OCT", "3D ResNet-34 (3D multitask CNN)", "3D OCT volume (128 B-scans)", "Primary", "primary validation (internal, patient-level random 20% split)", "NR", 0.937, 0.920, 0.954, 87.4, "82.7-91.6", 100.0, "100.0-100.0", 96.4, "95.1-97.6", ""),
    ("CIRRUS HD-OCT", "3D ResNet-34 (3D multitask CNN)", "3D OCT volume (128 B-scans)", "External 1 (Singapore, SERI)", "external test", "2365 volumes", 0.906, 0.947, 0.968, 81.4, "69.8-93.0", 99.8, "99.6-100.0", 99.5, "99.2-99.7", "Published AUROC CI (0.947-0.968) does not contain the point estimate 0.906; copied as printed."),
    ("CIRRUS HD-OCT", "3D ResNet-34 (3D multitask CNN)", "3D OCT volume (128 B-scans)", "External 2 (Hong Kong, AHNH)", "external test", "659 volumes", 0.929, 0.907, 0.947, 89.5, "85.1-93.0", 96.3, "93.9-97.9", 93.6, "91.5-95.4", ""),
    ("CIRRUS HD-OCT", "3D ResNet-34 (3D multitask CNN)", "3D OCT volume (128 B-scans)", "External 3 (U.S., Stanford)", "external test", "194 volumes", 0.930, 0.894, 0.965, 86.0, "78.5-92.5", 100.0, "100.0-100.0", 93.5, "90.0-96.5", ""),
    ("SPECTRALIS OCT", "2D ResNet-18 (2D multitask CNN), B-scan predictions aggregated by presence rule", "2D B-scans aggregated to volume", "Primary", "primary validation (internal, patient-level random 20% split)", "NR", 0.958, 0.930, 0.977, 92.7, "86.9-96.4", 98.9, "96.3-99.9", 96.3, "93.7-98.1", ""),
    ("SPECTRALIS OCT", "2D ResNet-18 (2D multitask CNN), B-scan predictions aggregated by presence rule", "2D B-scans aggregated to volume", "External 4 (China, Aier)", "external test", "202 volumes", 0.956, 0.935, 0.978, 100.0, "100.0-100.0", 91.3, "86.6-95.4", 92.6, "88.6-96.0", ""),
    ("SPECTRALIS OCT", "2D ResNet-18 (2D multitask CNN), B-scan predictions aggregated by presence rule", "2D B-scans aggregated to volume", "External 5 (Australia, WIMR)", "external test", "203 volumes", 0.936, 0.879, 0.994, 97.6, "95.2-99.4", 89.7, "75.9-100.0", 96.4, "93.8-98.5", ""),
    ("SPECTRALIS OCT", "2D ResNet-18 (2D multitask CNN), B-scan predictions aggregated by presence rule", "2D B-scans aggregated to volume", "External 6 (Hong Kong, UCH)", "external test", "399 volumes", 0.949, 0.922, 0.977, 96.4, "94.2-98.4", 93.4, "87.9-97.8", 95.7, "93.7-97.5", ""),
    ("Triton OCT", "2D ResNet-18 (2D multitask CNN), B-scan predictions aggregated by presence rule", "2D B-scans aggregated to volume", "Primary", "primary validation (internal, patient-level random 20% split)", "NR", 0.965, 0.948, 0.977, 94.3, "90.9-96.8", 98.6, "96.9-99.5", 96.9, "95.4-98.1", ""),
    ("Triton OCT", "2D ResNet-18 (2D multitask CNN), B-scan predictions aggregated by presence rule", "2D B-scans aggregated to volume", "External 7 (China, JSIEC)", "external test", "453 volumes", 0.954, 0.930, 0.971, 99.3, "97.9-99.9", 91.7, "77.5-98.3", 98.6, "97.1-99.5", ""),
]
for dev, model, inp, ds, role, n, a, lo, hi, se, seci, sp, spci, ac, acci, note in t2:
    L.append(row(study=T21, source_location="Main text Table 2 (volume scan level)", model=model, input_representation=inp,
                 task="Presence vs absence of DME (DME incl. retinal thickening by DRCR.net/Moorfields thresholds, cystoid spaces, SRF, hard exudates)",
                 positive_class="any DME", device=dev, dataset=ds, cohort_role=role, evaluation_unit="volume scan (per eye-visit)",
                 n_evaluated=n, accuracy=ac / 100, sensitivity=se / 100, specificity=sp / 100, f1="NR", auroc=a, auroc_ci_lo=lo, auroc_ci_hi=hi,
                 sensitivity_ci=seci, specificity_ci=spci, accuracy_ci=acci,
                 notes=(note + " n for external sets = all volumes of that set in Table 1 (externals are pure test sets); primary-validation counts not reported.").strip()))

# ---------------- Tang 2021, Supplementary Table S1 (B-scan level), presence vs absence of DME
s1 = [
    ("SPECTRALIS OCT", "Primary", "primary validation (internal, patient-level random 20% split)", "NR", 0.940, 0.934, 0.945, 88.7, "86.9-90.3", 99.2, "99.0-99.5", 97.2, "96.8-97.6", ""),
    ("SPECTRALIS OCT", "External 4 (China, Aier)", "external test", "3742 B-scans", 0.958, 0.951, 0.964, 92.9, "89.6-95.3", 98.7, "98.3-99.1", 98.2, "97.7-98.6", ""),
    ("SPECTRALIS OCT", "External 5 (Australia, WIMR)", "external test", "5090 B-scans", 0.959, 0.953, 0.964, 95.5, "94.5-96.4", 96.3, "95.6-96.9", 96.0, "95.5-96.6", ""),
    ("SPECTRALIS OCT", "External 6 (Hong Kong, UCH)", "external test", "9463 B-scans", 0.899, 0.893, 0.905, 80.4, "78.9-81.8", 99.4, "99.2-99.6", 93.6, "93.0-94.0", ""),
    ("Triton OCT", "Primary", "primary validation (internal, patient-level random 20% split)", "NR", 0.937, 0.931, 0.942, 88.5, "86.9-90.0", 98.8, "98.5-99.0", 96.8, "96.4-97.2", ""),
    ("Triton OCT", "External 7 (China, JSIEC)", "external test", "5468 B-scans", 0.947, 0.941, 0.953, 97.5, "97.0-98.0", 92.0, "90.4-100.0", 96.1, "95.6-96.1", "Published accuracy CI upper bound equals the point estimate (95.6-96.1); copied as printed."),
]
for dev, ds, role, n, a, lo, hi, se, seci, sp, spci, ac, acci, note in s1:
    L.append(row(study=T21, source_location="Supplementary Table S1 (B-scan level), figshare 10.2337/figshare.14710284", model="2D ResNet-18 (2D multitask CNN)",
                 input_representation="single 2D B-scan", task="Presence vs absence of DME (same definition as main text)", positive_class="any DME",
                 device=dev, dataset=ds, cohort_role=role, evaluation_unit="B-scan", n_evaluated=n, accuracy=ac / 100, sensitivity=se / 100,
                 specificity=sp / 100, f1="NR", auroc=a, auroc_ci_lo=lo, auroc_ci_hi=hi, sensitivity_ci=seci, specificity_ci=spci, accuracy_ci=acci,
                 notes=(note + " B-scans within a volume are correlated; whether CIs account for clustering is not stated. n for external sets = all B-scans of that set in Table 1.").strip()))

# ---------------- Zhu 2025, Table 3
z = [
    ("non-DME vs DME", "Testing", "hold-out test (internal, JSIEC, patient-level random 15%)", 0.961, 0.922, 0.936, 0.970, 0.990, 0.983, 0.996),
    ("non-DME vs DME", "External validation Chaozhou", "external validation", 0.864, 0.890, 0.873, 0.849, 0.933, 0.919, 0.947),
    ("non-DME vs DME", "External validation STU-2nd", "external validation", 0.729, 0.629, 0.458, 1.000, 0.904, 0.866, 0.942),
    ("non-DME vs DME", "External validation Total", "external validation", 0.842, 0.864, 0.818, 0.879, 0.916, 0.902, 0.930),
    ("non-CI-DME vs CI-DME", "Testing", "hold-out test (internal, JSIEC, patient-level random 15%)", 0.770, 0.795, 0.795, 0.737, 0.859, 0.812, 0.906),
    ("non-CI-DME vs CI-DME", "External validation Total", "external validation", 0.799, 0.833, 0.786, 0.821, 0.881, 0.859, 0.902),
]
nmap = {"External validation Chaozhou": "1269 volumes (column mapping inferred, see notes)", "External validation STU-2nd": "240 volumes (column mapping inferred, see notes)",
        "External validation Total": "1509 volumes", "Testing": "NR (15% of 6281 development volumes; exact count not reported)"}
for task, ds, role, ac, f1, se, sp, a, lo, hi in z:
    L.append(row(study=Z25, source_location="Table 3", model="Custom 3D CNN initialised from Models Genesis, two FC layers", input_representation="3D OCT volume (resampled to 128x128x128; split into two 64-slice cubes)",
                 task=task + " (ICO guideline definitions; hard exudates NOT counted as DME)", positive_class="DME" if task.startswith("non-DME") else "CI-DME (among DME eyes)",
                 device="Cirrus HD-OCT 4000 and Topcon 3D-OCT 2000 (pooled)", dataset=ds, cohort_role=role, evaluation_unit="OCT volume",
                 n_evaluated=nmap.get(ds, "NR"), accuracy=ac, sensitivity=se, specificity=sp, f1=f1, auroc=a, auroc_ci_lo=lo, auroc_ci_hi=hi,
                 sensitivity_ci="NR", specificity_ci="NR", accuracy_ci="NR",
                 notes="AUROC CI by DeLong (pROC 1.14.0). Table 2 lists 3 dataset labels over 4 device columns (Zeiss 1374 | Topcon 4907 | Zeiss 1269 | Topcon 240); JSIEC = 1374+4907 = 6281 matches the stated development total, so Chaozhou = 1269 (Zeiss) and STU-2nd = 240 (Topcon) is inferred. Per-device results are in online supplementary table 2, not retrieved (HTTP 403)."))

lit = pd.DataFrame(L)

# ---------------- This project (read from reconstructed outputs)
perf = pd.read_csv(DATA / "table2_within_project_performance.csv")
for _, r in perf[perf.threshold == 0.5].iterrows():
    if "ablation" in r.model:
        continue
    lit.loc[len(lit)] = row(study="This project (DME-deeplearning), reconstructed 2026-10-27", source_location=r.prediction_source, model=r.model + " (ImageNet-pretrained, fine-tuned)",
                            input_representation="single 2D B-scan (300x300 letterbox)", task="DME vs Normal (Kermany folder labels; CNV/DRUSEN excluded)", positive_class="DME",
                            device="Heidelberg Spectralis (Kermany et al. 2018, STAR Methods, verified from the published text; not recorded per image)", dataset="Kermany OCT2017 official test split",
                            cohort_role="internal held-out test (same source as training; patient-disjoint after exclusion)", evaluation_unit="B-scan",
                            n_evaluated=f"{int(r.n)} B-scans ({int(r.n_dme)} DME / {int(r.n_normal)} Normal)", accuracy=r.accuracy, sensitivity=r.sensitivity,
                            specificity=r.specificity, f1=r.f1, auroc=r.auroc, auroc_ci_lo=r.auroc_ci_lo, auroc_ci_hi=r.auroc_ci_hi,
                            sensitivity_ci=f"{r.sensitivity_ci_lo:.3f}-{r.sensitivity_ci_hi:.3f} (Wilson)", specificity_ci=f"{r.specificity_ci_lo:.3f}-{r.specificity_ci_hi:.3f} (Wilson)",
                            accuracy_ci=f"{r.accuracy_ci_lo:.3f}-{r.accuracy_ci_hi:.3f} (Wilson)", notes="Threshold 0.5, raw softmax. DeLong AUROC CI degenerate (perfect separation).")
lit.to_csv(AUDIT / "literature_extraction.csv", index=False)

# ---------------- Comparability matrix
CM = [
    dict(candidate="EfficientNet-B3 vs EfficientNet-B0 vs ResNet-50 (this project)", input="2D B-scan / same", task="DME vs Normal / same", population="Kermany OCT2017 / same",
         device="same", unit="B-scan / same", test_construction="identical 484 images", validation="internal / same", balance="242/242 / same", selection="val AUC (same split, same code); thresholds tuned on contaminated set",
         metric_scale="same code", uncertainty="computable from predictions", comparable="yes", category="A", plot="Figure 1, Figure 2, Table 2"),
    dict(candidate="Tang 2021 2D ResNet-18, B-scan level, primary validation (Suppl. Table S1)", input="2D B-scan / same", task="any DME (incl. thickening, hard exudates) vs no DME (diabetic eyes, may include non-DME abnormalities) / similar, NOT identical", population="Hong Kong diabetic cohort (CUHK-STDR) / different",
         device="SPECTRALIS (same vendor as Kermany) and Triton", unit="B-scan / same", test_construction="patient-level random 20% split", validation="internal / same type", balance="DME minority (Table 1) vs 50/50 here", selection="not described",
         metric_scale="AUROC 0-1, % for rates", uncertainty="95% CIs published", comparable="partially", category="B", plot="Table 3 and Figure 3 panel B (B-scan level, labelled cross-study)"),
    dict(candidate="Tang 2021 2D ResNet-18, B-scan level, external sets 4-7 (Suppl. Table S1)", input="2D B-scan / same", task="as above", population="external clinics / different",
         device="SPECTRALIS / Triton", unit="B-scan", test_construction="independent external sets", validation="EXTERNAL (this project has none)", balance="varies", selection="NR",
         metric_scale="as above", uncertainty="published", comparable="partially (external vs internal)", category="B", plot="Table 3 only (validation type differs)"),
    dict(candidate="Tang 2021, volume level (main Table 2), 2D CNN aggregated or 3D CNN", input="volume (3D CNN) or B-scans aggregated to volume / different", task="as above", population="different",
         device="CIRRUS / SPECTRALIS / Triton", unit="volume / different", test_construction="internal + 7 external", validation="internal + external", balance="varies", selection="NR",
         metric_scale="as above", uncertainty="published", comparable="no (unit differs)", category="C", plot="Table 3 contextual rows; Figure 3 separate contextual panel"),
    dict(candidate="Zhu 2025 3D CNN, non-DME vs DME (Table 3)", input="3D volume / different", task="DME vs non-DME, ICO definitions, hard exudates excluded / similar label, different definition", population="Shantou/Chaozhou Chinese clinics / different",
         device="Cirrus + Topcon / different", unit="volume / different", test_construction="15% hold-out + centre-split external", validation="internal + external", balance="NR for test set", selection="NR",
         metric_scale="AUROC 0-1, rates 0-1", uncertainty="AUROC CI (DeLong) only", comparable="no", category="C", plot="Table 3; Figure 3 separate contextual panel"),
    dict(candidate="Zhu 2025 3D CNN, CI-DME vs non-CI-DME", input="3D volume", task="centre involvement among DME / not addressable from one 2D B-scan", population="different",
         device="different", unit="volume", test_construction="as above", validation="as above", balance="NR", selection="NR",
         metric_scale="as above", uncertainty="AUROC CI only", comparable="no", category="D", plot="text only"),
    dict(candidate="Tang 2021 CI-DME vs non-CI-DME (all levels)", input="volume / B-scan", task="centre involvement / not addressed by this project", population="different",
         device="different", unit="various", test_construction="-", validation="-", balance="-", selection="-",
         metric_scale="-", uncertainty="published", comparable="no", category="D", plot="text only"),
    dict(candidate="This project's U-Net (Dice) vs any classification metric", input="-", task="segmentation vs classification", population="Duke vs Kermany",
         device="-", unit="pixel/image vs image", test_construction="-", validation="-", balance="-", selection="-", metric_scale="Dice vs accuracy", uncertainty="-",
         comparable="no", category="D", plot="never on the same axis"),
]
pd.DataFrame(CM).to_csv(AUDIT / "comparability_matrix.csv", index=False)
print(len(lit), "literature rows;", len(CM), "comparability rows")
