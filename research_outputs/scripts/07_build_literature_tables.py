"""
Phase C/D -- write research_audit/literature_extraction.csv and
research_audit/comparability_matrix.csv.

Every literature number below was transcribed by hand from the primary source
on 2026-10-27 and carries its exact table location. Sources read:
  Tang et al. 2021 (Diabetes Care 44(9):2078-2088): PMC8740924 full-text HTML,
    and figshare 10.2337/figshare.14710284 (Supplementary_Tables_Clean_14April2021.pdf,
    Supplementary_1_Clean_13April2021.pdf).
  Zhu et al. 2025 (BMJ Open 15(5):e099167): PMC12128415 full-text XML, checked
    against the publisher's 8-page article PDF (supplied by the user; Table 3
    values identical; Table 2 column layout read from the PDF), and the online
    supplement bmjopen-15-5-s001.docx (supplied by the user): Supplementary
    Table 1 confusion matrices (every published metric recomputes from them),
    Table 2 per-device results, Table 3 human-system comparison.
  Singh & Gorantla 2020 (PLoS ONE 15(2):e0220677), Rodríguez-Miguel et al. 2024
    (Sci Rep 14:17633) and Li F et al. 2019 (Biomed Opt Express 10(12):6204):
    article PDFs supplied by the user, read 2026-10-28 (text via pdfplumber;
    DMENet Fig 6A read from the rendered page), plus Rodríguez-Miguel's
    Supplementary Material 1 (docx from the article page). Authors, title,
    journal, volume, issue, first page/article number and date of all three
    were checked against Crossref on 2026-10-28 (all match); the BOE end page
    (6226) comes from the PDF, as Crossref lists the first page only.
Published values are copied as printed, including internally inconsistent
values (two CIs in Tang 2021; others in the three 2019-2024 papers), which are
flagged in the notes column, not corrected. comparability_category is set
explicitly for the later papers; Tang/Zhu rows are categorised in script 09.
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
        "sensitivity_ci", "specificity_ci", "accuracy_ci", "notes", "comparability_category"]


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
# Supplementary Table 1 confusion matrices (TN, FP, FN, TP), keyed by (task, dataset)
zcm = {("non-DME vs DME", "Testing"): (700, 22, 15, 219), ("non-DME vs DME", "External validation Chaozhou"): (398, 71, 102, 698),
       ("non-DME vs DME", "External validation STU-2nd"): (120, 0, 65, 55), ("non-DME vs DME", "External validation Total"): (518, 71, 167, 753),
       ("non-CI-DME vs CI-DME", "Testing"): (73, 26, 26, 101), ("non-CI-DME vs CI-DME", "External validation Total"): (275, 60, 125, 460)}
ZNOTE = ("AUROC CI by DeLong (pROC 1.14.0). Table 2 (article PDF layout): JSIEC spans Zeiss 1374 + Topcon 4907 (= 6281 development); "
         "Chaozhou = Zeiss 1269; STU-2nd = Topcon 240. TN/FP/FN/TP from Supplementary Table 1; published accuracy, F1, sensitivity and "
         "specificity all recompute from them.")
dmap = {"External validation Chaozhou": "Cirrus HD-OCT 4000 (Zeiss) only", "External validation STU-2nd": "Topcon 3D-OCT 2000 only"}
for task, ds, role, ac, f1, se, sp, a, lo, hi in z:
    L.append(row(study=Z25, source_location="Table 3", model="Custom 3D CNN initialised from Models Genesis, two FC layers", input_representation="3D OCT volume (resampled to 128x128x128; split into two 64-slice cubes)",
                 task=task + " (ICO guideline definitions; hard exudates NOT counted as DME)", positive_class="DME" if task.startswith("non-DME") else "CI-DME (among DME eyes)",
                 device=dmap.get(ds, "Cirrus HD-OCT 4000 and Topcon 3D-OCT 2000 (pooled)"), dataset=ds, cohort_role=role, evaluation_unit="OCT volume",
                 n_evaluated="{} volumes (TN {}, FP {}, FN {}, TP {})".format(sum(zcm[(task, ds)]), *zcm[(task, ds)]), accuracy=ac, sensitivity=se, specificity=sp, f1=f1, auroc=a, auroc_ci_lo=lo, auroc_ci_hi=hi,
                 sensitivity_ci="NR", specificity_ci="NR", accuracy_ci="NR",
                 notes=ZNOTE))

# ---------------- Zhu 2025, Supplementary Tables 2 and 3 (online supplement, bmjopen-15-5-s001.docx)
zs = [  # source, task, dataset, device, role, (TN, FP, FN, TP), acc, f1, sens, spec, auroc, lo, hi, note
    ("Supplementary Table 2", "non-DME vs DME", "Testing, Topcon only", "Topcon 3D-OCT 2000", "hold-out test (internal, JSIEC, patient-level random 15%)", (650, 13, 7, 90), 0.974, 0.900, 0.928, 0.980, 0.989, 0.977, 1.000, ""),
    ("Supplementary Table 2", "non-DME vs DME", "Testing, Zeiss only", "Cirrus HD-OCT 4000 (Zeiss)", "hold-out test (internal, JSIEC, patient-level random 15%)", (129, 9, 8, 50), 0.913, 0.855, 0.862, 0.935, 0.967, 0.944, 0.990, ""),
    ("Supplementary Table 3", "non-DME vs DME", "Human-system comparison set", "NR", "reader comparison set (source and sampling not stated in the supplement)", (104, 7, 8, 81), 0.925, 0.915, 0.910, 0.937, 0.971, 0.948, 0.993,
     "Three retina experts on the same 200 volumes: sensitivity 0.944 / 1.000 / 1.000, specificity 0.658 / 0.856 / 0.928 (AUROC NA). The printed 'experts' average' F1 (1.957; 1.871 for CI-DME) cannot exceed 1 and does not match its counts (0.887); copied in this note only."),
]
for src, task, ds, dev, role, cmx, ac, f1, se, sp, a, lo, hi, note in zs:
    L.append(row(study=Z25, source_location=f"{src} (online supplement bmjopen-15-5-s001.docx)", model="Custom 3D CNN initialised from Models Genesis, two FC layers", input_representation="3D OCT volume (resampled to 128x128x128; split into two 64-slice cubes)",
                 task=task + " (ICO guideline definitions; hard exudates NOT counted as DME)", positive_class="DME",
                 device=dev, dataset=ds, cohort_role=role, evaluation_unit="OCT volume",
                 n_evaluated="{} volumes (TN {}, FP {}, FN {}, TP {})".format(sum(cmx), *cmx), accuracy=ac, sensitivity=se, specificity=sp, f1=f1, auroc=a, auroc_ci_lo=lo, auroc_ci_hi=hi,
                 sensitivity_ci="NR", specificity_ci="NR", accuracy_ci="NR",
                 notes=("Metrics recompute from the printed counts. " + note).strip()))

# ---------------- Singh & Gorantla 2020 (PLoS ONE 15(2):e0220677), article PDF supplied by the user.
# Colour FUNDUS photographs (IDRiD + MESSIDOR), DME = hard exudates: different modality, category D.
S20 = "Singh & Gorantla 2020, PLoS ONE, doi:10.1371/journal.pone.0220677"
SNOTE = ("Colour fundus photographs, not OCT; DME graded from hard exudates near the macula (IDRiD/MESSIDOR grades 0-2). "
         "5-fold cross-validation at image level on augmented data; patient grouping not described. Different modality and label definition: not comparable.")
s20 = [  # source, dataset, n, acc, sens, spec, f1, auroc, note
    ("Table 3 and Fig 6A (Gamma Ensemble, stage 1)", "IDRiD + MESSIDOR pooled", "360 test images", 0.9667, 0.9750, 0.9563, 0.9701, 0.9654,
     "Fig 6A (row-normalised) shows No-DME correct 0.9653 and DME correct 0.9684, i.e. sensitivity 0.968 / specificity 0.965, not Table 3's 0.975 / 0.956. "
     "Counts consistent with both (inferred, not printed): TN 195, FP 7, FN 5, TP 153; Table 3's sensitivity, specificity and precision then equal NPV, PPV and specificity, "
     "and its F1 0.9701 equals the No-DME class F1 (DME-class F1 0.962). Copied as printed."),
    ("Table 4 / abstract (HE-CNN, both stages mapped to 3 grades)", "IDRiD + MESSIDOR, averaged", "NR", 0.9612, 0.9632, 0.9584, 0.9609, "NR",
     "Headline abstract figures; 3-grade result averaged over two datasets, not DME presence."),
    ("Table 5 (DMENet on MESSIDOR)", "MESSIDOR", "NR", 0.9547, 0.9468, 0.9719, "NR", "NR", ""),
    ("Table 6 (DMENet on IDRiD)", "IDRiD", "NR", 0.9676, 0.9788, 0.9449, "NR", "NR", ""),
]
for src, ds, n, ac, se, sp, f1, a, note in s20:
    L.append(row(study=S20, source_location=src, model="Hierarchical Ensemble of CNNs (ResNet/DenseNet/SE-ResNet/SqueezeNet learners with gating CNNs)",
                 input_representation="colour fundus photograph", task="DME (hard exudates) presence, then grade 1 vs 2", positive_class="DME (fundus grade 1-2)",
                 device="Kowa VX-10 alpha (IDRiD); Topcon TRC NW6 (MESSIDOR)", dataset=ds, cohort_role="5-fold cross-validation (image level)",
                 evaluation_unit="fundus image", n_evaluated=n, accuracy=ac, sensitivity=se, specificity=sp, f1=f1, auroc=a, auroc_ci_lo="NR", auroc_ci_hi="NR",
                 notes=(SNOTE + " " + note).strip(), comparability_category="D"))

# ---------------- Rodríguez-Miguel et al. 2024 (Sci Rep 14:17633), article PDF supplied by the user, and
# Supplementary Material 1 (41598_2024_68489_MOESM2_ESM.docx, downloaded from the article page 2026-10-28).
R24 = "Rodríguez-Miguel 2024, Sci Rep, doi:10.1038/s41598-024-68489-2"
RNOTE = ("Hybrid CNN-RNN on 22 automatically selected B-scans (12 foveal, every 2nd of slices 60-85; 10 parafoveal, every 10th) of a 128-slice Topcon cube; "
         "CNN backbone pre-trained on Kermany OCT2017 (the dataset this project trains on). Models were trained on a separate 650-cube ground-truth set "
         "(433/100/117 train/val/test, subject-disjoint); 7 of them were kept for 'the best generalizable metrics in the test set' (Supplementary Material 1). "
         "Evaluation cohort: retrospective real-world DR screening, 5,314 cubes from 4,408 subjects, prevalence 2.37%; reference = screening retina specialist "
         "(thickening >= 300 um plus anatomical signs). Thresholds were tuned over 1,000 values on this same cohort (Youden / highest sensitivity / highest specificity), "
         "so operating points are optimistic. Repeat screens of the same subject are included; the binomial CIs treat cubes as independent.")
r24 = [  # model, threshold label, sens, sens_ci, spec, spec_ci, auroc, lo, hi, n, note
    ("Model 2", "threshold 1 (Youden)", 0.905, "84.0-95.0", 0.908, "90.0-91.6", 0.94, 0.91, 0.97, "5314 cubes (126 DME / 5188 non-DME)", ""),
    ("Ensemble (mode vote of models 1-3)", "threshold 1 (Youden)", 0.889, "82.1-93.8", 0.933, "92.6-94.0", "NR", "NR", "NR", "5314 cubes (126 DME / 5188 non-DME)", ""),
    ("Ensemble (mode vote of models 1-3)", "threshold 2 (highest sensitivity)", 0.921, "82.1-93.8", 0.831, "82.1-84.1", "NR", "NR", "NR", "5314 cubes (126 DME / 5188 non-DME)",
     "All four threshold-2 rows print the same sensitivity CI (82.1-93.8), identical to the threshold-1 ensemble CI; for 116/126 (92.1%) the exact CI is 85.9-96.1 and the "
     "printed upper bound is barely above the point estimate. Likely a copy error; copied as printed."),
    ("Ensemble (mode vote of models 1-3)", "threshold 3 (highest specificity)", 0.810, "73.0-87.4", 0.970, "96.5-97.4", "NR", "NR", "NR",
     "5314 cubes (TN 5032, FP 156, FN 24, TP 102)",
     "FN 24 and FP 156 are printed in the Results; sensitivity, specificity and PPV 39.5% recompute from them. The false-negative review says 'just 4 DMEs (12.1%)' were missed: "
     "4/24 = 16.7%, while 12.1% = 4/33, so the denominator does not match the 24 false negatives."),
]
for model, thr, se, seci, sp, spci, a, lo, hi, n, note in r24:
    L.append(row(study=R24, source_location="Table 2 and Results text" + (" (AUROC: Results, Fig 1)" if a != "NR" else ""),
                 model=f"Hybrid CNN-RNN (22 parallel custom CNNs + bidirectional RNN), {model}, {thr}", input_representation="22-B-scan subset of a 3D OCT cube",
                 task="DME vs non-DME in diabetic screening attendees", positive_class="DME (retina specialist, screening programme)",
                 device="Topcon 3D OCT-Maestro 1", dataset="Real-world DR screening cohort (Madrid), Nov 2015 - Mar 2019",
                 cohort_role="retrospective screening cohort (separate from the training set; same programme, thresholds tuned on it)", evaluation_unit="OCT cube (screening visit)",
                 n_evaluated=n, accuracy="NR", sensitivity=se, specificity=sp, f1="NR", auroc=a, auroc_ci_lo=lo, auroc_ci_hi=hi,
                 sensitivity_ci=seci, specificity_ci=spci, accuracy_ci="NR", notes=(RNOTE + " " + note).strip(), comparability_category="C"))

# Supplementary Material 1, first table: ground-truth test split (117 cubes: 40 DME / 77 non-DME); no sensitivity/specificity/CIs printed.
SUPPNOTE = ("Supplementary Material 1 also: (i) the listed slice positions (1, 11, ..., 51, 61-85 every 2nd, 95, 115, 125) give 13 foveal + 9 parafoveal slices, "
            "not the stated 12 + 10 (slice 105 skipped), and the figure caption says a 126-slice cube; (ii) it describes the Kermany pre-training data as "
            "'acquired with different OCT manufacturers', whereas Kermany's STAR Methods report Spectralis; (iii) the pre-trained 4-class CNN scored "
            "accuracy 0.909 / AUROC 0.987 on the 1,000-image Kermany test set.")
r24s = [  # model, config, acc, auroc, auprc
    ("Model 1", "biRNN(10), 24 CNN output neurons", 0.915, 0.960, 0.934),
    ("Model 2", "biLSTM(11), 24 CNN output neurons", 0.890, 0.944, 0.894),
    ("Model 3", "biGRU(5), 20 CNN output neurons", 0.880, 0.962, 0.940),
]
for model, cfg, ac, a, ap in r24s:
    L.append(row(study=R24, source_location="Supplementary Material 1, table 'Training and test results of the CNN-RNN models over the ground truth dataset'",
                 model=f"Hybrid CNN-RNN, {model} ({cfg})", input_representation="22-B-scan subset of a 3D OCT cube",
                 task="DME vs non-DME (expert ground truth)", positive_class="DME (retina experts' ground truth)", device="Topcon 3D OCT-Maestro 1",
                 dataset="Ground-truth set test split (two Spanish DR screening programmes)",
                 cohort_role="internal held-out test (subject-disjoint), also used to select the 7 retained models", evaluation_unit="OCT cube",
                 n_evaluated="117 cubes (40 DME / 77 non-DME)", accuracy=ac, sensitivity="NR", specificity="NR", f1="NR", auroc=a, auroc_ci_lo="NR", auroc_ci_hi="NR",
                 notes=(f"AUPRC {ap}. Optimistic: this test split was used to choose which models to keep, and the final three were chosen on the screening cohort. "
                        "The three best models on the screening cohort (main-text models 1-3) carry the same numbers in the supplement's second table (AUROC 0.94 each), "
                        "so the numbering is taken to be shared across both tables. " + SUPPNOTE), comparability_category="C"))

# ---------------- Li F et al. 2019 (Biomed Opt Express 10(12):6204-6226), article PDF supplied by the user.
L19 = "Li F 2019, Biomed Opt Express, doi:10.1364/BOE.10.006204"
LMODEL = "Ensemble of four improved (dilated) ResNet50, ImageNet-pretrained"
LNOTE = ("Proprietary retrospective set (Shanghai Zhongshan + First People's hospitals, 2014-2019): 15,573 Spectralis B-scans from 2,796 patients after exclusions; "
         "test set 267 patients / 1,536 B-scans (395 CNV, 332 DME, 174 DRUSEN, 635 NORMAL), patient-disjoint from 10-fold training/validation. "
         "Labels by 5 junior + 3 senior ophthalmologists with adjudication. The paper defines sensitivity and specificity as correct abnormal / correct normal "
         "'divided by the total number of test images' (as printed). CI method not stated.")
l19 = [  # source, task, positive, dataset, role, unit, n, acc, accci, se, seci, sp, spci, a, lo, hi, cat, note
    ("Table 6 (3 ensembling binary classifiers, DME vs NORMAL)", "DME vs NORMAL (binary)", "DME", "Shanghai hospitals, independent test patients",
     "internal held-out test (patient-disjoint)", "B-scan", "NR (test set holds 332 DME / 635 NORMAL B-scans)", 0.988, "0.987-0.989", 0.986, "0.984-0.988", 0.991, "0.990-0.992",
     0.996, 0.995, 0.997, "B",
     "Closest published analogue of this project (single Spectralis B-scan, DME vs normal, patient-disjoint test). Inconsistencies: 0.986 is not k/332 (327/332 = 0.985, 328/332 = 0.988); "
     "the CIs are about 5x narrower than binomial for n = 332 / 635 (Wilson 0.965-0.994 for 327/332), so they cannot be per-image test-set CIs."),
    ("Table 3 (4-class model, abnormal vs normal)", "Any of CNV/DME/DRUSEN vs NORMAL (from a 4-class model)", "abnormal (CNV, DME or DRUSEN)", "Shanghai hospitals, independent test patients",
     "internal held-out test (patient-disjoint)", "B-scan", "1536 B-scans (901 abnormal / 635 NORMAL)", 0.973, "0.971-0.975", 0.963, "0.960-0.966", 0.985, "0.983-0.987",
     0.995, 0.994, 0.996, "D",
     "Wilson CI for 0.973 at n = 1536 is 0.964-0.980, wider than printed. A separate 'best model' (Table 4: acc 0.979, sens 0.968, spec 0.994, AUROC 0.998) was chosen as the best of 10 folds on the test set itself, so it is optimistically selected."),
    ("Table 8 (UCSD dataset, 'with pre-trained weights' row, DME class)", "DME one-vs-rest within CNV/DME/DRUSEN/NORMAL", "DME", "UCSD = Kermany OCT2017 (84,484 B-scans, 4,686 patients)",
     "6 sequential splits: train on 1/6, test on 5/6; mean +/- SD", "B-scan", "NR (5/6 of 84,484 per run)", 0.971, "SD 0.002", 0.901, "SD 0.012", 0.982, "SD 0.004", 0.978, "NR", "NR", "D",
     "Same source dataset as this project, but a 4-class one-vs-rest task (negatives include CNV and DRUSEN) and a different split; whether splits are patient-disjoint is not stated. "
     "Values printed in percent with SD over six runs (precision 94.3)."),
    ("Table 7 (DHU = Duke Srinivasan 2014, leave-three-out CV)", "AMD / DME / NORMAL (3-class), volume = mode of B-scan predictions", "DME", "Duke Srinivasan 2014 (15 AMD, 15 DME, 15 NORMAL volumes)",
     "45-fold leave-three-out cross-validation, fine-tuned on the other 42 volumes", "OCT volume", "30 volumes (15 DME / 15 NORMAL; AMD excluded here)", "NR", "NR", 1.0, "NR", 1.0, "NR",
     "NR", "NR", "NR", "C",
     "DME 15/15 and NORMAL 15/15 volumes correctly classified with weights pre-trained on the authors' data (14/15 NORMAL without them). 'Specificity' here = NORMAL volumes classified NORMAL in a 3-class task. "
     "Same 30 volumes as this project's Duke 2014 external set (report section 7.4), but fine-tuned on Duke by cross-validation, whereas this project applied its models without any Duke training."),
]
for src, task, pos, ds, role, unit, n, ac, acci, se, seci, sp, spci, a, lo, hi, cat, note in l19:
    L.append(row(study=L19, source_location=src, model=LMODEL, input_representation="single 2D B-scan (224x224)" if unit == "B-scan" else "2D B-scans aggregated to volume (mode)",
                 task=task, positive_class=pos, device="Heidelberg Spectralis", dataset=ds, cohort_role=role, evaluation_unit=unit, n_evaluated=n,
                 accuracy=ac, sensitivity=se, specificity=sp, f1="NR", auroc=a, auroc_ci_lo=lo, auroc_ci_hi=hi, sensitivity_ci=seci, specificity_ci=spci, accuracy_ci=acci,
                 notes=((LNOTE + " ") if ds.startswith("Shanghai") else "") + note, comparability_category=cat))

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
    dict(candidate="Li F 2019 ResNet50 ensemble, DME vs NORMAL binary (Table 6)", input="2D B-scan / same", task="DME vs NORMAL, ophthalmologist-graded / similar", population="Shanghai hospital patients / different",
         device="Heidelberg Spectralis / same vendor as Kermany", unit="B-scan / same", test_construction="267 test patients, patient-disjoint", validation="internal / same type",
         balance="332 DME / 635 NORMAL vs 242/242 here", selection="NR for the binary models; the 4-class 'best model' was picked on the test set",
         metric_scale="0-1", uncertainty="CIs published but too narrow for the stated n", comparable="partially (closest published analogue)", category="B", plot="Table 3 and Figure 3 (B-scan level, labelled cross-study)"),
    dict(candidate="Li F 2019 on UCSD/Kermany (Table 8) and on Duke 2014 (Table 7)", input="2D B-scan; volume by mode rule", task="4-class one-vs-rest (UCSD); 3-class volume (Duke) / different",
         population="Kermany OCT2017 (same source) and Duke 2014 (same set as this project's external Duke 2014)", device="Spectralis / same", unit="B-scan; volume",
         test_construction="train 1/6, test 5/6 (UCSD); leave-three-out CV fine-tuned on Duke", validation="internal resampling only", balance="different", selection="NR",
         metric_scale="% with SD; volume fractions", uncertainty="SD over runs; none for Duke", comparable="no (task and training protocol differ)", category="C/D", plot="Table 3 only"),
    dict(candidate="Rodríguez-Miguel 2024 CNN-RNN on 22-slice cubes (Table 2)", input="22 B-scans of a 3D cube / different", task="DME vs non-DME in unselected screening attendees / different population and prevalence (2.4%)",
         population="Madrid DR screening programme / different", device="Topcon 3D OCT-Maestro 1 / different", unit="OCT cube / different",
         test_construction="whole screening cohort, separate from the 650-cube training set", validation="temporal/real-world cohort, same programme", balance="126 / 5188",
         selection="7 models kept on the 117-cube ground-truth test split, final 3 and thresholds chosen on the evaluation cohort", metric_scale="%", uncertainty="binomial CIs (one copied interval flagged)", comparable="no (unit, device and spectrum differ)", category="C",
         plot="Table 3; Figure 3 contextual panel (model 2 AUROC)"),
    dict(candidate="Singh & Gorantla 2020 DMENet (fundus)", input="colour fundus photograph / different modality", task="hard-exudate DME grade / different definition", population="IDRiD + MESSIDOR / different",
         device="fundus cameras / different", unit="fundus image", test_construction="5-fold CV on augmented images", validation="internal", balance="augmented to balance", selection="NR",
         metric_scale="%", uncertainty="none", comparable="no", category="D", plot="Table 3 only, never on the same axis"),
    dict(candidate="This project's U-Net (Dice) vs any classification metric", input="-", task="segmentation vs classification", population="Duke vs Kermany",
         device="-", unit="pixel/image vs image", test_construction="-", validation="-", balance="-", selection="-", metric_scale="Dice vs accuracy", uncertainty="-",
         comparable="no", category="D", plot="never on the same axis"),
]
pd.DataFrame(CM).to_csv(AUDIT / "comparability_matrix.csv", index=False)
print(len(lit), "literature rows;", len(CM), "comparability rows")
