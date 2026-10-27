# Research Results Report: Audit and Statistical Evaluation of a 2D OCT Deep-Learning Classifier for Diabetic Macular Edema

Repository: DME-deeplearning (local working copy, branch `research-audit`). Audit date: 2026-10-27.
Every number in this report is read from a CSV or JSON file under `research_outputs/data/` or `research_audit/`, cited inline. Labels used below:

- **[E]** an existing result from the original project
- **[R]** independently reconstructed in this audit
- **[N]** a new analysis performed in this audit
- **[U]** remains unverified

---

## 1. Research objective

To detect diabetic macular edema (DME) versus normal retina from single 2D OCT B-scans with a deep convolutional classifier, and to evaluate that classifier rigorously: verified metrics with uncertainty, a leakage audit, calibration, and positioning against published DME systems.

## 2. Verified description of the implemented models

- **Classifiers [R]:** timm `tf_efficientnet_b3` (primary), `tf_efficientnet_b0` and `resnet50`. Each starts from ImageNet weights, has a 2-class head and dropout 0.3, and is fine-tuned in two phases: head only for 5 epochs (LR 1e-3), then the full network for up to 25 more epochs (LR 1e-5). Loss is class-weighted cross-entropy. An ablation `tf_efficientnet_b3` was trained with label smoothing 0.05.
- **Selected checkpoints [R]** (`checkpoint_metadata.json`): B3 at epoch 20 (validation AUROC 0.9958); B0 at epoch 19 (0.9950; *previously reported as "not recorded"*); ResNet-50 at epoch 30 (0.9961). Selection used the best validation AUROC.
- **Positive class:** DME (index 1). Scores are softmax P(DME).
- **Explainability [E]:** Grad-CAM on `conv_head`, with a classifier-weight randomisation sanity check (mean correlation −0.117). Not regenerated here.
- **Calibration [E/R]:** single-temperature scaling (§11).
- **Segmentation [R]:** a U-Net (`smp`, EfficientNet-B3 encoder initialised from the classifier) trained on Duke DME fluid masks (§10).
- **Not implemented:** 3D/volumetric models, ensembles. External data were not part of the original project; this audit added an external evaluation (§7.4).

## 3. Dataset and split

Source: the Kermany et al. 2018 OCT2017 release (83,484 training images), DME and NORMAL classes only (`research_outputs/tables/table1_dataset_setup.md`).

Per the paper's STAR Methods (verified from the published text) **[E]**: Spectralis OCT (Heidelberg Engineering); retrospective adult cohorts from five centres (Shiley Eye Institute UCSD, California Retinal Research Foundation, Medical Center Ophthalmology Associates, Shanghai First People's Hospital, Beijing Tongren Eye Center), July 2013 – March 2017; one horizontal foveal B-scan per exported image. Labels started from the patient's most recent EMR diagnosis and passed a three-tier grading: students (quality control, artifact exclusion), four ophthalmologists (independent grading), and two senior retinal specialists (final verification).

Two differences from the released data matter here. The paper describes 108,312 training images and a 1,000-image test set (250 per class), whereas this project's release has 83,484 training images and 242 test images per class. The paper does not describe the released test set's selection, which is consistent with the test-set differences found in §7.3.

- `train/`: 11,348 DME + 26,315 Normal images. Training and validation use a **patient-grouped** StratifiedGroupKFold (5 folds, seed 42, one fold held out for validation). Patient = the ID in the filename; eye and volume identity are not available.
- The official `test/` split (242 DME + 242 Normal; 368 patients) shares **305 patients** with `train/`. The project removed every `train/` image of those patients (7,503 images) before training, leaving a training pool of 30,160 images (Kaggle runs: 24,600 train / 5,560 validation).
- After exclusion: test↔training patient overlap 0 and exact duplicates 0 **[R]** (`leakage_summary.json`).

## 4. Evaluation protocol

- **Unit:** the individual B-scan. **Set:** the official 484-image test set, identical for all models (asserted).
- **Primary operating point:** threshold 0.5 on raw softmax probabilities (equivalent to argmax, so pre-specified).
- **Original pipeline:** per-image outputs were never saved. This audit regenerated them with a fresh forward pass of each unmodified checkpoint (`research_outputs/scripts/01_reinfer_classifiers.py`) **[R]**.
- **"Tuned" thresholds** in the original reports were chosen on a validation split that does not match the training-time split (§7.2). They are superseded and shown for reconciliation only. **Refit** tuned thresholds and temperatures, fitted on the rebuilt training-time validation split (n = 5,560), are reported in Table 2 and §7.2 **[N]**.

## 5. Verified model performance (official test set, threshold 0.5) [R]

Source: `research_outputs/data/table2_within_project_performance.csv`; `research_outputs/tables/table2_within_project_performance.md`.

| Model | TP / FP / TN / FN | Accuracy (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | F1 (95% CI) | AUROC |
|---|---|---|---|---|---|---|
| EfficientNet-B3 | 241 / 0 / 242 / 1 | 0.9979 (0.9884–0.9996) | 0.9959 (0.9770–0.9993) | 1.0000 (0.9844–1.0000) | 0.9979 (0.9938–1.0000) | 1.0000 |
| EfficientNet-B0 | 241 / 0 / 242 / 1 | 0.9979 (0.9884–0.9996) | 0.9959 (0.9770–0.9993) | 1.0000 (0.9844–1.0000) | 0.9979 (0.9938–1.0000) | 1.0000 |
| ResNet-50 | 241 / 0 / 242 / 1 | 0.9979 (0.9884–0.9996) | 0.9959 (0.9770–0.9993) | 1.0000 (0.9844–1.0000) | 0.9979 (0.9938–1.0000) | 1.0000 |

- All 80 reconstructed values (4 checkpoints × 2 thresholds × 10 metrics) equal the values stored in the original `metrics.json` files (`reconstructed_vs_reported.csv`).
- Precision = 1.000 (95% CI 0.984–1.000) for every model. PR-AUC = 1.000.
- AUROC = 1.000 means every DME image scored above every Normal image. The DeLong interval is degenerate (zero variance), and **no meaningful AUROC confidence interval exists on this test set**.
- **The models miss different images.** B3's only false negative is DME-8366524-1 (P = 0.309). B0 and ResNet-50 instead miss DME-6151003-1 (P = 0.079 and 0.030) (`test_misclassified_any_model.csv`).
- **Paired comparisons:** exact McNemar p = 1.00 for every pair (Holm-adjusted 1.00), with 0–2 discordant images per pair. Paired DeLong is undefined because both AUROCs are 1 (`paired_model_comparisons.csv`). **This test set provides no evidence that any architecture differs from another.** It is too small and too easy to rank them.

### 5b. Supplementary internal estimate on unseen images (new analysis) [N]

As a check on the 484-image test set, each checkpoint was also scored on the 7,503 `train/` images of the 305 test-set patients. These images were excluded from training and validation, so no model had seen them; they are different images from the official test set, but from the same patients (4,276 DME / 3,227 Normal). Patient-cluster bootstrap 95% CIs:

| Model | TP / FP / TN / FN | Accuracy (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | F1 (95% CI) | AUROC (95% CI) |
|---|---|---|---|---|---|---|
| EfficientNet-B3 | 4124 / 55 / 3172 / 152 | 0.9724 (0.9636–0.9801) | 0.9645 (0.9503–0.9770) | 0.9830 (0.9756–0.9896) | 0.9755 (0.9674–0.9823) | 0.9964 (0.9946–0.9978) |
| EfficientNet-B0 | 4094 / 34 / 3193 / 182 | 0.9712 (0.9624–0.9794) | 0.9574 (0.9437–0.9702) | 0.9895 (0.9837–0.9945) | 0.9743 (0.9664–0.9813) | 0.9966 (0.9949–0.9978) |
| ResNet-50 | 4128 / 38 / 3189 / 148 | 0.9752 (0.9664–0.9831) | 0.9654 (0.9516–0.9785) | 0.9882 (0.9828–0.9932) | 0.9780 (0.9705–0.9847) | 0.9970 (0.9954–0.9982) |

Error rates rise from 0.2% to about 2.5–2.9%, and AUROC is 0.996–0.997, matching the patient-disjoint validation AUROC logged during training. This is the more conservative internal estimate. It is **not** external validation: the images come from the same dataset, device family, labelling process and patients as the test set.

## 6. Statistical methods

- **Confusion-matrix metrics:** computed twice (manual counts and `sklearn.metrics.confusion_matrix`) and asserted equal. Undefined ratios are recorded as NaN; none occurred.
- **Proportions:** Wilson score 95% CIs. **F1:** class-stratified percentile bootstrap, B = 2,000, seed 42.
- **AUROC:** sklearn, cross-checked against an independent fast-DeLong implementation (Sun & Xu 2014) and a brute-force O(mn) implementation. 95% CI by DeLong (1988).
- **Supplementary pool:** patient-cluster percentile bootstrap (B = 2,000, seed 42), because patients contribute many images.
- **Paired comparisons:** exact McNemar on discordant classifications at 0.5, and paired DeLong for AUROC. Holm correction across the three pairwise comparisons.
- **Calibration:** Brier score, log loss, and ECE with 10 equal-width bins (the binning used by the original `evaluate.py`).
- **Not done, deliberately:** significance tests between this project and published studies (unrelated cohorts); CIs for published metrics that did not report them; any analysis on rounded published values.

## 7. Leakage audit findings

Details: `research_audit/leakage_audit.md`.

### 7.1 No leakage or shortcut explains the test result [R/N]

- Patient overlap test↔training pool is 0, and exact duplicates are 0.
- 5 perceptual-hash candidate near-duplicates (all Normal–Normal) were inspected visually; all are visibly different scans. (An earlier count of 10 came from lossy float-stored hashes; corrected.)
- The class mapping is correct, and the test set was never used for selection.
- A geometry-only classifier scores AUROC 0.31 on the test set (the cue reverses between train and test), so a dimension shortcut cannot produce the perfect test score.
- A white rotation-fill corner artifact is balanced by class in training (AUROC 0.49 as a feature).

### 7.2 Validation contamination of the tuned threshold and temperature [N]

- Training on Kaggle used scikit-learn 1.4.2: validation set 5,560 images.
- The local re-evaluation used scikit-learn 1.9.0 with the same code and seed, and produced a **different patient partition** (6,047 images).
- Temperature and "tuned" thresholds were fitted on this local partition. After rebuilding the Kaggle split (below), **5,096 of its 6,047 images (84.3%; 553 of 676 patients) turn out to be training images**.
- This explains the previously unresolved discrepancy between `val_n` 5,560 and 6,047. It invalidates the original tuned operating points (B3 0.971, B0 0.608, ResNet-50 0.628, B3+LS 0.890) but not the 0.5 or AUROC results.

**Fix [N]:** the training-time split was rebuilt by running the unmodified `utils.build_splits()` under scikit-learn 1.4.2 (script 11). The counts match the Kaggle log exactly (train 24,600, val 5,560: 1,254 DME / 4,306 Normal), and validation AUROC recomputed on it matches each checkpoint's stored `val_auc` to within 7×10⁻⁵. Temperature and threshold were then refit on those 5,560 images with `evaluate.py`'s own functions (script 12); no checkpoint was modified.

| Model | T (old → refit) | Threshold (old → refit) | Val sens / spec | Test TP / FP / TN / FN |
|---|---|---|---|---|
| EfficientNet-B3 | 0.917 → 1.549 | 0.971 → 0.181 | 0.9705 / 0.9666 | 242 / 0 / 242 / 0 |
| EfficientNet-B0 | 1.019 → 1.557 | 0.608 → 0.121 | 0.9705 / 0.9626 | 242 / 0 / 242 / 0 |
| ResNet-50 | 0.881 → 1.121 | 0.628 → 0.219 | 0.9705 / 0.9821 | 241 / 0 / 242 / 1 |
| B3 + LS 0.05 | 0.479 → 0.625 | 0.890 → 0.096 | 0.9705 / 0.9561 | 242 / 0 / 242 / 0 |

The refit operating points are valid, but the perfect test counts mostly reflect the easier test set (§7.3). At 97% validation sensitivity the validation specificity (0.956–0.982) is the more realistic estimate. Test sensitivity 242/242 has a Wilson 95% CI of 0.9844–1.0000.

### 7.3 The official test set is unlike the training distribution [N]

- The test set has **no 512×512 images**, versus 74.6% of DME training images.
- Its DME/Normal dimension mix is reversed relative to training.
- The white-corner artifact is enriched in test Normal images (82.6% vs 58–60% elsewhere).
- Patient-disjoint validation AUROC during training was 0.995–0.996, below the test AUROC of 1.000.

Together these suggest the 484 test images are a separately selected, comparatively easy subset (see also §5 supplementary results).

### 7.4 External evaluation on public non-Kermany data [N]

Frozen checkpoints, unchanged preprocessing, and thresholds fixed in advance: raw 0.5, and the refit temperature/threshold from §7.2, which were fitted on Kermany data only. Nothing was tuned on external data (`scripts/13_external_validation.py`).

**OCTDL** (Kulyabin et al., *Sci Data* 2024; CC BY 4.0). Optovue Avanti RTVue XR, a different manufacturer from Kermany's Spectralis; Russian clinical data. DME 147 B-scans (107 patients) vs normal 332 B-scans (110 patients); OCTDL's other classes were excluded. No exact copies of Kermany images were found, and the six nearest dHash matches (distance ≤ 4) were inspected visually and are different scans. CIs are patient-cluster bootstrap (B = 2,000).

| Model | Threshold | TP / FP / TN / FN | Sensitivity (95% CI) | Specificity (95% CI) | AUROC (cluster-bootstrap 95% CI) |
|---|---|---|---|---|---|
| EfficientNet-B3 | 0.5 raw | 141 / 1 / 331 / 6 | 0.959 (0.916–0.992) | 0.997 (0.990–1.000) | 0.9993 (0.9982–0.9999) |
| EfficientNet-B3 | 0.181 refit | 147 / 16 / 316 / 0 | 1.000 (degenerate) | 0.952 (0.928–0.972) | same |
| EfficientNet-B0 | 0.5 raw | 137 / 1 / 331 / 10 | 0.932 (0.882–0.972) | 0.997 (0.990–1.000) | 0.9934 (0.9793–1.0000) |
| EfficientNet-B0 | 0.121 refit | 142 / 2 / 330 / 5 | 0.966 (0.927–0.994) | 0.994 (0.984–1.000) | same |
| ResNet-50 | 0.5 raw | 137 / 0 / 332 / 10 | 0.932 (0.887–0.969) | 1.000 (degenerate) | 0.9981 (0.9961–0.9994) |
| ResNet-50 | 0.219 refit | 140 / 2 / 330 / 7 | 0.952 (0.915–0.981) | 0.994 (0.984–1.000) | same |
| B3 + LS 0.05 | 0.5 raw | 140 / 1 / 331 / 7 | 0.952 (0.909–0.987) | 0.997 (0.990–1.000) | 0.9986 (0.9964–0.9999) |
| B3 + LS 0.05 | 0.096 refit | 146 / 16 / 316 / 1 | 0.993 (0.978–1.000) | 0.952 (0.929–0.972) | same |

- Ranking is not stable enough to separate the architectures; their intervals overlap.
- At 0.5, sensitivity drops from 0.996 (Kermany test) to 0.93–0.96, with almost no false positives. The refit thresholds recover sensitivity at the cost of specificity, most clearly for B3 and B3+LS (16 false positives each).
- B3's six misses at 0.5 show hyperreflective foci, hard exudates or mild thickening without obvious cysts (visual inspection, not graded). This is plausibly a label-definition gap with Kermany's DME class rather than a device effect, but it was not tested. Its one false positive is a myopic normal image that carries an on-screen arrow overlay.
- Patient level (patient score = max P over that patient's images, 0.5): sensitivity 0.94–0.97, specificity 0.99–1.00, AUROC 0.996–0.999 (`external_validation_patient_level.csv`).

**Duke Srinivasan 2014** (Srinivasan et al., *BOE* 2014; research/education use, not redistributed). Spectralis, the same device family as Kermany but different centres (Duke, Harvard, Michigan). 15 DME and 15 normal volumes, one per subject, 2,508 B-scans; the AMD volumes were excluded. No exact copies of Kermany images; the two nearest dHash matches (distance 4) are different scans. Labels are per volume, so the pre-specified primary analysis is volume level, with volume score = max P(DME) over its B-scans (the presence rule used by Tang 2021). Mean P(DME) was the pre-specified secondary rule. CIs are Wilson (n = 15 per class) and DeLong.

| Model | AUROC, max rule (and mean rule) | Max rule @0.5: sens / spec | Max rule @refit: spec | Mean rule @0.5: sens / spec | Mean rule @refit: sens / spec |
|---|---|---|---|---|---|
| EfficientNet-B3 | 1.000 (1.000) | 15/15 / 7/15 (0.47; 0.25–0.70) | 0/15 | 12/15 / 15/15 | 15/15 / 15/15 |
| EfficientNet-B0 | 1.000 (1.000) | 15/15 / 11/15 (0.73; 0.48–0.89) | 0/15 | 11/15 / 15/15 | 15/15 / 15/15 |
| ResNet-50 | 1.000 (1.000) | 15/15 / 6/15 (0.40; 0.20–0.64) | 1/15 | 13/15 / 15/15 | 15/15 / 15/15 |
| B3 + LS 0.05 | 1.000 (1.000) | 15/15 / 7/15 (0.47; 0.25–0.70) | 0/15 | 13/15 / 15/15 | 15/15 / 13/15 |

- **Ranking is perfect, but the pre-specified operating point fails on normal eyes.** Every model ranks all 15 DME volumes above all 15 normal volumes (AUROC 1.000; n = 30, so the interval is wide in reality and DeLong is degenerate). With the max rule, B-scan thresholds tuned for single images flag 4–9 of 15 normal volumes at 0.5 and nearly all of them at the lower refit thresholds.
- **Why.** At B-scan level only 0.7–1.4% of normal B-scans score ≥ 0.5, compared with 70–78% of B-scans in DME volumes. Taking the maximum over ~97 slices still catches those few. The highest-scoring normal B-scans are all off-centre slices without a foveal pit (visual inspection). Kermany's training images are single foveal B-scans (STAR Methods), so off-centre slices are out of distribution for these models.
- **Mean rule.** It separates the two classes at the refit thresholds for three models (15/15 and 15/15), but this was a secondary rule, and its good result should not be promoted after the fact.
- **POST HOC (decided after seeing the above):** scoring only each volume's central B-scan, a proxy for the fovea, gives sensitivity 12–13/15, specificity 15/15 and AUROC 0.991–1.000 at 0.5 (`external_validation_duke2014_posthoc.csv`).
- **Implication.** These are B-scan classifiers for foveal images. Applying them to whole volumes needs a slice-selection or aggregation rule validated on separate data; none has been validated here.

**Duke DME 2015** (Chiu et al., *BOE* 2015; Spectralis; 10 DME subjects, 110 grader-annotated B-scans). There are no normal eyes, so only sensitivity is estimable. On the 78 B-scans with grader-1 fluid, sensitivity at 0.5 is 0.949–0.962 (e.g. B3 75/78, Wilson 0.893–0.987; CIs ignore within-subject clustering). All 10 subjects are flagged by the max rule (Wilson 0.72–1.00). 44–69% of fluid-free B-scans from these DME eyes are also called DME; these are not negatives and only describe the score distribution.

**Interpretation.** Image-level discrimination transfers to a second manufacturer and site (OCTDL AUROC ≥ 0.99) and to volume-level ranking on a second Spectralis cohort (Duke 2014 AUROC 1.000). Operating points do not transfer automatically, though: whole-volume use produces false positives from off-centre slices. All three external sets are small, curated, public B-scan collections with their own label definitions. This is evidence of transfer beyond Kermany, not a clinical or prospective validation, and it does not make the results comparable to Tang 2021 or Zhu 2025, which are volume-level studies with clinically graded diabetic cohorts.

## 8. Literature comparison

Sources were read in full: Tang et al. 2021 (PMC full text plus figshare supplement), Zhu et al. 2025 (PMC XML full text, article PDF and supplement), and the article PDFs of Li F et al. 2019, Rodríguez-Miguel et al. 2024 and Singh & Gorantla 2020. Extraction: `research_audit/literature_extraction.csv`. Comparability: `research_audit/comparability_matrix.csv`. Table: `research_outputs/tables/table3_literature_comparison.md`.

**Tang et al. 2021, Diabetes Care 44(9):2078–2088 (primary reference).**

- Design: a 3D ResNet-34 for CIRRUS volumes and a 2D ResNet-18 for SPECTRALIS/Triton B-scans, both multitask (no DME / non-CI-DME / CI-DME, plus non-DME abnormality).
- Main-text Table 2 is entirely **volume-level**. For SPECTRALIS and Triton, B-scan predictions are aggregated by a presence rule.
- **B-scan-level 2D results appear only in Supplementary Table S1**, presence vs absence of DME:
  - SPECTRALIS primary validation: AUROC 0.940 (0.934–0.945), sensitivity 88.7%, specificity 99.2%.
  - Triton primary validation: AUROC 0.937 (0.931–0.942).
  - External sets 4–7: AUROC 0.899–0.959.
- These are the only rows that qualify as **category B**: same input unit and a similar task, but a different population, device mix and label definition. Their "DME" includes thickening and hard exudates in diabetic eyes, and their negatives can include other retinal disease.
- Two published CIs are internally inconsistent (External 1 AUROC 0.906 with CI 0.947–0.968; S1 External 7 accuracy CI 95.6–96.1 around 96.1). They are copied as printed and flagged.

**Zhu et al. 2025, BMJ Open 15(5):e099167 (secondary, category C).**

- Design: a 3D CNN (Models Genesis initialisation) on Cirrus and Topcon volumes. Labels follow ICO definitions and exclude hard exudates.
- DME vs non-DME, AUROC 0.990 (0.983–0.996) on the 15% JSIEC hold-out test set and 0.916 (0.902–0.930) on centre-split external validation (Table 3). CI-DME vs non-CI-DME: 0.859 and 0.881.
- The evaluation unit (volume), input (3D) and labels differ from this project, so it supports discussion only. Main-text values were checked against the publisher's article PDF (identical). Each external site used one device (Chaozhou: Cirrus, 1,269 volumes; STU-2nd: Topcon, 240 volumes; Table 2), so the per-site rows are also per-device rows for external data. The online supplement (`bmjopen-15-5-s001.docx`) was also checked **[E]**:
  - **Confusion matrices.** Supplementary Table 1 gives the confusion matrices; every published accuracy, F1, sensitivity and specificity recomputes from them. The hold-out test set is 956 volumes (234 DME).
  - **Per device (Supplementary Table 2).** On the hold-out test set, Topcon reaches AUROC 0.989 (0.977–1.000; n = 760) and Cirrus 0.967 (0.944–0.990; n = 196). Externally, Cirrus/Chaozhou gives 0.933 and Topcon/STU-2nd gives 0.904, with sensitivity 0.458 and 65 of 120 DME volumes missed. The main text's statement that Cirrus performed better therefore holds for the external data but not for the hold-out test set, where Topcon is higher; the external comparison also confounds device with site.
  - **Readers (Supplementary Table 3).** On a 200-volume comparison set, the system has sensitivity 0.910 and specificity 0.937 (AUROC 0.971), while three retina experts have sensitivity 0.944–1.000 and specificity 0.658–0.928.
  - **Two printing errors, copied as printed and flagged.** The experts' average F1 is printed as 1.957 and 1.871, which is impossible; the counts give 0.887 for DME. Supplementary Table 4 is captioned as covering testing and external validation, but its totals (71 FP / 102 FN) equal Chaozhou alone.

**Three further studies (article PDFs supplied by the user, read in full 2026-10-28) [E].** They were extracted the same way as Zhu 2025: design, device, unit, cohort role, n and metrics with table locations, NR where not reported, and every printed metric checked against counts where counts could be derived. Rodríguez-Miguel's online Supplementary Material 1 was downloaded from the article page and read. Bibliographic details of all three were checked against Crossref: authors, title, journal, volume, issue, first page or article number, and date all match. All three studies are in `literature_extraction.csv` (15 rows) and Table 3.

**Li F et al. 2019, Biomed Opt Express 10(12):6204 (category B for one row; the closest published analogue).**

- Design: an ensemble of four dilated ResNet50s on single Heidelberg Spectralis B-scans from two Shanghai hospitals (15,573 graded B-scans, 2,796 patients). There is a patient-disjoint test set of 267 patients / 1,536 B-scans (332 DME, 635 NORMAL), and 10-fold patient-level cross-validation for training and validation.
- **DME vs NORMAL binary classifier (Table 6):** accuracy 0.988, sensitivity 0.986, specificity 0.991, AUROC 0.996 (0.995–0.997), at B-scan level. This is the same input unit, device vendor and binary task as this project, but a different population and label process (ophthalmologist-graded, adjudicated). Its test prevalence is 34% rather than 50%.
- **Internal inconsistencies, copied as printed and flagged:**
  - 0.986 is not attainable as k/332 (327/332 = 0.985, 328/332 = 0.988).
  - Every CI is about 5× narrower than a binomial interval for the stated test size (Wilson 0.965–0.994 for 327/332), so the CIs cannot be per-image test-set intervals. The method is not stated.
  - The text defines sensitivity and specificity as correct counts "divided by the total number of test images".
  - The 4-class "best model" (Table 4) was chosen as the best of 10 folds on the test set itself.
- **Kermany/UCSD (Table 8):** DME one-vs-rest within a 4-class model gives accuracy 97.1%, sensitivity 90.1%, specificity 98.2% and AUROC 0.978, as mean ± SD over six runs that train on 1/6 and test on 5/6. This is the same source data as this project, but a different task (negatives include CNV and drusen) and split, and patient grouping is not stated. Category D.
- **Duke Srinivasan 2014 (Table 7):** 15/15 DME and 15/15 NORMAL volumes correct by leave-three-out cross-validation, after fine-tuning on Duke, with volume = mode of B-scan predictions. These are the same 30 volumes as this project's Duke 2014 external set (§7.4). That set was used here **without any Duke training**: AUROC 1.000, but the pre-specified max rule failed on normal volumes. The protocols differ, so the two results are not comparable as performance figures (category C).

**Rodríguez-Miguel et al. 2024, Sci Rep 14:17633 (category C).**

- Design: a hybrid CNN-RNN reads 22 automatically selected B-scans from each 128-slice Topcon 3D OCT-Maestro cube, and its CNN backbone is pre-trained on Kermany OCT2017. Models were trained on a separate 650-cube ground-truth set.
- Evaluation was on an unselected real-world DR screening cohort: 5,314 cubes from 4,408 subjects, 126 DME (2.37%), with the screening retina specialist's diagnosis as reference.
- Model 2 AUROC 0.94 (0.91–0.97). Ensemble at the Youden threshold: sensitivity 88.9% (82.1–93.8), specificity 93.3% (92.6–94.0). At the high-specificity threshold: 81.0% / 97.0%, with PPV 39.5%. The high-specificity counts (FN 24, FP 156) are printed and recompute.
- **Caveats:**
  - All three thresholds were tuned on the evaluation cohort itself, so the operating points are optimistic.
  - Repeat screens of the same subjects are included, and the CIs treat cubes as independent.
- **Inconsistencies, flagged:**
  - Every threshold-2 row prints the same sensitivity CI (82.1–93.8). For 116/126 the exact interval is 85.9–96.1, so this looks like a copy error.
  - The false-negative review gives "4 DMEs (12.1%)", which corresponds to 4/33, not the 24 false negatives.
- **Supplementary Material 1:**
  - On the ground-truth test split (117 cubes: 40 DME / 77 non-DME), main-text models 1–3 reach accuracy 0.915 / 0.890 / 0.880 and AUROC 0.960 / 0.944 / 0.962. No sensitivity, specificity or CIs are given.
  - That split was also used to choose the seven retained models, and the final three were picked on the screening cohort. Both stages of selection are optimistic.
  - The listed slice positions give 13 foveal + 9 parafoveal slices, not the stated 12 + 10 (slice 105 is skipped), and the figure caption says 126 slices.
  - It describes Kermany as multi-manufacturer, whereas Kermany's STAR Methods report Spectralis (§2).
  - The Kermany-pretrained 4-class CNN scored accuracy 0.909 / AUROC 0.987 on Kermany's 1,000-image test set.
- This is the only study here evaluated at realistic screening prevalence. It shows how far PPV falls below what a balanced test set suggests, which is directly relevant to this project's 50%-prevalence test set (§15).

**Singh & Gorantla 2020, PLoS ONE 15(2):e0220677 (DMENet; category D, not plotted).**

- Design: **colour fundus photographs** (IDRiD, MESSIDOR), where DME is graded from hard exudates near the macula. It uses a hierarchical CNN ensemble with 5-fold image-level cross-validation on augmented data. This is a different modality and label definition from OCT DME, so it is not comparable.
- Stage-1 DME detection (Table 3, 360 test images): accuracy 96.67%, sensitivity 97.50%, specificity 95.63%, AUROC 0.9654. The abstract's 96.12% / 96.32% / 95.84% is the 3-grade result averaged over both datasets.
- **Inconsistency, flagged:** the paper's own confusion matrix (Fig 6A) shows sensitivity 0.968 and specificity 0.965, not Table 3's values. Counts that fit both (TN 195, FP 7, FN 5, TP 153; inferred, not printed) make Table 3's "sensitivity", "specificity" and "precision" the NPV, PPV and specificity. Its F1 (0.9701) is the No-DME class F1.

**Figure 3** therefore shows published AUROCs as a grouped forest-style display:

- this project (internal test sets, and the external OCTDL and Duke 2014 sets);
- Tang 2021 and Li 2019 B-scan rows (category B);
- volume- or cube-level results (Tang, Zhu, Rodríguez-Miguel), separated and marked as context only (category C);
- Singh & Gorantla 2020 (DMENet), fundus photographs, as a different-modality reference only (category D).

It includes **no cross-study statistical test and no relative-improvement claim**. The external-validation results of these papers come from data this project has never been evaluated on (except Li 2019's Duke 2014 rows, which use a different protocol).

### 8.1 This project and each paper, value by value [N]

Every figure and table of this report now places this project and the five papers in the same graph or table: Figures 1–3, Table 3 and Table 6. Figures 4, 5 and A1 carry a note instead, because none of the five papers reports learning curves, calibration or a fluid-segmentation Dice; their full texts were searched for these. All values come from `research_outputs/data/per_paper_comparison.csv`, built by `scripts/14_per_paper_comparison.py` from `literature_extraction.csv` and the project's result CSVs. Nothing is typed in by hand.

**How the pairs are formed.** The rules were fixed before plotting:

- **Headline row per paper:** the internal B-scan/image-level result if the paper reports one (Tang Suppl. S1 Spectralis, Li Table 6, Singh Table 3). Otherwise it is the paper's main-text primary result (Zhu Table 3 Testing, Rodríguez-Miguel Table 2 Model 2).
- **All other rows** of the paper are kept. They are shown as dots and as a min–max range, so no row is dropped.
- **Unit-matched project value:** an image-level headline is set beside the project's official test (with the unseen pool and OCTDL also shown). A volume-level headline is set beside the project's Duke 2014 volume-level result (EfficientNet-B3, threshold 0.5, max rule, n = 30).

Every pair differs in data, population, device, label definition and often evaluation unit. The numbers are placed side by side for reading, not ranked, and no cross-study test was computed.

| Source | Unit | Accuracy | Sensitivity | Specificity | F1 | AUROC |
|---|---|---|---|---|---|---|
| **This project**, EfficientNet-B3, official test (n = 484) | B-scan | 0.998 | 0.996 | 1.000 | 0.998 | 1.000 |
| This project, unseen pool (n = 7,503) | B-scan | see Table 2 | | | | 0.996 (0.995–0.998) |
| This project, OCTDL external (n = 479) | B-scan | | 0.959 | 0.997 | | 0.999 |
| This project, Duke 2014 external (n = 30 volumes, max rule) | volume | | 1.000 | 0.467 | | 1.000 |
| **Tang 2021** headline (Suppl. S1, Spectralis, internal) | B-scan | 0.972 | 0.887 | 0.992 | NR | 0.940 (0.934–0.945) |
| **Zhu 2025** headline (Table 3, hold-out test) | volume | 0.961 | 0.936 | 0.970 | 0.922 | 0.990 (0.983–0.996) |
| **Li F 2019** headline (Table 6, DME vs NORMAL) | B-scan | 0.988 | 0.986 | 0.991 | NR | 0.996 (0.995–0.997) |
| **Rodríguez-Miguel 2024** headline (Table 2, Model 2, screening cohort) | cube | NR | 0.905 | 0.908 | NR | 0.940 (0.910–0.970) |
| **Singh & Gorantla 2020** headline (Table 3, fundus) | fundus image | 0.967 | 0.975 | 0.956 | 0.970* | 0.965 |

\* Singh & Gorantla's F1 is the No-DME class F1 (§8).

**Tang 2021 (16 rows; Figures 1–3).**

- Tang's B-scan rows: internal AUROC 0.940 and 0.937, external 0.899–0.959. The project's B-scan values are 1.000 (official test), 0.996 (unseen pool) and 0.999 (OCTDL).
- Tang's volume rows: AUROC 0.906–0.965. The project's single volume-level set (Duke 2014, 30 volumes) gives AUROC 1.000, but specificity only 0.467 at 0.5 with the max rule.
- **Why the numbers differ:**
  - Tang's DME label includes retinal thickening and hard exudates in clinically graded diabetic eyes, on three devices and six external centres.
  - The project's sets are curated public B-scans with folder labels.
  - The higher project values therefore do not show a better model. They are measured on easier, different data.

**Zhu 2025 (7 rows, volume level, category C).**

- Zhu: hold-out AUROC 0.990, sensitivity 0.936, specificity 0.970. External AUROC is 0.904–0.933, and sensitivity drops to 0.458 at one centre.
- Project, Duke 2014 volume level: AUROC 1.000 and sensitivity 1.000, but specificity 0.467 with the pre-specified max rule. The secondary mean rule gives 0.800 / 1.000.
- The project has no validated volume aggregation rule, and its volume evidence is 30 volumes. Zhu's is 956 test and 1,509 external volumes.

**Li F 2019 (3 DME rows; the closest analogue).**

- Li's Table 6 values (0.988 / 0.986 / 0.991, AUROC 0.996) are close to the project's unseen-pool AUROC (0.996) and below its official-test values (1.000). Li's population is different, and its CIs cannot be per-image test intervals (§8).
- On the same 30 Duke 2014 volumes, Li reports 30/30 correct after fine-tuning on Duke with leave-three-out cross-validation. The project, with no Duke training, gives AUROC 1.000 but specificity 0.467 at the max rule.
- The protocols differ, so these are not the same measurement.

**Rodríguez-Miguel 2024 (7 rows, cube level).**

- At real screening prevalence (2.37% DME, 5,314 cubes), sensitivity is 0.810–0.921 and specificity 0.831–0.970, depending on the threshold, with AUROC 0.940. On the ground-truth test split, AUROC is 0.944–0.962.
- This is the only one of the six sources evaluated at realistic prevalence. All project test sets are balanced or near-balanced, so the project's predictive values would fall at screening prevalence (§15).

**Singh & Gorantla 2020 (4 rows, fundus; category D).**

- Accuracy 0.955–0.968, sensitivity 0.947–0.979, specificity 0.945–0.972, AUROC 0.965.
- These are fundus-photograph results with a hard-exudate-based DME label, and its Table 3 does not match its own confusion matrix (§8). It is shown in Figures 1–3 for completeness, marked as a different modality, and is not comparable with any OCT result.

**What the papers report that the project does not, and the reverse.**

- Tang, Zhu, Rodríguez-Miguel and Li report external, multi-device or screening-cohort results at the volume level, which the project has only on 30 Duke volumes.
- The project reports learning curves (Figure 4), calibration (Figure 5) and fluid segmentation (§10). None of the five papers reports any of these (Table 6, NR / NA).

## 9. 2D versus 3D OCT

- A single B-scan samples one cross-section of a ~6 mm macular cube (Tang 2021 used 128 B-scans per CIRRUS volume; Zhu 2025 resampled volumes to 128 slices).
- Volume-level labels and models can use lesion continuity across slices, and can determine centre involvement relative to the fovea. A single B-scan cannot reliably do this, and the project correctly refuses to infer centre involvement.
- Volume-level and B-scan-level metrics are not interchangeable. In Tang 2021, the same 2D network scores AUROC 0.958 at volume level vs 0.940 at B-scan level on SPECTRALIS primary data: aggregation changes the numbers.
- B-scan test sets also contain correlated images from the same eye, which inflates the apparent sample size. Kermany provides no volume IDs, so this project cannot correct for it beyond patient grouping.

## 10. Segmentation

### 10.1 Original checkpoint (audited) [R]

- The reported test Dice / IoU of 0.0090 / 0.0045 are reproduced exactly (`segmentation_summary.json`).
- **The model is collapsed:** it labels 99.99% of every 320×320 frame as fluid, including padding, with mean probability 0.98. The checkpoint is from epoch 1 of 6.
- The phase-4 sign-off's explanation ("predicting almost no foreground") is incorrect.
- The ground-truth definition counts ungraded pixels as background. Only a central column band (500–542 of 768 columns, where the manual layers are traced) is graded. The ungraded margins are stored as 0, not NaN, so `segment.py` cannot tell them apart from "no fluid".
- Restricted to that graded band, test Dice is 0.020 (val 0.030), still a collapsed model. An earlier version of this audit masked NaN pixels instead, which covers the whole frame, and reported 0.014 as graded-only (corrected 2026-10-28).
- Human inter-grader Dice on the same test scans, within the graded band, is 0.758 (0.620 on fluid-positive scans). No fluid is marked outside the band, so this value does not depend on the mask correction.
- The original checkpoint's performance is therefore **not reported as a result**. Audit Figure A1 documents the failure instead, and no Table 4 is produced for it.

### 10.2 Retrained model (new analysis, separate session) [N]

After this audit, the U-Net was retrained in a separate session (branch `claude/seg-retrain-227f8e`). Full details, including an erratum, are in `research_outputs/segmentation_retrain_report.md`. The original checkpoint was not modified.

**There were two retrains. v2 is the current model; v1 is superseded.**

- **v1 (superseded; code `da997e8`, results in `research_outputs/segmentation_retrain/`).** The loss masked out the letterbox padding and the ungraded side margins. On the graded band its test pooled Dice was 0.706 (fluid-positive mean 0.633). On a whole frame, however, it predicted fluid on about 40% of the ungraded margins and 94% of the padding (val subjects), so it is not usable in `app.py`. Its graded-band numbers are correct as stated but describe a model that is not used.
- **v2 (current; code `65d4b78`, merged as `7f8fe26`; results in `research_outputs/segmentation_retrain_v2/`).** The loss covers the whole frame, so padding and ungraded margins count as background. This accepts some label noise, because any unmarked fluid in the margins is learned as background. The checkpoint is git-ignored at `artifacts/segmentation_retrain_v2/checkpoint_best.pt`. `app.py` uses it at threshold 0.95.

- **What is the same in both:**
  - Metrics use graded pixels only (the layer band of §10.1), at native 496×768 resolution.
  - The loss adds BCE with `pos_weight` 10 plus a soft Dice term, fixed in advance.
  - The epoch and the probability threshold are selected jointly on validation only. v2: epoch 16 of 31, threshold 0.95.
  - All-foreground and all-background baselines are reported.
  - Same patient split as before (seed 42; test subjects 06 and 07).
- **Test results, v2 (22 scans, 14 with fluid):**

| | Pooled Dice | Mean Dice, fluid-positive scans |
|---|---|---|
| Retrained model v2 (t = 0.95) | **0.638** | **0.545** |
| Grader 2 vs grader 1 | 0.694 | 0.615 |
| All-foreground baseline | 0.020 | 0.031 |
| All-background baseline | 0.000 | 0.000 |

- Validation (v2): pooled Dice 0.789, fluid-positive mean 0.529.
- **Whole-frame check (v2):** 0.0% predicted fluid in the ungraded margins and in the padding on both val and test, and no fluid-free scan with any predicted fluid (0/5 val, 0/8 test).
- **Threshold at the grid edge:** 0.95 is the top of the pre-specified grid, and val pooled Dice was still rising there (0.783 at 0.90). The grid was not extended.
- **Test-set reuse, disclosed:** the two test subjects have now scored two models (v1 and v2). The retrain was triggered by v1's whole-frame behaviour, which was equally visible on val, not by any test Dice, and nothing in v2 was tuned on test. Even so, the test split is no longer strictly untouched.
- **Audit check [R]:** every value in both summary JSONs that can be derived from the committed per-image rows (counts, mean, pooled and fluid-positive Dice and IoU, baselines, inter-grader values, per-subject pooled Dice, the threshold choice) was recomputed and matches (`research_audit/value_ledger_project.csv`). Inference was not rerun by the audit; boundary distances need the masks and are taken from the source.
- **Interpretation:** v2 is far above the trivial baselines, so the collapse of §10.1 is fixed, and it no longer predicts fluid outside the graded band. On these two test subjects its agreement with grader 1 is **below** the human inter-grader agreement (pooled Dice 0.638 vs 0.694).
- **Limits:**
  - The test set is two subjects and 22 scans. Per-subject pooled Dice is 0.612 (subject 06) and 0.799 (subject 07), and subject 07 has only 4 fluid-positive scans.
  - One test scan with a 354-pixel grader-1 region scores Dice 0.
  - Mean boundary distance on test is 7.8 pixels, against 5.7 between graders.
  - All data are from one public Spectralis dataset (Duke 2015), with no external segmentation test.
  - Inter-grader values differ slightly from §10.1 (e.g. mean Dice 0.755 vs 0.758) because the retrain scores at native resolution, not the 320×320 letterbox.


## 11. Calibration and explainability

Source: `calibration_summary.csv`, Figure 5.

- **Calibration [N]:** temperatures are now refit on the training-time validation split (§7.2). On that validation set, scaling reduces ECE for every model (e.g. B3 0.0133 → 0.0054).
- **On the test set, refit scaling does not help B3 or B0.** ECE rises (B3 0.0028 → 0.0052; B0 0.0038 → 0.0073) and so does log loss; Brier barely moves. The models are over-confident on validation patients (T > 1), but the test set is easier, so softening them costs calibration there. ResNet-50 is essentially unchanged (ECE 0.0023 → 0.0021).
- **The ablation is the one consistent effect.** Label smoothing produces under-confident raw probabilities (test ECE 0.031), which temperature scaling repairs (0.0070).
- The originally reported "ECE 0.0028 → 0.0027" used the superseded temperature (§7.2); those values are retained in `calibration_summary.csv` for reconciliation only.
- **These calibration numbers carry little information.** 482–483 of 484 predictions fall in the two outermost bins.
- Calibration cannot change AUROC; this was confirmed.
- **Grad-CAM [E]:** examples are selected deterministically (most confident correct DME and Normal, worst-margin error). The randomisation test shows dependence on the trained classifier weights, but not clinical validity of the highlighted regions. No heatmap is offered as evidence of clinical reasoning.

## 12. Figure captions

**Figure 1.** Performance of this project's models and the five published DME classifiers, metric by metric (one graph).
Accuracy, sensitivity, specificity, F1 and AUROC.
- **Project bars:** EfficientNet-B3, EfficientNet-B0 and ResNet-50 at threshold 0.5 on the official Kermany OCT2017 test split (242 DME / 242 Normal B-scans; DME = positive class). Error bars are Wilson 95% CIs (F1: class-stratified bootstrap).
- **Paper bars:** each paper's headline row (§8.1). The black bracket gives the min–max over all of that paper's DME-presence rows; NR marks a metric the headline row does not report.
- **Dots:** every other row of the same source. For the project, these are the same model on the unseen pool, OCTDL and Duke 2014 (volume level).
- Zhu and Rodríguez-Miguel headlines are volume level, and Singh & Gorantla use fundus photographs.
The three project architectures are statistically indistinguishable (McNemar p = 1.00). Across sources this is not a head-to-head comparison, and no cross-study test was performed.

**Figure 2.** Receiver operating characteristic: this project's curves and the five papers' published operating points (one graph).
(a) Project ROC curves for all three models: official test set (thin; all follow the left and top axes, AUROC 1.000) and supplementary unseen pool (thick). EfficientNet-B3's operating points at 0.5 are marked on the official test, unseen pool, OCTDL and Duke 2014 (volume) sets. Every published DME-presence row with both sensitivity and specificity is plotted as a point: one marker shape and colour per paper, filled = image level, hollow = volume level, large black-edged = headline row.
(b) The same graph, zoomed.
Project curves are computed from continuous raw softmax P(DME); temperature scaling would not change them. The papers publish operating points, not ROC coordinates, and their printed ROC images were not digitised. The dashed diagonal is a random classifier.

**Figure 3.** Reported performance of published DME classification systems and the present 2D OCT model.
AUROC with 95% CI as published (Tang 2021 Supplementary Table S1 and main Table 2; Li 2019 Table 6; Zhu 2025 Table 3; Rodríguez-Miguel 2024 Results; Singh & Gorantla 2020 Table 3) or computed (this project: official test, unseen pool, OCTDL and Duke 2014). Li 2019's printed CIs are narrower than its test size supports (§8). DMENet uses fundus photographs and is shown as a different-modality reference only (category D). Circles = B-scan-level evaluation; squares = volume-level evaluation (context only).
Every row uses a different dataset, population, device, label definition and protocol. The figure shows research context, not a ranking, and no cross-study test was performed.

**Figure 4.** Training and validation performance across training epochs.
(a) EfficientNet-B3 training and validation loss, parsed from the Kaggle console log, with the phase-A/B boundary and the selected epoch (20) marked.
(b) Validation AUROC (the checkpoint-selection metric) for all four models.
(c) ResNet-50 training and validation loss (selected epoch 30).
(d) EfficientNet-B0 and B3 + label-smoothing training and validation loss (selected epochs 19 and 13), parsed from their Kaggle console logs (`20260920T095427Z`, `20260922T160023Z`). Both runs early-stopped (24 and 18 epochs). The label-smoothed loss is on a different scale.
The selected epochs agree with the checkpoints: LS peaks at epoch 13 (0.9952), and B0's epoch 19 is one of four epochs tied at 0.9950 (to four decimals). Validation is the patient-disjoint training-time split; the test set was never used.
Comparison with the five papers: none reports learning curves (full texts searched), so there are no published values to place on this figure (Table 6).

**Figure 5.** Calibration on the official test set, raw vs temperature-scaled probabilities. Reliability points use 10 equal-width bins, with images per bin shown under each axis. Nearly all predictions lie in the outermost bins. Temperatures were refit on the training-time validation split (n = 5,560); the test set was not used for fitting. Comparison with the five papers: none reports a calibration analysis (reliability diagram, ECE or Brier), so there are no published values to place on this figure (Table 6).

**Audit Figure A1.** U-Net fluid segmentation collapsed to all-foreground (not a performance figure). Duke DME held-out subjects 06 and 07: input, grader-1 mask, prediction and probability map for the first fluid-positive and first fluid-free test scan, plus predicted vs true fluid fraction for all validation and test scans. Comparison with the five papers: all five are classification-only (Tang 2021 states it is segmentation-free), so no published Dice exists to compare with.

## 13. Table captions

- **Table 1.** Dataset and experimental setup (`tables/table1_dataset_setup.md`).
- **Table 2.** Verified within-project model performance, with 95% CIs, including the supplementary unseen pool, refit tuned operating points and the superseded originals (`tables/table2_within_project_performance.md`).
- **Table 3.** Published DME classifiers and the present model, with evaluation unit, cohort and comparability category; not a head-to-head comparison (`tables/table3_literature_comparison.md`).
- **Table 4.** Not produced for the original checkpoint, whose evaluation is invalid (`tables/table4_segmentation_NOT_PRODUCED.md`). The note points to the retrained model's results (§10.2), which are reported in `segmentation_retrain_report.md`.
- **Table 5.** Reproducibility and limitations (`tables/table5_reproducibility_limitations.md`).
- **Table 6.** This project and each of the five papers side by side: modality, unit, devices, number of rows, headline result with n, headline accuracy / sensitivity / specificity / F1 / AUROC, the min–max of each metric over all rows, the unit-matched project value, and whether learning curves, calibration and segmentation are reported (`tables/table6_per_paper_comparison.md`). Not a head-to-head comparison.

## 14. Scientific interpretation

- On the official Kermany test split, three ImageNet-pretrained CNNs separate DME from Normal B-scans perfectly by AUROC and misclassify one DME image each. The result is reproducible from the saved checkpoints and is not explained by patient leakage, duplicated images, label-encoding shortcuts or evaluation bugs.
- It is, however, a measurement on a small (n = 484), artificially balanced, apparently curated test subset from the same public dataset the models were trained on. Its image-format distribution differs from the training data.
- The supplementary evaluation on unseen same-dataset images (§5) gives a more conservative internal estimate.
- The architecture comparison is uninformative: the models are indistinguishable.
- Relative to the literature, the project's figures **cannot** be read as better than Tang 2021, Zhu 2025 or Rodríguez-Miguel 2024. Those studies used clinically graded diabetic cohorts with harder label definitions, volume-level or multi-device evaluation, real-world prevalence, and genuine external or screening test sets.
- Paper by paper (§8.1, Table 6), the project's B-scan values sit at or above the published B-scan values. At the volume level, its 30-volume Duke result has much lower specificity (0.467 at the max rule) than the volume-level papers. The B-scan gap reflects easier, curated data rather than a stronger model.
- The closest published analogue, Li F 2019 (single Spectralis B-scans, DME vs NORMAL, patient-disjoint test), reports AUROC 0.996 in a different population. Results of this magnitude on curated B-scan sets are therefore not unusual. Its printed CIs cannot be reconciled with its test size, and no cross-study test was performed.

## 15. Limitations

- External evidence is limited to three small public datasets (§7.4): OCTDL (Optovue, DME vs normal B-scans), Duke 2014 (Spectralis, 15 DME / 15 normal volumes) and Duke 2015 (Spectralis, DME only, so sensitivity only). None is a clinically graded, prospective cohort. The models are foveal B-scan classifiers, and no volume-level aggregation rule has been validated.
- The official test set is small, balanced (prevalence 50%), and distributionally unlike the training data. PPV does not transfer to screening prevalence.
- Patient is the finest grouping available; correlated B-scans from one eye or volume cannot be identified.
- The original tuned thresholds and temperatures were not valid (§7.2). They have been refit on the rebuilt training-time split; the refit depends on reproducing the scikit-learn 1.4.2 partition, which was confirmed by exact count matching rather than a saved manifest. (Future runs now save one, and scikit-learn is pinned to 1.4.2; see §18.)
- Per-image outputs were not saved originally; the reconstructed outputs come from a different software stack (agreement with every stored aggregate was confirmed).
- The audited B0 run folder's log is empty; its exclusion of test patients is verified through an identical-weights run whose log survives (`20260920T095427Z`). The B3 per-epoch log is a console reconstruction without accuracy.
- Labels come from the public dataset folders. Kermany "DME" vs "NORMAL" is a different label definition from clinically graded DME in diabetic cohorts, and negatives exclude other pathology.
- The original segmentation checkpoint is non-functional (§10.1). The retrained model in use (v2, §10.2) is far above trivial baselines but below human inter-grader agreement (test pooled Dice 0.638 vs 0.694), on only two test subjects (22 scans) that have now scored two models, from one public dataset; boundary accuracy is worse than between graders, and there is no external test.
- The single B-scan design cannot assess centre involvement.

## 16. Reproducibility

Run from the repository root, in the local environment. Data must be present at `data/raw/OCT2017/{train,test}/{DME,NORMAL}` and `data/raw/duke_dme/2015_BOE_Chiu/`, with checkpoints under `artifacts/kaggle/`.

```bash
python research_outputs/scripts/01_reinfer_classifiers.py
```
```bash
python research_outputs/scripts/02_leakage_scan.py
```
```bash
python research_outputs/scripts/03_reinfer_excluded_pool.py
```
```bash
python research_outputs/scripts/04_compute_metrics.py
```
```bash
python research_outputs/scripts/05_segmentation_audit.py
```
```bash
python research_outputs/scripts/06_border_artifact_scan.py
```
```bash
python research_outputs/scripts/07_build_literature_tables.py
```
```bash
python research_outputs/scripts/08_make_figures.py
```
```bash
python research_outputs/scripts/09_make_tables.py
```
```bash
python research_outputs/scripts/10_final_checks.py
```

Software versions, seeds and hardware are listed in `tables/table5_reproducibility_limitations.md`. Script 03 is the slow step (several CPU-hours on a 12-thread CPU).

**Threshold/temperature refit (scripts 11–12).** Script 11 needs scikit-learn 1.4.2, which has no build for the project's Python 3.14, so it runs in an isolated Python 3.12 environment (created with uv). Script 12 then runs in the project environment. Rerun 04 and 08–10 afterwards; 04 picks up `refit_calibration_threshold.csv` automatically.

```bash
uv venv --python 3.12 .venv-sk142
```
```bash
uv pip install --python .venv-sk142 "scikit-learn==1.4.2" "numpy==1.26.*" "pandas==2.2.*"
```
```bash
.venv-sk142/Scripts/python research_outputs/scripts/11_rebuild_training_split.py
```
```bash
python research_outputs/scripts/12_refit_calibration_threshold.py
```

**External evaluation (script 13).** Download OCTDL from Mendeley Data (doi:10.17632/sncdhf53xc.4; files `OCTDL.zip` and `OCTDL_labels.csv`) into `data/external/octdl/` and unzip it there. Download Duke 2014 (`2014_BOE_Srinivasan.zip`, 626 MB, from people.duke.edu/~sf59) into `data/external/duke2014/` and unzip it there. Duke 2015 is the local segmentation dataset. A full run takes about 20 minutes on CPU; `--posthoc-only` recomputes the post-hoc Duke 2014 analysis from the saved predictions.

```bash
python research_outputs/scripts/13_external_validation.py
```

## 17. Results by provenance

| Category | Items |
|---|---|
| Existing results [E] | `metrics.json` aggregates; Grad-CAM randomisation check; training logs; segmentation log values |
| Independently reconstructed [R] | all test-set confusion matrices and metrics, AUROC, calibration metrics; segmentation Dice/IoU; split counts |
| New analyses [N] | Retrained segmentation U-Net (§10.2; aggregates re-checked from per-image rows); external evaluation on OCTDL, Duke 2014 and Duke 2015; CIs and paired tests; duplicate and patient-overlap scan; dimension and border-artifact analyses; validation-split mismatch diagnosis and measured contamination (84.3%); training-time split rebuild and threshold/temperature refit; supplementary unseen-pool evaluation; segmentation failure-mode analysis; literature extraction |
| Unverified [U] | None outstanding for the literature extraction. Not tested: the reason for the OCTDL misses (label definition vs device) |

## 18. Recommended future experiments

1. ~~Rebuild the training-time split with scikit-learn 1.4.2 and refit the threshold and temperature.~~ **Done** (§7.2, scripts 11–12). `requirements.txt` now pins `scikit-learn==1.4.2`, and `train.py` writes `split_manifest.csv` (path, label, patient, split, scikit-learn version) at the start of every run.
2. ~~Evaluate on an external DME dataset from a different site or device~~ **Partly done** (§7.4: OCTDL, Duke 2015). Duke Srinivasan 2014 was added later (§7.4). Remaining: a clinically graded diabetic cohort, and a volume aggregation or foveal slice-selection rule fixed on separate data. The Isfahan MISP Basel set (swept-source OCT, 40 DME / 50 healthy volumes) was identified but not downloaded.
3. ~~Save per-image predictions in `evaluate.py` for every future run.~~ **Done:** `evaluate.py` writes `predictions.csv` (validation and test: path, patient, label, both logits, raw and calibrated P(DME)) next to `metrics.json`. Existing checkpoints and results were not rerun.
4. ~~Re-train the U-Net with ungraded pixels masked out of the loss and metric, a positive-class weight, and a sanity check against all-foreground and all-background baselines.~~ **Done** (§10.2): the current model (v2, whole-frame loss) reaches test pooled Dice 0.638, against 0.694 between graders; the first retrain (v1, 0.706) was superseded because it predicted fluid outside the graded band. Remaining: evaluate on more subjects and on an external segmentation set (e.g. RETOUCH), and check boundary accuracy.
5. If volumes become available, evaluate at the volume level to allow category-A/B comparison with Tang 2021.

## 19. References

1. Tang F, Wang X, Ran AR, et al. A Multitask Deep-Learning System to Classify Diabetic Macular Edema for Different Optical Coherence Tomography Devices: A Multicenter Analysis. *Diabetes Care*. 2021;44(9):2078–2088. doi:10.2337/dc20-3064. PMID 34315698; PMCID PMC8740924. Supplement: doi:10.2337/figshare.14710284.
2. Zhu H, Ji J, Lin JW, et al. Development and validation of a 3-D deep learning system for diabetic macular oedema classification on optical coherence tomography images. *BMJ Open*. 2025;15(5):e099167. doi:10.1136/bmjopen-2025-099167. PMID 40449950; PMCID PMC12128415.
3. Kermany DS, Goldbaum M, Cai W, et al. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*. 2018;172(5):1122–1131.e9. doi:10.1016/j.cell.2018.02.010.
4. Chiu SJ, Allingham MJ, Mettu PS, Cousins SW, Izatt JA, Farsiu S. Kernel regression based segmentation of optical coherence tomography images with diabetic macular edema. *Biomed Opt Express*. 2015;6(4):1172. doi:10.1364/BOE.6.001172.
5. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*. 1988;44(3):837. doi:10.2307/2531595.
6. Sun X, Xu W. Fast Implementation of DeLong's Algorithm for Comparing the Areas Under Correlated Receiver Operating Characteristic Curves. *IEEE Signal Process Lett*. 2014;21(11):1389–1393. doi:10.1109/LSP.2014.2337313.
7. Li F, Chen H, Liu Z, Zhang XD, Jiang MS, Wu ZZ, Zhou KQ. Deep learning-based automated detection of retinal diseases using optical coherence tomography images. *Biomed Opt Express*. 2019;10(12):6204–6226. doi:10.1364/BOE.10.006204.
8. Rodríguez-Miguel A, Arruabarrena C, Allendes G, Olivera M, Zarranz-Ventura J, Teus MA. Hybrid deep learning models for the screening of Diabetic Macular Edema in optical coherence tomography volumes. *Sci Rep*. 2024;14:17633. doi:10.1038/s41598-024-68489-2.
9. Singh RK, Gorantla R. DMENet: Diabetic Macular Edema diagnosis using Hierarchical Ensemble of CNNs. *PLoS ONE*. 2020;15(2):e0220677. doi:10.1371/journal.pone.0220677.

Full author lists: `research_audit/references.bib`.
