**Table 6. This project and each of the five papers side by side (not a head-to-head comparison)**

| Item | This project | Tang 2021 | Zhu 2025 | Li F 2019 | Rodríguez-Miguel 2024 | Singh & Gorantla 2020 |
|---|---|---|---|---|---|---|
| Input modality | OCT | OCT | OCT | OCT | OCT | fundus photograph |
| Evaluation unit(s) | image, volume | image, volume | volume | image, volume | volume | image |
| Device(s) | Spectralis (Kermany, Duke 2014); Optovue (OCTDL) | CIRRUS HD-OCT; SPECTRALIS OCT; Triton OCT | Cirrus HD-OCT 4000 and Topcon 3D-OCT 2000 (pooled); Cirrus HD-OCT 4000 (Zeiss) only; Topcon 3D-OCT 2000 only; Topcon 3D-OCT 2000; Cirrus HD-OCT 4000 (Zeiss); NR | Heidelberg Spectralis | Topcon 3D OCT-Maestro 1 | Kowa VX-10 alpha (IDRiD); Topcon TRC NW6 (MESSIDOR) |
| Rows (DME presence) | 12 (6 external / separate cohort) | 16 (11 external / separate cohort) | 7 (3 external / separate cohort) | 3 (0 external / separate cohort) | 7 (4 external / separate cohort) | 4 (0 external / separate cohort) |
| Comparability category | A | B, C | C | B, C, D | C | D |
| Headline result | EfficientNet-B3 · Kermany official test (B-scan, internal, n=484) | 2D ResNet-18 (2D multitask CNN) · Primary [Supplementary Table S1 (B-scan level), figshare 10.2337/figshare.14710284] | Custom 3D CNN initialised from Models Genesis, two FC layers · Testing [Table 3] | Ensemble of four improved (dilated) ResNet50, ImageNet-pretrained · Shanghai hospitals, independent test patients [Table 6 (3 ensembling binary classifiers, DME vs NORMAL)] | Hybrid CNN-RNN (22 parallel custom CNNs + bidirectional RNN), Model 2, threshold 1 (Youden) · Real-world DR screening cohort (Madrid), Nov 2015 - Mar 2019 [Table 2 and Results text (AUROC: Results, Fig 1)] | Hierarchical Ensemble of CNNs (ResNet/DenseNet/SE-ResNet/SqueezeNet learners with gating CNNs) · IDRiD + MESSIDOR pooled [Table 3 and Fig 6A (Gamma Ensemble, stage 1)] |
| Headline n | 484 | NR | 956 volumes (TN 700, FP 22, FN 15, TP 219) | NR (test set holds 332 DME / 635 NORMAL B-scans) | 5314 cubes (126 DME / 5188 non-DME) | 360 test images |
| Headline Accuracy | 0.998 | 0.972 | 0.961 | 0.988 | NR | 0.967 |
| Headline Sensitivity | 0.996 | 0.887 | 0.936 | 0.986 | 0.905 | 0.975 |
| Headline Specificity | 1.000 | 0.992 | 0.970 | 0.991 | 0.908 | 0.956 |
| Headline F1 | 0.998 | NR | 0.922 | NR | NR | 0.970 |
| Headline AUROC | 1.000 | 0.940 (0.934–0.945) | 0.990 (0.983–0.996) | 0.996 (0.995–0.997) | 0.940 (0.910–0.970) | 0.965 |
| Range Accuracy (k) | 0.700–0.998 (12) | 0.926–0.995 (16) | 0.729–0.974 (7) | 0.971–0.988 (2) | 0.880–0.915 (3) | 0.955–0.968 (4) |
| Range Sensitivity (k) | 0.932–1.000 (12) | 0.804–1.000 (16) | 0.458–0.936 (7) | 0.901–1.000 (3) | 0.810–0.921 (4) | 0.947–0.979 (4) |
| Range Specificity (k) | 0.400–1.000 (12) | 0.897–1.000 (16) | 0.849–1.000 (7) | 0.982–1.000 (3) | 0.831–0.970 (4) | 0.945–0.972 (4) |
| Range F1 (k) | 0.769–0.998 (12) | NR | 0.629–0.922 (7) | NR | NR | 0.961–0.970 (2) |
| Range AUROC (k) | 0.993–1.000 (12) | 0.899–0.965 (16) | 0.904–0.990 (7) | 0.978–0.996 (2) | 0.940–0.962 (4) | 0.965–0.965 (1) |
| Unit-matched project value (EfficientNet-B3, 0.5) | NA | Kermany official test (B-scan, internal, n=484): Sens 0.996, Spec 1.000, AUROC 1.000 | Duke 2014 Spectralis (volume, max rule, external, n=30): Sens 1.000, Spec 0.467, AUROC 1.000 | Kermany official test (B-scan, internal, n=484): Sens 0.996, Spec 1.000, AUROC 1.000 | Duke 2014 Spectralis (volume, max rule, external, n=30): Sens 1.000, Spec 0.467, AUROC 1.000 | Kermany official test (B-scan, internal, n=484): Sens 0.996, Spec 1.000, AUROC 1.000 |
| Learning curves | Figure 4 | NR | NR | NR | NR | NR |
| Calibration (ECE / Brier) | Figure 5, Table 2 | NR | NR | NR | NR | NR |
| Fluid segmentation (Dice) | Report §10.2 (retrained U-Net v2, test pooled Dice 0.638; graders 0.694) | NA (classification only) | NA (classification only) | NA (classification only) | NA (classification only) | NA (classification only) |

*Values from research_outputs/data/per_paper_comparison.csv (script 14). Headline = internal B-scan/image-level result if the paper reports one, otherwise its main-text primary result; range = min–max over every DME-presence row of that source (k = rows reporting the metric). Unit-matched project value: same evaluation unit as the paper's headline (image → official test; volume → Duke 2014, max rule, n=30). Data, populations, devices and labels differ in every column; no cross-study test. NR = not reported; NA = not applicable.*
