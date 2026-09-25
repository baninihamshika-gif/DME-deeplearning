# DME OCT Screening

**Deep Learning-Based AI Detection of Macular Edema Using Optical Coherence Tomography (OCT) Images**

Final-year project, Kalasalingam Academy of Research and Education. Guide: Dr. R. Murugeswari. Team: M Bharadhwaj · M Ghangaatharen Vishakh · S A Banini Hamshika · P Harani Hajandikaa.

## What it does

Takes a single 2D retinal OCT B-scan and returns:

1. A classification — DME or Normal
2. A calibrated confidence value, with a referral flag when confidence is low
3. A Grad-CAM heatmap showing which retinal regions drove the decision

Two extended objectives were also built, run, and are now **surfaced in the interface as informational, clearly-disclaimed readouts** — neither one changes the classification or referral decision above:

4. A fluid segmentation mask (U-Net, trained on the separate Duke DME dataset) — time-boxed per the project brief; the real result is honest "future work" (see Results below), shown with an explicit reliability disclaimer, not usable output
5. A relative retinal thickening index (classical image processing, run on the same Kermany dataset) — complete, with a real but weak DME-vs-Normal separation (see Results below), also shown with an explicit reliability disclaimer

This is a **screening and triage aid, not a diagnostic device**. The word "diagnosis" does not appear anywhere in the interface, this README, or the project report. OCT scans are acquired faster than retina specialists can grade them; the goal is to help a technician or general ophthalmologist decide, in about two seconds, whether a scan needs specialist review — not to replace that review.

A single 2D B-scan cannot distinguish centre-involving from non-centre-involving DME (which determines whether anti-VEGF therapy is indicated). The system does not attempt this — it flags centre involvement as indeterminate and recommends volumetric follow-up. This is designed behaviour, not a hidden limitation. See `report/progress_report.md` for the full rationale and results.

## Results summary

EfficientNet-B3, evaluated once on a held-out 484-image test set (242 DME / 242 Normal):

| Threshold | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| 0.5 (default) | 99.79% | 99.59% | 100.00% | 1.0000 |
| 0.971 (tuned to a 0.97 sensitivity target) | 99.17% | 98.35% | 100.00% | 1.0000 |

Calibrated via temperature scaling (ECE 0.0028 → 0.0027, T=0.917). Grad-CAM (targeting `model.conv_head`) passed its randomisation sanity check — heatmaps changed substantially when the classifier's weights were randomised, confirming the explanation depends on the trained weights rather than input structure alone. Two smaller/older baselines (EfficientNet-B0, ResNet-50) matched B3 exactly on test accuracy and AUC — a real, plainly-reported finding, not a failed experiment. Full tables, figures, and the investigation into why test AUC reaches 1.0000 (patient leakage ruled out; a weak, non-explanatory image-dimension confound quantified) are in `report/progress_report.md`.

**Extended objectives (E1 segmentation, E2 thickening index)** — both run and reported honestly, not fabricated to look complete:

| Objective | Status | Headline result |
|---|---|---|
| E1 — fluid/layer segmentation (U-Net, Duke DME dataset) | Time-boxed per brief | Test Dice ≈ 0.0090 — not usable; treated as disclosed future work |
| E2 — relative thickening index (classical CV, Kermany dataset) | Complete | Normal 0.906 ± 0.419 vs DME 1.049 ± 0.458; correct direction, heavy overlap (AUC 0.601) |

Both are wired into the live `app.py` interface as informational readouts, each shown with the disclaimer above — neither changes the classifier's referral decision. Full methodology, the boundary-detection iteration history, and the honest diagnostic reading of both results are in `report/progress_report.md` §2.8/§2.9 and §3.7/§3.8.

## Repository layout

```
├── PROJECT_BRIEF.md       # full project specification and phase-by-phase plan
├── requirements.txt
├── config.py              # all hyperparameters and paths — nothing hardcoded elsewhere
├── utils.py                # patient parsing, patient-grouped splits, preprocessing, seeding
├── train.py                 # classifier training (two-phase fine-tuning), resumable
├── evaluate.py              # test evaluation, threshold tuning, calibration, TTA
├── explain.py                # Grad-CAM + randomisation sanity check
├── prepare_data.py           # Phase 1: splits, sample curation, preprocessing figure
├── app.py                    # Gradio clinical triage interface (Phase 6)
├── segment.py                 # extended objective E1: U-Net fluid/layer segmentation (offline pipeline; app.py also loads its checkpoint for live inference)
├── thickness.py                # extended objective E2: classical-CV relative thickening index (offline pipeline; app.py also imports it for live inference)
├── kaggle/
│   ├── run.py                 # push / poll / fetch automation for training on Kaggle
│   ├── entry.py                # thin kernel bootstrap (no logic — see below)
│   ├── run_segment.py           # Kaggle push/poll/fetch automation for E1 segmentation training
│   ├── entry_segment.py          # thin kernel bootstrap for E1 (no logic — see below)
│   └── kernel-metadata.json
├── tests/
│   └── test_thickness.py        # regression tests for the E2 boundary-detection pipeline
├── artifacts/                 # checkpoints, figures, metrics.json (gitignored except this report references it)
│   └── thickness/               # E2 outputs: per-image thickness profiles, thickness_summary.json
├── data/
│   ├── raw/                    # gitignored — see Data below
│   └── sample/                  # curated ~60-image subset for local development, gitignored
└── report/
    └── progress_report.md       # full Phase 7 report: methodology, results, figures, limitations
```

All logic lives in importable modules under version control; `kaggle/entry.py` is a thin bootstrap with no logic of its own, so a Kaggle session dying mid-run never loses anything that wasn't already in git.

## Setup

Python 3.10+, PyTorch 2.x. Training requires a GPU and is designed to run on Kaggle's free tier (~30 GPU hrs/week); the local machine is used for code editing and git only, plus CPU-only inference via `app.py`.

```
pip install -r requirements.txt
```

**Note on the `gradio` pin:** `requirements.txt` pins `gradio==4.*` for reproducibility, matching the version used during development. On some local environments (notably Python 3.14 on Windows, as of this writing) `gradio==4.*`'s `pillow<11.0` dependency has no prebuilt wheel and fails to build from source. Since `gradio` is never imported during Kaggle training, installing an unpinned, newer `gradio` locally (`pip install --upgrade gradio`) does not affect training reproducibility and is a safe workaround for running the interface specifically — see `report/progress_report.md` §6 for the full note.

## Data

Kermany et al. (Cell, 2018), OCT2017 / Mendeley V2 — **specifically the 83,484-training-image release** (a second release with 108,312 images and a different NORMAL count exists and will silently invalidate every expected-count assertion in this codebase if used instead). Place it under `data/raw/` so that `data/raw/OCT2017/{train,test}/{DME,NORMAL}/` resolves. Only the DME and NORMAL classes are used.

On Kaggle, the dataset is attached as `paultimothymooney/kermany2018` and resolved automatically via `config.DATA_DIR`.

## Reproduction

### Phase 1 — splits, sample curation, preprocessing figure (local, no GPU needed)

```
python prepare_data.py --train-dir data/raw/OCT2017/train --test-dir data/raw/OCT2017/test
```

### Phase 2 / 2.5 — training on Kaggle

```
python kaggle/run.py push --smoke          # always run a smoke test first, on the same source SHA
python kaggle/run.py poll baninihamshika/dme-oct-train
python kaggle/run.py push                   # full run, once the smoke test passes
python kaggle/run.py fetch baninihamshika/dme-oct-train
```

`push` refuses to run a full job without a passing smoke run on the current source SHA, and refuses to push if the projected GPU time would exceed the configured budget (default 6h/push, tracked against the real 30h/week free-tier cap).

To train locally instead (small-scale / debugging only — no GPU guardrails):

```
python train.py --train-dir data/raw/OCT2017/train --test-dir data/raw/OCT2017/test --smoke
```

### Phase 3 — evaluation, calibration, Grad-CAM

```
python evaluate.py --train-dir data/raw/OCT2017/train --test-dir data/raw/OCT2017/test \
    --checkpoint artifacts/kaggle/<run-timestamp>/artifacts/checkpoint_best.pt --tta

python explain.py --test-dir data/raw/OCT2017/test \
    --checkpoint artifacts/kaggle/<run-timestamp>/artifacts/checkpoint_best.pt
```

`evaluate.py` writes the tuned threshold and calibration temperature back into the checkpoint, so `app.py` (below) never needs them hardcoded. For a baseline or ablation checkpoint, pass a distinct `--artifacts-dir` (e.g. `artifacts/baselines/<model_name>/`) so its `metrics.json` doesn't overwrite the primary model's.

### Phase 6 — the interface

```
python app.py --checkpoint artifacts/kaggle/<run-timestamp>/artifacts/checkpoint_best.pt
```

Opens at `http://127.0.0.1:7860`. Defaults to the signed-off primary checkpoint if `--checkpoint` is omitted (see `DEFAULT_CHECKPOINT` in `app.py`), and to the real E1/E2 artifacts below if `--segmentation-checkpoint`/`--thickness-summary` are omitted (see `DEFAULT_SEGMENTATION_CHECKPOINT`/`DEFAULT_THICKNESS_SUMMARY` in `app.py`) — either falls back to an explicit "not available" placeholder in the UI if its file is missing, rather than failing to launch. Full acceptance-criteria verification (greyscale legibility, keyboard navigation, 1280×720 usability, all five view tabs, the confidence scale, referral states) is in `report/phase6-gradio-signoff.md`; a condensed version is also referenced from `report/progress_report.md`.

### Extended objectives — E1 / E2 (offline; E1 needs a GPU to train, E2 is CPU-only)

```
python segment.py --duke-dir data/raw/2015_BOE_Chiu \
    --classifier-checkpoint artifacts/kaggle/<run-timestamp>/artifacts/checkpoint_best.pt

python thickness.py --train-dir data/raw/OCT2017/train --test-dir data/raw/OCT2017/test
```

Both scripts support `--verify-only` (generate the verification figure only, no GPU/full run) and `--smoke` (a small sanity-check run before the full one), mirroring the conventions of `train.py`/`evaluate.py`. These are the scripts that produced the real, once-off training run and full-corpus reference behind each objective; `app.py` (below) then loads that same E1 checkpoint and E2 reference at startup for live, per-scan inference in the interface. See `report/progress_report.md` §2.8/§2.9 (methodology) and §3.7/§3.8 (results) for the full picture, including why E1's result is disclosed as unusable rather than hidden.

## Limitations

- Screening/triage aid, not a diagnostic device — all findings require ophthalmologist review.
- Single 2D B-scan only; cannot determine centre involvement (flagged as indeterminate by design, not silently assumed).
- Near-perfect test-set performance (AUC 1.0000) was investigated for leakage and shortcut learning before being accepted; the dataset's apparent separability may not generalise to OCT images from other scanners, sites, or acquisition protocols without external validation.
- EfficientNet-B3 does not meaningfully outperform smaller (B0) or older (ResNet-50) baselines on this dataset.
- Extended objective E1 (fluid/layer segmentation) was time-boxed per the project brief; the real trained result (test Dice ≈ 0.0090) is not usable and is disclosed as future work rather than fabricated or hidden.
- Extended objective E2 (relative thickening index) is complete and real, but weak: DME vs. Normal separation is in the correct direction with heavy overlap (AUC 0.601), not clinically discriminative on its own.
- Both E1 and E2 are wired into the live `app.py` interface as informational, explicitly-disclaimed readouts (real model/pipeline output, not fabricated) — neither one feeds into or overrides the primary classifier's referral decision.

Full details, all results tables, the complete figure index, and every disclosed data gap: `report/progress_report.md`.
