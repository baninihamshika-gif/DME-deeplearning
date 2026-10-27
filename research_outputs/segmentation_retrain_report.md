# Fluid segmentation retrain (Duke DME 2015): results note

Branch `claude/seg-retrain-227f8e`. Code: `segment.py` (commit da997e8). Machine-readable results are in
`research_outputs/segmentation_retrain/`: `segmentation_summary.json`, `segmentation_per_image.csv`,
the training log and the console log. The checkpoint and the overlay figures are in the git-ignored
`artifacts/segmentation_retrain/`. The figures are not committed because the Duke data may not be
redistributed. The earlier checkpoint `artifacts/kaggle/20260923T181925Z/.../checkpoint_best.pt` was not touched.

## What was wrong and what changed

| Audit finding | Change |
|---|---|
| Predicted fluid on ~100% of pixels; best checkpoint was from epoch 1 | Epoch and probability threshold are now selected jointly, on val only, by pooled Dice over a fixed grid (0.05-0.95, step 0.05). Early-stopping patience is 15 epochs, with at most 100 epochs. Batch size is 4, giving about 17 steps per epoch (the old setting gave 5). |
| Ungraded pixels counted as background | The loss and all metrics use graded pixels only. **Correction to the audit's description:** on the 110 annotated scans `manualFluid1/2` contain *no* NaN. The ungraded side margins are stored as 0. The graded extent is the band of columns that carries manual layer coordinates: about 500-550 of the 768 columns, identical for both graders, with no fluid from either grader outside it. So graded = non-NaN AND inside that band, with letterbox padding also excluded. The audit's graded-only metric removed only the padding. |
| ~1% fluid: no imbalance handling | The loss is BCE with `pos_weight=10` plus a soft Dice term pooled over the batch, both on graded pixels. These were fixed in advance, not tuned. |
| No sanity baselines | All-foreground and all-background predictions are reported, scored with the same metrics. |

Other choices:
- Metrics are computed at native 496x768 resolution: probabilities are un-letterboxed and resized back.
- The encoder starts from the EfficientNet-B3 classifier (`20260917T035646Z`); all 572/572 encoder keys matched.
- The patient-level split is unchanged (seed 42): train 02 03 04 05 09 10, val 01 08, test 06 07.
- Training ran on CPU; the run is deterministic, and the smoke run reproduced the full run's epochs 1-2 exactly.
- The test split was evaluated once, after the checkpoint and threshold were fixed. The script refuses to overwrite an existing final summary.

**Selection:** best epoch 17 of 32 (early stop). Chosen threshold **0.80**; val pooled Dice is flat between 0.65 and 0.85 (0.792-0.796), and is 0.775 at 0.5.

## Results (graded pixels only, ground truth = grader 1)

Dice convention: a scan with no fluid in either the ground truth or the prediction scores Dice 1. That is
why the all-background baseline gets a non-zero *mean* Dice, and why the fluid-positive-only mean is the
more informative per-image number. Pooled = pixel counts summed over all scans of the split.

### Validation (subjects 01, 08; 22 scans, 17 with fluid)

| | Pooled Dice | Pooled IoU | Mean Dice (all) | Mean Dice (fluid+) | Mean IoU (fluid+) |
|---|---|---|---|---|---|
| **Model (t=0.80)** | **0.796** | 0.662 | 0.687 | **0.595** | 0.475 |
| All-foreground | 0.031 | 0.016 | 0.030 | 0.039 | 0.020 |
| All-background | 0.000 | 0.000 | 0.227 | 0.000 | 0.000 |
| Grader 2 vs grader 1 | 0.701 | 0.540 | 0.520 | 0.556 | 0.415 |

### Test (subjects 06, 07; 22 scans, 14 with fluid), evaluated once

| | Pooled Dice | Pooled IoU | Mean Dice (all) | Mean Dice (fluid+) | Mean IoU (fluid+) |
|---|---|---|---|---|---|
| **Model (t=0.80)** | **0.706** | 0.545 | 0.721 | **0.633** | 0.491 |
| All-foreground | 0.020 | 0.010 | 0.020 | 0.031 | 0.016 |
| All-background | 0.000 | 0.000 | 0.364 | 0.000 | 0.000 |
| Grader 2 vs grader 1 | 0.694 | 0.531 | 0.755 | 0.615 | 0.471 |

- **Per subject (model, pooled Dice):** val 01 0.824, 08 0.641; test 06 0.694, 07 0.797. Subject 07 has only 4 fluid-positive scans.
- **Threshold 0.5 instead of 0.8, for reference only (not selected):** test pooled Dice 0.677, mean Dice (fluid+) 0.635.
- **Model vs grader 2, mean Dice on grader-1 fluid-positive scans:** val 0.574, test 0.629.
- **False positives on fluid-free scans:** the model predicted some fluid on 0/5 val and 1/8 test scans with no grader-1 fluid. The test case is Subject_06 scan 15, with 196 px.
- **Boundary distance:** mean symmetric surface distance on fluid-positive scans was 9.8 px on val and 26.9 px on test. Grader 2 vs grader 1 was 24.5 px and 5.7 px. These figures are driven by a few scans with small or misplaced regions; for example, test Subject_06 scan 10 has Dice 0 against a 354 px grader-1 region.
- **Audit figures for the earlier checkpoint, for context:** test Dice 0.009 and val 0.014, with fluid predicted on 99.995% of the frame.

## Interpretation (cautious)

- **Clear improvement over the trivial baselines.** On both splits the model scores well above predicting fluid everywhere (pooled Dice 0.71 vs 0.02 on test) and predicting nothing (0.71 vs 0). It predicts fluid on about 1.2-1.7% of graded pixels, close to the 1.0-1.6% marked by grader 1. The earlier collapse to near-all-foreground is gone.
- **Roughly in the same range as human inter-grader agreement on these 4 held-out subjects.** On test, pooled Dice was 0.71 vs 0.69 and fluid-positive mean Dice 0.63 vs 0.61. This should *not* be read as "human-level":
  - The model was trained on grader 1's style and scored against grader 1, while grader 2 is an independent human. Against grader 2 the model's fluid-positive mean Dice is 0.63 on test and 0.57 on val, about the same as the grader-vs-grader numbers.
  - With 2 test subjects and 14 fluid-positive test scans, a difference of a few Dice points is well inside the noise.
  - The val numbers are optimistic, because both the epoch and the threshold were chosen on val.
- **Qualitative failure mode.** In the test overlays (`artifacts/segmentation_retrain/segmentation_test_examples.png`) the model finds the fluid region in the right place. However, it merges neighbouring small cysts into one smooth region and does not reproduce the septa between them. Small or faint fluid areas (a few hundred pixels) are sometimes missed entirely.

## Limits

- **Very small dataset.** There are 10 subjects in total: 6 for training (66 scans), 2 for validation (22) and 2 for test (22), all from one device (Spectralis) and one centre. Results describe these 4 held-out eyes only. They give no evidence about other devices, other populations, or eyes without DME.
- **Uncertainty is wide and not formally estimated.** With 2 test subjects, a subject-level confidence interval is meaningless (NR), and an image-level bootstrap would understate the between-subject variation. The per-subject spread (0.64-0.82 pooled Dice) is the most honest indication.
- **One run, one seed.** Seed-to-seed variance was not measured (NR).
- **Hyperparameters were fixed in advance and not searched:** `pos_weight`, loss weights, learning rate, batch size, augmentation and patience. The threshold was the only tuned quantity, on val. A different reasonable configuration could do better or worse.
- **Selection noise.** The validation set has only 2 subjects, and its inter-grader agreement is low (mean Dice 0.52). Epoch selection is therefore noisy: val pooled Dice moved between 0.77 and 0.80 from epoch 7 onwards.
- **Not reviewed for clinical use.** The fluid mask is a research output only. It has not been assessed for any diagnostic or clinical use, and nothing here supports clinical claims.
- **`app.py` is unchanged.** It still points at the earlier checkpoint and uses a fixed 0.5 threshold. Switching it to this checkpoint (threshold 0.80) is a separate decision.
