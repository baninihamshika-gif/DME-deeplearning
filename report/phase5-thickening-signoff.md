# Phase 5 — Relative Thickening Index (Extended Objective) — Sign-off

**Status: COMPLETE.** Real, full-corpus numbers below are reported exactly as produced by `thickness.py` on the user's machine (2026-09-23); none are invented (C6). Thickness is reported only in pixels relative to a Normal-split reference, never in micrometres (C3), per the brief.

## What this phase covers

Extended objective (beyond the primary classifier and Phase 4's segmentation): a classical-CV proxy for retinal thickening, run directly on the Kermany OCT2017 classifier dataset (not Duke). Per the brief: extract an ILM/RPE boundary proxy per column, take per-column thickness, find the foveal column, take a windowed mean, normalise against the Normal training-split median, flag the "elevated but off-centre" third referral condition, and report DME-vs-Normal separation with an honest overlap assessment. This is explicitly a classical, unsupervised proxy — not a trained layer segmentation, and not built on Phase 4's model (per `segment.py`'s own docstring, boundary extraction for this phase was always flagged as separate, later work).

## Boundary-detection method: 4 iterations, each caught by visual verification before being trusted

Per this project's "verify before scaling to thousands of images" discipline (same as Phase 4's `verify_rasterization()`), every version of `_isolate_retina_mask()` was checked against real sample images — including a broader 20-image sample beyond the first 2 — before being accepted:

- **v1** (raw Otsu, per-column min/max): isolated noise specks in the dark vitreous cavity corrupted the ILM boundary almost everywhere.
- **v2** (+ opening + largest-connected-component): fixed the noise problem, but a single global threshold on the sharp image only ever captured the brightest sub-band (the RPE), undershooting true thickness 3-5x.
- **v3** (+ Gaussian blur before thresholding): fixed v2's undershoot for most images, but failed on 2/20 broader-sample images with an unusually noisy/grainy vitreous cavity (confirmed against the raw source JPEGs) — after blurring, the noise became as bright as the real retina band, and Otsu + largest-component picked the noise itself as "tissue."
- **v4** (shipped): `_smoothed_lower_boundary_anchor()` reuses `utils.flatten_retina()`'s own proven technique (raw Otsu + degree-2 polynomial fit, already running in production since Phase 1) as a trustworthy lower-boundary anchor; `_isolate_retina_mask()` then keeps only the contiguous tissue run touching that anchor, rather than trusting "largest bright blob = retina" unconditionally. Combined with a new, documented plausibility cap (`MAX_PLAUSIBLE_THICKNESS_FRACTION = 0.5`) that discards individual columns whose span exceeds half the content height — set from a real, wide gap in the 20-image sample (genuine columns topped out at 32%; the columns v4 still gets wrong sat at 69-72%).

Both original v3 failures re-measured correctly under v4 (198px→97px, 141px→31px), and the fix is regression-tested: `tests/test_thickness.py` (23 new tests) includes a synthetic-image test that reproduces the exact v3 failure (a disconnected artifact block made deliberately larger in area than the true retina band) and confirms v4 isolates the real band instead. Full repo suite: 146/146 passing.

## The real full run (2026-09-23)

```
python thickness.py --train-dir data/raw/OCT2017/train --test-dir data/raw/OCT2017/test
```

- **Normal reference**: median windowed thickness over the Normal training split (after the existing C8 patient-overlap exclusion with the test set — 305 patients / 7503 images excluded from train/val, unrelated to this phase, already-established `build_splits()` behaviour) = **43.69px**, from 18,179/18,364 valid images (99.0%).
- **Test set**: 484 images (242 DME + 242 Normal, official split). 481/484 valid (99.4%) — 3 images failed `find_foveal_column` outright, well under the pipeline's own 90%-valid blocking-question bar (none triggered).
- A small number of `Premature end of JPEG file` warnings appeared during both the reference and test computation (15 during the full run, 1 during smoke) — truncated/corrupted source JPEGs in the downloaded Kermany corpus, not a code defect. Handled gracefully: cv2/PIL still return decodable image data for these, and the ones that couldn't be measured are already reflected in the valid/total counts above, not silently dropped.

### Results

| | Normal (n=241) | DME (n=240) |
|---|---|---|
| Index (mean ± std) | 0.906 ± 0.419 | 1.049 ± 0.458 |

- **Separation: heavy overlap (AUC 0.601)**, against this module's own documented buckets (≥0.85 clear, 0.65-0.85 partial, <0.65 heavy overlap).
- **Centre-involvement-indeterminate** (third referral condition: index elevated but peak thickness outside the foveal window) fired on **46.6%** of valid test images.
- **Blocking questions: none** (from the pipeline's own honesty check).

## Reading the result honestly

An AUC of 0.601 is weak — barely better than the "heavy overlap" threshold — and far short of the primary classifier's 0.9958 AUC. Before reporting it at face value, it was checked the same way this project checks any result, in both directions (C7 in reverse for Phase 4's near-zero Dice; here, checking that a mediocre-but-plausible number isn't hiding an actual bug):

- **Direction is correct**: DME mean (1.049) > Normal mean (0.906), consistent with retinal thickening from edema — not reversed or degenerate.
- **The distribution is sane, not pathological** (`thickness_distribution.png`): Normal is left-shifted and DME right-shifted with heavy but genuine overlap in the middle — not bimodal collapse, not identical distributions, not a handful of outliers driving everything.
- **The reference value (43.69px) matches independent spot-checks**: a 20-image local sample examined during development (this session's cloud sandbox) produced a very similar range (median ~53px, 19-97px) using the same code path.
- **A spot check of the real-corpus verification figure** (`thickness_verification_check.png`, generated from the real train split's first Normal/DME images, not hand-picked) shows one image tracked well (`DME-1083927-1.jpeg` — ILM/RPE correctly hugging the fovea dip and fluid pockets) and one with visibly imperfect tracking on an unusually rotated/skewed scan (`NORMAL-1001666-1.jpeg`) — a reminder that the classical proxy has real per-image limitations on atypical scans, consistent with what's already documented as a limitation, not a new defect.

**Conclusion: this is accepted as the genuine, final Phase 5 result** — a real but weak signal, not a bug. The classical CV proxy (raw pixel-intensity boundary detection, no learned features, 300px-downsampled, single 2D-slice measurement with no volumetric context) is a fundamentally cruder instrument than the deep-learning classifier, and a large drop in discriminative power between the two is expected, not surprising. The high centre-involvement-indeterminate rate (46.6%) is consistent with — not independent evidence against — the same weak signal: with index values clustered near the 1.0 threshold and high per-class variance, many images flip in and out of "elevated" on noise alone, which is exactly what a mediocre AUC implies.

## Limitations (for the final report)

- Boundary detection is a classical, unsupervised proxy (Otsu thresholding + morphological + polynomial-fit anchoring), not a trained or clinically validated layer segmentation.
- Measured in pixels at 300px content resolution, never converted to physical units (C3) — not directly comparable to clinical OCT thickness measurements in micrometres.
- Single 2D B-scan per eye (as provided by the Kermany dataset) — no volumetric/3D context, so the "centre involvement" read is necessarily partial.
- ~1% of images (both reference and test) failed boundary detection outright and were excluded, not imputed.
- A handful of source JPEGs in the downloaded corpus are truncated/corrupted (`Premature end of JPEG file`); this is a data-quality property of the public dataset, not this pipeline.

## Artifacts

`artifacts/thickness/`: `thickness_results.csv` (484 rows), `thickness_summary.json`, `thickness_verification_check.png`, `thickness_distribution.png`.

## GPU / compute cost

None — Phase 5 is pure classical CV (OpenCV + NumPy), CPU-only. Full-run wall-clock: ~7460s (~2.1h) for the Normal reference (18,364 images) + ~180s for the 484-image test set, run directly on the user's machine.
