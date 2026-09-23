# DME OCT Project — Progress Report

**Deep Learning-Based AI Detection of Macular Edema Using OCT Images**
Kalasalingam Academy of Research and Education · Guide: Dr. R. Murugeswari
Team: M Bharadhwaj · M Ghangaatharen Vishakh · S A Banini Hamshika · P Harani Hajandikaa

Report generated: 2026-09-22, from the current state of the repository and Kaggle run logs. Pulled directly from `quota_log.json`, `last_smoke.json`, fetched training logs, and evaluation artifacts — nothing in this report is estimated or assumed.

---

## Summary

| | Status |
|---|---|
| Primary objectives (P1–P4) | Functionally complete, pending final ablation entry |
| Phase 3 sign-off | **Not yet reached** — one experiment outstanding |
| Interface, freeze, documentation, extended objectives | Not started |

---

## Completed phases

### Phase 0 — Scaffolding
`requirements.txt`, `config.py`, `.gitignore`, `set_seed()` all in place and unchanged since setup.

### Phase 1 — Data handling & preprocessing
Patient-grouped splitting (`build_splits()`, `StratifiedGroupKFold`), aspect-preserving letterbox resize to 300×300, retinal flattening with fallback logging, `PreprocessCache`, and `fig_preprocessing_stages.png` are all implemented and verified in every subsequent run's log output (count assertions and patient-overlap checks pass identically on every run).

### Phase 2 — Classifier training (primary)
EfficientNet-B3 trained to completion on Kaggle. Phase A ran the full 5/5 epochs; Phase B early-stopped at epoch 20/25 (5 epochs without improvement). Best global epoch 20, best validation AUC 0.9958. Total training time 319.77 minutes (5.33 h). Checkpoint: `artifacts/kaggle/20260917T035646Z/artifacts/checkpoint_best.pt`.

### Phase 2.5 — Kaggle automation
`kaggle/run.py` (push / poll / fetch), GPU quota tracking, and smoke-gating are working. Two known limitations, both understood and worked around rather than hidden:
- `poll()` has repeatedly shown a local wall-clock/network gap (status briefly reads `unknown` for extended periods) on long runs. Where this previously produced an inflated `actual_minutes`, it was corrected against the run's own log timestamps and documented with a `correction_note` in `quota_log.json`. On the most recent (2026-09-21) smoke run it self-corrected properly.
- The quota **guardrail** compares cumulative 7-day usage against a small default per-push budget (6 h / 360 min), separate from the real 30 h/week free-tier cap — it needs an explicit `--budget-min` override once cumulative usage passes 6 h, which is expected behaviour, not a bug.

### Phase 3a–3f — Evaluation, calibration, Grad-CAM
All complete on the primary B3 checkpoint: sensitivity-targeted threshold tuning (validation only), full test-set evaluation at both default and tuned thresholds, temperature scaling with ECE reported before/after, and the Grad-CAM grid plus its randomisation sanity check. All corresponding figures exist under `artifacts/`.

### Phase 3g — Baselines table
EfficientNet-B0 and ResNet-50 evaluated under identical preprocessing, splits, and evaluation methodology as B3. Finding, reported plainly per the brief's own instruction: **B3 does not meaningfully beat either baseline** — all three match on test accuracy (99.79%) and AUC (1.0000) at the default threshold.

A side investigation was carried out before this near-perfect result was accepted (project rule C7: 100% accuracy is treated as a probable bug until checked): patient leakage was ruled out by code review and confirmed at zero overlap in every run; an image-dimension/border confound was found to be real but weak (AUC 0.69 from geometry alone, and a naive dimension-based rule scored *worse* than chance on the test set because the pattern reverses between train and test); and the fact that three independently trained architectures converge on the identical result supports genuine class separability over a shared bug. Full writeup: `phase3g-baseline-comparison.md`.

---

## In progress

### Phase 3g — Ablation table
Two techniques are in scope (the other four — 5-fold ensemble, EMA, cosine annealing, progressive resolution — are explicitly descoped: not implemented in code, and not worth the GPU budget they'd cost, per an earlier scope decision):

| Technique | Status | Result |
|---|---|---|
| A1 — Test-time augmentation | **Done** | No measurable change vs. no-TTA (acc 99.79%, AUC 1.0000 either way) — an honest "didn't help" result |
| A5 — Label smoothing (0.05) | **Full run status unknown** | Smoke test passed (2026-09-21). Full retrain was pushed (2026-09-21T19:44 UTC, expected ~6.1 h) but the local process never recorded a completion — `quota_log.json` still shows `status: "pushed"`, and no new artifacts folder has been fetched. **This needs to be checked**: either re-poll the kernel (`python kaggle\run.py poll baninihamshika/dme-oct-train`) or check the Kaggle web page directly to see whether it finished, failed, or is still running. |

**Important note for whoever runs the eventual evaluation:** the label-smoothing checkpoint uses the same architecture name (`tf_efficientnet_b3`) as the primary model, so `evaluate.py`'s automatic output-folder namespacing will **not** separate it from the primary model's results — running it with the default `--artifacts-dir` would overwrite the real primary `metrics.json`. It needs an explicit distinct `--artifacts-dir` (e.g. `artifacts/ablation/label_smoothing_005/`) when the time comes.

Phase 3 cannot be signed off until this is resolved.

---

## Not started

### Phase 6 — Interface
No `app.py`. Per the team's agreed execution order, this comes immediately after Phase 3 sign-off — before Phase 4/5 — using the already-completed classification, calibration, and Grad-CAM, built as a Gradio app in the OCT-report visual style specified in the brief.

### Version freeze
No tag/checkpoint yet marking a complete primary system. Planned as `v1.0-primary-complete` once Phase 6 works, before Phase 7 documentation begins.

### Phase 7 — Report & deck assets
`report/` folder exists but is empty. No `README.md` yet. Planned to be generated from the frozen version's real artifacts only — no placeholder numbers.

### Phase 4 — Segmentation (extended objective E1)
No `segment.py`. No Duke dataset ingestion has begun. Sequenced after Phase 6 and the version freeze, with its own 7-day timebox once started.

### Phase 5 — Relative thickening index (extended objective E2)
Depends entirely on Phase 4's segmentation output — not started.

### Final integration & documentation
Adding Phase 4/5 outputs into the Phase 6 UI, and the final report/deck refresh, both come last.

---

## GPU quota (informational)

Recorded usage over the last 7 days as of the most recent check: **974.2 minutes (~16.2 h)** of the 1800-minute (30 h) free-tier weekly cap, not counting the pending A5 full run (expected ~367.7 min / ~6.1 h if it ran to completion) — real headroom remains, but confirming the A5 run's actual outcome is the immediate priority before anything else is scheduled.

---

*This report reflects only what is verifiable from repository state and Kaggle run logs at generation time — no metric, status, or duration in this document has been estimated or invented.*
