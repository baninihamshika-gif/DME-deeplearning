# Phase 6 — Gradio Clinical Triage Interface — Sign-off

`app.py` serves the Phase 3 primary checkpoint (`artifacts/kaggle/20260917T035646Z/artifacts/checkpoint_best.pt`, tf_efficientnet_b3, threshold 0.971, temperature 0.917) behind a single-screen triage UI, per `PROJECT_BRIEF.md` Phase 6.

## Verification method

The app runs on the user's Windows machine (`python app.py`, `http://127.0.0.1:7860`). Rather than relying on the user to copy-paste terminal output and screenshots back and forth, verification was automated end-to-end using the Claude Browser pane connected directly to that local server: test images were injected via a synthetic `File`/`DataTransfer` upload (no manual file picker needed), and the resulting DOM/screenshots were inspected directly.

Three real OCT B-scans from `data/raw/OCT2017/test/` were run through the live app:

| Scan | True label | Predicted | Probability | Referral state |
|---|---|---|---|---|
| NORMAL-101880-1 | Normal | Within normal limits | 0.09 | ✓ no referral |
| NORMAL-1017237-1 | Normal | Within normal limits | 0.21 | ✓ no referral |
| DME-1081406-1 | DME | DME detected | 1.00 | ✗ refer |

## Confidence-bar CSS: investigated and cleared

Earlier in this session, screenshots the user sent appeared to show a "Confidence 0.00" label next to what looked like a large filled bar, and a first round of live DOM inspection (via the browser pane) found `.confidence-track` and its ancestor chain reporting `getBoundingClientRect().width === 0`. That measurement was taken while the browser pane was in its hidden state, which turned out to be the actual cause — not a real CSS bug. Re-measuring with the pane visible, across all three probabilities above, showed the fill/marker/threshold-tick positioned exactly proportionally to the real probability in every case (e.g. prob 0.09 → marker at ~9% of the track; prob 1.00 → marker at the far right; threshold tick fixed at 97%). The defensive CSS reset added earlier (`box-sizing`/`display`/`flex` pin on `.confidence-track`) is harmless and left in place, but was not the actual fix — there was nothing to fix. What the user likely saw was the neutral grey track (always full width by design, since it's the 0–1 scale) being mistaken for a "filled" bar next to a genuinely low, correctly-rounded probability.

## Acceptance checklist

- Outputs wired: label / confidence+threshold / Grad-CAM / referral state / segmentation+thickening-index placeholders — **yes**, all five wired; Phase 4/5 slots show explicit "not available (Phase N pending)" text, never a fabricated mask or number. (As of this sign-off, Phase 4/5 had not yet run — see `phase4-segmentation-signoff.md` / `phase5-thickening-signoff.md` for what actually shipped once they did, and `report/progress_report.md` for whether/how those outputs were later wired into this interface.)
- Threshold and temperature loaded from checkpoint (never hardcoded): **yes** — `load_model()` reads `ckpt.get("threshold")` / `ckpt.get("temperature")` and refuses to launch if either is `None`.
- Greyscale legibility check: **passed** — tested by applying `filter: grayscale(1)` to the live page; the DME/Normal states stay distinguishable by icon glyph (✗ vs ✓) and text, not color alone.
- 1280×720 check: **passed** — viewport emulated at 1280×720, layout has no overflow or cut-off content, footer disclaimer fully visible.
- Keyboard nav (←/→ cycle view tabs, space toggles heatmap↔original): **passed**, verified via dispatched keydown events against the live page.
- Heatmap opacity slider: **passed** — moved from 50 to 100, Grad-CAM overlay correctly went from blended to fully opaque.
- All 4 view tabs: **passed** — original, preprocessed, and heatmap all render real image data; fluid mask correctly shows no image plus the explicit "Segmentation mask not available (Phase 4 pending)" note.
- DME-positive and Normal-negative cases both exercised end-to-end.

## PHASE 6 COMPLETE

```
PHASE 6 COMPLETE
Outputs wired: label / confidence+threshold / Grad-CAM / referral state / seg+thickening-index placeholders
Threshold and temperature loaded from checkpoint: yes
Greyscale legibility check: passed
1280×720 check: passed
Blocking questions: none
```

Next per the locked execution order: freeze `v1.0-primary-complete` git tag, then Phase 7 (report), then Phase 4 (segmentation), then Phase 5 (thickening index), then integration, then final documentation.
