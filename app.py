"""
Phase 6: Gradio clinical triage interface for DME screening.

Serves the Phase 2/3 primary checkpoint (tf_efficientnet_b3, signed off in
Phase 3 -- see the project's phase3-ablation-and-signoff.md) behind a
single-screen triage UI. Per PROJECT_BRIEF.md Phase 6: ground the aesthetic
in OCT reporting itself (printed-report white, grayscale scan as anchor,
two semantic colours used only on state indicators), load the model once
at startup, and read the threshold/temperature from the checkpoint --
never hardcode them.

Design decisions made here, stated up front (same "explain before coding"
convention evaluate.py/explain.py follow -- see e.g. explain.py's
RANDOMIZATION_CORRELATION_THRESHOLD docstring for the same pattern):

  - The brief specifies three mutually exclusive referral states but the
    model outputs one continuous calibrated probability. "Below confidence
    threshold -- refer for manual review" is triggered for probabilities
    that land just under the tuned threshold (itself tuned for
    config.SENSITIVITY_TARGET = 0.97, see evaluate.py sweep_threshold) --
    i.e. cases the tuned operating point called Normal by only a small
    margin. LOW_CONFIDENCE_MARGIN below is that margin, in probability
    space. It is a judgment call, not fitted on any data or specified in
    the brief -- easy to change, always disclosed here rather than left
    implicit in the code.

  - Grad-CAM (reused directly from explain.py's generate_cam) targets
    model.conv_head, which exists on the EfficientNet family this
    project's primary checkpoint uses, but not on every architecture this
    codebase can train (e.g. resnet50, Phase 3g baseline, has no
    conv_head). This app is built to serve the signed-off primary
    checkpoint; if a checkpoint without conv_head is loaded, Grad-CAM is
    disabled with an explicit UI message instead of crashing or faking a
    heatmap.

  - At deployment the true label is unknown (unlike explain.py's offline
    grid, which explains the TRUE class because ground truth is known from
    the test set). Here Grad-CAM explains the PREDICTED class -- "what did
    the model attend to for the class it called."

  - Phase 4 (segmentation / fluid mask) and Phase 5 (thickening index,
    centre involvement) are not implemented. Their UI slots are shown as
    explicit "not yet available" placeholders -- never a fabricated mask
    or number (PROJECT_BRIEF.md C7 / master-prompt Section 11).

Usage:
    python app.py
    python app.py --checkpoint artifacts/kaggle/<run>/artifacts/checkpoint_best.pt
"""

import argparse
import datetime as dt
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import gradio as gr

import config
from evaluate import _softmax_positive
from train import MODEL_NAME, _eval_transform, build_model, load_checkpoint
from utils import preprocess_image

# ---------------------------------------------------------------------------
# Judgment-call constants (see module docstring)
# ---------------------------------------------------------------------------
LOW_CONFIDENCE_MARGIN = 0.10

DEFAULT_CHECKPOINT = config.ARTIFACTS_DIR / "kaggle" / "20260917T035646Z" / "artifacts" / "checkpoint_best.pt"

# ---------------------------------------------------------------------------
# Copy (verbatim per PROJECT_BRIEF.md Phase 6 -- "diagnosis" appears nowhere)
# ---------------------------------------------------------------------------
EMPTY_STATE_TEXT = "Upload an OCT B-scan to begin."
ERROR_STATE_TEXT = "That file isn't a readable image. Upload a JPEG or PNG B-scan."
FOOTER_TEXT = "Screening aid for triage. Not a diagnostic device. All findings require ophthalmologist review."

VIEW_CHOICES = ["original", "preprocessed", "heatmap", "fluid mask"]


# ---------------------------------------------------------------------------
# Model loading -- once at startup, threshold/temperature read from the
# checkpoint, never hardcoded.
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: Path):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[app] loading checkpoint: {checkpoint_path}")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Pass --checkpoint pointing at a checkpoint_best.pt that evaluate.py has already "
            "run on (so it carries a real threshold/temperature, not None)."
        )
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    model_name = ckpt.get("model_name", MODEL_NAME)
    model = build_model(model_name=model_name).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    threshold = ckpt.get("threshold")
    temperature = ckpt.get("temperature")
    if threshold is None or temperature is None:
        raise RuntimeError(
            f"{checkpoint_path} has no threshold/temperature (Phase 3c/3e not yet run on it). "
            "Refusing to invent a default operating point -- run evaluate.py on this checkpoint "
            "first, then relaunch the app."
        )

    gradcam_available = hasattr(model, "conv_head")
    print(
        f"[app] model={model_name} device={device} threshold={threshold:.4f} "
        f"temperature={temperature:.4f} gradcam_available={gradcam_available}"
    )
    return {
        "model": model,
        "device": device,
        "threshold": float(threshold),
        "temperature": float(temperature),
        "model_name": model_name,
        "gradcam_available": gradcam_available,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_screening(ctx: dict, bgr: np.ndarray) -> dict:
    """
    ctx: the dict returned by load_model(). bgr: an image array as
    cv2.imread returns it (BGR if 3-channel), matching the convention every
    other script in this repo uses -- preprocess_image() assumes the same.
    """
    import torch

    preprocessed_rgb, _content_bbox, fallback_triggered = preprocess_image(
        bgr, image_size=config.IMAGE_SIZE, flatten=config.RETINAL_FLATTENING_ENABLED
    )
    original_rgb = (
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    )

    transform = _eval_transform()
    tensor = transform(image=preprocessed_rgb)["image"].unsqueeze(0).to(ctx["device"])

    model = ctx["model"]
    with torch.no_grad():
        logits = model(tensor).cpu().numpy()

    # Calibrated probability vs the tuned threshold -- this is what would actually
    # ship (same convention as evaluate.py's "Test @ tuned" block; see its module
    # docstring for why calibration happens before thresholding, not after).
    prob = float(_softmax_positive(logits / ctx["temperature"])[0])
    prediction = "DME" if prob >= ctx["threshold"] else "Normal"

    heatmap = None
    if ctx["gradcam_available"]:
        from explain import generate_cam

        target_class = 1 if prediction == "DME" else 0
        try:
            heatmap = generate_cam(model, tensor, target_class)
        except Exception as exc:  # surfaced via heatmap=None -> UI message, never a crash
            print(f"[app] Grad-CAM generation failed: {exc}")
            heatmap = None

    return {
        "original_rgb": original_rgb,
        "preprocessed_rgb": preprocessed_rgb,
        "heatmap": heatmap,
        "probability": prob,
        "prediction": prediction,
        "fallback_triggered": fallback_triggered,
    }


def render_view(result: dict, view: str, opacity: float):
    """Returns the numpy image to display for the selected view, or None (fluid mask -- not built)."""
    if result is None:
        return None
    if view == "original":
        return result["original_rgb"]
    if view == "preprocessed":
        return result["preprocessed_rgb"]
    if view == "heatmap":
        if result["heatmap"] is None:
            return result["preprocessed_rgb"]
        from pytorch_grad_cam.utils.image import show_cam_on_image

        # opacity: 0 = pure preprocessed image, 100 = pure heatmap. image_weight is
        # pytorch_grad_cam's own blend parameter (weight of the ORIGINAL image), so invert.
        image_weight = 1.0 - (float(opacity) / 100.0)
        return show_cam_on_image(
            result["preprocessed_rgb"].astype(np.float32) / 255.0,
            result["heatmap"],
            use_rgb=True,
            image_weight=image_weight,
        )
    if view == "fluid mask":
        return None
    return result["original_rgb"]


# ---------------------------------------------------------------------------
# HTML rendering -- the confidence scale is a manually-rendered gr.HTML(),
# per the brief's Implementation section.
# ---------------------------------------------------------------------------


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def render_header_html(scan_id: Optional[str]) -> str:
    today = dt.date.today()
    date_str = f"{today.day} {today.strftime('%b %Y')}"
    meta = "Upload a scan to begin" if scan_id is None else f"Scan {scan_id}&nbsp;&nbsp;·&nbsp;&nbsp;{date_str}"
    return f'''
    <div id="app-header">
      <span class="title">DME screening</span>
      <span class="meta">{meta}</span>
    </div>
    '''


def render_label_html(prediction: str, prob: float) -> str:
    if prediction == "DME":
        text, color = "DME detected", "var(--signal-high)"
    else:
        text, color = "Within normal limits", "var(--signal-none)"
    return f'<div class="result-label" style="color:{color}">{text}</div>'


def render_confidence_html(prob: float, threshold: float) -> str:
    prob_pct = max(0.0, min(1.0, prob)) * 100
    thr_pct = max(0.0, min(1.0, threshold)) * 100
    marker_color = "var(--signal-high)" if prob >= threshold else "var(--signal-none)"
    return f'''
    <div class="confidence-block">
      <div class="confidence-row">
        <span class="confidence-label">Confidence</span>
        <span class="confidence-value">{_fmt(prob)}</span>
      </div>
      <div class="confidence-track">
        <div class="confidence-fill" style="width:{prob_pct:.2f}%"></div>
        <div class="confidence-threshold-tick" style="left:{thr_pct:.2f}%"></div>
        <div class="confidence-marker" style="left:{prob_pct:.2f}%; background:{marker_color}"></div>
      </div>
      <div class="confidence-scale-labels">
        <span class="scale-min">0</span>
        <span class="threshold-label" style="left:{thr_pct:.2f}%">{_fmt(threshold)}</span>
        <span class="scale-max">1.0</span>
      </div>
    </div>
    '''


def render_extended_html() -> str:
    # Phase 4/5 not implemented -- explicit placeholders, never a fabricated value.
    return '''
    <div class="extended-row">
      <span class="extended-label">Thickening index</span>
      <span class="extended-value placeholder">not available (Phase 5 pending)</span>
    </div>
    <div class="extended-row">
      <span class="extended-label">Centre involvement</span>
      <span class="extended-value placeholder">not available (Phase 5 pending)</span>
    </div>
    '''


def render_referral_html(prediction: str, prob: float, threshold: float, margin: float) -> str:
    # Icons are distinct glyph shapes, not just colour -- greyscale/colour-blind legible
    # per the acceptance criterion.
    if prediction == "DME":
        state, icon, text = "dme", "&#10007;", "DME detected, centre involvement indeterminate &mdash; refer"
    elif prob >= threshold - margin:
        state, icon, text = "borderline", "!", "Below confidence threshold &mdash; refer for manual review"
    else:
        state, icon, text = "normal", "&#10003;", "Within normal limits &mdash; no referral"
    return f'''
    <div class="referral-state referral-{state}">
      <span class="referral-icon" aria-hidden="true">{icon}</span>
      <span class="referral-text">{text}</span>
    </div>
    '''


def render_results_html(result: dict, threshold: float) -> str:
    if result is None:
        return f'<div class="empty-state">{EMPTY_STATE_TEXT}</div>'
    label_html = render_label_html(result["prediction"], result["probability"])
    confidence_html = render_confidence_html(result["probability"], threshold)
    extended_html = render_extended_html()
    referral_html = render_referral_html(result["prediction"], result["probability"], threshold, LOW_CONFIDENCE_MARGIN)
    fallback_note = (
        '<div class="fallback-note">Retinal flattening fell back to the unflattened image for this scan '
        "(automatic quality guard, not an error).</div>"
        if result.get("fallback_triggered")
        else ""
    )
    return f'<div class="fade-in">{label_html}{confidence_html}{extended_html}{referral_html}{fallback_note}</div>'


# ---------------------------------------------------------------------------
# CSS -- OCT-report idiom per the brief: printed-white page, grayscale scan
# as the dark anchor, two semantic colours used only on state indicators.
# Explicitly NOT: dark dashboard with one bright accent, rounded-card grid,
# gradient washes, all-caps eyebrow labels.
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
:root {
  --paper: #FCFCFA;
  --ink: #16181C;
  --graphite: #5A6069;
  --rule: #DFE1E0;
  --scan-bg: #0A0C0F;
  --signal-high: #C0392B;
  --signal-none: #1B7F5A;
  --signal-defer: #B8860B;
}

/* Defensive reset -- gr.HTML() content sits inside Gradio's own base
   stylesheet, which can impose its own box-sizing/display defaults on bare
   divs. Pin these explicitly so the confidence bar's percentage-based
   width/left offsets are computed against a predictable box model. */
.confidence-track, .confidence-track * {
  box-sizing: border-box;
  display: block;
  flex: none;
}

.gradio-container {
  background: var(--paper) !important;
  color: var(--ink) !important;
  font-family: 'Inter', 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  font-variant-numeric: tabular-nums;
  max-width: 1280px !important;
}

#app-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  border-bottom: 1px solid var(--rule);
  padding-bottom: 12px;
  margin-bottom: 18px;
}
#app-header .title { font-size: 20px; font-weight: 600; color: var(--ink); }
#app-header .meta { font-size: 13px; color: var(--graphite); font-variant-numeric: tabular-nums; }

#scan-display { background: var(--scan-bg) !important; border-radius: 2px !important; }
#scan-display img { object-fit: contain !important; }

.result-label {
  font-size: 32px;
  font-weight: 600;
  line-height: 1.2;
  font-variant-numeric: tabular-nums;
  margin-bottom: 14px;
}

.confidence-block { margin-bottom: 18px; }
.confidence-row { display: flex; justify-content: space-between; font-size: 15px; margin-bottom: 8px; }
.confidence-label { color: var(--graphite); }
.confidence-value { font-variant-numeric: tabular-nums; font-weight: 600; color: var(--ink); }
.confidence-track {
  position: relative;
  height: 8px;
  background: var(--rule);
  border-radius: 2px;
  margin: 10px 2px 6px 2px;
}
.confidence-fill {
  position: absolute; left: 0; top: 0; height: 100%;
  background: var(--graphite);
  border-radius: 2px;
}
.confidence-threshold-tick {
  position: absolute; top: -5px; width: 2px; height: 18px;
  background: var(--ink);
  transform: translateX(-1px);
}
.confidence-marker {
  position: absolute; top: -6px; width: 10px; height: 20px;
  border-radius: 2px;
  transform: translateX(-5px);
  border: 2px solid var(--paper);
  box-shadow: 0 0 0 1px var(--ink);
}
.confidence-scale-labels {
  position: relative; height: 16px; font-size: 13px; color: var(--graphite);
  font-variant-numeric: tabular-nums;
}
.confidence-scale-labels .scale-min { position: absolute; left: 0; }
.confidence-scale-labels .scale-max { position: absolute; right: 0; }
.threshold-label { position: absolute; transform: translateX(-50%); font-weight: 600; color: var(--ink); }

.extended-row {
  display: flex; justify-content: space-between; align-items: baseline;
  font-size: 15px; padding: 8px 0; border-top: 1px solid var(--rule);
}
.extended-label { color: var(--graphite); }
.extended-value { font-variant-numeric: tabular-nums; }
.extended-value.placeholder { color: var(--graphite); font-style: italic; font-size: 13px; }

.referral-state {
  display: flex; align-items: center; gap: 10px;
  padding: 12px 14px;
  background: var(--paper);
  border: 1px solid var(--rule);
  border-left: 4px solid currentColor;
  margin: 18px 0 14px 0;
  font-size: 15px;
}
.referral-dme { color: var(--signal-high); }
.referral-borderline { color: var(--signal-defer); }
.referral-normal { color: var(--signal-none); }
.referral-icon { font-size: 16px; font-weight: 700; width: 18px; text-align: center; }
.referral-text { color: var(--ink); }

.fallback-note { font-size: 13px; color: var(--graphite); margin-top: 8px; }

.footer { font-size: 13px; color: var(--graphite); border-top: 1px solid var(--rule); padding-top: 12px; margin-top: 22px; }

.empty-state { font-size: 15px; color: var(--graphite); padding: 48px 0; text-align: center; }
.empty-state.error { color: var(--signal-high); }

@media (prefers-reduced-motion: no-preference) {
  .fade-in { animation: dme-fadein 200ms ease-out; }
}
@keyframes dme-fadein { from { opacity: 0; } to { opacity: 1; } }

button:focus-visible, input:focus-visible, [tabindex]:focus-visible {
  outline: 2px solid var(--ink) !important;
  outline-offset: 2px !important;
}
"""

# Left/right arrow cycles the view radio; space toggles heatmap<->original.
# Client-side only (no Python round trip needed to change the selection --
# clicking the underlying radio input fires Gradio's own change event, which
# re-renders the view exactly as a mouse click would).
KEYBOARD_NAV_JS = """
() => {
  function getRadios() {
    const container = document.getElementById('view-radio');
    if (!container) return [];
    return Array.from(container.querySelectorAll('input[type=radio]'));
  }
  document.addEventListener('keydown', (e) => {
    const tag = (e.target && e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea') return;
    const radios = getRadios();
    if (radios.length === 0) return;
    let idx = radios.findIndex((r) => r.checked);
    if (idx === -1) idx = 0;
    if (e.key === 'ArrowRight') {
      e.preventDefault();
      radios[(idx + 1) % radios.length].click();
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      radios[(idx - 1 + radios.length) % radios.length].click();
    } else if (e.key === ' ' || e.code === 'Space') {
      e.preventDefault();
      const current = radios[idx];
      const targetValue = current && current.value === 'heatmap' ? 'original' : 'heatmap';
      const target = radios.find((r) => r.value === targetValue);
      if (target) target.click();
    }
  });
}
"""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def build_demo(ctx: dict) -> gr.Blocks:
    threshold = ctx["threshold"]

    def on_upload(path):
        if not path:
            return None, None, render_header_html(None), render_results_html(None, threshold), gr.update(visible=False)
        bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            return (
                None,
                None,
                render_header_html(None),
                f'<div class="empty-state error">{ERROR_STATE_TEXT}</div>',
                gr.update(visible=False),
            )
        try:
            result = run_screening(ctx, bgr)
        except Exception as exc:
            print(f"[app] inference failed: {exc}")
            return (
                None,
                None,
                render_header_html(None),
                f'<div class="empty-state error">Could not process this image: {exc}</div>',
                gr.update(visible=False),
            )
        scan_id = Path(path).stem[:24]
        display = render_view(result, "original", 50)
        # Diagnostic: the confidence bar and the printed number below must always
        # agree, since render_results_html derives both from this same value --
        # logged here so a mismatch on screen is verifiable against the real
        # computed probability, not guessed at from a screenshot.
        print(
            f"[app] scan={scan_id} prediction={result['prediction']} "
            f"probability={result['probability']:.6f} threshold={threshold:.6f}"
        )
        return result, display, render_header_html(scan_id), render_results_html(result, threshold), gr.update(visible=False)

    def on_view_change(result, view, opacity):
        img = render_view(result, view, opacity)
        return img, gr.update(visible=(view == "fluid mask"))

    with gr.Blocks(title="DME screening") as demo:
        state = gr.State(None)

        header_html = gr.HTML(render_header_html(None))

        with gr.Row():
            with gr.Column(scale=6):
                image_input = gr.Image(
                    type="filepath", label="Upload OCT B-scan", sources=["upload"], elem_id="upload-image"
                )
                display_image = gr.Image(
                    type="numpy", show_label=False, interactive=False, elem_id="scan-display"
                )
                opacity_slider = gr.Slider(0, 100, value=50, step=1, label="Heatmap opacity")
                view_radio = gr.Radio(
                    choices=VIEW_CHOICES, value="original", label="View", elem_id="view-radio"
                )
                fluid_note = gr.HTML(
                    '<div class="extended-value placeholder">Segmentation mask not available (Phase 4 pending).</div>',
                    visible=False,
                )
            with gr.Column(scale=5):
                results_html = gr.HTML(render_results_html(None, threshold))
                gr.HTML(f'<div class="footer">{FOOTER_TEXT}</div>')

        image_input.change(
            on_upload,
            inputs=[image_input],
            outputs=[state, display_image, header_html, results_html, fluid_note],
        )
        view_radio.change(
            on_view_change, inputs=[state, view_radio, opacity_slider], outputs=[display_image, fluid_note]
        )
        opacity_slider.change(
            on_view_change, inputs=[state, view_radio, opacity_slider], outputs=[display_image, fluid_note]
        )

        demo.load(fn=None, inputs=None, outputs=None, js=KEYBOARD_NAV_JS)

    return demo


def main():
    parser = argparse.ArgumentParser(description="Phase 6: DME screening interface.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="checkpoint_best.pt to serve (must already carry threshold/temperature from evaluate.py).")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    ctx = load_model(args.checkpoint)
    demo = build_demo(ctx)
    demo.launch(server_port=args.server_port, share=args.share, theme=gr.themes.Base(), css=CUSTOM_CSS)


if __name__ == "__main__":
    main()
