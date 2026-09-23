"""
Phase 3f: Grad-CAM explainability + the model-parameter randomization
sanity check.

Grad-CAM targets model.conv_head. Per train.py's PHASE_A_UNFROZEN_MODULES
comment, EfficientNet-B3 here ends ... -> conv_head (1x1 conv) -> bn2 ->
global pool -> classifier (final FC to 2 classes). conv_head is the last
layer with spatial (H, W, C) structure -- Grad-CAM needs a spatial
activation map to turn into a heatmap, and classifier (post-pooling) has
none, so conv_head is the only layer downstream of the backbone that
makes sense as a target.

Design decision, stated up front per the team's "explain before coding"
rule: PROJECT_BRIEF.md Section 5 3f asks for a randomization check against
"the final layer's" weights. That's model.classifier -- the network's
actual last layer -- NOT conv_head (which is Grad-CAM's target layer for
reading activations, a different role). Randomizing classifier's weights
makes the backprop signal Grad-CAM uses to weight conv_head's activations
effectively random, so a healthy Grad-CAM implementation is expected to
produce a substantially different heatmap. This matches the literature
(Adebayo et al., "Sanity Checks for Saliency Maps": Grad-CAM is one of the
methods that DOES depend on the trained weights, unlike e.g. plain Guided
Backprop). If a real run here instead shows "did not change", that's a
genuine, reportable finding -- not something to explain away or re-run
until it looks right (PROJECT_BRIEF.md C7 / Section 10.5).

"Changed substantially" is operationalised as: the Pearson correlation
between the original and randomized-classifier CAM (flattened, per
example) drops below 0.5 -- less than half the variance shared. That
threshold is a judgment call, not a standard from the literature; it's
easy to change (see cam_correlation / the module-level constant below)
and the raw per-example correlations are always reported alongside the
verdict, never hidden behind the single yes/no call.

Usage:
    python explain.py --test-dir data/raw/OCT2017/test \\
        --checkpoint artifacts/kaggle/<run>/artifacts/checkpoint_best.pt
"""

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np

import config
from utils import PreprocessCache, load_class_folder_df, set_seed

RANDOMIZATION_CORRELATION_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Pure helpers -- no torch/grad-cam import here, testable standalone.
# ---------------------------------------------------------------------------


def select_grid_examples(labels, probs, threshold: float = 0.5) -> dict:
    """
    Pick one row index each for the 3x3 Grad-CAM grid's three rows: the
    most-confidently-correct DME image, the most-confidently-correct
    Normal image, and the single WORST-margin misclassification (not just
    any misclassification) -- the clearest case of the model being wrong,
    not an arbitrary pick.

    Indices are positions into labels/probs (0-based, matching
    run_inference's output order), not a DataFrame's own pandas index.
    Raises rather than silently substituting a different row when a
    category is empty (e.g. zero misclassifications at this threshold) --
    that's a real finding about the model, not something to paper over.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)

    dme_correct = np.where((labels == 1) & (preds == 1))[0]
    normal_correct = np.where((labels == 0) & (preds == 0))[0]
    misclassified = np.where(preds != labels)[0]

    if len(dme_correct) == 0:
        raise AssertionError("No correctly classified DME image found -- can't build the Grad-CAM grid's DME row.")
    if len(normal_correct) == 0:
        raise AssertionError("No correctly classified Normal image found -- can't build the Grad-CAM grid's Normal row.")
    if len(misclassified) == 0:
        raise AssertionError(
            "No misclassified test image found at this threshold -- the Grad-CAM grid's third row "
            "(misclassified) can't be built. This is a real finding (the model made zero mistakes at "
            "this threshold on this test set), not a bug -- report it, don't substitute a different row."
        )

    dme_idx = int(dme_correct[np.argmax(probs[dme_correct])])
    normal_idx = int(normal_correct[np.argmin(probs[normal_correct])])
    # "Worst" = probability furthest onto the wrong side of the threshold.
    margins = np.where(
        labels[misclassified] == 1,
        threshold - probs[misclassified],
        probs[misclassified] - threshold,
    )
    misclassified_idx = int(misclassified[np.argmax(margins)])

    return {"dme": dme_idx, "normal": normal_idx, "misclassified": misclassified_idx}


def cam_correlation(cam_a: np.ndarray, cam_b: np.ndarray) -> float:
    """
    Pearson correlation between two same-shaped Grad-CAM heatmaps,
    flattened. ~1.0 means the two maps are essentially the same pattern;
    low or negative means they're substantially different. Turns the
    randomization check into one reportable number per example instead of
    an eyeballed image comparison. Returns NaN if either map is constant
    (undefined correlation), rather than a misleading 0.0 or 1.0.
    """
    a, b = cam_a.flatten().astype(float), cam_b.flatten().astype(float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Torch-dependent (model, Grad-CAM library, checkpoint I/O, orchestration)
# ---------------------------------------------------------------------------


def randomize_final_layer(model):
    """
    Returns a DEEP COPY of `model` with only the final layer's
    (model.classifier) weights reinitialised -- the original `model` is
    left untouched, and every other layer (including conv_head, Grad-CAM's
    target layer) is identical between the two. See the module docstring
    for why classifier, not conv_head, is "the final layer" being
    randomised here.
    """
    import torch.nn as nn

    randomized = copy.deepcopy(model)
    layer = randomized.classifier
    if hasattr(layer, "reset_parameters"):
        layer.reset_parameters()
    else:
        # Fallback for a layer type without reset_parameters(): re-init like a fresh nn.Linear.
        nn.init.kaiming_uniform_(layer.weight, a=5**0.5)
        if getattr(layer, "bias", None) is not None:
            fan_in = layer.weight.shape[1]
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(layer.bias, -bound, bound)
    return randomized


def generate_cam(model, image_tensor, target_class: int) -> np.ndarray:
    """
    Grad-CAM heatmap (H, W), values in [0, 1], for `image_tensor`
    (1, C, H, W), targeting model.conv_head, explaining `target_class`
    (0=Normal, 1=DME per C5 class_to_idx ordering).
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model.eval()
    cam = GradCAM(model=model, target_layers=[model.conv_head])
    grayscale_cam = cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(target_class)])
    return grayscale_cam[0]


def load_raw_rgb(path) -> np.ndarray:
    """The untouched source image (no denoise/flatten/CLAHE/letterbox), for the grid's 'original' column."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def run_explain(model, test_df, cache: PreprocessCache, class_to_idx: dict, device, threshold: float, artifacts_dir: Path, selection: dict = None):
    """
    Builds and saves gradcam_grid.png (3x3: dme/normal/misclassified rows
    x original/preprocessed/heatmap-overlay columns), runs the
    randomization sanity check, saves gradcam_randomization_check.png
    (original vs randomized-classifier overlays side by side) and
    gradcam_randomization.json (per-example correlations + verdict).

    `selection` can be passed in directly (row indices into test_df) to
    skip re-running inference when the caller already has it (or, in
    tests, to force a specific deterministic choice); otherwise it's
    computed here by the caller before this function is invoked -- see
    main() for the real (inference-driven) path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from train import _eval_transform

    if selection is None:
        raise ValueError("selection is required (compute it via select_grid_examples() first)")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    transform = _eval_transform()
    randomized_model = randomize_final_layer(model)

    row_keys = ["dme", "normal", "misclassified"]
    row_titles = {"dme": "DME (correct)", "normal": "Normal (correct)", "misclassified": "Misclassified"}

    from pytorch_grad_cam.utils.image import show_cam_on_image

    fig, axes = plt.subplots(3, 3, figsize=(9, 9.5))
    fig_rand, axes_rand = plt.subplots(3, 2, figsize=(6.5, 9.5))
    correlations = {}

    for row_i, key in enumerate(row_keys):
        idx = selection[key]
        row = test_df.iloc[idx]
        raw_rgb = load_raw_rgb(row["path"])
        preprocessed_rgb, _content_bbox, _fallback = cache.get_or_compute(row["path"])
        tensor = transform(image=preprocessed_rgb)["image"].unsqueeze(0).to(device)

        # Every row explains the TRUE class, including the misclassified row --
        # the point is "what did the model attend to for the class it got wrong",
        # not "what did it attend to for the class it guessed".
        target_class = class_to_idx[row["label"]]

        cam_map = generate_cam(model, tensor, target_class)
        overlay = show_cam_on_image(preprocessed_rgb.astype(np.float32) / 255.0, cam_map, use_rgb=True)

        randomized_cam_map = generate_cam(randomized_model, tensor, target_class)
        randomized_overlay = show_cam_on_image(preprocessed_rgb.astype(np.float32) / 255.0, randomized_cam_map, use_rgb=True)
        correlations[key] = cam_correlation(cam_map, randomized_cam_map)

        for col_i, (img, col_title) in enumerate(
            [(raw_rgb, "original"), (preprocessed_rgb, "preprocessed"), (overlay, "Grad-CAM overlay")]
        ):
            ax = axes[row_i, col_i]
            ax.imshow(img)
            if row_i == 0:
                ax.set_title(col_title)
            if col_i == 0:
                ax.set_ylabel(row_titles[key], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        for col_i, (img, col_title) in enumerate([(overlay, "original weights"), (randomized_overlay, "randomized classifier")]):
            ax = axes_rand[row_i, col_i]
            ax.imshow(img)
            if row_i == 0:
                ax.set_title(col_title)
            if col_i == 0:
                ax.set_ylabel(row_titles[key], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    grid_path = artifacts_dir / "gradcam_grid.png"
    fig.savefig(grid_path, dpi=150)
    plt.close(fig)

    fig_rand.tight_layout()
    rand_path = artifacts_dir / "gradcam_randomization_check.png"
    fig_rand.savefig(rand_path, dpi=150)
    plt.close(fig_rand)

    mean_corr = float(np.nanmean(list(correlations.values())))
    changed_substantially = mean_corr < RANDOMIZATION_CORRELATION_THRESHOLD
    verdict = "changed substantially" if changed_substantially else "did not change"

    payload = {
        "selection": selection,
        "threshold_used_for_selection": threshold,
        "randomization_correlations": correlations,
        "randomization_mean_correlation": mean_corr,
        "randomization_correlation_threshold": RANDOMIZATION_CORRELATION_THRESHOLD,
        "randomization_verdict": verdict,
    }
    json_path = artifacts_dir / "gradcam_randomization.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    return {"grid_path": grid_path, "randomization_check_path": rand_path, "json_path": json_path, **payload}


def main():
    import torch

    from evaluate import _softmax_positive, run_inference
    from train import build_class_to_idx, build_model, load_checkpoint

    parser = argparse.ArgumentParser(description="Phase 3f: Grad-CAM + randomization sanity check.")
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint_best.pt from Phase 2 training (ideally already Phase-3c/e updated by evaluate.py).")
    parser.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Threshold used only to pick the misclassified example for the grid (default 0.5 -- "
             "illustrating a mistake, not reproducing deployment behaviour, so this is deliberately "
             "NOT the Phase 3c tuned threshold unless you pass it explicitly).",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(config.SEED)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.artifacts_dir / "preprocess_cache")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    class_to_idx = build_class_to_idx()
    print(f"[explain] loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_df = load_class_folder_df(args.test_dir)
    cache = PreprocessCache(cache_dir=cache_dir, image_size=config.IMAGE_SIZE)

    print(f"[explain] running inference on test set ({len(test_df)} images) to select grid examples...")
    logits, labels = run_inference(model, test_df, cache, class_to_idx, device, num_workers=args.num_workers)
    probs = _softmax_positive(logits)

    selection = select_grid_examples(labels, probs, threshold=args.threshold)
    print(f"[explain] selected rows (positions in test_df): {selection}")

    result = run_explain(model, test_df, cache, class_to_idx, device, args.threshold, args.artifacts_dir, selection=selection)

    print(f"[explain] wrote {result['grid_path']}")
    print(f"[explain] wrote {result['randomization_check_path']}")
    for key, corr in result["randomization_correlations"].items():
        print(f"[explain] randomization correlation ({key}): {corr:.4f}")
    print(f"[explain] mean correlation={result['randomization_mean_correlation']:.4f} "
          f"(threshold {RANDOMIZATION_CORRELATION_THRESHOLD}) -> verdict: {result['randomization_verdict']}")
    print(f"[explain] wrote {result['json_path']}")


if __name__ == "__main__":
    main()
