"""
Phase 3 (stage 1): evaluation, threshold tuning, calibration, and the A1
test-time-augmentation ablation.

Scope note (see PROJECT_BRIEF.md Section 5, Phase 3): this file covers 3c
(threshold tuning), 3d (single test-set evaluation at both thresholds), 3e
(temperature-scaling calibration), and A1 (TTA). Grad-CAM (3f) and the
Baselines/Ablation retraining tables (3b/3g, which need more Kaggle GPU
runs) are deliberately NOT here -- they're a separate follow-up, agreed
with the team before writing any of this.

Design decision, stated up front per the team's "explain before coding"
rule (not left implicit in the code):
  - "Test @ 0.5" uses RAW (uncalibrated) softmax probabilities -- this is
    the model's literal default behaviour (argmax over logits), untouched
    by anything this script decides.
  - "Test @ tuned" uses CALIBRATED probabilities (temperature fit on
    validation) and the sensitivity-targeted threshold swept on validation.
    This is what would actually ship, so it's evaluated the way it would
    actually be used.
  - Calibration doesn't change ROC/AUC or which (sensitivity, specificity)
    operating points are reachable (temperature scaling is a monotonic
    transform of the positive-class probability), it only changes which
    raw probability VALUE corresponds to a given operating point -- hence
    threshold tuning happens on the calibrated probabilities, since that's
    the probability scale actually used at decision time.
  - The threshold sweep is done on the VALIDATION set only (3c); the test
    set is touched exactly once, in the 3d block below (never re-run,
    never used to pick a better-looking number -- see PROJECT_BRIEF.md C7
    / Section 10.5 "no fabricated output, no test-set tuning").

Everything above the "torch-dependent" marker is pure NumPy/SciPy and is
unit-tested directly (tests/test_phase3.py) without needing a live model.
Below that marker, functions need torch + the trained model and are
exercised in tests with a tiny real (untrained) EfficientNet checkpoint,
not synthetic data -- there's no way to test "does inference actually run"
any other way.

Usage:
    python evaluate.py --train-dir data/raw/OCT2017/train \\
        --test-dir data/raw/OCT2017/test \\
        --checkpoint artifacts/kaggle/<run>/artifacts/checkpoint_best.pt \\
        --tta
"""

import argparse
import json
from pathlib import Path

import numpy as np

import config
from utils import PreprocessCache, build_splits, load_class_folder_df, set_seed

# ---------------------------------------------------------------------------
# Pure NumPy/SciPy helpers -- no torch import here, testable standalone.
# ---------------------------------------------------------------------------


def _softmax_positive(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax of an (N, 2) logits array, returning column 1 (DME, C5 ordering)."""
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    softmax = exp / exp.sum(axis=1, keepdims=True)
    return softmax[:, 1]


def compute_metrics(labels, probs, threshold: float) -> dict:
    """
    Accuracy/sensitivity/specificity/precision/F1/AUC-ROC at a given
    probability threshold. AUC doesn't depend on `threshold` (it's a
    ranking metric) but is included here so every row of the 3d table is
    self-contained.
    """
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())

    accuracy = (tp + tn) / max(1, len(labels))
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
    auc = roc_auc_score(labels, probs) if len(set(labels.tolist())) > 1 else float("nan")

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auc": float(auc),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "n": len(labels),
    }


def sweep_threshold(labels, probs, sensitivity_target: float, n_thresholds: int = 1001):
    """
    3c: sweep candidate thresholds over [0, 1] and return the one with the
    highest specificity among those meeting `sensitivity_target`.

    Always satisfiable on real data: threshold=0.0 predicts every image
    positive, giving sensitivity=1.0 >= any target <= 1.0. Raises instead
    of silently returning a threshold that misses the target, since a
    caller silently getting the wrong number here is exactly the kind of
    thing this project's "never fabricate/never silently degrade" rule
    exists to catch.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs, dtype=float)
    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    best = None
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        sensitivity = tp / max(1, tp + fn)
        specificity = tn / max(1, tn + fp)
        if sensitivity >= sensitivity_target and (best is None or specificity > best[2]):
            best = (float(t), sensitivity, specificity)

    if best is None:
        raise AssertionError(
            f"No threshold in [0, 1] reached sensitivity_target={sensitivity_target} -- "
            "this should be impossible on real labels/probs (threshold=0.0 always gives "
            "sensitivity=1.0). Check that `labels` actually contains positive (DME) cases."
        )
    return best  # (threshold, sensitivity, specificity)


def expected_calibration_error(labels, probs, n_bins: int = 10):
    """
    3e: standard equal-width-bin ECE. Returns (ece, bin_details) where
    bin_details is a list of per-bin {lo, hi, count, confidence, accuracy}
    dicts, used to draw the reliability diagram -- kept separate from the
    scalar so the figure and the number are provably built from the same
    binning, not recomputed twice and allowed to drift apart.
    """
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    n = len(labels)
    ece = 0.0
    bin_details = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs <= hi) if i == n_bins - 1 else (probs >= lo) & (probs < hi)
        count = int(mask.sum())
        if count == 0:
            bin_details.append({"lo": float(lo), "hi": float(hi), "count": 0, "confidence": None, "accuracy": None})
            continue
        confidence = float(probs[mask].mean())
        accuracy = float(labels[mask].mean())
        ece += (count / n) * abs(accuracy - confidence)
        bin_details.append({"lo": float(lo), "hi": float(hi), "count": count, "confidence": confidence, "accuracy": accuracy})

    return float(ece), bin_details


def fit_temperature(logits, labels, bounds=(0.05, 10.0)) -> float:
    """
    3e: fit a single scalar temperature T minimizing the NLL of
    softmax(logits / T) against `labels`, on the VALIDATION set only (the
    test set must never be used to fit anything, only to report the
    result once). Bounded scalar minimization (not gradient descent on a
    torch tensor) because T is one number and this needs to run without a
    GPU or even torch being present at all.
    """
    from scipy.optimize import minimize_scalar

    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)

    def nll(temperature: float) -> float:
        scaled = logits / temperature
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        softmax = exp / exp.sum(axis=1, keepdims=True)
        correct_probs = np.clip(softmax[np.arange(len(labels)), labels], 1e-12, 1.0)
        return float(-np.mean(np.log(correct_probs)))

    result = minimize_scalar(nll, bounds=bounds, method="bounded")
    return float(result.x)


def resolve_output_dir(artifacts_dir: Path, model_name: str, primary_model_name: str) -> Path:
    """
    Where metrics.json/figures for this evaluate.py run get written.
    Namespaces by model so a Baseline (Phase 3g: EfficientNet-B0, ResNet-50)
    run can never silently overwrite the primary model's already-signed-off
    metrics.json/gradcam figures -- both would otherwise land at the same
    --artifacts-dir path by default. The primary model keeps writing to the
    top-level artifacts_dir unchanged, so every path already referenced in
    the Phase 3 stage 1 / 3f sign-off stays valid. Does NOT create the
    directory -- callers create it if/when they're about to write into it.
    """
    if model_name == primary_model_name:
        return artifacts_dir
    return artifacts_dir / "baselines" / model_name


def tta_variants(rgb: np.ndarray) -> list:
    """
    A1: original + horizontal flip + +/-5 degree rotation. Deliberately no
    vertical flip, ever (C4) -- retinal layer order is anatomically fixed,
    same rule train.py's `_train_augmentation` follows.
    """
    import cv2

    h, w = rgb.shape[:2]
    center = (w / 2, h / 2)
    variants = [rgb, cv2.flip(rgb, 1)]
    for angle in (5, -5):
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(rgb, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        variants.append(rotated)
    return variants


# ---------------------------------------------------------------------------
# Torch-dependent (model inference, checkpoint I/O, orchestration)
# ---------------------------------------------------------------------------


def run_inference(model, df, cache: PreprocessCache, class_to_idx: dict, device, batch_size: int = 32, num_workers: int = 0):
    """
    Batched forward pass over `df` using the same eval transform training
    uses (no augmentation, just Normalize + ToTensor). Returns
    (logits (N, 2) ndarray, labels (N,) ndarray) -- raw logits, not
    softmax, since callers need both the calibration fit (needs logits)
    and plain probabilities (derived from them via _softmax_positive).
    """
    import torch
    from torch.utils.data import DataLoader

    from train import OCTDataset

    loader = DataLoader(
        OCTDataset(df, cache, class_to_idx, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device)).cpu().numpy()
            all_logits.append(logits)
            all_labels.extend(labels.numpy().tolist())
    return np.concatenate(all_logits, axis=0), np.array(all_labels)


def run_inference_tta(model, df, cache: PreprocessCache, class_to_idx: dict, device):
    """
    A1: per-image (unbatched) forward pass averaging logits across
    tta_variants(). Deliberately not batched like run_inference -- the
    four variants per image would need equal-sized batches of augmented
    copies, and this ablation only ever runs on the 484-image test set, so
    the simpler, obviously-correct loop is worth the speed cost.
    """
    import torch

    from train import _eval_transform

    transform = _eval_transform()
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for _, row in df.iterrows():
            rgb, _content_bbox, _fallback = cache.get_or_compute(row["path"])
            variant_logits = [
                model(transform(image=variant)["image"].unsqueeze(0).to(device)).cpu().numpy()[0]
                for variant in tta_variants(rgb)
            ]
            all_logits.append(np.mean(variant_logits, axis=0))
            all_labels.append(class_to_idx[row["label"]])
    return np.array(all_logits), np.array(all_labels)


def _save_figures(artifacts_dir: Path, test_labels, test_probs_raw, test_probs_calibrated,
                   metrics_at_default: dict, metrics_at_tuned: dict,
                   bins_before: list, bins_after: list, training_log_csv: Path = None) -> None:
    """
    150dpi figures per PROJECT_BRIEF.md Section 5 3g: confusion matrices
    (both thresholds, side by side), ROC with AUC annotated, reliability
    diagrams before/after calibration, and training curves IF a real
    training_log.csv was supplied (never fabricated placeholder curves --
    skipped with a printed note if not available).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # --- confusion matrices, @0.5 (raw) and @tuned (calibrated), side by side ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, metrics, title in (
        (axes[0], metrics_at_default, f"Test @ 0.5 (threshold={metrics_at_default['threshold']:.3f})"),
        (axes[1], metrics_at_tuned, f"Test @ tuned (threshold={metrics_at_tuned['threshold']:.3f})"),
    ):
        cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        ax.set_xticks([0, 1], ["Normal", "DME"])
        ax.set_yticks([0, 1], ["Normal", "DME"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(artifacts_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    # --- ROC curve (raw probabilities -- AUC is threshold/calibration-invariant) ---
    fpr, tpr, _ = roc_curve(test_labels, test_probs_raw)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {metrics_at_default['auc']:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC -- test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(artifacts_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    # --- reliability diagrams, before/after calibration ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, bins, title in ((axes[0], bins_before, "Before calibration"), (axes[1], bins_after, "After calibration")):
        confidences = [b["confidence"] for b in bins if b["count"] > 0]
        accuracies = [b["accuracy"] for b in bins if b["count"] > 0]
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.bar(confidences, accuracies, width=0.08, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(artifacts_dir / "reliability_diagrams.png", dpi=150)
    plt.close(fig)

    # --- training curves, only if the real log was supplied ---
    if training_log_csv is not None and Path(training_log_csv).exists():
        import pandas as pd

        log_df = pd.read_csv(training_log_csv)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].plot(log_df["global_epoch"], log_df["train_loss"], label="train_loss")
        axes[0].plot(log_df["global_epoch"], log_df["val_loss"], label="val_loss")
        axes[0].set_xlabel("Global epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].set_title("Loss")
        axes[1].plot(log_df["global_epoch"], log_df["val_auc"], label="val_auc", color="green")
        axes[1].set_xlabel("Global epoch")
        axes[1].set_ylabel("val_auc")
        axes[1].legend()
        axes[1].set_title("Validation AUC")
        fig.tight_layout()
        fig.savefig(artifacts_dir / "training_curves.png", dpi=150)
        plt.close(fig)
    else:
        print(
            "[evaluate] --training-log-csv not given (or not found) -- skipping training_curves.png "
            "rather than fabricating one. Pass --training-log-csv <path to the fetched training_log.csv> to include it."
        )


def main():
    import torch

    from train import assert_class_weight_alignment, build_class_to_idx, build_model, load_checkpoint, save_checkpoint

    parser = argparse.ArgumentParser(description="Phase 3 (stage 1): evaluation, threshold tuning, calibration, TTA ablation.")
    parser.add_argument("--train-dir", type=Path, required=True, help="Same --train-dir used for training, to reconstruct the identical val split (C1).")
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint_best.pt from Phase 2 training.")
    parser.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR, help="Where metrics.json and figures are written.")
    parser.add_argument("--cache-dir", type=Path, default=None,
                         help="Local PreprocessCache dir (default: <artifacts-dir>/preprocess_cache). A "
                              "Kaggle-downloaded preprocess_cache CANNOT be reused here -- its cache keys "
                              "are hashed from Kaggle container paths, not local ones -- so a fresh cache "
                              "builds automatically on first run.")
    parser.add_argument("--training-log-csv", type=Path, default=None, help="Real training_log.csv to plot (skipped, not fabricated, if omitted).")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows/CPU-only).")
    parser.add_argument("--tta", action="store_true", help="Also compute the A1 TTA ablation on the test set (slower).")
    args = parser.parse_args()

    set_seed(config.SEED)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.artifacts_dir / "preprocess_cache")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    class_to_idx = build_class_to_idx()
    assert_class_weight_alignment(class_to_idx)

    print(f"[evaluate] loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    # Baseline checkpoints (Phase 3g: EfficientNet-B0, ResNet-50) carry their own
    # model_name; older checkpoints from before that field existed (e.g. the
    # Phase 2 B3 run this project already signed off) fall back to train.MODEL_NAME,
    # which is what they were actually trained as.
    import train as _train_module

    model_name = ckpt.get("model_name", _train_module.MODEL_NAME)
    model = build_model(model_name=model_name).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[evaluate] checkpoint: model={model_name} phase={ckpt.get('phase')} global_epoch={ckpt.get('global_epoch')} val_auc={ckpt.get('val_auc')}")

    # cache_dir is NOT namespaced by model -- PreprocessCache's key is
    # (source_path, image_size, flatten) only, independent of architecture, so
    # the same cache is correctly shared and reused across every model
    # evaluated against the same images.
    output_dir = resolve_output_dir(args.artifacts_dir, model_name, _train_module.MODEL_NAME)
    if output_dir != args.artifacts_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[evaluate] model_name={model_name} != primary ({_train_module.MODEL_NAME}) -- writing outputs under {output_dir}")

    # Same seed + same build_splits() call as training -> identical patient-grouped
    # val split (C1), reconstructed rather than needing it saved to disk separately.
    result = build_splits(args.train_dir, test_dir=args.test_dir)
    val_df = result.val_df
    test_df = load_class_folder_df(args.test_dir)

    cache = PreprocessCache(cache_dir=cache_dir, image_size=config.IMAGE_SIZE)

    print(f"[evaluate] running inference on validation set ({len(val_df)} images)...")
    val_logits, val_labels = run_inference(model, val_df, cache, class_to_idx, device, num_workers=args.num_workers)
    val_probs_raw = _softmax_positive(val_logits)

    print(f"[evaluate] running inference on test set ({len(test_df)} images)...")
    test_logits, test_labels = run_inference(model, test_df, cache, class_to_idx, device, num_workers=args.num_workers)
    test_probs_raw = _softmax_positive(test_logits)

    # --- 3e: calibration, fit on validation only ---
    temperature = fit_temperature(val_logits, val_labels)
    val_probs_calibrated = _softmax_positive(val_logits / temperature)
    test_probs_calibrated = _softmax_positive(test_logits / temperature)

    ece_before, bins_before = expected_calibration_error(test_labels, test_probs_raw)
    ece_after, bins_after = expected_calibration_error(test_labels, test_probs_calibrated)
    print(f"[evaluate] temperature={temperature:.4f}  ECE before={ece_before:.4f} after={ece_after:.4f}")

    # --- 3c: threshold tuning, swept on validation (calibrated probabilities) ---
    tuned_threshold, tuned_val_sens, tuned_val_spec = sweep_threshold(val_labels, val_probs_calibrated, config.SENSITIVITY_TARGET)
    print(f"[evaluate] tuned threshold={tuned_threshold:.4f} (val sensitivity={tuned_val_sens:.4f} specificity={tuned_val_spec:.4f})")

    # --- 3d: test set touched exactly once, at both thresholds ---
    metrics_at_default = compute_metrics(test_labels, test_probs_raw, 0.5)
    metrics_at_tuned = compute_metrics(test_labels, test_probs_calibrated, tuned_threshold)
    print(f"[evaluate] Test @ 0.5:   acc={metrics_at_default['accuracy']:.4f} sens={metrics_at_default['sensitivity']:.4f} "
          f"spec={metrics_at_default['specificity']:.4f} AUC={metrics_at_default['auc']:.4f}")
    print(f"[evaluate] Test @ tuned: acc={metrics_at_tuned['accuracy']:.4f} sens={metrics_at_tuned['sensitivity']:.4f} "
          f"spec={metrics_at_tuned['specificity']:.4f} AUC={metrics_at_tuned['auc']:.4f}")

    # --- A1: TTA ablation, test set only ---
    tta_metrics = None
    if args.tta:
        print(f"[evaluate] running TTA ablation on test set ({len(test_df)} images, unbatched)...")
        tta_logits, tta_labels = run_inference_tta(model, test_df, cache, class_to_idx, device)
        tta_metrics = compute_metrics(tta_labels, _softmax_positive(tta_logits), 0.5)
        print(f"[evaluate] A1 TTA @ 0.5: acc={tta_metrics['accuracy']:.4f} sens={tta_metrics['sensitivity']:.4f} "
              f"spec={tta_metrics['specificity']:.4f} AUC={tta_metrics['auc']:.4f}  "
              f"(vs no-TTA acc={metrics_at_default['accuracy']:.4f}, AUC={metrics_at_default['auc']:.4f})")

    # --- write threshold/temperature back into the checkpoint (placeholders since Phase 2) ---
    ckpt["threshold"] = tuned_threshold
    ckpt["temperature"] = temperature
    save_checkpoint(args.checkpoint, ckpt)
    print(f"[evaluate] wrote threshold={tuned_threshold:.4f} temperature={temperature:.4f} back into {args.checkpoint}")

    metrics_payload = {
        "checkpoint": str(args.checkpoint),
        "model_name": model_name,
        "val_n": len(val_df),
        "test_n": len(test_df),
        "temperature": temperature,
        "tuned_threshold": tuned_threshold,
        "tuned_threshold_val_sensitivity": tuned_val_sens,
        "tuned_threshold_val_specificity": tuned_val_spec,
        "test_at_default_threshold": metrics_at_default,
        "test_at_tuned_threshold": metrics_at_tuned,
        "calibration": {"ece_before": ece_before, "ece_after": ece_after, "bins_before": bins_before, "bins_after": bins_after},
    }
    if tta_metrics is not None:
        metrics_payload["tta_ablation"] = tta_metrics

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[evaluate] wrote {metrics_path}")

    _save_figures(
        output_dir, test_labels, test_probs_raw, test_probs_calibrated,
        metrics_at_default, metrics_at_tuned, bins_before, bins_after,
        training_log_csv=args.training_log_csv,
    )
    print(f"[evaluate] wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
