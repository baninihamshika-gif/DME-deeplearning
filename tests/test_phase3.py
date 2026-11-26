"""
Phase 3 (stage 1) tests for evaluate.py: threshold tuning (3c), test
evaluation (3d), calibration (3e), and the A1 TTA ablation.

Two tiers, mirroring test_phase2.py's split:
  - "Pure helpers": sweep_threshold, compute_metrics, expected_calibration_error,
    fit_temperature, tta_variants, _softmax_positive -- exercised against
    synthetic NumPy arrays, no torch/model needed.
  - "Torch-dependent": run_inference / run_inference_tta / _save_figures --
    exercised against a real (untrained, pretrained=False -- same reasoning
    as test_phase2.py's _build_model_no_pretrained_download: no route to
    HuggingFace Hub in this sandbox, and architecture/shape behaviour is
    identical either way) EfficientNet-B3 and small real image files, since
    "does inference actually run end-to-end" can't be checked any other way.

These do not and cannot substitute for running evaluate.py against the real
checkpoint and the real 484-image test set on the team's machine -- that is
the whole reason Phase 3 sign-off numbers must come from there, not from
this file (see PROJECT_BRIEF.md C7 / Section 10.5).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

import config
import train
from evaluate import (
    _save_figures,
    _softmax_positive,
    compute_metrics,
    expected_calibration_error,
    fit_temperature,
    resolve_output_dir,
    run_inference,
    run_inference_tta,
    sweep_threshold,
    tta_variants,
)
from utils import PreprocessCache

# ---------------------------------------------------------------------------
# resolve_output_dir (Phase 3g: Baseline runs must never clobber the primary
# model's already-signed-off metrics.json/gradcam figures)
# ---------------------------------------------------------------------------


def test_resolve_output_dir_primary_model_uses_artifacts_dir_unchanged():
    artifacts_dir = Path("artifacts")
    assert resolve_output_dir(artifacts_dir, "tf_efficientnet_b3", primary_model_name="tf_efficientnet_b3") == artifacts_dir


def test_resolve_output_dir_baseline_model_gets_namespaced_subdir():
    artifacts_dir = Path("artifacts")
    result = resolve_output_dir(artifacts_dir, "resnet50", primary_model_name="tf_efficientnet_b3")
    assert result == artifacts_dir / "baselines" / "resnet50"
    assert result != artifacts_dir  # never collides with the primary model's output path


def test_resolve_output_dir_different_baselines_get_different_subdirs():
    artifacts_dir = Path("artifacts")
    b0_dir = resolve_output_dir(artifacts_dir, "tf_efficientnet_b0", primary_model_name="tf_efficientnet_b3")
    resnet_dir = resolve_output_dir(artifacts_dir, "resnet50", primary_model_name="tf_efficientnet_b3")
    assert b0_dir != resnet_dir  # two baselines evaluated in sequence can't clobber each other either


# ---------------------------------------------------------------------------
# _softmax_positive
# ---------------------------------------------------------------------------


def test_softmax_positive_matches_manual_softmax():
    logits = np.array([[0.0, 0.0], [1.0, 3.0], [5.0, 0.0]])
    probs = _softmax_positive(logits)
    expected = np.array(
        [0.5, np.exp(3.0) / (np.exp(1.0) + np.exp(3.0)), np.exp(0.0) / (np.exp(5.0) + np.exp(0.0))]
    )
    assert np.allclose(probs, expected, atol=1e-8)


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_perfect_classifier():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(labels, probs, threshold=0.5)
    assert m["accuracy"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["precision"] == 1.0
    assert m["f1"] == 1.0
    assert m["auc"] == 1.0
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (2, 0, 2, 0)


def test_compute_metrics_threshold_shifts_predictions():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.3, 0.4, 0.6, 0.9])
    # A very low threshold predicts everything positive: perfect sensitivity, zero specificity.
    m_low = compute_metrics(labels, probs, threshold=0.0)
    assert m_low["sensitivity"] == 1.0
    assert m_low["specificity"] == 0.0
    # A very high threshold predicts everything negative: zero sensitivity, perfect specificity.
    m_high = compute_metrics(labels, probs, threshold=1.0)
    assert m_high["sensitivity"] == 0.0
    assert m_high["specificity"] == 1.0
    # AUC is threshold-independent -- both should report the same ranking-based value.
    assert m_low["auc"] == m_high["auc"]


def test_compute_metrics_handles_single_class_auc_as_nan():
    labels = np.array([1, 1, 1])
    probs = np.array([0.6, 0.7, 0.9])
    m = compute_metrics(labels, probs, threshold=0.5)
    assert np.isnan(m["auc"])


# ---------------------------------------------------------------------------
# sweep_threshold (3c)
# ---------------------------------------------------------------------------


def test_sweep_threshold_meets_sensitivity_target():
    rng = np.random.RandomState(0)
    n = 2000
    labels = rng.randint(0, 2, size=n)
    # Probabilities correlated with the label but noisy, so there's a real
    # sensitivity/specificity trade-off to sweep over (not a degenerate 0/1 case).
    probs = np.clip(labels * 0.6 + rng.normal(0.3, 0.25, size=n), 0, 1)

    threshold, sensitivity, specificity = sweep_threshold(labels, probs, sensitivity_target=0.97)
    assert sensitivity >= 0.97
    # Re-deriving metrics at the returned threshold must reproduce the same sensitivity.
    m = compute_metrics(labels, probs, threshold)
    assert abs(m["sensitivity"] - sensitivity) < 1e-9
    assert abs(m["specificity"] - specificity) < 1e-9


def test_sweep_threshold_prefers_higher_specificity_among_qualifying():
    # Construct labels/probs where sensitivity=1.0 is reachable at more than
    # one threshold (a gap with no positives in it) -- the sweep must pick
    # the highest such threshold (best specificity), not just the first one found.
    labels = np.array([0, 0, 0, 1, 1])
    probs = np.array([0.05, 0.10, 0.15, 0.50, 0.90])
    threshold, sensitivity, specificity = sweep_threshold(labels, probs, sensitivity_target=1.0)
    assert sensitivity == 1.0
    # Best specificity achievable while keeping both positives (>=0.50) correctly
    # classified is by pushing the threshold as close to 0.50 as the grid allows,
    # which also pushes all three negatives below it -> specificity 1.0.
    assert specificity == 1.0
    assert 0.15 < threshold <= 0.50


def test_sweep_threshold_raises_when_target_unreachable_on_bad_input():
    # sensitivity_target > 1.0 can never be satisfied by any threshold, including 0.0.
    labels = np.array([0, 1])
    probs = np.array([0.3, 0.7])
    with pytest.raises(AssertionError):
        sweep_threshold(labels, probs, sensitivity_target=1.5)


# ---------------------------------------------------------------------------
# expected_calibration_error (3e)
# ---------------------------------------------------------------------------


def test_expected_calibration_error_zero_when_perfectly_calibrated():
    # 10 probabilities exactly at each bin center, with the fraction of
    # positives in each "bin" of copies matching the confidence exactly.
    probs = np.repeat(np.linspace(0.05, 0.95, 10), 20)
    labels = np.concatenate([np.array([1] * int(round(p * 20)) + [0] * (20 - int(round(p * 20)))) for p in np.linspace(0.05, 0.95, 10)])
    ece, bins = expected_calibration_error(labels, probs, n_bins=10)
    assert ece < 0.02  # rounding in the synthetic construction, not exactly 0
    assert len(bins) == 10
    assert sum(b["count"] for b in bins) == len(labels)


def test_expected_calibration_error_positive_when_miscalibrated():
    # Every prediction is confident (0.95) but half are wrong -> large ECE.
    probs = np.full(100, 0.95)
    labels = np.array([1] * 50 + [0] * 50)
    ece, bins = expected_calibration_error(labels, probs, n_bins=10)
    assert ece > 0.4


# ---------------------------------------------------------------------------
# fit_temperature (3e)
# ---------------------------------------------------------------------------


def test_fit_temperature_recovers_known_overconfidence():
    rng = np.random.RandomState(42)
    n = 3000
    true_logit_diff = rng.normal(0, 2.0, size=n)  # logit(DME) - logit(Normal), well-calibrated scale
    true_probs = 1 / (1 + np.exp(-true_logit_diff))
    labels = (rng.uniform(size=n) < true_probs).astype(int)

    true_logits = np.stack([np.zeros(n), true_logit_diff], axis=1)
    known_temperature = 3.0
    overconfident_logits = true_logits * known_temperature  # miscalibrated by a known factor

    fitted = fit_temperature(overconfident_logits, labels)
    assert abs(fitted - known_temperature) < 0.5


def test_fit_temperature_stays_within_bounds():
    labels = np.array([0, 1, 0, 1])
    logits = np.array([[0.0, 0.0]] * 4)  # degenerate: no signal at all
    fitted = fit_temperature(logits, labels, bounds=(0.05, 10.0))
    assert 0.05 <= fitted <= 10.0


# ---------------------------------------------------------------------------
# tta_variants (A1, C4)
# ---------------------------------------------------------------------------


def test_tta_variants_returns_four_including_original_and_hflip():
    rgb = np.random.randint(0, 255, size=(300, 300, 3), dtype=np.uint8)
    variants = tta_variants(rgb)
    assert len(variants) == 4
    assert np.array_equal(variants[0], rgb)  # original, unmodified
    assert np.array_equal(variants[1], cv2.flip(rgb, 1))  # horizontal flip
    for v in variants:
        assert v.shape == rgb.shape


def test_tta_variants_never_vertically_flips():
    # A distinctive top-row marker: if any variant is a pure vertical flip,
    # the marker would move to the bottom row. None of the 4 variants should
    # do that (C4 -- retinal layer order is anatomically fixed).
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[0, :, :] = 255  # marker on the top row only
    vertical_flip = cv2.flip(rgb, 0)
    for v in tta_variants(rgb):
        assert not np.array_equal(v, vertical_flip)


# ---------------------------------------------------------------------------
# Torch-dependent: run_inference / run_inference_tta / _save_figures
# ---------------------------------------------------------------------------


def _build_model_no_pretrained_download():
    # Same reasoning as test_phase2.py's helper of the same name: this
    # sandbox has no route to HuggingFace Hub, and shape/inference behaviour
    # is identical whether or not ImageNet weights are loaded.
    import timm

    return timm.create_model(train.MODEL_NAME, pretrained=False, num_classes=len(config.CLASS_NAMES), drop_rate=config.DROP_RATE)


def _write_real_image(path: Path, width: int, height: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.random.randint(0, 255, size=(height, width), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _build_tiny_test_df(root: Path, n_per_class: int = 3) -> pd.DataFrame:
    records = []
    for i in range(n_per_class):
        patient = 7_000_000 + i
        path = root / "DME" / f"DME-{patient}-1.jpeg"
        _write_real_image(path, width=320, height=280)
        records.append({"path": str(path), "label": "DME", "patient": str(patient)})
    for i in range(n_per_class):
        patient = 8_000_000 + i
        path = root / "NORMAL" / f"NORMAL-{patient}-1.jpeg"
        _write_real_image(path, width=320, height=280)
        records.append({"path": str(path), "label": "Normal", "patient": str(patient)})
    return pd.DataFrame.from_records(records)


def test_run_inference_returns_logits_and_labels_for_every_row(tmp_path):
    model = _build_model_no_pretrained_download()
    df = _build_tiny_test_df(tmp_path / "test")
    cache = PreprocessCache(cache_dir=tmp_path / "cache", image_size=config.IMAGE_SIZE)
    class_to_idx = train.build_class_to_idx()

    logits, labels = run_inference(model, df, cache, class_to_idx, torch.device("cpu"), batch_size=4, num_workers=0)

    assert logits.shape == (len(df), 2)
    assert labels.shape == (len(df),)
    assert set(labels.tolist()) == {0, 1}


def test_run_inference_tta_returns_logits_and_labels_for_every_row(tmp_path):
    model = _build_model_no_pretrained_download()
    df = _build_tiny_test_df(tmp_path / "test", n_per_class=2)
    cache = PreprocessCache(cache_dir=tmp_path / "cache", image_size=config.IMAGE_SIZE)
    class_to_idx = train.build_class_to_idx()

    logits, labels = run_inference_tta(model, df, cache, class_to_idx, torch.device("cpu"))

    assert logits.shape == (len(df), 2)
    assert labels.shape == (len(df),)


def test_run_inference_and_tta_agree_on_original_variant_only_run(tmp_path):
    # Sanity check the two inference paths aren't wired to fundamentally
    # different preprocessing: averaging TTA over just the identity variant
    # (monkeypatched) must reproduce plain run_inference's logits exactly.
    import evaluate as evaluate_module

    model = _build_model_no_pretrained_download()
    model.eval()
    df = _build_tiny_test_df(tmp_path / "test", n_per_class=2)
    cache = PreprocessCache(cache_dir=tmp_path / "cache", image_size=config.IMAGE_SIZE)
    class_to_idx = train.build_class_to_idx()
    device = torch.device("cpu")

    plain_logits, plain_labels = run_inference(model, df, cache, class_to_idx, device, batch_size=1, num_workers=0)

    original_tta_variants = evaluate_module.tta_variants
    evaluate_module.tta_variants = lambda rgb: [rgb]  # identity-only, for this test
    try:
        tta_logits, tta_labels = run_inference_tta(model, df, cache, class_to_idx, device)
    finally:
        evaluate_module.tta_variants = original_tta_variants

    assert np.array_equal(plain_labels, tta_labels)
    assert np.allclose(plain_logits, tta_logits, atol=1e-5)


def test_save_figures_writes_expected_files(tmp_path):
    rng = np.random.RandomState(0)
    n = 100
    labels = rng.randint(0, 2, size=n)
    probs_raw = np.clip(labels * 0.5 + rng.normal(0.25, 0.2, size=n), 0, 1)
    probs_cal = np.clip(labels * 0.5 + rng.normal(0.25, 0.15, size=n), 0, 1)

    metrics_default = compute_metrics(labels, probs_raw, 0.5)
    tuned_threshold, _sens, _spec = sweep_threshold(labels, probs_cal, sensitivity_target=0.8)
    metrics_tuned = compute_metrics(labels, probs_cal, tuned_threshold)
    _ece_before, bins_before = expected_calibration_error(labels, probs_raw)
    _ece_after, bins_after = expected_calibration_error(labels, probs_cal)

    artifacts_dir = tmp_path / "artifacts"
    _save_figures(artifacts_dir, labels, probs_raw, probs_cal, metrics_default, metrics_tuned, bins_before, bins_after)

    assert (artifacts_dir / "confusion_matrices.png").exists()
    assert (artifacts_dir / "roc_curve.png").exists()
    assert (artifacts_dir / "reliability_diagrams.png").exists()
    assert not (artifacts_dir / "training_curves.png").exists()  # no training_log_csv given -> skipped, not fabricated


def test_save_figures_includes_training_curves_when_log_given(tmp_path):
    log_path = tmp_path / "training_log.csv"
    pd.DataFrame(
        {
            "global_epoch": [1, 2, 3],
            "train_loss": [0.6, 0.4, 0.3],
            "val_loss": [0.7, 0.5, 0.35],
            "val_auc": [0.8, 0.9, 0.95],
        }
    ).to_csv(log_path, index=False)

    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = compute_metrics(labels, probs, 0.5)
    _ece, bins = expected_calibration_error(labels, probs)

    artifacts_dir = tmp_path / "artifacts"
    _save_figures(artifacts_dir, labels, probs, probs, metrics, metrics, bins, bins, training_log_csv=log_path)

    assert (artifacts_dir / "training_curves.png").exists()


# ---------------------------------------------------------------------------
# Checkpoint round trip: threshold/temperature written back without
# clobbering the rest of the payload (model_state_dict, phase, etc.)
# ---------------------------------------------------------------------------


def test_checkpoint_threshold_temperature_round_trip_preserves_other_fields(tmp_path):
    ckpt_path = tmp_path / "checkpoint_best.pt"
    original = {
        "model_state_dict": {"fake": torch.tensor([1.0, 2.0])},
        "phase": "B",
        "epoch_in_phase": 15,
        "global_epoch": 20,
        "val_auc": 0.9958,
        "threshold": None,
        "temperature": None,
    }
    train.save_checkpoint(ckpt_path, original)

    ckpt = train.load_checkpoint(ckpt_path)
    ckpt["threshold"] = 0.62
    ckpt["temperature"] = 1.8
    train.save_checkpoint(ckpt_path, ckpt)

    reloaded = train.load_checkpoint(ckpt_path)
    assert reloaded["threshold"] == 0.62
    assert reloaded["temperature"] == 1.8
    assert reloaded["phase"] == "B"
    assert reloaded["global_epoch"] == 20
    assert reloaded["val_auc"] == 0.9958
    assert torch.equal(reloaded["model_state_dict"]["fake"], original["model_state_dict"]["fake"])
