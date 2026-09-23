"""
Tests for explain.py (Phase 3f: Grad-CAM + randomization sanity check).

Same two-tier split as test_phase3.py: pure helpers (select_grid_examples,
cam_correlation) against synthetic arrays, then torch/grad-cam-dependent
pieces (randomize_final_layer, generate_cam, run_explain end-to-end)
against a real (untrained, pretrained=False -- see test_phase2.py's
_build_model_no_pretrained_download for why) EfficientNet-B3 and small
real image files. These verify the pipeline runs correctly and produces
self-consistent output; they cannot and do not substitute for running
explain.py against the real trained checkpoint on the team's machine,
where the actual Grad-CAM heatmaps and randomization verdict must come
from (PROJECT_BRIEF.md C7 -- never fabricate output).
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
from explain import (
    cam_correlation,
    generate_cam,
    load_raw_rgb,
    randomize_final_layer,
    run_explain,
    select_grid_examples,
)
from utils import PreprocessCache

# ---------------------------------------------------------------------------
# select_grid_examples
# ---------------------------------------------------------------------------


def test_select_grid_examples_picks_most_confident_correct_and_worst_miss():
    labels = np.array([1, 1, 1, 0, 0, 0])
    #                  DME:0.95(correct,most conf)  DME:0.6(correct)  DME:0.3(WRONG, worst margin)
    #                  Normal:0.05(correct,most conf)  Normal:0.2(correct)  Normal:0.7(WRONG)
    probs = np.array([0.95, 0.6, 0.3, 0.05, 0.2, 0.7])
    selection = select_grid_examples(labels, probs, threshold=0.5)

    assert selection["dme"] == 0  # highest-prob correctly classified DME
    assert selection["normal"] == 3  # lowest-prob correctly classified Normal
    # Two misclassifications: idx 2 (DME predicted Normal, margin 0.5-0.3=0.2) and
    # idx 5 (Normal predicted DME, margin 0.7-0.5=0.2) -- tied; either is a valid "worst".
    assert selection["misclassified"] in (2, 5)


def test_select_grid_examples_raises_when_no_misclassification():
    labels = np.array([1, 0])
    probs = np.array([0.9, 0.1])  # both correct at threshold 0.5
    with pytest.raises(AssertionError):
        select_grid_examples(labels, probs, threshold=0.5)


def test_select_grid_examples_raises_when_no_correct_dme():
    labels = np.array([1, 0])
    probs = np.array([0.1, 0.1])  # DME misclassified as Normal, Normal correct
    with pytest.raises(AssertionError):
        select_grid_examples(labels, probs, threshold=0.5)


def test_select_grid_examples_raises_when_no_correct_normal():
    labels = np.array([1, 0])
    probs = np.array([0.9, 0.9])  # DME correct, Normal misclassified as DME
    with pytest.raises(AssertionError):
        select_grid_examples(labels, probs, threshold=0.5)


# ---------------------------------------------------------------------------
# cam_correlation
# ---------------------------------------------------------------------------


def test_cam_correlation_identical_maps_is_one():
    rng = np.random.RandomState(0)
    a = rng.rand(20, 20)
    assert cam_correlation(a, a.copy()) == pytest.approx(1.0)


def test_cam_correlation_anticorrelated_maps_is_negative():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = -a
    assert cam_correlation(a, b) == pytest.approx(-1.0)


def test_cam_correlation_constant_map_is_nan():
    a = np.ones((5, 5))
    b = np.random.rand(5, 5)
    assert np.isnan(cam_correlation(a, b))


# ---------------------------------------------------------------------------
# Torch-dependent: randomize_final_layer / generate_cam / run_explain
# ---------------------------------------------------------------------------


def _build_model_no_pretrained_download():
    import timm

    return timm.create_model(train.MODEL_NAME, pretrained=False, num_classes=len(config.CLASS_NAMES), drop_rate=config.DROP_RATE)


def test_randomize_final_layer_only_changes_classifier():
    model = _build_model_no_pretrained_download()
    randomized = randomize_final_layer(model)

    # classifier differs...
    assert not torch.equal(model.classifier.weight, randomized.classifier.weight)
    # ...but conv_head (Grad-CAM's target layer) and everything else is untouched.
    assert torch.equal(model.conv_head.weight, randomized.conv_head.weight)
    # Original model itself must be unmodified (deep copy, not in-place).
    original_classifier_weight = model.classifier.weight.clone()
    randomize_final_layer(model)
    assert torch.equal(model.classifier.weight, original_classifier_weight)


def test_generate_cam_returns_valid_heatmap_shape_and_range():
    model = _build_model_no_pretrained_download()
    model.eval()
    tensor = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)

    cam_map = generate_cam(model, tensor, target_class=1)

    assert cam_map.shape == (config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert cam_map.min() >= 0.0 - 1e-6
    assert cam_map.max() <= 1.0 + 1e-6


def test_load_raw_rgb_reads_grayscale_as_three_channel(tmp_path):
    path = tmp_path / "img.jpeg"
    cv2.imwrite(str(path), np.random.randint(0, 255, size=(100, 120), dtype=np.uint8))
    rgb = load_raw_rgb(path)
    assert rgb.shape == (100, 120, 3)


def test_load_raw_rgb_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_rgb(tmp_path / "does_not_exist.jpeg")


def _write_real_image(path: Path, width: int, height: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.random.randint(0, 255, size=(height, width), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_run_explain_writes_grid_and_randomization_outputs(tmp_path):
    model = _build_model_no_pretrained_download()
    model.eval()
    class_to_idx = train.build_class_to_idx()

    test_root = tmp_path / "test"
    records = []
    dme_path = test_root / "DME" / "DME-7000001-1.jpeg"
    _write_real_image(dme_path, 320, 280)
    records.append({"path": str(dme_path), "label": "DME", "patient": "7000001"})
    normal_path = test_root / "NORMAL" / "NORMAL-8000001-1.jpeg"
    _write_real_image(normal_path, 320, 280)
    records.append({"path": str(normal_path), "label": "Normal", "patient": "8000001"})
    misclassified_path = test_root / "DME" / "DME-7000002-1.jpeg"
    _write_real_image(misclassified_path, 320, 280)
    records.append({"path": str(misclassified_path), "label": "DME", "patient": "7000002"})
    test_df = pd.DataFrame.from_records(records)

    cache = PreprocessCache(cache_dir=tmp_path / "cache", image_size=config.IMAGE_SIZE)
    artifacts_dir = tmp_path / "artifacts"
    device = torch.device("cpu")

    # Force the selection rather than deriving it from this untrained model's
    # (essentially random) predictions -- this test is about run_explain's
    # figure-writing/randomization-check plumbing, not about re-testing
    # select_grid_examples (already covered above with synthetic arrays).
    selection = {"dme": 0, "normal": 1, "misclassified": 2}

    result = run_explain(model, test_df, cache, class_to_idx, device, threshold=0.5, artifacts_dir=artifacts_dir, selection=selection)

    assert result["grid_path"].exists()
    assert result["randomization_check_path"].exists()
    assert result["json_path"].exists()
    assert set(result["randomization_correlations"].keys()) == {"dme", "normal", "misclassified"}
    assert result["randomization_verdict"] in ("changed substantially", "did not change")


def test_run_explain_raises_without_precomputed_selection(tmp_path):
    model = _build_model_no_pretrained_download()
    class_to_idx = train.build_class_to_idx()
    cache = PreprocessCache(cache_dir=tmp_path / "cache", image_size=config.IMAGE_SIZE)
    with pytest.raises(ValueError):
        run_explain(model, pd.DataFrame(), cache, class_to_idx, torch.device("cpu"), threshold=0.5, artifacts_dir=tmp_path / "artifacts", selection=None)
