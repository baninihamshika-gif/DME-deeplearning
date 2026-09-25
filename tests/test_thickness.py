"""
Tests for thickness.py (Phase 5, extended objective E2).

SCOPE: the pure geometry/windowing functions are tested directly against
synthetic 1D thickness arrays -- no image I/O, no torch, no real OCT data.
_isolate_retina_mask() and _smoothed_lower_boundary_anchor() are tested
against small synthetic *images* built at roughly real content scale (so
the tuned Gaussian blur sigmas behave the way they do on real scans),
specifically regression-testing the v4 fix described in
_isolate_retina_mask()'s own docstring: a brighter, disconnected region
elsewhere in the frame must not be picked over the real anchored tissue
band. compute_thickness_profile()'s plausibility cap and
compute_normal_reference()'s C6 refusal are tested with
_isolate_retina_mask() / analyze_scan() monkeypatched out, so they don't
depend on that CV pipeline succeeding for any particular synthetic input.
This does not and cannot exercise the pipeline against the real Kermany
corpus -- see thickness.py's own module docstring and
verify_thickness_extraction()'s role in that.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import thickness


# ---------------------------------------------------------------------------
# find_foveal_column / windowed_region / windowed_mean_thickness / find_peak_column
# ---------------------------------------------------------------------------


def test_find_foveal_column_is_the_minimum_within_the_central_third():
    # width 30 -> central third is columns [10, 20); the true minimum (at
    # column 3) sits outside that window and must be ignored.
    thickness_arr = np.full(30, 100.0)
    thickness_arr[3] = 1.0
    thickness_arr[15] = 20.0
    assert thickness.find_foveal_column(thickness_arr) == 15


def test_find_foveal_column_none_when_central_third_is_all_nan():
    thickness_arr = np.full(30, 100.0)
    thickness_arr[10:20] = np.nan
    assert thickness.find_foveal_column(thickness_arr) is None


def test_find_foveal_column_none_for_empty_array():
    assert thickness.find_foveal_column(np.array([])) is None


def test_windowed_region_centers_on_column_and_clips_to_bounds():
    lo, hi = thickness.windowed_region(content_width=100, center_col=50, window_fraction=0.2)
    assert lo < 50 < hi
    assert hi - lo == pytest.approx(20, abs=1)

    lo, hi = thickness.windowed_region(content_width=100, center_col=2, window_fraction=0.2)
    assert lo == 0  # clipped at the left edge, not negative

    lo, hi = thickness.windowed_region(content_width=100, center_col=98, window_fraction=0.2)
    assert hi == 100  # clipped at the right edge, not past content width


def test_windowed_mean_thickness_averages_only_the_window_and_ignores_nan():
    thickness_arr = np.full(20, np.nan)
    thickness_arr[8:13] = [10.0, 20.0, np.nan, 40.0, 50.0]
    mean_px, (lo, hi) = thickness.windowed_mean_thickness(thickness_arr, center_col=10, window_fraction=0.5)
    assert (lo, hi) == (5, 16)
    assert mean_px == pytest.approx(30.0)  # mean of 10, 20, 40, 50 -- the NaN is excluded, not zeroed


def test_windowed_mean_thickness_nan_when_window_entirely_invalid():
    thickness_arr = np.full(20, np.nan)
    mean_px, _ = thickness.windowed_mean_thickness(thickness_arr, center_col=10)
    assert np.isnan(mean_px)


def test_find_peak_column_searches_the_whole_width_not_just_the_central_third():
    thickness_arr = np.full(30, 10.0)
    thickness_arr[1] = 999.0  # outside the central third -- must still be found
    assert thickness.find_peak_column(thickness_arr) == 1


def test_find_peak_column_none_for_all_nan():
    assert thickness.find_peak_column(np.full(10, np.nan)) is None


# ---------------------------------------------------------------------------
# is_centre_involvement_indeterminate -- the brief's third referral condition
# ---------------------------------------------------------------------------


def test_centre_involvement_indeterminate_true_when_elevated_and_peak_outside_window():
    assert thickness.is_centre_involvement_indeterminate(index=1.5, peak_col=90, window_bounds=(40, 60)) is True


def test_centre_involvement_not_indeterminate_when_peak_inside_window():
    # elevated, but the thickening IS centred -- ordinary DME-at-the-fovea, not this condition.
    assert thickness.is_centre_involvement_indeterminate(index=1.5, peak_col=50, window_bounds=(40, 60)) is False


def test_centre_involvement_not_indeterminate_when_not_elevated():
    assert thickness.is_centre_involvement_indeterminate(index=0.9, peak_col=90, window_bounds=(40, 60)) is False


def test_centre_involvement_not_indeterminate_for_nan_or_missing_index():
    assert thickness.is_centre_involvement_indeterminate(index=float("nan"), peak_col=90, window_bounds=(40, 60)) is False
    assert thickness.is_centre_involvement_indeterminate(index=1.5, peak_col=None, window_bounds=(40, 60)) is False


# ---------------------------------------------------------------------------
# compute_thickness_profile -- the MAX_PLAUSIBLE_THICKNESS_FRACTION cap
# (_isolate_retina_mask itself is monkeypatched out here: this is testing
# the cap, not the CV pipeline that feeds it)
# ---------------------------------------------------------------------------


def _rgb_stub(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_compute_thickness_profile_discards_implausibly_large_spans(monkeypatch):
    h, w = 200, 10
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:60, 0:5] = 255  # plausible span: rows 20..59 -> 39px, ~20% of content height
    mask[0:150, 5:10] = 255  # implausible span: rows 0..149 -> 149px, ~75% of content height
    monkeypatch.setattr(thickness, "_isolate_retina_mask", lambda gray: mask)

    result = thickness.compute_thickness_profile(_rgb_stub(h, w), content_bbox=(0, h, 0, w))
    assert result[:5].tolist() == [39.0] * 5  # kept -- under the cap
    assert np.all(np.isnan(result[5:]))  # discarded -- over the cap, not silently trusted


def test_compute_thickness_profile_nan_for_columns_with_no_tissue(monkeypatch):
    h, w = 100, 4
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:30, 0:2] = 255  # only the first two columns have any isolated tissue -- rows 10..29 -> 19px
    monkeypatch.setattr(thickness, "_isolate_retina_mask", lambda gray: mask)

    result = thickness.compute_thickness_profile(_rgb_stub(h, w), content_bbox=(0, h, 0, w))
    assert result[:2].tolist() == [19.0, 19.0]
    assert np.all(np.isnan(result[2:]))


def test_compute_thickness_profile_empty_for_zero_size_content_bbox():
    result = thickness.compute_thickness_profile(_rgb_stub(50, 50), content_bbox=(10, 10, 0, 50))  # y0 == y1
    assert result.size == 0


# ---------------------------------------------------------------------------
# _isolate_retina_mask / _smoothed_lower_boundary_anchor -- the v4 fix itself
# ---------------------------------------------------------------------------


def _synthetic_scan(h=300, w=80, artifact_rows=None, retina_rows=(200, 240), background=20, bright=160, artifact_value=255):
    """
    A synthetic B-scan at roughly real content scale: dark background
    throughout, a bright "retina band" at retina_rows, and, if given, an
    unrelated bright "artifact" block at artifact_rows (disconnected from
    the retina band by a wide dark gap -- wide enough that the tuned blur
    sigmas can't bridge it, unlike the genuine failure this regression-
    tests, which had no gap at all in some columns).
    """
    img = np.full((h, w), background, dtype=np.uint8)
    img[retina_rows[0] : retina_rows[1], :] = bright
    if artifact_rows is not None:
        img[artifact_rows[0] : artifact_rows[1], :] = artifact_value
    return img


def test_isolate_retina_mask_prefers_the_anchored_band_over_a_larger_disconnected_artifact():
    # The artifact block (100 rows) is far larger in area than the real
    # retina band (40 rows) -- a plain "largest bright connected
    # component" approach (v3) would pick the artifact. This is exactly
    # the failure _isolate_retina_mask()'s own docstring documents (v3's
    # 2/20-sample failure): the fix must isolate the retina band anyway,
    # by anchoring on the trustworthy lower-boundary curve.
    gray = _synthetic_scan(artifact_rows=(0, 100), retina_rows=(200, 240))
    mask = thickness._isolate_retina_mask(gray)

    mid_col = mask.shape[1] // 2
    rows = np.nonzero(mask[:, mid_col])[0]
    assert rows.size > 0
    assert rows.min() >= 180  # top of the isolated span is near the retina band, not the artifact
    assert rows.max() <= 260


def test_isolate_retina_mask_isolates_a_lone_band_with_no_artifact_present():
    gray = _synthetic_scan(artifact_rows=None, retina_rows=(120, 160))
    mask = thickness._isolate_retina_mask(gray)
    mid_col = mask.shape[1] // 2
    rows = np.nonzero(mask[:, mid_col])[0]
    assert rows.size > 0
    span = rows.max() - rows.min()
    assert 20 <= span <= 60  # roughly the true 40px band, allowing for blur-edge softening


def test_smoothed_lower_boundary_anchor_tracks_the_bright_bands_bottom_edge():
    gray = _synthetic_scan(artifact_rows=None, retina_rows=(120, 160))
    anchor = thickness._smoothed_lower_boundary_anchor(gray)
    assert anchor is not None
    assert np.all(np.abs(anchor - 159) < 5)  # flat band -> flat anchor near row 159 (0-indexed bottom)


def test_smoothed_lower_boundary_anchor_none_when_coverage_too_sparse():
    # Foreground present in well under half the columns -- same bail-out
    # bar utils.flatten_retina() itself uses.
    gray = np.full((200, 100), 20, dtype=np.uint8)
    gray[80:120, 0:10] = 200  # only 10/100 columns have any bright pixels
    anchor = thickness._smoothed_lower_boundary_anchor(gray)
    assert anchor is None


# ---------------------------------------------------------------------------
# compute_normal_reference -- C6: refuse to fabricate a reference from 0 valid images
# ---------------------------------------------------------------------------


def test_compute_normal_reference_raises_when_no_valid_images(monkeypatch):
    monkeypatch.setattr(thickness, "analyze_scan", lambda path, cache: {"valid": False, "windowed_thickness_px": float("nan")})
    df = pd.DataFrame({"path": ["a.jpeg", "b.jpeg"]})
    with pytest.raises(RuntimeError, match="0 valid images"):
        thickness.compute_normal_reference(df, cache=None)


def test_compute_normal_reference_medians_only_the_valid_images(monkeypatch):
    fake_results = {"a.jpeg": 10.0, "b.jpeg": 20.0, "c.jpeg": 30.0}

    def fake_analyze(path, cache):
        if path == "c.jpeg":
            return {"valid": False, "windowed_thickness_px": float("nan")}
        return {"valid": True, "windowed_thickness_px": fake_results[path]}

    monkeypatch.setattr(thickness, "analyze_scan", fake_analyze)
    df = pd.DataFrame({"path": ["a.jpeg", "b.jpeg", "c.jpeg"]})
    median_px, n_valid, n_total = thickness.compute_normal_reference(df, cache=None)
    assert median_px == pytest.approx(15.0)
    assert (n_valid, n_total) == (2, 3)


# ---------------------------------------------------------------------------
# summarize_separation -- AUC buckets
# ---------------------------------------------------------------------------


def _results_df(normal_idx, dme_idx):
    rows = [{"path": f"n{i}", "label": "Normal", "valid": True, "index": v, "centre_involvement_indeterminate": False} for i, v in enumerate(normal_idx)]
    rows += [{"path": f"d{i}", "label": "DME", "valid": True, "index": v, "centre_involvement_indeterminate": False} for i, v in enumerate(dme_idx)]
    return pd.DataFrame(rows)


def test_summarize_separation_clear_when_perfectly_separated():
    df = _results_df(normal_idx=[0.9, 0.95, 1.0, 1.0], dme_idx=[1.8, 2.0, 2.2, 2.5])
    summary = thickness.summarize_separation(df)
    assert summary["separation"] == "clear"
    assert summary["separation_auc"] == pytest.approx(1.0)


def test_summarize_separation_undetermined_when_one_class_missing():
    df = _results_df(normal_idx=[0.9, 1.0], dme_idx=[])
    summary = thickness.summarize_separation(df)
    assert summary["separation"] == "undetermined (one class had 0 valid images)"
    assert np.isnan(summary["separation_auc"])
