"""
Phase 1 acceptance tests.

These exercise parse_patient_id, build_splits, the preprocessing pipeline,
PreprocessCache, and the data/sample curation logic against small
synthetic fixtures -- NOT the real Kermany OCT2017 dataset, which isn't
available locally yet. That means these tests verify the code is correct;
they do not (and cannot) stand in for the real split sizes, class
balance, or fallback rate the Phase 1 sign-off block asks for, which can
only come from a run against the actual dataset (see PROJECT_BRIEF.md
Section 5 / C6: never invent numbers).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `import config`/`utils`

import cv2
import numpy as np
import pandas as pd
import pytest

import config
from prepare_data import curate_sample, generate_preprocessing_figure, write_manifest
from utils import (
    PreprocessCache,
    _letterbox_resize,
    build_splits,
    flatten_retina,
    load_class_folder_df,
    parse_patient_id,
    preprocess_image,
)


# ---------------------------------------------------------------------------
# parse_patient_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename,expected_patient",
    [
        ("DME-1072015-11.jpeg", "1072015"),
        ("NORMAL-956449-1.jpeg", "956449"),
        ("dme-1072015-11.jpeg", "1072015"),  # case-insensitive
        ("CNV-9925708-2.jpeg", "9925708"),
        ("DRUSEN-8086027-2.png", "8086027"),
    ],
)
def test_parse_patient_id_valid(filename, expected_patient):
    assert parse_patient_id(f"/some/dir/{filename}") == expected_patient


@pytest.mark.parametrize(
    "filename",
    [
        "1072015-11.jpeg",          # missing label
        "DME_1072015_11.jpeg",      # wrong separators
        "DME-1072015-11.bmp",       # unsupported extension
        "DME-abc123-11.jpeg",       # non-numeric patient id
        "random_file.jpeg",
    ],
)
def test_parse_patient_id_rejects_malformed_names(filename):
    with pytest.raises(ValueError):
        parse_patient_id(filename)


# ---------------------------------------------------------------------------
# Letterbox resize (C2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h,w", [(60, 200), (200, 50), (300, 300)])
def test_letterbox_resize_preserves_aspect_and_pads(h, w):
    image = np.random.randint(1, 255, size=(h, w), dtype=np.uint8)  # nonzero, so padding is distinguishable
    target = 300
    canvas, (y0, y1, x0, x1) = _letterbox_resize(image, target_size=target)

    assert canvas.shape == (target, target)

    scale = target / max(h, w)
    assert y1 - y0 == max(1, round(h * scale))
    assert x1 - x0 == max(1, round(w * scale))

    # Regions outside the content bbox must be untouched padding (zero).
    if y0 > 0:
        assert np.all(canvas[:y0, :] == 0)
    if y1 < target:
        assert np.all(canvas[y1:, :] == 0)
    if x0 > 0:
        assert np.all(canvas[:, :x0] == 0)
    if x1 < target:
        assert np.all(canvas[:, x1:] == 0)


# ---------------------------------------------------------------------------
# Retinal flattening (fallback safety)
# ---------------------------------------------------------------------------


def test_flatten_retina_falls_back_on_sparse_coverage():
    # Bright block confined to the first 20% of columns -> binary coverage
    # well under the 50%-of-width threshold -> must fall back untouched.
    image = np.zeros((100, 100), dtype=np.uint8)
    image[40:60, 0:20] = 255

    result, fallback_triggered = flatten_retina(image, enabled=True)

    assert fallback_triggered is True
    assert np.array_equal(result, image)


def test_flatten_retina_flattens_a_full_width_sloped_band():
    h, w = 100, 100
    image = np.zeros((h, w), dtype=np.uint8)
    for col in range(w):
        row = int(30 + col * 0.3)  # sloped band, full width coverage
        image[row : row + 10, col] = 255

    result, fallback_triggered = flatten_retina(image, enabled=True)

    assert fallback_triggered is False
    assert result.shape == image.shape
    assert not np.array_equal(result, image)  # it actually did something


def test_flatten_retina_disabled_is_a_no_op():
    image = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)
    result, fallback_triggered = flatten_retina(image, enabled=False)
    assert fallback_triggered is False
    assert np.array_equal(result, image)


# ---------------------------------------------------------------------------
# preprocess_image end-to-end
# ---------------------------------------------------------------------------


def test_preprocess_image_output_shape_and_dtype():
    image = np.random.randint(0, 255, size=(120, 400), dtype=np.uint8)
    rgb, content_bbox, fallback_triggered = preprocess_image(image, image_size=config.IMAGE_SIZE)

    assert rgb.shape == (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)
    assert rgb.dtype == np.uint8
    assert isinstance(fallback_triggered, bool)
    y0, y1, x0, x1 = content_bbox
    assert 0 <= y0 < y1 <= config.IMAGE_SIZE
    assert 0 <= x0 < x1 <= config.IMAGE_SIZE


# ---------------------------------------------------------------------------
# PreprocessCache
# ---------------------------------------------------------------------------


def test_preprocess_cache_roundtrip(tmp_path):
    src = tmp_path / "scan.png"
    cv2.imwrite(str(src), np.random.randint(0, 255, size=(80, 250), dtype=np.uint8))

    cache = PreprocessCache(cache_dir=tmp_path / "cache", image_size=config.IMAGE_SIZE)

    assert cache.get(src) is None  # miss before first compute

    rgb1, bbox1, fallback1 = cache.get_or_compute(src)
    cache_files = list((tmp_path / "cache").glob("*.npz"))
    assert len(cache_files) == 1

    cached = cache.get(src)
    assert cached is not None
    rgb2, bbox2, fallback2 = cached
    assert np.array_equal(rgb1, rgb2)
    assert bbox1 == bbox2
    assert fallback1 == fallback2


def test_preprocess_cache_keys_differ_by_image_size(tmp_path):
    src = tmp_path / "scan.png"
    cv2.imwrite(str(src), np.random.randint(0, 255, size=(80, 250), dtype=np.uint8))

    cache_dir = tmp_path / "cache"
    PreprocessCache(cache_dir=cache_dir, image_size=224).get_or_compute(src)
    PreprocessCache(cache_dir=cache_dir, image_size=300).get_or_compute(src)

    assert len(list(cache_dir.glob("*.npz"))) == 2


# ---------------------------------------------------------------------------
# build_splits (C1, C8)
# ---------------------------------------------------------------------------


def _write_synthetic_class_folder(
    root: Path, label_folder: str, n_patients: int, images_per_patient: int, id_offset: int = 0
):
    # id_offset keeps DME's and NORMAL's patient ID ranges disjoint by
    # default, since that's the common case; tests that want a patient
    # straddling both classes add one explicitly.
    class_dir = root / label_folder
    class_dir.mkdir(parents=True, exist_ok=True)
    for patient_idx in range(n_patients):
        patient_id = 1_000_000 + id_offset + patient_idx
        for image_idx in range(images_per_patient):
            (class_dir / f"{label_folder}-{patient_id}-{image_idx + 1}.jpeg").write_bytes(b"\x00")


def test_build_splits_synthetic_dataset_passes_all_invariants(tmp_path):
    train_dir = tmp_path / "train"
    _write_synthetic_class_folder(train_dir, "DME", n_patients=40, images_per_patient=3)
    _write_synthetic_class_folder(train_dir, "NORMAL", n_patients=40, images_per_patient=3, id_offset=10_000)

    result = build_splits(
        train_dir,
        val_split=0.25,
        seed=42,
        expected_counts={"DME": 120, "NORMAL": 120},
    )

    assert len(result.train_df) + len(result.val_df) == 240
    assert result.train_val_patient_overlap == 0
    assert set(result.train_df["patient"]).isdisjoint(set(result.val_df["patient"]))
    assert set(result.train_df["label"]) == {"DME", "Normal"}
    assert set(result.val_df["label"]) == {"DME", "Normal"}


def test_build_splits_raises_on_count_mismatch(tmp_path):
    train_dir = tmp_path / "train"
    _write_synthetic_class_folder(train_dir, "DME", n_patients=10, images_per_patient=2)
    _write_synthetic_class_folder(train_dir, "NORMAL", n_patients=10, images_per_patient=2, id_offset=10_000)

    with pytest.raises(AssertionError):
        build_splits(train_dir, expected_counts={"DME": 999, "NORMAL": 20}, seed=42)


def test_build_splits_detects_patient_overlap_with_test_set(tmp_path):
    train_dir = tmp_path / "train"
    _write_synthetic_class_folder(train_dir, "DME", n_patients=20, images_per_patient=2)
    _write_synthetic_class_folder(train_dir, "NORMAL", n_patients=20, images_per_patient=2, id_offset=10_000)

    test_dir = tmp_path / "test"
    # Deliberately reuse patient 1000000 (already in the DME train pool).
    (test_dir / "DME").mkdir(parents=True, exist_ok=True)
    (test_dir / "DME" / "DME-1000000-99.jpeg").write_bytes(b"\x00")
    (test_dir / "NORMAL").mkdir(parents=True, exist_ok=True)
    (test_dir / "NORMAL" / "NORMAL-9999999-1.jpeg").write_bytes(b"\x00")  # not in train

    with pytest.warns(UserWarning):
        result = build_splits(
            train_dir, test_dir=test_dir, expected_counts={"DME": 40, "NORMAL": 40}, seed=42
        )

    # test_patient_overlap reports what was FOUND (before the fix); by
    # default build_splits then excludes those patients from train/val, so
    # the returned frames are actually clean (see PROJECT_BRIEF.md Phase 1
    # discussion: the official 484-image test set is kept intact, the
    # overlap is fixed from the training side instead).
    assert result.test_patient_overlap == 1
    assert result.excluded_test_overlap_patients == 1
    assert result.excluded_test_overlap_images == 2  # patient 1000000's 2 DME images
    assert "1000000" not in set(result.train_df["patient"]) | set(result.val_df["patient"])


def test_build_splits_can_skip_test_overlap_exclusion(tmp_path):
    train_dir = tmp_path / "train"
    _write_synthetic_class_folder(train_dir, "DME", n_patients=20, images_per_patient=2)
    _write_synthetic_class_folder(train_dir, "NORMAL", n_patients=20, images_per_patient=2, id_offset=10_000)

    test_dir = tmp_path / "test"
    (test_dir / "DME").mkdir(parents=True, exist_ok=True)
    (test_dir / "DME" / "DME-1000000-99.jpeg").write_bytes(b"\x00")
    (test_dir / "NORMAL").mkdir(parents=True, exist_ok=True)
    (test_dir / "NORMAL" / "NORMAL-9999999-1.jpeg").write_bytes(b"\x00")

    with pytest.warns(UserWarning):
        result = build_splits(
            train_dir,
            test_dir=test_dir,
            expected_counts={"DME": 40, "NORMAL": 40},
            seed=42,
            exclude_test_overlap_patients=False,
        )

    assert result.test_patient_overlap == 1
    assert result.excluded_test_overlap_patients == 0
    assert "1000000" in set(result.train_df["patient"]) | set(result.val_df["patient"])


def test_build_splits_keeps_cross_class_patient_in_one_split(tmp_path):
    # On the real Kermany data, some patient IDs appear under BOTH DME and
    # NORMAL (graded differently across visits/eyes) -- 339 of them in the
    # real train set. A patient split per-class independently would happily
    # put that patient's DME images in train and NORMAL images in val,
    # which is a genuine C1 violation even though it doesn't fail an
    # unqualified "no overlap within DME" or "no overlap within NORMAL"
    # check. build_splits groups across the whole DME+Normal pool, so this
    # patient must land entirely in one split.
    train_dir = tmp_path / "train"
    _write_synthetic_class_folder(train_dir, "DME", n_patients=30, images_per_patient=3)
    _write_synthetic_class_folder(train_dir, "NORMAL", n_patients=30, images_per_patient=3, id_offset=10_000)

    # One extra patient, with a fresh ID not used by either class above,
    # who has images under BOTH DME and NORMAL.
    shared_patient = 1_999_999
    for image_idx in range(3):
        (train_dir / "DME" / f"DME-{shared_patient}-{image_idx + 1}.jpeg").write_bytes(b"\x00")
    for image_idx in range(3):
        (train_dir / "NORMAL" / f"NORMAL-{shared_patient}-{image_idx + 1}.jpeg").write_bytes(b"\x00")

    result = build_splits(
        train_dir,
        val_split=0.25,
        seed=42,
        expected_counts={"DME": 93, "NORMAL": 93},
        # This test is about grouping correctness, not ratio balance -- with
        # only 31 patient-groups per class, per-group quantization alone can
        # push the ratio past the default 5% tolerance; the real 4,600+
        # patient dataset doesn't have that problem (see prepare_data.py's
        # actual run). Loosened here so the test isn't flaky on group count.
        ratio_tolerance=0.15,
    )

    assert result.train_val_patient_overlap == 0
    assert set(result.train_df["patient"]).isdisjoint(set(result.val_df["patient"]))

    shared_patient_rows = pd.concat(
        [
            result.train_df[result.train_df["patient"] == str(shared_patient)],
            result.val_df[result.val_df["patient"] == str(shared_patient)],
        ]
    )
    # All 6 of that patient's images (3 DME + 3 NORMAL) landed in the same split.
    assert len(shared_patient_rows) == 6
    assert {"DME", "Normal"} == set(shared_patient_rows["label"])
    in_train = shared_patient_rows["path"].isin(result.train_df["path"])
    assert in_train.all() or (~in_train).all()


def test_build_splits_no_overlap_with_disjoint_test_set(tmp_path):
    train_dir = tmp_path / "train"
    _write_synthetic_class_folder(train_dir, "DME", n_patients=20, images_per_patient=2)
    _write_synthetic_class_folder(train_dir, "NORMAL", n_patients=20, images_per_patient=2, id_offset=10_000)

    test_dir = tmp_path / "test"
    (test_dir / "DME").mkdir(parents=True, exist_ok=True)
    (test_dir / "DME" / "DME-5000000-1.jpeg").write_bytes(b"\x00")
    (test_dir / "NORMAL").mkdir(parents=True, exist_ok=True)
    (test_dir / "NORMAL" / "NORMAL-5000001-1.jpeg").write_bytes(b"\x00")

    result = build_splits(train_dir, test_dir=test_dir, expected_counts={"DME": 40, "NORMAL": 40}, seed=42)
    assert result.test_patient_overlap == 0


# ---------------------------------------------------------------------------
# load_class_folder_df (Phase 3: loading the held-out test set, unsplit)
# ---------------------------------------------------------------------------


def test_load_class_folder_df_returns_unsplit_frame(tmp_path):
    test_dir = tmp_path / "test"
    _write_synthetic_class_folder(test_dir, "DME", n_patients=5, images_per_patient=2)
    _write_synthetic_class_folder(test_dir, "NORMAL", n_patients=5, images_per_patient=2, id_offset=10_000)

    df = load_class_folder_df(test_dir, expected_counts={"DME": 10, "NORMAL": 10})

    assert len(df) == 20  # no splitting -- every discovered image comes back
    assert set(df["label"]) == {"DME", "Normal"}
    assert set(df.columns) == {"path", "label", "patient"}


def test_load_class_folder_df_raises_on_wrong_expected_counts(tmp_path):
    test_dir = tmp_path / "test"
    _write_synthetic_class_folder(test_dir, "DME", n_patients=5, images_per_patient=2)
    _write_synthetic_class_folder(test_dir, "NORMAL", n_patients=5, images_per_patient=2, id_offset=10_000)

    with pytest.raises(AssertionError):
        load_class_folder_df(test_dir, expected_counts={"DME": 999, "NORMAL": 10})


def test_load_class_folder_df_skips_assertion_when_expected_counts_is_none(tmp_path):
    test_dir = tmp_path / "test"
    _write_synthetic_class_folder(test_dir, "DME", n_patients=3, images_per_patient=1)
    _write_synthetic_class_folder(test_dir, "NORMAL", n_patients=3, images_per_patient=1, id_offset=10_000)

    df = load_class_folder_df(test_dir, expected_counts=None)
    assert len(df) == 6


def test_load_class_folder_df_defaults_to_expected_test_counts(monkeypatch, tmp_path):
    # The default argument is bound to EXPECTED_TEST_COUNTS at import time
    # (mirroring build_splits's own default binding to EXPECTED_TRAIN_COUNTS),
    # so this confirms the *real* default -- not a value the test supplies --
    # still catches a wrong/incomplete download the same way build_splits does.
    test_dir = tmp_path / "test"
    _write_synthetic_class_folder(test_dir, "DME", n_patients=3, images_per_patient=1)
    _write_synthetic_class_folder(test_dir, "NORMAL", n_patients=3, images_per_patient=1, id_offset=10_000)

    with pytest.raises(AssertionError):
        load_class_folder_df(test_dir)  # real EXPECTED_TEST_COUNTS (242/242) won't match 3/3


# ---------------------------------------------------------------------------
# curate_sample / manifest / figure -- against synthetic real image files
# ---------------------------------------------------------------------------


def _write_real_image(path: Path, width: int, height: int, sloped_band: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width), dtype=np.uint8)
    if sloped_band:
        for col in range(width):
            row = int(height * 0.3 + col * 0.05) % max(1, height - 10)
            image[row : row + 8, col] = 255
    else:
        # Sparse block -> triggers the flattening fallback (proxy for "pathology").
        image[height // 3 : height // 2, 0 : max(1, width // 6)] = 255
    cv2.imwrite(str(path), image)


def _build_synthetic_sample_df(root: Path) -> pd.DataFrame:
    records = []
    # DME: 10 patients, most with 1 image, a couple with 2 (for multi-scan), varying widths.
    widths = [200, 400, 600, 800, 300, 350, 500, 700, 250, 450]
    for i, width in enumerate(widths):
        patient = 2_000_000 + i
        path = root / "DME" / f"DME-{patient}-1.jpeg"
        _write_real_image(path, width=width, height=200, sloped_band=(i % 3 != 0))
        records.append({"path": str(path), "label": "DME", "patient": str(patient)})
        if i < 2:  # give the first two patients a second scan
            path2 = root / "DME" / f"DME-{patient}-2.jpeg"
            _write_real_image(path2, width=width, height=200, sloped_band=True)
            records.append({"path": str(path2), "label": "DME", "patient": str(patient)})

    for i, width in enumerate(widths):
        patient = 3_000_000 + i
        path = root / "NORMAL" / f"NORMAL-{patient}-1.jpeg"
        _write_real_image(path, width=width, height=200, sloped_band=True)
        records.append({"path": str(path), "label": "Normal", "patient": str(patient)})

    return pd.DataFrame.from_records(records)


def test_curate_sample_honors_selection_criteria(tmp_path):
    df = _build_synthetic_sample_df(tmp_path / "raw")
    sample_dir = tmp_path / "sample"

    rows = curate_sample(
        df, sample_dir, per_class=4, n_pathology=2, min_multi_scan_patients=2, seed=1
    )

    assert len(rows) > 0
    for row in rows:
        assert (sample_dir / Path(row["path"]).name).exists()

    for label in ["DME", "Normal"]:
        reasons = [r["reason"] for r in rows if r["class"] == label]
        assert any("widest" in r for r in reasons)
        assert any("narrowest" in r for r in reasons)

    patient_counts = {}
    for row in rows:
        patient_counts[row["patient"]] = patient_counts.get(row["patient"], 0) + 1
    assert sum(1 for n in patient_counts.values() if n > 1) >= 2

    manifest_path = write_manifest(rows, sample_dir)
    assert manifest_path.exists()
    manifest_df = pd.read_csv(manifest_path)
    assert len(manifest_df) == len(rows)
    assert set(manifest_df.columns) == {"path", "class", "patient", "width", "height", "reason"}


def test_generate_preprocessing_figure_writes_file(tmp_path):
    src = tmp_path / "scan.jpeg"
    _write_real_image(src, width=400, height=150, sloped_band=True)

    out_path = generate_preprocessing_figure(src, tmp_path / "artifacts" / "fig_preprocessing_stages.png")

    assert out_path.exists()
    assert out_path.stat().st_size > 0
