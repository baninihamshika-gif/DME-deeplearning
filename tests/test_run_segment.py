"""
Tests for kaggle/run_segment.py's own logic: staging, quota/smoke-gate
bookkeeping, and log parsing. The real `kaggle` CLI calls are monkeypatched
out everywhere (same reasoning as kaggle/run.py's own tests in
tests/test_phase2_5.py, which this file's fixtures mirror) -- this has not
been exercised against the live Kaggle API; see run_segment.py's own
module docstring.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))

import config
import run as base
import run_segment


@pytest.fixture(autouse=True)
def _isolated_artifacts_dir(tmp_path, monkeypatch):
    """Every test gets its own scratch artifacts dir so nothing here ever
    touches the real quota_log.json / segment_last_smoke.json on disk."""
    artifacts_dir = tmp_path / "kaggle_artifacts"
    monkeypatch.setattr(config, "KAGGLE_ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(config, "KAGGLE_QUOTA_LOG", artifacts_dir / "quota_log.json")
    monkeypatch.setattr(config, "KAGGLE_SEGMENT_LAST_SMOKE", artifacts_dir / "segment_last_smoke.json")
    yield artifacts_dir


@pytest.fixture(autouse=True)
def _mock_kaggle_cli(monkeypatch):
    """No real `kaggle` subprocess calls anywhere in this file."""
    calls = []
    monkeypatch.setattr(base, "_run_kaggle_cli", lambda cmd, ctx, **kw: (calls.append((ctx, cmd)) or ""))
    monkeypatch.setattr(base, "_dataset_exists", lambda slug: False)
    monkeypatch.setattr(base, "wait_for_dataset_ready", lambda slug, **kw: None)
    monkeypatch.setattr(base, "check_username_configured", lambda: None)
    return calls


@pytest.fixture
def duke_dir(tmp_path):
    d = tmp_path / "duke"
    d.mkdir()
    for i in range(1, 4):
        (d / f"Subject_{i:02d}.mat").write_bytes(b"fake")
    return d


@pytest.fixture
def fake_checkpoint(tmp_path):
    ckpt = tmp_path / "some_run_name_checkpoint.pt"  # deliberately NOT already named checkpoint_best.pt
    ckpt.write_bytes(b"fake")
    return ckpt


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------


def test_stage_segment_source_dataset_includes_segment_py_and_both_markers(tmp_path):
    stage = tmp_path / "stage"
    run_segment._stage_segment_source_dataset(config.REPO_ROOT, stage, smoke=True)
    files = {p.name for p in stage.iterdir()}
    assert "segment.py" in files
    assert (stage / "SEGMENT_SMOKE_MODE").read_text().strip() == "true"
    # Must not delete the classifier pipeline's own markers from the shared
    # dataset -- written here with train.py's neutral defaults instead.
    assert (stage / "SMOKE_MODE").read_text().strip() == "false"
    assert json.loads((stage / "TRAIN_ARGS.json").read_text()) == {"model_name": "tf_efficientnet_b3", "label_smoothing": 0.0}


def test_push_single_file_dataset_normalizes_destination_name(tmp_path, fake_checkpoint):
    stage = tmp_path / "ckpt_stage"
    run_segment._push_single_file_dataset(
        "someuser/some-ckpt", "some-ckpt", fake_checkpoint, stage, "v1", dest_filename="checkpoint_best.pt"
    )
    assert (stage / "checkpoint_best.pt").exists()
    assert not (stage / fake_checkpoint.name).exists() or fake_checkpoint.name == "checkpoint_best.pt"


def test_push_duke_dataset_stages_all_mat_files_flat(tmp_path, duke_dir, monkeypatch):
    monkeypatch.setattr(config, "KAGGLE_ARTIFACTS_DIR", tmp_path)
    run_segment._push_duke_dataset(duke_dir, "v1")
    staged = {p.name for p in (tmp_path / "_duke_stage").iterdir()}
    assert staged == {"Subject_01.mat", "Subject_02.mat", "Subject_03.mat", "dataset-metadata.json"}


def test_stage_segment_kernel_push_folder_dataset_sources_order(tmp_path):
    run_segment._stage_segment_kernel_push_folder(tmp_path)
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["dataset_sources"] == [
        config.KAGGLE_SRC_DATASET_SLUG,
        config.KAGGLE_DUKE_DATASET_SLUG,
        config.KAGGLE_CLASSIFIER_CKPT_DATASET_SLUG,
    ]
    assert meta["code_file"] == "entry_segment.py"
    assert (tmp_path / "entry_segment.py").exists()
    # entry.py must NOT be staged here -- confirmed on the first real push
    # that `kaggle kernels push` only uploads the code_file for a
    # script-type kernel; a sibling entry.py would silently never reach
    # the kernel, so entry_segment.py is self-contained instead (see its
    # own module docstring).
    assert not (tmp_path / "entry.py").exists()


# ---------------------------------------------------------------------------
# Smoke gate / quota bookkeeping
# ---------------------------------------------------------------------------


def test_full_push_refused_without_a_passed_smoke(duke_dir, fake_checkpoint):
    with pytest.raises(base.SmokeGateError):
        run_segment.push_segment(config.REPO_ROOT, smoke=False, duke_dir=duke_dir, classifier_checkpoint=fake_checkpoint)


def test_full_push_proceeds_after_a_passed_smoke_on_same_source(duke_dir, fake_checkpoint):
    src_hash = base.git_sha_or_source_hash(config.REPO_ROOT)
    config.KAGGLE_SEGMENT_LAST_SMOKE.parent.mkdir(parents=True, exist_ok=True)
    config.KAGGLE_SEGMENT_LAST_SMOKE.write_text(json.dumps({"source_hash": src_hash, "status": "passed", "epoch_seconds": 10.0}))
    slug = run_segment.push_segment(config.REPO_ROOT, smoke=False, duke_dir=duke_dir, classifier_checkpoint=fake_checkpoint)
    assert slug == config.KAGGLE_SEGMENT_KERNEL_SLUG


def test_estimate_segment_run_minutes_prefers_real_history_for_this_kernel_slug():
    quota_log = [
        {"kernel_slug": "someone/other-kernel", "smoke": False, "status": "complete", "actual_minutes": 999.0},
        {"kernel_slug": config.KAGGLE_SEGMENT_KERNEL_SLUG, "smoke": False, "status": "complete", "actual_minutes": 100.0},
    ]
    assert run_segment.estimate_segment_run_minutes(quota_log, None) == pytest.approx(115.0)


def test_estimate_segment_run_minutes_falls_back_to_smoke_extrapolation():
    last_smoke = {"status": "passed", "epoch_seconds": 20.0}
    from segment import EPOCHS_SEG
    expected = (20.0 * EPOCHS_SEG / 60.0) * 1.3
    assert run_segment.estimate_segment_run_minutes([], last_smoke) == pytest.approx(expected)


def test_estimate_segment_run_minutes_none_without_any_history():
    assert run_segment.estimate_segment_run_minutes([], None) is None
    assert run_segment.estimate_segment_run_minutes([], {"status": "failed"}) is None


def test_quota_log_is_shared_across_classifier_and_segment_pushes(duke_dir, fake_checkpoint):
    """The 7-day GPU cap is one real Kaggle-account-wide resource -- both
    pipelines must write into the SAME quota_log.json for the guardrail to
    mean anything real (see config.py's KAGGLE_SEGMENT_LAST_SMOKE comment)."""
    run_segment.push_segment(config.REPO_ROOT, smoke=True, duke_dir=duke_dir, classifier_checkpoint=fake_checkpoint)
    quota_log = json.loads(config.KAGGLE_QUOTA_LOG.read_text())
    assert len(quota_log) == 1
    assert quota_log[0]["kernel_slug"] == config.KAGGLE_SEGMENT_KERNEL_SLUG


# ---------------------------------------------------------------------------
# Verdict-line extraction from the fetched log
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "log_text,expected",
    [
        ('[{"stream_name": "stdout", "time": 1.0, "data": "Verdict: ship\\n"}]', "Verdict: ship"),
        ('[{"stream_name": "stdout", "time": 1.0, "data": "Verdict: future work\\n"}]', "Verdict: future work"),
        ('[\n  {\n    "data": "Verdict: ship\\n"\n  }\n]', "Verdict: ship"),
        ("", None),
        ('[{"stream_name": "stdout", "time": 1.0, "data": "no verdict here\\n"}]', None),
    ],
)
def test_find_phase4_verdict_line(log_text, expected):
    assert run_segment._find_phase4_verdict_line(log_text) == expected
