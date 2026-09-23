"""
Phase 2.5 structural tests for kaggle/run.py.

SCOPE: everything here is testable without kaggle.com or real credentials
-- source hashing, quota math, the smoke-gate and quota guardrails, status
parsing, failure classification, and the push/create-vs-version routing
decision (mocked). It deliberately does NOT and cannot exercise a real
`kaggle datasets create/version` or `kaggle kernels push/status/output`
call: no automated shell available while building this had network access
to kaggle.com. The real push/poll/fetch cycle can only be verified by
actually running it -- see PHASE 2.5's sign-off, "Blocking questions".
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))

import config
import run


# ---------------------------------------------------------------------------
# Source hashing
# ---------------------------------------------------------------------------


def _fake_repo(tmp_path, contents: dict) -> Path:
    for rel, text in contents.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return tmp_path


def test_source_hash_deterministic_for_same_content(tmp_path):
    repo_a = _fake_repo(tmp_path / "a", {"config.py": "X=1", "utils.py": "Y=2"})
    repo_b = _fake_repo(tmp_path / "b", {"config.py": "X=1", "utils.py": "Y=2"})
    with patch.object(run, "SOURCE_FILES", ["config.py", "utils.py"]):
        assert run.source_hash(repo_a) == run.source_hash(repo_b)


def test_source_hash_changes_when_content_changes(tmp_path):
    repo = _fake_repo(tmp_path, {"config.py": "X=1", "utils.py": "Y=2"})
    with patch.object(run, "SOURCE_FILES", ["config.py", "utils.py"]):
        h1 = run.source_hash(repo)
        (repo / "config.py").write_text("X=2")
        h2 = run.source_hash(repo)
    assert h1 != h2


def test_source_hash_skips_missing_files_without_crashing(tmp_path):
    repo = _fake_repo(tmp_path, {"config.py": "X=1"})
    with patch.object(run, "SOURCE_FILES", ["config.py", "does_not_exist.py"]):
        h = run.source_hash(repo)
    assert isinstance(h, str) and len(h) == 12


def test_git_sha_or_source_hash_falls_back_when_no_git(tmp_path):
    repo = _fake_repo(tmp_path, {"config.py": "X=1"})
    with patch.object(run, "SOURCE_FILES", ["config.py"]):
        result = run.git_sha_or_source_hash(repo)
    # tmp_path is not a git repo (or if it happens to be nested in one with
    # no commits, `git rev-parse HEAD` still fails) -- either way this must
    # never silently return something that looks like a real git sha.
    assert result.startswith("contenthash-")


def test_git_sha_or_source_hash_prefers_real_git_sha(tmp_path):
    repo = _fake_repo(tmp_path, {"config.py": "X=1"})
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    result = run.git_sha_or_source_hash(repo)
    assert not result.startswith("contenthash-")
    assert len(result) == 12


# ---------------------------------------------------------------------------
# Quota math
# ---------------------------------------------------------------------------


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def test_gpu_minutes_used_sums_within_7_day_window():
    log = [
        {"timestamp": _iso(1), "expected_minutes": 10, "actual_minutes": None},
        {"timestamp": _iso(24 * 3), "expected_minutes": 20, "actual_minutes": 25},  # actual overrides expected
        {"timestamp": _iso(24 * 10), "expected_minutes": 999, "actual_minutes": None},  # outside window
    ]
    used = run.gpu_minutes_used_last_7_days(log)
    assert used == pytest.approx(10 + 25)


def test_estimate_full_run_minutes_none_without_smoke_timing():
    assert run.estimate_full_run_minutes([], None) is None
    assert run.estimate_full_run_minutes([], {"status": "passed"}) is None  # no epoch_seconds
    assert run.estimate_full_run_minutes([], {"status": "failed", "epoch_seconds": 30}) is None


def test_estimate_full_run_minutes_uses_real_epoch_budget_when_no_full_run_history():
    """The necessarily-rough first-ever-full-push estimate, extrapolated
    from the smoke run -- only reached when quota_log has no completed
    full run yet."""
    last_smoke = {"status": "passed", "epoch_seconds": 60.0}
    total_epochs = config.EPOCHS_PHASE_A + config.EPOCHS_PHASE_B
    expected = (60.0 * total_epochs / 60.0) * 1.3
    assert run.estimate_full_run_minutes([], last_smoke) == pytest.approx(expected)


def test_estimate_full_run_minutes_prefers_real_full_run_history():
    """Confirmed on a real push (2026-09-16): extrapolating from the
    smoke run's ~500-image subset undershot a real full run by ~9.7x
    (32.8min estimated vs 319.77min actual). Once a completed full run is
    on record, its real actual_minutes (+15% margin) must be used instead
    of the smoke extrapolation, even if a last_smoke entry is also
    present."""
    quota_log = [
        {"smoke": True, "status": "complete", "actual_minutes": 8.0},
        {"smoke": False, "status": "timeout", "actual_minutes": 682.0},  # must be ignored: not complete
        {"smoke": False, "status": "complete", "actual_minutes": 319.77},
    ]
    last_smoke = {"status": "passed", "epoch_seconds": 60.0}  # would give a very different number if used
    result = run.estimate_full_run_minutes(quota_log, last_smoke)
    assert result == pytest.approx(319.77 * 1.15)


def test_estimate_full_run_minutes_ignores_incomplete_full_runs():
    quota_log = [{"smoke": False, "status": "error", "actual_minutes": 200.0}]
    last_smoke = {"status": "passed", "epoch_seconds": 60.0}
    total_epochs = config.EPOCHS_PHASE_A + config.EPOCHS_PHASE_B
    expected = (60.0 * total_epochs / 60.0) * 1.3
    assert run.estimate_full_run_minutes(quota_log, last_smoke) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _real_duration_minutes_from_log
#
# Confirmed the hard way (2026-09-16/17): recording poll()'s local
# wall-clock time as actual_minutes silently counted a ~9 hour local
# network/sleep outage as "GPU minutes consumed". The fetched log's own
# timestamps are ground truth and immune to that -- prefer them.
# ---------------------------------------------------------------------------


def test_real_duration_minutes_from_log_uses_last_timestamp():
    log_text = json.dumps([
        {"stream_name": "stdout", "time": 1.5, "data": "start\n"},
        {"stream_name": "stdout", "time": 19186.45, "data": "done\n"},
    ])
    assert run._real_duration_minutes_from_log(log_text) == pytest.approx(19186.45 / 60.0)


def test_real_duration_minutes_from_log_none_on_empty_text():
    assert run._real_duration_minutes_from_log("") is None


def test_real_duration_minutes_from_log_none_on_garbage():
    assert run._real_duration_minutes_from_log("not json at all") is None


def test_real_duration_minutes_from_log_none_on_empty_array():
    assert run._real_duration_minutes_from_log("[]") is None


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def test_quota_guardrail_passes_under_budget():
    run.check_quota_guardrail(expected_minutes=10, quota_log=[], budget_min=360)  # must not raise


def test_quota_guardrail_refuses_over_budget():
    with pytest.raises(run.QuotaExceeded):
        run.check_quota_guardrail(expected_minutes=400, quota_log=[], budget_min=360)


def test_quota_guardrail_accounts_for_prior_usage_this_week():
    log = [{"timestamp": _iso(1), "expected_minutes": 350, "actual_minutes": None}]
    with pytest.raises(run.QuotaExceeded):
        run.check_quota_guardrail(expected_minutes=20, quota_log=log, budget_min=360)


def test_smoke_gate_refuses_with_no_smoke_recorded():
    with pytest.raises(run.SmokeGateError):
        run.check_smoke_gate("abc123", None)


def test_smoke_gate_refuses_on_source_mismatch():
    with pytest.raises(run.SmokeGateError):
        run.check_smoke_gate("current_hash", {"source_hash": "old_hash", "status": "passed"})


def test_smoke_gate_refuses_on_failed_smoke():
    with pytest.raises(run.SmokeGateError):
        run.check_smoke_gate("abc123", {"source_hash": "abc123", "status": "failed"})


def test_smoke_gate_passes_on_matching_passed_smoke():
    run.check_smoke_gate("abc123", {"source_hash": "abc123", "status": "passed"})  # must not raise


# ---------------------------------------------------------------------------
# Phase 3g Baselines: smoke gate / full-run estimate must be keyed on
# model_name (and, for the gate, label_smoothing) too -- --model-name and
# --label-smoothing are CLI-only push args, not part of source_hash, so
# without this a smoke pass on one architecture would silently authorize a
# full (expensive) push of a different, never-smoke-tested architecture as
# long as the source tree was unchanged.
# ---------------------------------------------------------------------------


def test_smoke_gate_refuses_on_model_name_mismatch():
    last_smoke = {"source_hash": "abc123", "model_name": "resnet50", "label_smoothing": 0.0, "status": "passed"}
    with pytest.raises(run.SmokeGateError):
        run.check_smoke_gate("abc123", last_smoke, model_name="tf_efficientnet_b0", label_smoothing=0.0)


def test_smoke_gate_refuses_on_label_smoothing_mismatch():
    last_smoke = {"source_hash": "abc123", "model_name": "tf_efficientnet_b3", "label_smoothing": 0.0, "status": "passed"}
    with pytest.raises(run.SmokeGateError):
        run.check_smoke_gate("abc123", last_smoke, model_name="tf_efficientnet_b3", label_smoothing=0.05)


def test_smoke_gate_passes_on_matching_model_name_and_label_smoothing():
    last_smoke = {"source_hash": "abc123", "model_name": "resnet50", "label_smoothing": 0.0, "status": "passed"}
    run.check_smoke_gate("abc123", last_smoke, model_name="resnet50", label_smoothing=0.0)  # must not raise


def test_smoke_gate_treats_legacy_smoke_record_without_model_name_as_primary():
    """A smoke record from before Phase 3g (no model_name/label_smoothing
    fields at all) must satisfy a default (tf_efficientnet_b3, 0.0) push --
    that's what every pre-Baselines smoke run actually was."""
    last_smoke = {"source_hash": "abc123", "status": "passed"}
    run.check_smoke_gate("abc123", last_smoke)  # defaults model_name="tf_efficientnet_b3", label_smoothing=0.0


def test_estimate_full_run_minutes_ignores_different_architecture_history():
    """A completed ResNet-50 run's real timing must not be reused as the
    estimate for an EfficientNet-B0 push -- their per-epoch compute costs
    differ substantially. Without model_name filtering this would silently
    misinform the quota guardrail."""
    quota_log = [{"smoke": False, "status": "complete", "actual_minutes": 500.0, "model_name": "resnet50"}]
    last_smoke = {"status": "passed", "epoch_seconds": 60.0}
    total_epochs = config.EPOCHS_PHASE_A + config.EPOCHS_PHASE_B
    expected = (60.0 * total_epochs / 60.0) * 1.3  # falls back to smoke extrapolation, not the ResNet-50 number
    result = run.estimate_full_run_minutes(quota_log, last_smoke, model_name="tf_efficientnet_b0")
    assert result == pytest.approx(expected)
    assert result != pytest.approx(500.0 * 1.15)


def test_estimate_full_run_minutes_matches_same_architecture_history():
    quota_log = [{"smoke": False, "status": "complete", "actual_minutes": 500.0, "model_name": "resnet50"}]
    result = run.estimate_full_run_minutes(quota_log, None, model_name="resnet50")
    assert result == pytest.approx(500.0 * 1.15)


def test_estimate_full_run_minutes_legacy_entries_without_model_name_match_default_b3():
    """Every quota_log entry recorded before Phase 3g has no model_name
    field -- they were all real tf_efficientnet_b3 runs, so the default
    lookup (model_name="tf_efficientnet_b3") must still find them."""
    quota_log = [{"smoke": False, "status": "complete", "actual_minutes": 319.77}]
    result = run.estimate_full_run_minutes(quota_log, None)
    assert result == pytest.approx(319.77 * 1.15)


def test_username_guardrail_refuses_placeholder():
    with patch.object(config, "KAGGLE_USERNAME", "REPLACE_WITH_KAGGLE_USERNAME"):
        with pytest.raises(RuntimeError):
            run.check_username_configured()


def test_username_guardrail_passes_once_set():
    with patch.object(config, "KAGGLE_USERNAME", "someuser"):
        run.check_username_configured()  # must not raise


# ---------------------------------------------------------------------------
# Status parsing / failure classification
# ---------------------------------------------------------------------------


def _fake_completed(stdout="", stderr="", returncode=0):
    result = subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)
    return result


# ---------------------------------------------------------------------------
# _run_subprocess: two distinct Windows encoding failures found on two
# separate real runs -- decoding the child's captured output as UTF-8 with
# errors="replace" (parent-side), and forcing PYTHONIOENCODING in the
# child's own environment so the `kaggle` CLI process (itself Python)
# doesn't crash internally trying to print outside cp1252 before we ever
# get to decode anything.
# ---------------------------------------------------------------------------


def test_run_subprocess_sets_pythonioencoding_in_child_env():
    with patch("subprocess.run", return_value=_fake_completed()) as mock_run:
        run._run_subprocess(["kaggle", "kernels", "logs", "someuser/dme-oct-train"])
    passed_env = mock_run.call_args.kwargs["env"]
    assert passed_env["PYTHONIOENCODING"] == "utf-8:replace"


def test_run_subprocess_preserves_rest_of_current_environment():
    """Must not drop the child's PATH etc. -- only add the one variable."""
    with patch.dict(os.environ, {"SOME_UNRELATED_VAR": "keepme"}), \
         patch("subprocess.run", return_value=_fake_completed()) as mock_run:
        run._run_subprocess(["kaggle", "kernels", "status", "someuser/dme-oct-src"])
    passed_env = mock_run.call_args.kwargs["env"]
    assert passed_env["SOME_UNRELATED_VAR"] == "keepme"


def test_run_subprocess_does_not_override_explicit_pythonioencoding():
    with patch("subprocess.run", return_value=_fake_completed()) as mock_run:
        run._run_subprocess(["kaggle", "kernels", "logs", "someuser/dme-oct-train"], env={"PYTHONIOENCODING": "latin-1"})
    passed_env = mock_run.call_args.kwargs["env"]
    assert passed_env["PYTHONIOENCODING"] == "latin-1"


def test_get_status_parses_success():
    with patch("subprocess.run", return_value=_fake_completed(stdout='someuser/dme-oct-train has status "complete"\n')):
        status, msg = run.get_status("someuser/dme-oct-train")
    assert status == "complete"
    assert msg is None


def test_get_status_parses_failure_message():
    stdout = (
        'someuser/dme-oct-train has status "error"\n'
        'Failure message: "Kernel session terminated unexpectedly"\n'
    )
    with patch("subprocess.run", return_value=_fake_completed(stdout=stdout)):
        status, msg = run.get_status("someuser/dme-oct-train")
    assert status == "error"
    assert "terminated unexpectedly" in msg


def test_get_status_unknown_when_unparseable():
    with patch("subprocess.run", return_value=_fake_completed(stdout="garbage output")):
        status, msg = run.get_status("someuser/dme-oct-train")
    assert status == "unknown"
    assert msg is None


def test_get_status_normalizes_enum_repr_form():
    """Regression test: a real account's `kaggle` CLI printed
    'has status "KernelWorkerStatus.ERROR"' (a Python Enum's default str()),
    not the bare 'error' this module was originally tested against. Before
    _normalize_status() existed, poll() compared this raw string against
    TERMINAL_FAILURE = {"error", ...} and never matched, so a run that had
    already failed was polled as if still in progress for the full
    KAGGLE_MAX_POLL_MIN (6 hours) instead of being recognized immediately."""
    stdout = 'someuser/dme-oct-train has status "KernelWorkerStatus.ERROR"\n'
    with patch("subprocess.run", return_value=_fake_completed(stdout=stdout)):
        status, msg = run.get_status("someuser/dme-oct-train")
    assert status == "error"
    assert status in run.TERMINAL_FAILURE


def test_get_status_normalizes_enum_repr_running_is_not_terminal():
    stdout = 'someuser/dme-oct-train has status "KernelWorkerStatus.RUNNING"\n'
    with patch("subprocess.run", return_value=_fake_completed(stdout=stdout)):
        status, msg = run.get_status("someuser/dme-oct-train")
    assert status == "running"
    assert status not in run.TERMINAL_SUCCESS
    assert status not in run.TERMINAL_FAILURE


def test_get_status_normalizes_enum_repr_complete():
    stdout = 'someuser/dme-oct-train has status "KernelWorkerStatus.COMPLETE"\n'
    with patch("subprocess.run", return_value=_fake_completed(stdout=stdout)):
        status, msg = run.get_status("someuser/dme-oct-train")
    assert status == "complete"
    assert status in run.TERMINAL_SUCCESS


def test_poll_recognizes_enum_repr_error_as_terminal_immediately():
    """The exact scenario that hung in production: status stays
    'KernelWorkerStatus.ERROR' on every poll -- must return on the FIRST
    check, not loop until max_minutes."""
    with patch.object(run, "get_status", return_value=("error", "boom")), patch("time.sleep") as mock_sleep:
        status, msg = run.poll("someuser/dme-oct-train", interval_sec=60, max_minutes=360)
    assert status == "error"
    mock_sleep.assert_not_called()


def test_classify_failure_traceback_is_code_error():
    log_text = "some output\nTraceback (most recent call last):\nValueError: bad\n"
    assert run.classify_failure("error", None, log_text) == "code_error"


def test_classify_failure_session_keyword_is_infra():
    assert run.classify_failure("error", "Kernel session terminated unexpectedly", "") == "infra"


def test_classify_failure_timeout_status_is_infra():
    assert run.classify_failure("timeout", "polling exceeded max minutes", "") == "infra"


def test_classify_failure_unclear_defaults_to_code_error_not_retried():
    """Ambiguous failures must default to the non-retrying classification -- retrying blind
    on an unrecognised failure burns GPU quota on a run that will likely fail the same way."""
    assert run.classify_failure("error", "something odd happened", "") == "code_error"


def test_classify_failure_gpu_torch_arch_mismatch_is_infra_not_code_error():
    """The exact real regression from push #5 (confirmed against
    Kaggle/docker-python#1546): a genuine Python traceback, but caused by
    Kaggle allocating a P100 that its own preinstalled torch build doesn't
    support -- not a bug in our training code. The generic
    "any traceback -> code_error" rule must not win here."""
    log_text = (
        "[phase transition] Phase A trainable params: 595970\n"
        "Traceback (most recent call last):\n"
        "  File \"/kaggle/working/train.py\", line 480, in main\n"
        "    global_epoch, best_val_auc, best_epoch, epochs_no_improve, _complete = run_phase(\n"
        "torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device\n"
        "Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the current PyTorch installation.\n"
    )
    assert run.classify_failure("error", None, log_text) == "infra"


def test_classify_failure_unrelated_traceback_still_code_error():
    """Regression guard: the new GPU/torch-arch-mismatch rule must not
    swallow ordinary code bugs into the retryable bucket."""
    log_text = "Traceback (most recent call last):\nZeroDivisionError: division by zero\n"
    assert run.classify_failure("error", None, log_text) == "code_error"


# ---------------------------------------------------------------------------
# Push routing: create vs version (mocked -- no real kaggle.com call)
# ---------------------------------------------------------------------------


def test_dataset_exists_true_on_zero_exit():
    with patch("subprocess.run", return_value=_fake_completed(returncode=0)):
        assert run._dataset_exists("someuser/dme-oct-src") is True


def test_dataset_exists_false_on_nonzero_exit():
    with patch("subprocess.run", return_value=_fake_completed(returncode=1)):
        assert run._dataset_exists("someuser/dme-oct-src") is False


def test_push_source_dataset_routes_to_create_when_absent(tmp_path):
    repo = _fake_repo(tmp_path, {"config.py": "X=1", "requirements.txt": "torch\n"})
    with patch.object(run, "SOURCE_FILES", ["config.py", "requirements.txt"]), \
         patch.object(config, "KAGGLE_ARTIFACTS_DIR", tmp_path / "artifacts" / "kaggle"), \
         patch.object(run, "_dataset_exists", return_value=False), \
         patch.object(run, "_run_kaggle_cli") as mock_cli, \
         patch.object(run, "wait_for_dataset_ready") as mock_wait:
        run.push_source_dataset(repo, smoke=True, version_message="abc123")
    called_cmd = mock_cli.call_args[0][0]
    assert "create" in called_cmd
    mock_wait.assert_called_once()  # must block on readiness before returning to the caller


def test_push_source_dataset_routes_to_version_when_present(tmp_path):
    repo = _fake_repo(tmp_path, {"config.py": "X=1", "requirements.txt": "torch\n"})
    with patch.object(run, "SOURCE_FILES", ["config.py", "requirements.txt"]), \
         patch.object(config, "KAGGLE_ARTIFACTS_DIR", tmp_path / "artifacts" / "kaggle"), \
         patch.object(run, "_dataset_exists", return_value=True), \
         patch.object(run, "_run_kaggle_cli") as mock_cli, \
         patch.object(run, "wait_for_dataset_ready") as mock_wait:
        run.push_source_dataset(repo, smoke=False, version_message="abc123")
    called_cmd = mock_cli.call_args[0][0]
    assert "version" in called_cmd
    assert "abc123" in called_cmd
    mock_wait.assert_called_once()


# ---------------------------------------------------------------------------
# Dataset readiness wait (the bug found on the first real smoke push:
# kernel pushed before the dataset version finished processing)
# ---------------------------------------------------------------------------


def test_get_dataset_status_parses_json():
    stdout = '{"status": "ready", "currentVersionNumber": 3}\n'
    with patch("subprocess.run", return_value=_fake_completed(stdout=stdout)):
        assert run.get_dataset_status("someuser/dme-oct-src") == "ready"


def test_get_dataset_status_unknown_on_nonzero_exit():
    with patch("subprocess.run", return_value=_fake_completed(returncode=1)):
        assert run.get_dataset_status("someuser/dme-oct-src") == "unknown"


def test_get_dataset_status_unknown_on_unparseable_output():
    with patch("subprocess.run", return_value=_fake_completed(stdout="not json")):
        assert run.get_dataset_status("someuser/dme-oct-src") == "unknown"


def test_wait_for_dataset_ready_returns_immediately_when_ready():
    with patch.object(run, "get_dataset_status", return_value="ready"), patch("time.sleep") as mock_sleep:
        run.wait_for_dataset_ready("someuser/dme-oct-src", poll_interval_sec=1, max_wait_minutes=1)
    mock_sleep.assert_not_called()


def test_wait_for_dataset_ready_raises_on_error_status():
    with patch.object(run, "get_dataset_status", return_value="error"), patch("time.sleep"):
        with pytest.raises(RuntimeError, match="failed processing"):
            run.wait_for_dataset_ready("someuser/dme-oct-src", poll_interval_sec=1, max_wait_minutes=1)


def test_wait_for_dataset_ready_polls_until_ready():
    """The exact scenario that broke the first real push: status isn't
    'ready' the instant the upload call returns -- must poll and wait."""
    with patch.object(run, "get_dataset_status", side_effect=["unknown", "unknown", "ready"]), \
         patch("time.sleep") as mock_sleep:
        run.wait_for_dataset_ready("someuser/dme-oct-src", poll_interval_sec=1, max_wait_minutes=5)
    assert mock_sleep.call_count == 2


def test_wait_for_dataset_ready_times_out():
    with patch.object(run, "get_dataset_status", return_value="unknown"), \
         patch("time.time", side_effect=[0, 0, 10 * 60, 10 * 60]), \
         patch("time.sleep"):
        with pytest.raises(RuntimeError, match="did not report 'ready'"):
            run.wait_for_dataset_ready("someuser/dme-oct-src", poll_interval_sec=1, max_wait_minutes=5)


# ---------------------------------------------------------------------------
# fetch() resilience: a Windows console UnicodeEncodeError inside the
# `kaggle kernels output` subprocess call once crashed this function with
# an uncaught RuntimeError before logs(slug) ever ran -- costing the one
# artifact that actually diagnoses a failed run. fetch() must now always
# attempt the log fetch, even when the output download itself fails.
# ---------------------------------------------------------------------------


def test_fetch_writes_output_and_log_on_success(tmp_path):
    out_dir = tmp_path / "out"
    with patch.object(run, "_run_kaggle_cli") as mock_cli, \
         patch.object(run, "logs", return_value="training log contents\n") as mock_logs:
        result = run.fetch("someuser/dme-oct-train", out_dir=out_dir)
    assert result == out_dir
    mock_cli.assert_called_once()
    mock_logs.assert_called_once_with("someuser/dme-oct-train")
    assert (out_dir / "run.log").read_text() == "training log contents\n"


def test_fetch_still_writes_log_when_output_download_fails(tmp_path):
    """The exact regression: _run_kaggle_cli raising RuntimeError for the
    `kaggle kernels output` step must not prevent logs(slug) from running."""
    out_dir = tmp_path / "out"
    with patch.object(run, "_run_kaggle_cli", side_effect=RuntimeError("kaggle kernels output failed (exit 1)")), \
         patch.object(run, "logs", return_value="the real traceback we needed\n") as mock_logs:
        result = run.fetch("someuser/dme-oct-train", out_dir=out_dir)
    mock_logs.assert_called_once_with("someuser/dme-oct-train")
    assert (result / "run.log").read_text() == "the real traceback we needed\n"


def test_fetch_writes_log_with_explicit_utf8_encoding(tmp_path):
    """The third distinct Windows encoding bug found on a real run: this
    write_text() call had no explicit encoding, so it defaulted to
    locale.getpreferredencoding() (cp1252 on Windows) -- a totally separate
    code path from the sys.stdout / subprocess fixes elsewhere in this
    file, which don't touch file I/O at all. This crashed on the very
    first fully-successful smoke run, on real training-log content
    containing a character cp1252 can't represent (an em dash), even
    though everything up to that point -- including printing this same
    text to the console -- had already succeeded. Content containing a
    character outside cp1252 (but valid UTF-8) proves the fix even when
    this test runs on a UTF-8-default platform."""
    out_dir = tmp_path / "out"
    non_cp1252_text = "training log with an em dash — and more\n"
    with patch.object(run, "_run_kaggle_cli"), \
         patch.object(run, "logs", return_value=non_cp1252_text):
        result = run.fetch("someuser/dme-oct-train", out_dir=out_dir)
    assert (result / "run.log").read_text(encoding="utf-8") == non_cp1252_text


def test_stage_source_dataset_writes_smoke_marker(tmp_path):
    repo = _fake_repo(tmp_path / "repo", {"config.py": "X=1"})
    stage_dir = tmp_path / "stage"
    with patch.object(run, "SOURCE_FILES", ["config.py"]):
        run._stage_source_dataset(repo, stage_dir, smoke=True)
    # Flat at the dataset root, not nested under a subfolder -- `kaggle
    # datasets create/version` silently skips subfolders by default
    # (confirmed against a real account: "Skipping folder: kaggle; use
    # '--dir-mode' to upload folders"), which is exactly how the smoke
    # marker went missing on the first two real pushes.
    assert (stage_dir / "SMOKE_MODE").read_text() == "true"
    assert not (stage_dir / "kaggle").exists()
    assert (stage_dir / "dataset-metadata.json").exists()
    assert (stage_dir / "config.py").exists()


def test_stage_source_dataset_never_creates_subfolders(tmp_path):
    """No file in SOURCE_FILES should land under a subfolder of the staged
    dataset dir -- kaggle datasets create/version skips directories by
    default, so anything nested here silently never uploads."""
    repo = _fake_repo(tmp_path / "repo", {"config.py": "X=1", "utils.py": "Y=2"})
    stage_dir = tmp_path / "stage"
    run._stage_source_dataset(repo, stage_dir, smoke=False)
    top_level_dirs = [p for p in stage_dir.iterdir() if p.is_dir()]
    assert top_level_dirs == []


def test_stage_source_dataset_writes_train_args_json_with_defaults(tmp_path):
    repo = _fake_repo(tmp_path / "repo", {"config.py": "X=1"})
    stage_dir = tmp_path / "stage"
    with patch.object(run, "SOURCE_FILES", ["config.py"]):
        run._stage_source_dataset(repo, stage_dir, smoke=False)
    train_args = json.loads((stage_dir / "TRAIN_ARGS.json").read_text())
    assert train_args == {"model_name": "tf_efficientnet_b3", "label_smoothing": 0.0}


def test_stage_source_dataset_writes_train_args_json_for_baseline(tmp_path):
    repo = _fake_repo(tmp_path / "repo", {"config.py": "X=1"})
    stage_dir = tmp_path / "stage"
    with patch.object(run, "SOURCE_FILES", ["config.py"]):
        run._stage_source_dataset(repo, stage_dir, smoke=True, model_name="resnet50", label_smoothing=0.05)
    train_args = json.loads((stage_dir / "TRAIN_ARGS.json").read_text())
    assert train_args == {"model_name": "resnet50", "label_smoothing": 0.05}
    # flat at the dataset root, same as SMOKE_MODE -- not nested under a subfolder.
    top_level_dirs = [p for p in stage_dir.iterdir() if p.is_dir()]
    assert top_level_dirs == []


def test_stage_kernel_push_folder_uses_config_username_not_placeholder(tmp_path):
    with patch.object(config, "KAGGLE_USERNAME", "someuser"), \
         patch.object(config, "KAGGLE_KERNEL_SLUG", "someuser/dme-oct-train"), \
         patch.object(config, "KAGGLE_SRC_DATASET_SLUG", "someuser/dme-oct-src"):
        stage_dir = tmp_path / "kstage"
        run._stage_kernel_push_folder(stage_dir)
        metadata = json.loads((stage_dir / "kernel-metadata.json").read_text())
    assert metadata["id"] == "someuser/dme-oct-train"
    assert "someuser/dme-oct-src" in metadata["dataset_sources"]
    assert config.KAGGLE_KERMANY_DATASET_SLUG in metadata["dataset_sources"]
    assert (stage_dir / "entry.py").exists()


# ---------------------------------------------------------------------------
# training_log.csv epoch-time parsing (used for full-run duration estimate)
# ---------------------------------------------------------------------------


def test_parse_epoch_seconds_averages_rows(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "training_log.csv").write_text(
        "phase,epoch_time_sec\nA,10.0\nA,20.0\nB,30.0\n"
    )
    avg = run._parse_epoch_seconds_from_training_log(tmp_path)
    assert avg == pytest.approx(20.0)


def test_parse_epoch_seconds_none_when_missing(tmp_path):
    assert run._parse_epoch_seconds_from_training_log(tmp_path) is None


# ---------------------------------------------------------------------------
# poll() loop (mocked get_status + time.sleep -- no real waiting)
# ---------------------------------------------------------------------------


def test_poll_loops_until_terminal_status():
    statuses = [("queued", None), ("running", None), ("complete", None)]
    with patch.object(run, "get_status", side_effect=statuses), patch("time.sleep") as mock_sleep:
        status, msg = run.poll("someuser/dme-oct-train", interval_sec=60, max_minutes=60)
    assert status == "complete"
    assert mock_sleep.call_count == 2  # slept between the two non-terminal checks


def test_poll_times_out_when_never_terminal():
    with patch.object(run, "get_status", return_value=("running", None)), \
         patch("time.time", side_effect=[0, 0, 100 * 60, 100 * 60]), \
         patch("time.sleep"):
        status, msg = run.poll("someuser/dme-oct-train", interval_sec=60, max_minutes=5)
    assert status == "timeout"
    assert "polling exceeded" in msg


# ---------------------------------------------------------------------------
# push_and_run() orchestration (push/poll/fetch mocked -- exercises the
# retry/classification/state-writing control flow for real)
# ---------------------------------------------------------------------------


@pytest.fixture
def _kaggle_artifacts(tmp_path):
    kdir = tmp_path / "artifacts" / "kaggle"
    with patch.object(config, "KAGGLE_ARTIFACTS_DIR", kdir), \
         patch.object(config, "KAGGLE_QUOTA_LOG", kdir / "quota_log.json"), \
         patch.object(config, "KAGGLE_LAST_SMOKE", kdir / "last_smoke.json"):
        yield kdir


def _fake_out_dir(tmp_path, log_text: str, epoch_time_sec: float = 12.0) -> Path:
    out_dir = tmp_path / "fetched"
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (out_dir / "run.log").write_text(log_text, encoding="utf-8")  # matches fetch()'s real write path
    (out_dir / "artifacts" / "training_log.csv").write_text(
        f"phase,epoch_time_sec\nA,{epoch_time_sec}\n"
    )
    return out_dir


def test_push_and_run_smoke_success_records_last_smoke(tmp_path, _kaggle_artifacts):
    out_dir = _fake_out_dir(tmp_path, "training...\n[C5] class_to_idx vs CLASS_WEIGHTS alignment: PASSED\n")
    with patch.object(run, "push", return_value="someuser/dme-oct-train") as mock_push, \
         patch.object(run, "poll", return_value=("complete", None)), \
         patch.object(run, "fetch", return_value=out_dir), \
         patch.object(run, "git_sha_or_source_hash", return_value="abc123"):
        result = run.push_and_run(Path("/fake/repo"), smoke=True)

    assert result["status"] == "complete"
    assert result["c5_line"] is not None
    assert result["attempts"] == 1
    mock_push.assert_called_once()

    last_smoke = json.loads((config.KAGGLE_LAST_SMOKE).read_text())
    assert last_smoke["status"] == "passed"
    assert last_smoke["source_hash"] == "abc123"
    assert last_smoke["c5_passed"] is True
    assert last_smoke["epoch_seconds"] == pytest.approx(12.0)
    # Phase 3g Baselines: defaults recorded even when the caller doesn't
    # pass --model-name/--label-smoothing, so a later full push's smoke
    # gate check has something concrete to compare against.
    assert last_smoke["model_name"] == "tf_efficientnet_b3"
    assert last_smoke["label_smoothing"] == 0.0


def test_push_and_run_smoke_records_non_default_model_name_and_label_smoothing(tmp_path, _kaggle_artifacts):
    out_dir = _fake_out_dir(tmp_path, "training...\n[C5] class_to_idx vs CLASS_WEIGHTS alignment: PASSED\n")
    with patch.object(run, "push", return_value="someuser/dme-oct-train") as mock_push, \
         patch.object(run, "poll", return_value=("complete", None)), \
         patch.object(run, "fetch", return_value=out_dir), \
         patch.object(run, "git_sha_or_source_hash", return_value="abc123"):
        run.push_and_run(Path("/fake/repo"), smoke=True, model_name="resnet50", label_smoothing=0.05)

    mock_push.assert_called_once_with(Path("/fake/repo"), True, None, model_name="resnet50", label_smoothing=0.05)
    last_smoke = json.loads((config.KAGGLE_LAST_SMOKE).read_text())
    assert last_smoke["model_name"] == "resnet50"
    assert last_smoke["label_smoothing"] == 0.05


def test_push_and_run_records_actual_minutes_in_quota_log(tmp_path, _kaggle_artifacts):
    """actual_minutes was written as None by push() and nothing ever
    updated it afterward -- confirmed by reading a real quota_log.json
    after a genuinely successful smoke push, where it still showed null
    for every entry, meaning gpu_minutes_used_last_7_days() was silently
    falling back to the static expected_minutes estimate instead of real
    usage. This exercises the FALLBACK path specifically: the fake log
    here isn't the real JSON format, so _real_duration_minutes_from_log()
    returns None and push_and_run() must fall back to poll()'s wall-clock
    duration (see test_push_and_run_prefers_real_log_duration below for
    the preferred path)."""
    quota_log_path = config.KAGGLE_QUOTA_LOG
    quota_log_path.parent.mkdir(parents=True, exist_ok=True)
    quota_log_path.write_text(json.dumps([
        {
            "timestamp": "2026-09-13T18:05:34.924126+00:00",
            "source_hash": "abc123",
            "kernel_slug": "someuser/dme-oct-train",
            "smoke": True,
            "expected_minutes": 10.0,
            "actual_minutes": None,
            "status": "pushed",
        }
    ]))
    out_dir = _fake_out_dir(tmp_path, "[C5] class_to_idx vs CLASS_WEIGHTS alignment: PASSED\n")
    with patch.object(run, "push", return_value="someuser/dme-oct-train"), \
         patch.object(run, "poll", return_value=("complete", None)), \
         patch.object(run, "fetch", return_value=out_dir), \
         patch.object(run, "git_sha_or_source_hash", return_value="abc123"), \
         patch("time.time", side_effect=[1000.0, 1000.0 + 13.0 * 60]):
        run.push_and_run(Path("/fake/repo"), smoke=True)

    quota_log = json.loads(quota_log_path.read_text())
    assert quota_log[-1]["actual_minutes"] == pytest.approx(13.0, abs=0.01)
    assert quota_log[-1]["status"] == "complete"


def test_push_and_run_prefers_real_log_duration_over_wall_clock(tmp_path, _kaggle_artifacts):
    """The preferred path: when the fetched log is real, parseable JSON
    with timestamps, push_and_run() must use the log's own last timestamp
    for actual_minutes -- NOT poll()'s wall-clock time. This is what
    protects a future run from a repeat of the real incident where a ~9
    hour local network/sleep outage during polling got recorded as ~682
    "actual" GPU minutes for a push that really completed in 319.77."""
    quota_log_path = config.KAGGLE_QUOTA_LOG
    quota_log_path.parent.mkdir(parents=True, exist_ok=True)
    quota_log_path.write_text(json.dumps([
        {
            "timestamp": "2026-09-16T16:20:25.434734+00:00",
            "source_hash": "abc123",
            "kernel_slug": "someuser/dme-oct-train",
            "smoke": False,
            "expected_minutes": 32.8,
            "actual_minutes": None,
            "status": "pushed",
        }
    ]))
    real_log = json.dumps([
        {"stream_name": "stdout", "time": 1.5, "data": "start\n"},
        {"stream_name": "stdout", "time": 19186.45, "data": "[C5] class_to_idx vs CLASS_WEIGHTS alignment: PASSED\n"},
    ])
    out_dir = _fake_out_dir(tmp_path, real_log)
    with patch.object(run, "push", return_value="someuser/dme-oct-train"), \
         patch.object(run, "poll", return_value=("complete", None)), \
         patch.object(run, "fetch", return_value=out_dir), \
         patch.object(run, "git_sha_or_source_hash", return_value="abc123"), \
         patch("time.time", side_effect=[1000.0, 1000.0 + 682.0 * 60]):  # wall-clock says 682min
        run.push_and_run(Path("/fake/repo"), smoke=False)

    quota_log = json.loads(quota_log_path.read_text())
    assert quota_log[-1]["actual_minutes"] == pytest.approx(19186.45 / 60.0, abs=0.01)  # not 682.0
    assert quota_log[-1]["status"] == "complete"


def test_push_and_run_reads_run_log_with_non_ascii_content(tmp_path, _kaggle_artifacts):
    """The exact real training log that first fully succeeded contained a
    C8 warning with an em dash -- confirms the run.log read path (not just
    fetch()'s write) round-trips non-cp1252 content without raising."""
    log_text = (
        "(C8) 305 patient(s) appear in both train/val and the official test split "
        "— this compromises the evaluation regardless of our own split hygiene.\n"
        "[C5] class_to_idx vs CLASS_WEIGHTS alignment: PASSED\n"
    )
    out_dir = _fake_out_dir(tmp_path, log_text)
    with patch.object(run, "push", return_value="someuser/dme-oct-train"), \
         patch.object(run, "poll", return_value=("complete", None)), \
         patch.object(run, "fetch", return_value=out_dir), \
         patch.object(run, "git_sha_or_source_hash", return_value="abc123"):
        result = run.push_and_run(Path("/fake/repo"), smoke=True)
    assert result["status"] == "complete"
    assert result["c5_line"] is not None


def test_push_and_run_code_error_raises_without_retry(tmp_path, _kaggle_artifacts):
    out_dir = _fake_out_dir(tmp_path, "Traceback (most recent call last):\nValueError: boom\n")
    with patch.object(run, "push", return_value="someuser/dme-oct-train") as mock_push, \
         patch.object(run, "poll", return_value=("error", "some message")), \
         patch.object(run, "fetch", return_value=out_dir), \
         patch.object(run, "git_sha_or_source_hash", return_value="abc123"):
        with pytest.raises(run.RunFailed, match="code error"):
            run.push_and_run(Path("/fake/repo"), smoke=True)
    assert mock_push.call_count == 1  # never retried

    last_smoke = json.loads((config.KAGGLE_LAST_SMOKE).read_text())
    assert last_smoke["status"] == "failed"
    assert last_smoke["failure_type"] == "code_error"
    assert last_smoke["model_name"] == "tf_efficientnet_b3"  # recorded even on the failure path


def test_push_and_run_infra_error_retries_then_succeeds(tmp_path, _kaggle_artifacts):
    fail_dir = _fake_out_dir(tmp_path / "1", "no traceback here\n")
    ok_dir = _fake_out_dir(tmp_path / "2", "[C5] class_to_idx vs CLASS_WEIGHTS alignment: PASSED\n")
    with patch.object(run, "push", return_value="someuser/dme-oct-train"), \
         patch.object(run, "poll", side_effect=[("error", "session terminated unexpectedly"), ("complete", None)]), \
         patch.object(run, "fetch", side_effect=[fail_dir, ok_dir]), \
         patch.object(run, "git_sha_or_source_hash", return_value="abc123"):
        result = run.push_and_run(Path("/fake/repo"), smoke=True)
    assert result["status"] == "complete"
    assert result["attempts"] == 2


def test_push_and_run_infra_error_exhausts_retry_budget(tmp_path, _kaggle_artifacts):
    dirs = [_fake_out_dir(tmp_path / str(i), "no traceback\n") for i in range(5)]
    with patch.object(run, "push", return_value="someuser/dme-oct-train") as mock_push, \
         patch.object(run, "poll", return_value=("timeout", "polling exceeded max minutes")), \
         patch.object(run, "fetch", side_effect=dirs), \
         patch.object(run, "git_sha_or_source_hash", return_value="abc123"):
        with pytest.raises(run.RunFailed, match="retry budget"):
            run.push_and_run(Path("/fake/repo"), smoke=True)
    # KAGGLE_MAX_INFRA_RETRIES retries -> MAX_INFRA_RETRIES + 1 total push attempts
    assert mock_push.call_count == config.KAGGLE_MAX_INFRA_RETRIES + 1
