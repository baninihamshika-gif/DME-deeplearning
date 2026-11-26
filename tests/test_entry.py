"""
Tests for the parts of kaggle/entry.py that don't require a real Kaggle
container: the input-dataset search/wait loop and the requirements.txt
filter. Everything else in entry.py (the actual /kaggle/input paths, pip
install subprocess, train.main() call) only runs for real inside a live
Kaggle kernel -- see the module's own docstring.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))

import entry


# ---------------------------------------------------------------------------
# _find_input_dataset_dir / _wait_for_input_dataset
#
# History: the first real smoke push assumed /kaggle/input/<slug> and
# failed with FileNotFoundError. A wait-and-retry (in case it was a mount
# propagation lag) ruled that out on push #2 -- the diagnostic listing
# showed /kaggle/input contains a single "datasets" wrapper directory, not
# the slugs directly, so the fixed-path assumption itself was wrong. These
# functions search by name instead of assuming a fixed depth.
# ---------------------------------------------------------------------------


def test_find_input_dataset_dir_finds_direct_child(tmp_path):
    input_dir = tmp_path / "input"
    target = input_dir / "dme-oct-src"
    target.mkdir(parents=True)
    with patch.object(entry, "INPUT_DIR", input_dir):
        assert entry._find_input_dataset_dir("dme-oct-src") == target


def test_find_input_dataset_dir_finds_nested_child():
    """The exact real-world case that broke the fixed-path assumption:
    /kaggle/input/datasets/<slug>, not /kaggle/input/<slug>."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        input_dir = Path(tmp) / "input"
        target = input_dir / "datasets" / "dme-oct-src"
        target.mkdir(parents=True)
        with patch.object(entry, "INPUT_DIR", input_dir):
            assert entry._find_input_dataset_dir("dme-oct-src") == target


def test_find_input_dataset_dir_none_when_absent(tmp_path):
    input_dir = tmp_path / "input"
    (input_dir / "datasets" / "kermany2018").mkdir(parents=True)
    with patch.object(entry, "INPUT_DIR", input_dir):
        assert entry._find_input_dataset_dir("dme-oct-src") is None


def test_find_input_dataset_dir_respects_max_depth(tmp_path):
    input_dir = tmp_path / "input"
    # 4 levels deep: input/a/b/c/dme-oct-src
    target = input_dir / "a" / "b" / "c" / "dme-oct-src"
    target.mkdir(parents=True)
    with patch.object(entry, "INPUT_DIR", input_dir):
        assert entry._find_input_dataset_dir("dme-oct-src", max_depth=2) is None
        assert entry._find_input_dataset_dir("dme-oct-src", max_depth=4) == target


def test_find_input_dataset_dir_never_descends_into_matched_directory(tmp_path):
    """Must not walk into files/subdirs of a directory once it's already
    matched by name -- entering a matched dataset's own huge class-image
    folders would defeat the whole point of bounding the search."""
    input_dir = tmp_path / "input"
    target = input_dir / "kermany2018"
    (target / "oct2017" / "train" / "DME").mkdir(parents=True)

    queried = []
    real_iterdir = Path.iterdir

    def tracking_iterdir(self):
        queried.append(self)
        return real_iterdir(self)

    with patch.object(entry, "INPUT_DIR", input_dir), patch.object(Path, "iterdir", tracking_iterdir):
        result = entry._find_input_dataset_dir("kermany2018")

    assert result == target
    assert not any(target in q.parents or q == target for q in queried), (
        f"search descended into the matched directory: {queried}"
    )


def test_wait_for_input_dataset_returns_immediately_when_present(tmp_path):
    input_dir = tmp_path / "input"
    (input_dir / "datasets" / "dme-oct-src").mkdir(parents=True)
    with patch.object(entry, "INPUT_DIR", input_dir), patch("time.sleep") as mock_sleep:
        result = entry._wait_for_input_dataset("dme-oct-src", timeout_sec=60, poll_sec=1)
    assert result == input_dir / "datasets" / "dme-oct-src"
    mock_sleep.assert_not_called()


def test_wait_for_input_dataset_retries_until_it_appears(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    target = input_dir / "datasets" / "dme-oct-src"

    call_count = {"n": 0}

    def fake_sleep(_seconds):
        call_count["n"] += 1
        if call_count["n"] == 2:  # appears on the 2nd sleep, so the 3rd check finds it
            target.mkdir(parents=True)

    with patch.object(entry, "INPUT_DIR", input_dir), patch("time.sleep", side_effect=fake_sleep):
        result = entry._wait_for_input_dataset("dme-oct-src", timeout_sec=60, poll_sec=1)
    assert result == target
    assert call_count["n"] == 2


def test_wait_for_input_dataset_raises_with_diagnostic_tree_after_timeout(tmp_path):
    input_dir = tmp_path / "input"
    (input_dir / "datasets" / "kermany2018").mkdir(parents=True)  # present, but not what we're waiting for
    with patch.object(entry, "INPUT_DIR", input_dir), \
         patch("time.sleep"), \
         patch("time.time", side_effect=[0, 0, 200, 200]):
        with pytest.raises(FileNotFoundError, match="dme-oct-src"):
            entry._wait_for_input_dataset("dme-oct-src", timeout_sec=180, poll_sec=5)


# ---------------------------------------------------------------------------
# _requirements_to_install
# ---------------------------------------------------------------------------


def _write_requirements(working_dir: Path, text: str) -> None:
    working_dir.mkdir(parents=True, exist_ok=True)
    (working_dir / "requirements.txt").write_text(text)


def test_requirements_to_install_skips_torch_torchvision_kaggle(tmp_path):
    _write_requirements(
        tmp_path,
        "torch==2.3.*\ntorchvision==0.18.*\ntimm==0.9.*\nkaggle==1.6.*\nalbumentations==1.4.*\n",
    )
    packages = entry._requirements_to_install(tmp_path)
    assert packages == ["timm==0.9.*", "albumentations==1.4.*"]


def test_requirements_to_install_skips_comments_and_blank_lines(tmp_path):
    _write_requirements(tmp_path, "# a comment\n\ntimm==0.9.*  # inline note\n\n")
    packages = entry._requirements_to_install(tmp_path)
    assert packages == ["timm==0.9.*"]


def test_requirements_to_install_is_case_insensitive_on_skip_list(tmp_path):
    _write_requirements(tmp_path, "Torch==2.3.*\ntimm==0.9.*\n")
    packages = entry._requirements_to_install(tmp_path)
    assert packages == ["timm==0.9.*"]


# ---------------------------------------------------------------------------
# _detect_gpu_name / _ensure_gpu_torch_compat
#
# Real push #5 hit a genuine Kaggle-side bug (confirmed against
# Kaggle/docker-python#1546): the preinstalled torch build on Kaggle's GPU
# image doesn't support the P100's CUDA compute capability (sm_60), so any
# real op crashes with "CUDA error: no kernel image is available for
# execution on the device". These functions detect that GPU by name via
# nvidia-smi -- BEFORE torch is ever imported in this process -- and install
# a known-compatible build instead of trusting the (for P100) broken default.
# ---------------------------------------------------------------------------


def _fake_nvidia_smi_result(stdout="", returncode=0):
    from subprocess import CompletedProcess
    return CompletedProcess(args=["nvidia-smi"], returncode=returncode, stdout=stdout, stderr="")


def test_detect_gpu_name_returns_stripped_name_on_success():
    with patch("subprocess.run", return_value=_fake_nvidia_smi_result(stdout="Tesla P100-PCIE-16GB\n")):
        assert entry._detect_gpu_name() == "Tesla P100-PCIE-16GB"


def test_detect_gpu_name_returns_empty_on_nonzero_exit():
    with patch("subprocess.run", return_value=_fake_nvidia_smi_result(returncode=1)):
        assert entry._detect_gpu_name() == ""


def test_detect_gpu_name_returns_empty_when_nvidia_smi_missing():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert entry._detect_gpu_name() == ""


def test_detect_gpu_name_returns_empty_on_timeout():
    import subprocess as sp
    with patch("subprocess.run", side_effect=sp.TimeoutExpired(cmd="nvidia-smi", timeout=30)):
        assert entry._detect_gpu_name() == ""


def test_ensure_gpu_torch_compat_reinstalls_torch_when_p100_detected():
    with patch.object(entry, "_detect_gpu_name", return_value="Tesla P100-PCIE-16GB"), \
         patch.object(entry, "_verify_gpu_torch_compat") as mock_verify, \
         patch("subprocess.run") as mock_run:
        entry._ensure_gpu_torch_compat()
    mock_run.assert_called_once()
    called_cmd = mock_run.call_args[0][0]
    assert entry.COMPATIBLE_TORCH_INDEX_URL in called_cmd
    for pkg in entry.COMPATIBLE_TORCH_PACKAGES:
        assert pkg in called_cmd
    # --force-reinstall is load-bearing here, not decoration: confirmed the
    # hard way on a real push that without it, pip treats an already-
    # installed "2.10.0+cu128" as satisfying a bare "torch==2.10.0"
    # requirement and silently does nothing, leaving the broken build in
    # place.
    assert "--force-reinstall" in called_cmd
    mock_verify.assert_called_once()  # must confirm the reinstall actually worked


def test_ensure_gpu_torch_compat_leaves_other_gpus_alone():
    """No evidence T4 (or anything else) is broken -- reinstalling
    unnecessarily just burns startup time on every push."""
    with patch.object(entry, "_detect_gpu_name", return_value="Tesla T4"), \
         patch.object(entry, "_verify_gpu_torch_compat") as mock_verify, \
         patch("subprocess.run") as mock_run:
        entry._ensure_gpu_torch_compat()
    mock_run.assert_not_called()
    mock_verify.assert_not_called()


def test_ensure_gpu_torch_compat_noop_when_no_gpu_detected():
    with patch.object(entry, "_detect_gpu_name", return_value=""), \
         patch.object(entry, "_verify_gpu_torch_compat") as mock_verify, \
         patch("subprocess.run") as mock_run:
        entry._ensure_gpu_torch_compat()
    mock_run.assert_not_called()
    mock_verify.assert_not_called()


# ---------------------------------------------------------------------------
# _verify_gpu_torch_compat
#
# Added after a real reinstall attempt silently no-op'd (see above) and the
# same crash recurred ~15 minutes into a real run. This check runs in
# seconds at startup instead, so a still-broken build fails fast and
# clearly rather than burning GPU quota to rediscover the same crash deep
# in a forward pass.
# ---------------------------------------------------------------------------


def _fake_verify_result(stdout="", returncode=0):
    from subprocess import CompletedProcess
    return CompletedProcess(args=[sys.executable, "-c", "..."], returncode=returncode, stdout=stdout, stderr="")


def test_verify_gpu_torch_compat_passes_silently_when_compatible():
    with patch("subprocess.run", return_value=_fake_verify_result(
        stdout="[verify] torch 2.10.0+cu126, GPU arch sm_60, supported ['sm_60', 'sm_70']\n", returncode=0,
    )):
        entry._verify_gpu_torch_compat()  # must not raise


def test_verify_gpu_torch_compat_raises_clearly_when_still_incompatible():
    with patch("subprocess.run", return_value=_fake_verify_result(
        stdout="[verify] torch 2.10.0+cu128, GPU arch sm_60, supported ['sm_70', 'sm_75']\n", returncode=1,
    )):
        with pytest.raises(RuntimeError, match="GPU/torch compatibility check failed"):
            entry._verify_gpu_torch_compat()


# ---------------------------------------------------------------------------
# _read_train_args (Phase 3g Baselines: --model-name/--label-smoothing
# passthrough from kaggle/run.py's TRAIN_ARGS.json marker file, staged flat
# at the dataset root -- same reasoning as SMOKE_MODE above.)
# ---------------------------------------------------------------------------


def test_read_train_args_defaults_when_marker_file_absent(tmp_path):
    """A source tree staged by a pre-Baselines run.py has no TRAIN_ARGS.json
    at all -- must fall back to train.py's own defaults, not crash."""
    result = entry._read_train_args(tmp_path)
    assert result == {"model_name": "tf_efficientnet_b3", "label_smoothing": 0.0}


def test_read_train_args_reads_staged_marker_file(tmp_path):
    (tmp_path / "TRAIN_ARGS.json").write_text(json.dumps({"model_name": "resnet50", "label_smoothing": 0.05}))
    result = entry._read_train_args(tmp_path)
    assert result == {"model_name": "resnet50", "label_smoothing": 0.05}
