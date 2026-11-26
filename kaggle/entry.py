"""
Thin Kaggle kernel bootstrap. Per PROJECT_BRIEF.md Section 7 ("notebooks and
entry.py contain no logic"), this file does exactly four things and nothing
else -- all real logic lives in the importable modules it calls into:

  1. Copy the source tree from the attached `dme-oct-src` dataset into
     /kaggle/working (writable) so config.REPO_ROOT resolves somewhere
     checkpoints/artifacts can actually be written -- /kaggle/input is
     read-only, so running train.py directly out of the input dataset
     would fail the first time it tries to save a checkpoint.
  2. Install the non-preinstalled dependencies.
  3. Locate the DME/NORMAL train and test folders inside the attached
     kermany2018 dataset (search, not a hardcoded path -- see note below).
  4. Import and call train.main(), forwarding --smoke if kaggle/run.py
     staged a smoke marker file.

Most of this file only runs inside a real Kaggle kernel, against real
Kaggle mount paths, so it can't be meaningfully unit-tested -- tests/test_
entry.py covers what can be (the input-directory search/wait loop and the
requirements.txt filter) with a fake INPUT_DIR, not a live kernel. Three
real smoke pushes so far have found real bugs this reasoning-without-a-
live-account approach missed; see the comments on INPUT_SEARCH_MAX_DEPTH
below for the most recent one.
"""

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

INPUT_DIR = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")
SRC_DATASET_DIRNAME = "dme-oct-src"
KERMANY_DATASET_DIRNAME = "kermany2018"

# The first three real smoke pushes all failed trying to reach
# /kaggle/input/<dataset-slug> directly. A wait-and-retry (in case it was a
# mount-propagation lag) ruled that out on push #2: the diagnostic listing
# printed on every retry showed /kaggle/input contains exactly one entry,
# a directory literally named "datasets" -- not the dataset slugs
# themselves. That means the long-assumed convention
# (/kaggle/input/<dataset-slug>/) is wrong for this account/image, full
# stop, not a timing issue. Since this project's training data has a
# January 2026 cutoff and this session is running in September 2026,
# Kaggle's mount layout may simply have changed since -- rather than guess
# a new fixed path from here (unverifiable without a live account) and
# risk a fourth wrong guess, _find_input_dataset_dir() below SEARCHES for
# the target directory by name under /kaggle/input, bounded to a depth
# shallow enough to never reach into kermany2018's actual class folders
# (DME/NORMAL, tens of thousands of image files each -- see
# _locate_split_dir below, several levels deeper still). The wait/retry
# loop stays, now wrapped around the search instead of a fixed path, in
# case there's *also* a genuine propagation lag on top of the layout
# difference.
INPUT_SEARCH_MAX_DEPTH = 5
INPUT_MOUNT_WAIT_TIMEOUT_SEC = 180
INPUT_MOUNT_WAIT_POLL_SEC = 5

# Packages NOT pip-installed here, even though requirements.txt pins them:
# torch, torchvision, kaggle. Kaggle GPU images ship a preinstalled
# torch/torchvision build matched to the image's CUDA driver -- pip
# installing requirements.txt's pinned torch==2.3.* over that risks
# silently resolving a wheel built for a different CUDA version (or a
# CPU-only wheel), which would either break GPU acceleration outright or
# fail in a way that burns GPU-hour quota without training anything.
# `kaggle` (the CLI/API package) has no reason to run inside its own
# kernel. Every other pinned package is safe to install because it doesn't
# touch the CUDA runtime. This is a deliberate deviation from a plain
# "pip install -r requirements.txt" and should be verified against the
# real Kaggle image's preinstalled torch version on the first smoke run.
PIP_SKIP_PACKAGES = ("torch", "torchvision", "kaggle")


def _requirements_to_install(working_dir: Path) -> list[str]:
    req_path = working_dir / "requirements.txt"
    lines = []
    for raw in req_path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        pkg_name = re.split(r"[=<>~!]", line, maxsplit=1)[0].strip()
        if pkg_name.lower() in PIP_SKIP_PACKAGES:
            continue
        lines.append(line)
    return lines


def _find_input_dataset_dir(dirname: str, max_depth: int = INPUT_SEARCH_MAX_DEPTH):
    """
    Search (directories only, bounded to max_depth) under INPUT_DIR for a
    directory literally named `dirname`, instead of assuming a fixed mount
    path -- see the module-level note above on why. Depth-bounded so it
    can never wander into kermany2018's class-image folders however deep
    the real mount nests things: a dataset ROOT name should turn up within
    a few levels under /kaggle/input in any plausible layout, well before
    reaching individual class folders with tens of thousands of files.
    Returns None if not found within max_depth.
    """
    if not INPUT_DIR.exists():
        return None
    stack = [(INPUT_DIR, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            children = [c for c in current.iterdir() if c.is_dir()]
        except OSError:
            continue
        for child in children:
            if child.name == dirname:
                return child
            if depth + 1 <= max_depth:
                stack.append((child, depth + 1))
    return None


def _list_input_tree(max_depth: int = INPUT_SEARCH_MAX_DEPTH) -> list:
    """Directory-only listing under INPUT_DIR, same depth bound as the search above -- for diagnostics."""
    lines = []
    stack = [(INPUT_DIR, 0)]
    while stack and INPUT_DIR.exists():
        current, depth = stack.pop()
        try:
            children = sorted((c for c in current.iterdir() if c.is_dir()), key=lambda p: str(p))
        except OSError:
            continue
        for child in children:
            lines.append(str(child))
            if depth + 1 <= max_depth:
                stack.append((child, depth + 1))
    return sorted(lines)


def _wait_for_input_dataset(
    dirname: str, timeout_sec: int = INPUT_MOUNT_WAIT_TIMEOUT_SEC, poll_sec: int = INPUT_MOUNT_WAIT_POLL_SEC
) -> Path:
    start = time.time()
    attempt = 0
    while True:
        elapsed = time.time() - start
        found = _find_input_dataset_dir(dirname)
        if found is not None:
            if attempt > 0:
                print(f"[entry] found {found} after {elapsed:.0f}s (attempt {attempt + 1})")
            return found
        tree = _list_input_tree()
        print(f"[entry] no directory named {dirname!r} found yet under {INPUT_DIR} "
              f"(attempt {attempt + 1}, {elapsed:.0f}s elapsed). Current tree: {tree}")
        if elapsed >= timeout_sec:
            raise FileNotFoundError(
                f"No directory named {dirname!r} found anywhere under {INPUT_DIR} within {timeout_sec}s "
                f"(searched {INPUT_SEARCH_MAX_DEPTH} levels deep). Tree at timeout: {tree}"
            )
        attempt += 1
        time.sleep(poll_sec)


def _copy_source_tree() -> Path:
    src_dir = _wait_for_input_dataset(SRC_DATASET_DIRNAME)
    print(f"[entry] copying source tree: {src_dir} -> {WORKING_DIR}")
    for item in src_dir.iterdir():
        dest = WORKING_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    return WORKING_DIR


def _pip_install(working_dir: Path) -> None:
    packages = _requirements_to_install(working_dir)
    print(f"[entry] pip installing from requirements.txt, skipping {PIP_SKIP_PACKAGES} "
          f"(torch/torchvision left as the Kaggle image's preinstalled build): {packages}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet"] + packages,
        check=True,
    )


# Confirmed via Kaggle/docker-python#1546 (2026-09), a real Kaggle-side bug,
# not ours: the preinstalled PyTorch build on Kaggle's GPU image (2.10.0+cu128
# at time of writing) was compiled without CUDA compute capability sm_60
# support, so it crashes ("CUDA error: no kernel image is available for
# execution on the device") the moment a real op runs on a Tesla P100 --
# a GPU Kaggle still allocates to free-tier kernels. Unresolved as of this
# session. There is also no confirmed way to request a different GPU type
# (T4 vs P100) via the Kaggle API -- it's an open, unaddressed feature
# request -- so this can't be worked around by hoping a retry lands on a
# different GPU. Instead, detect the P100 by name via nvidia-smi BEFORE
# torch is imported anywhere in this process (swapping the installed torch
# build out from under an already-imported, CUDA-initialized torch is not
# reliable) and install the community-confirmed working build in its place.
# Any other GPU (e.g. T4) is left on the preinstalled build -- there's no
# evidence that one is broken, and reinstalling unnecessarily just burns
# startup time.
KNOWN_INCOMPATIBLE_GPU_NAME = "P100"
COMPATIBLE_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu126"
COMPATIBLE_TORCH_PACKAGES = ["torch==2.10.0", "torchvision", "torchaudio"]


def _detect_gpu_name() -> str:
    """Query nvidia-smi directly, before torch (or anything that might import
    it) has run. Returns "" if nvidia-smi is unavailable or reports nothing
    (e.g. no GPU attached to this kernel) -- callers should treat that as
    "nothing to fix" rather than an error."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


_VERIFY_TORCH_GPU_SNIPPET = (
    "import sys, torch\n"
    "cap = torch.cuda.get_device_capability(0)\n"
    "arch = f'sm_{cap[0]}{cap[1]}'\n"
    "supported = torch.cuda.get_arch_list()\n"
    "print(f'[verify] torch {torch.__version__}, GPU arch {arch}, supported {supported}')\n"
    "sys.exit(0 if arch in supported else 1)\n"
)


def _ensure_gpu_torch_compat() -> None:
    gpu_name = _detect_gpu_name()
    print(f"[entry] detected GPU: {gpu_name!r}")
    if KNOWN_INCOMPATIBLE_GPU_NAME not in gpu_name:
        return
    print(f"[entry] GPU name contains {KNOWN_INCOMPATIBLE_GPU_NAME!r} -- Kaggle's preinstalled torch build is "
          f"known to lack CUDA kernels for this GPU (Kaggle/docker-python#1546, unresolved as of 2026-09). "
          f"Installing a known-compatible build before torch is ever imported: {COMPATIBLE_TORCH_PACKAGES} "
          f"from {COMPATIBLE_TORCH_INDEX_URL}")
    # --force-reinstall is load-bearing, not decoration: confirmed the hard
    # way on a real push -- PyTorch wheels carry their CUDA build as a PEP
    # 440 *local* version segment (e.g. "2.10.0+cu128" vs "2.10.0+cu126"),
    # and `pip install torch==2.10.0` (no local segment) treats an already-
    # installed "2.10.0+cu128" as satisfying that requirement. Without
    # --force-reinstall, pip printed "Requirement already satisfied" for
    # all three packages and did nothing -- the broken cu128 build stayed
    # in place, and training crashed with the exact same CUDA error as
    # before, just a bit later. This is fast to run either way (no-network
    # "already satisfied" or a real download), so forcing it costs little.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--force-reinstall",
         "--index-url", COMPATIBLE_TORCH_INDEX_URL] + COMPATIBLE_TORCH_PACKAGES,
        check=True,
    )
    _verify_gpu_torch_compat()


def _verify_gpu_torch_compat() -> None:
    """
    Confirm the reinstall actually fixed things, in a FRESH subprocess --
    not by importing torch in this process, which wouldn't reliably reflect
    a package that was just reinstalled out from under an interpreter that
    may have already touched it, and wouldn't give an honest read of what
    entry.py's own later `import train` (which imports torch) will get
    either way. Fails fast with a clear, specific message here (seconds,
    at startup) instead of letting a still-incompatible build crash
    confusingly deep inside a forward pass tens of minutes into training,
    as happened for real on the run this reinstall was added to fix.
    """
    result = subprocess.run(
        [sys.executable, "-c", _VERIFY_TORCH_GPU_SNIPPET],
        capture_output=True, text=True,
    )
    if result.stdout:
        print(f"[entry] {result.stdout.strip()}")
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            "GPU/torch compatibility check failed even after installing the known-compatible build "
            f"({COMPATIBLE_TORCH_PACKAGES} from {COMPATIBLE_TORCH_INDEX_URL}) -- see the "
            "[verify] line above (or stderr) for the detected GPU arch vs. torch's supported arch list."
        )


def _read_train_args(working_dir: Path) -> dict:
    """
    Phase 3g Baselines: kaggle/run.py stages TRAIN_ARGS.json flat at the
    dataset root (same reasoning as SMOKE_MODE above -- `kaggle datasets
    create/version` silently skips subfolders by default). Defaults here
    match train.py's own argparse defaults, so a source tree staged by an
    older run.py (pre-Baselines, no TRAIN_ARGS.json written yet) still
    trains the original primary configuration unchanged.
    """
    path = working_dir / "TRAIN_ARGS.json"
    if not path.exists():
        return {"model_name": "tf_efficientnet_b3", "label_smoothing": 0.0}
    return json.loads(path.read_text())


def _locate_split_dir(base: Path, split_name: str) -> Path:
    """
    Find the `split_name` (e.g. "train") folder that directly contains
    DME/ and NORMAL/ subfolders, searching rather than hardcoding a path.
    Kermany's zip has a known trailing-space folder name quirk
    ("OCT2017 ") and a duplicate nested branch -- the Kaggle dataset's
    exact internal layout was not independently verified from this
    session (no network route to kaggle.com from here), so a brittle
    hardcoded path would silently break on whatever the real layout turns
    out to be. This picks the FIRST match and prints every candidate
    found, with repr() so a trailing space is visible in the log.
    """
    candidates = [
        p for p in base.rglob(split_name)
        if p.is_dir() and (p / "DME").is_dir() and (p / "NORMAL").is_dir()
    ]
    print(f"[entry] candidates for split={split_name!r} under {base}: {[repr(str(p)) for p in candidates]}")
    if not candidates:
        raise FileNotFoundError(
            f"No {split_name}/ folder containing both DME/ and NORMAL/ subfolders found under {base}. "
            "Print the full tree and update this search if the dataset's layout differs from expected."
        )
    if len(candidates) > 1:
        print(f"[entry] WARNING: multiple {split_name}/ candidates found, using the first: {candidates[0]!r}")
    return candidates[0]


def main() -> None:
    working_dir = _copy_source_tree()
    _ensure_gpu_torch_compat()
    _pip_install(working_dir)

    sys.path.insert(0, str(working_dir))
    import config  # noqa: E402  (must import after sys.path is set up)
    import train  # noqa: E402

    print(f"[entry] config.ON_KAGGLE = {config.ON_KAGGLE}")
    assert config.ON_KAGGLE, "entry.py is running but config.ON_KAGGLE is False -- /kaggle/input not detected?"

    kermany_root = _wait_for_input_dataset(KERMANY_DATASET_DIRNAME)
    train_dir = _locate_split_dir(kermany_root, "train")
    test_dir = _locate_split_dir(kermany_root, "test")

    # Flat at the dataset root, not "kaggle/SMOKE_MODE" -- kaggle/run.py
    # stages it that way because `kaggle datasets create/version` silently
    # skips subfolders by default (confirmed the hard way).
    smoke_marker = working_dir / "SMOKE_MODE"
    smoke = smoke_marker.exists() and smoke_marker.read_text().strip().lower() == "true"
    print(f"[entry] smoke mode: {smoke} (marker file: {smoke_marker}, exists={smoke_marker.exists()})")

    train_args = _read_train_args(working_dir)
    print(f"[entry] train_args (from TRAIN_ARGS.json): {train_args}")
    argv = [
        "train.py",
        "--train-dir", str(train_dir),
        "--test-dir", str(test_dir),
        "--artifacts-dir", str(config.ARTIFACTS_DIR),
        "--model-name", str(train_args["model_name"]),
    ]
    if train_args.get("label_smoothing"):
        argv += ["--label-smoothing", str(train_args["label_smoothing"])]
    if smoke:
        argv.append("--smoke")

    resume_ckpt = config.ARTIFACTS_DIR / "checkpoint_last.pt"
    if resume_ckpt.exists():
        print(f"[entry] found existing checkpoint at {resume_ckpt}, resuming")
        argv += ["--resume", str(resume_ckpt)]

    print(f"[entry] invoking: {argv}")
    sys.argv = argv
    train.main()


if __name__ == "__main__":
    main()
