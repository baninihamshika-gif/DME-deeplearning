"""
Thin Kaggle kernel bootstrap for Phase 4 (extended objective E1, segmentation).
Same "no logic of its own" convention as kaggle/entry.py (PROJECT_BRIEF.md
Section 7).

DELIBERATELY SELF-CONTAINED, not importing entry.py -- an earlier version of
this file did `import entry` to reuse entry.py's already-hardened helpers
(source-tree copy, GPU/torch compat, pip install, the /kaggle/input search)
rather than duplicating them. That failed on the first real push:

    ModuleNotFoundError: No module named 'entry'

Root cause, confirmed from that real kernel's own traceback (not guessed):
`kaggle kernels push -p <dir>` for a kernel_type="script" kernel uploads
ONLY the single file named as kernel-metadata.json's code_file -- sibling
.py files sitting in the same local push folder are NOT bundled onto the
kernel's Python path, even though they're staged right next to it locally.
This is exactly why entry.py itself was never written to import any
sibling local module (only config/train, which come from the COPIED
SOURCE TREE after _copy_source_tree() runs, i.e. from the dme-oct-src
dataset, not from the kernel's own code folder) -- it just wasn't
documented as a reason until this failure made it one. So the helpers
below are DUPLICATED from entry.py, not imported, and must be kept in
sync by hand if entry.py's versions change -- flagged here so that's a
deliberate choice, not a silently-drifting copy.

This file does five things:

  1. Copy the source tree from the attached dme-oct-src dataset into
     /kaggle/working (writable), install dependencies, and apply the same
     P100/CUDA-torch-compat fix as entry.py -- verbatim duplicates of
     entry.py's own hardened logic (see module docstring above).
  2. Locate the Duke dataset (dme-oct-duke-src) under /kaggle/input.
  3. Locate the classifier checkpoint dataset (dme-oct-classifier-ckpt)
     under /kaggle/input and find the .pt file inside it (searched, not a
     hardcoded path -- same reasoning as entry.py's own input-mount
     lesson).
  4. Read this pipeline's OWN smoke marker (SEGMENT_SMOKE_MODE, staged
     flat at the dme-oct-src dataset root by run_segment.py) -- a
     separate marker from entry.py's SMOKE_MODE, since the two pipelines
     share that one source dataset and must not stomp each other's flag.
  5. Import and call segment.main(), forwarding --smoke if that marker
     says so.

Like entry.py, most of this can't be meaningfully unit-tested outside a
real Kaggle container. What CAN be tested (the input-dataset search, the
checkpoint-file search) is covered in tests/test_entry_segment.py with a
fake mount, not a live kernel.
"""

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

INPUT_DIR = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")
SRC_DATASET_DIRNAME = "dme-oct-src"
DUKE_DATASET_DIRNAME = "dme-oct-duke-src"
CKPT_DATASET_DIRNAME = "dme-oct-classifier-ckpt"

# See entry.py's own comment on this exact constant -- duplicated here
# because a Kaggle mount layout wrong-assumption (/kaggle/input/<slug>/ vs
# a real nested layout) was found and fixed there once already; searching
# rather than hardcoding avoids repeating that specific mistake here too.
INPUT_SEARCH_MAX_DEPTH = 5
INPUT_MOUNT_WAIT_TIMEOUT_SEC = 180
INPUT_MOUNT_WAIT_POLL_SEC = 5

# Same list and same reasoning as entry.py: torch/torchvision come
# preinstalled on the Kaggle GPU image, matched to its CUDA driver; `kaggle`
# has no reason to run inside its own kernel.
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
                print(f"[entry_segment] found {found} after {elapsed:.0f}s (attempt {attempt + 1})")
            return found
        tree = _list_input_tree()
        print(f"[entry_segment] no directory named {dirname!r} found yet under {INPUT_DIR} "
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
    print(f"[entry_segment] copying source tree: {src_dir} -> {WORKING_DIR}")
    for item in src_dir.iterdir():
        dest = WORKING_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    return WORKING_DIR


def _pip_install(working_dir: Path) -> None:
    packages = _requirements_to_install(working_dir)
    print(f"[entry_segment] pip installing from requirements.txt, skipping {PIP_SKIP_PACKAGES}: {packages}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet"] + packages,
        check=True,
    )


# --- GPU/torch compatibility fix -- verbatim duplicate of entry.py's own,
# see that file for the full history (Kaggle/docker-python#1546). ---
KNOWN_INCOMPATIBLE_GPU_NAME = "P100"
COMPATIBLE_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu126"
COMPATIBLE_TORCH_PACKAGES = ["torch==2.10.0", "torchvision", "torchaudio"]


def _detect_gpu_name() -> str:
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
    print(f"[entry_segment] detected GPU: {gpu_name!r}")
    if KNOWN_INCOMPATIBLE_GPU_NAME not in gpu_name:
        return
    print(f"[entry_segment] GPU name contains {KNOWN_INCOMPATIBLE_GPU_NAME!r} -- installing known-compatible "
          f"build before torch is ever imported: {COMPATIBLE_TORCH_PACKAGES} from {COMPATIBLE_TORCH_INDEX_URL}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--force-reinstall",
         "--index-url", COMPATIBLE_TORCH_INDEX_URL] + COMPATIBLE_TORCH_PACKAGES,
        check=True,
    )
    _verify_gpu_torch_compat()


def _verify_gpu_torch_compat() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _VERIFY_TORCH_GPU_SNIPPET],
        capture_output=True, text=True,
    )
    if result.stdout:
        print(f"[entry_segment] {result.stdout.strip()}")
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            "GPU/torch compatibility check failed even after installing the known-compatible build "
            f"({COMPATIBLE_TORCH_PACKAGES} from {COMPATIBLE_TORCH_INDEX_URL}) -- see the [verify] line above."
        )


# --- Segmentation-specific ---

# Flat at the dme-oct-src dataset root, like entry.py's own SMOKE_MODE --
# `kaggle datasets create/version` silently skips subfolders by default.
# A SEPARATE name from entry.py's own SMOKE_MODE marker: both pipelines
# share the one dme-oct-src source dataset, so reusing SMOKE_MODE would
# let a segmentation push's smoke flag silently overwrite the classifier
# pipeline's own flag, or vice versa.
SEGMENT_SMOKE_MARKER_NAME = "SEGMENT_SMOKE_MODE"


def _find_checkpoint_file(mount_dir: Path) -> Path:
    """Searches (not hardcodes) for a .pt file under the mounted checkpoint
    dataset directory. run_segment.py always stages the checkpoint as
    'checkpoint_best.pt' at that dataset's root, but the exact mount depth
    under /kaggle/input has already been wrong once for a different
    dataset (see entry.py's INPUT_SEARCH_MAX_DEPTH comment) -- searching a
    few levels deep costs nothing for a single small file and avoids
    repeating that exact mistake here."""
    candidates = sorted(mount_dir.rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No .pt checkpoint file found anywhere under {mount_dir}. "
            f"Expected 'checkpoint_best.pt' staged by run_segment.py's push_segment()."
        )
    if len(candidates) > 1:
        print(f"[entry_segment] WARNING: multiple .pt files found under {mount_dir}, using the first: {candidates[0]}")
    return candidates[0]


def main() -> None:
    working_dir = _copy_source_tree()
    _ensure_gpu_torch_compat()
    _pip_install(working_dir)

    sys.path.insert(0, str(working_dir))
    import config  # noqa: E402  (must import after sys.path is set up)
    import segment  # noqa: E402

    print(f"[entry_segment] config.ON_KAGGLE = {config.ON_KAGGLE}")
    assert config.ON_KAGGLE, "entry_segment.py is running but config.ON_KAGGLE is False -- /kaggle/input not detected?"

    duke_dir = _wait_for_input_dataset(DUKE_DATASET_DIRNAME)
    ckpt_mount_dir = _wait_for_input_dataset(CKPT_DATASET_DIRNAME)
    ckpt_path = _find_checkpoint_file(ckpt_mount_dir)
    print(f"[entry_segment] Duke dataset: {duke_dir}")
    print(f"[entry_segment] classifier checkpoint: {ckpt_path}")

    smoke_marker = working_dir / SEGMENT_SMOKE_MARKER_NAME
    smoke = smoke_marker.exists() and smoke_marker.read_text().strip().lower() == "true"
    print(f"[entry_segment] smoke mode: {smoke} (marker file: {smoke_marker}, exists={smoke_marker.exists()})")

    argv = [
        "segment.py",
        "--duke-dir", str(duke_dir),
        "--classifier-checkpoint", str(ckpt_path),
        "--artifacts-dir", str(config.ARTIFACTS_DIR / "segmentation"),
    ]
    if smoke:
        argv.append("--smoke")

    print(f"[entry_segment] invoking: {argv}")
    sys.argv = argv
    segment.main()


if __name__ == "__main__":
    main()
