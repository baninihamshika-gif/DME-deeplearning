"""
Phase 2.5: Kaggle automation -- push / poll / fetch / logs / status.

    python kaggle/run.py push --smoke
    python kaggle/run.py push
    python kaggle/run.py poll <slug>
    python kaggle/run.py fetch <slug> [--out DIR]
    python kaggle/run.py logs <slug>
    python kaggle/run.py status

IMPORTANT -- where this has to run: the `kaggle` CLI needs (a) network
access to kaggle.com and (b) the team's real ~/.kaggle/kaggle.json API
credentials. From the automated shells available while building this
project, kaggle.com was unreachable (network egress allowlists PyPI, not
general web) and moving personal API credentials into any of those shells
would be the wrong way to handle them. So this script needs to be run from
a real terminal on a machine that has both -- most likely a team member's
own machine with `pip install -r requirements.txt` and `kaggle.json` in
place. It has not been exercised against the live Kaggle API; see PHASE
2.5's sign-off "Blocking questions" for exactly what that first real run
needs to confirm.

Design notes (read once, matters for every function below):

  * Dataset-version message: PROJECT_BRIEF.md says tag each `dme-oct-src`
    version with "<git sha>". This repo's local working copy has no real
    commit history yet, so git_sha_or_source_hash() falls back to a
    content hash over the tracked source files, prefixed "contenthash-" so
    it's never mistaken for an actual commit. Once the team starts
    committing for real, this automatically switches back to using the
    git sha with no code change needed.

  * Retry classification: Kaggle's API status doesn't cleanly distinguish
    "the platform broke" from "the code threw an exception" -- both often
    surface as status "error". classify_failure() uses the fetched log (a
    Python traceback is the strongest signal of a code bug) and falls back
    to keyword-matching the failure message. This is a judgment call, not
    a documented Kaggle behaviour -- verify it against the first real
    failure you see and tighten it if it misclassifies.

  * Cross-push resume: entry.py resumes from a checkpoint if one exists in
    its own /kaggle/working, but a *fresh* `kaggle kernels push` always
    starts from an empty working directory -- resuming a run that crashed
    on a previous push would require attaching that previous push's output
    as an additional dataset_source, which this script does not automate.
    Given Phase A+B together are expected to run well inside a single
    Kaggle session's time limit for this dataset size, this is left as a
    known gap rather than solved speculatively -- revisit if a real run
    ever needs it.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Windows' console defaults to a legacy codepage (cp1252), not UTF-8. The
# `kaggle` CLI's own output can contain characters outside that codepage --
# confirmed the hard way: `kaggle kernels output` crashed the whole script
# with UnicodeEncodeError while printing its progress, right as it was
# fetching the one artifact (the failed run's log) needed to diagnose a
# real failure. `errors="replace"` means an unprintable character becomes
# "?" in the console instead of crashing the script -- acceptable, since
# nothing here parses printed text for meaning; only the captured
# subprocess stdout/stderr strings (decoded separately, not via the
# console) are ever parsed.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

# `python kaggle/run.py ...` puts kaggle/ (not the repo root) on sys.path[0],
# so config.py wouldn't otherwise be importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

SOURCE_FILES = [
    "config.py",
    "utils.py",
    "train.py",
    "prepare_data.py",
    "segment.py",
    "requirements.txt",
]
# evaluate.py / explain.py / thickness.py / app.py belong here once their
# phases land. _existing_source_files() skips (and warns about) anything not
# on disk yet rather than failing, since Phase 2.5 is being built before
# those files exist. segment.py added for Phase 4 -- run_segment.py's own
# _stage_segment_source_dataset() re-stages the SAME dme-oct-src dataset
# (now including segment.py) rather than a second source dataset, so it
# needs to be in this shared list to ride along either way a push happens.
#
# entry.py is deliberately NOT in this list: it ships to Kaggle as the
# kernel's own code_file via `kaggle kernels push` (see
# _stage_kernel_push_folder), not through the dme-oct-src dataset, so it
# doesn't need to be staged here too. Anything that WAS put under a
# subfolder here (this used to include "kaggle/entry.py", and the smoke
# marker used to live at "kaggle/SMOKE_MODE") silently never uploaded --
# `kaggle datasets create/version` skips directories by default ("Skipping
# folder: kaggle; use '--dir-mode' to upload folders" prints right in the
# CLI output, easy to miss scrolling past it). Confirmed the hard way on
# the first two real pushes. Keep everything staged into this dataset flat
# at its root unless --dir-mode is deliberately added to the upload calls.

STATUS_RE = re.compile(r'has status "([^"]+)"')
FAILURE_MSG_RE = re.compile(r'Failure message: "([^"]*)"')
TERMINAL_SUCCESS = {"complete"}
TERMINAL_FAILURE = {"error", "cancelacknowledged", "cancelled"}


class QuotaExceeded(Exception):
    pass


class SmokeGateError(Exception):
    pass


class RunFailed(Exception):
    pass


def _run_subprocess(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    """
    Two DIFFERENT Windows encoding failures live here, confirmed the hard
    way on two separate real runs:

    1. subprocess.run(..., text=True) decodes the CHILD's captured output
       with locale.getpreferredencoding(), which on Windows is typically
       cp1252, not UTF-8 -- the `kaggle` CLI's own output isn't guaranteed
       to fit that. encoding="utf-8", errors="replace" fixes decoding on
       OUR (the parent's) side.

    2. That fix alone did NOT stop a second, later crash with the same
       symptom ('charmap' codec can't encode characters ...): the `kaggle`
       CLI is itself a Python process, spawned as a child here, and ITS own
       stdout/stderr encoding is controlled by ITS OWN environment, not
       ours -- our sys.stdout.reconfigure() only affects this process.
       When that child's stdout is redirected to a pipe (as it always is
       here, via capture_output=True), Python still defaults to
       locale.getpreferredencoding() for it absent PYTHONIOENCODING, so the
       child crashed trying to print something outside cp1252 *before* we
       ever got to decode anything -- we only saw the tail of its own
       UnicodeEncodeError message. Setting PYTHONIOENCODING in the child's
       environment fixes encoding at the source instead of hoping the
       parent-side decode is the only place this can go wrong.
    """
    env = kwargs.pop("env", None)
    if env is None:
        env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8:replace")
    return subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", env=env, **kwargs)


# ---------------------------------------------------------------------------
# Source hashing / versioning
# ---------------------------------------------------------------------------


def _existing_source_files(repo_root: Path) -> list:
    paths = []
    for rel in SOURCE_FILES:
        p = repo_root / rel
        if p.exists():
            paths.append(p)
        else:
            print(f"[run] note: {rel} does not exist yet, skipping (expected before its phase is built)")
    return paths


def source_hash(repo_root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(_existing_source_files(repo_root)):
        h.update(str(p.relative_to(repo_root)).encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def git_sha_or_source_hash(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return f"contenthash-{source_hash(repo_root)}"


# ---------------------------------------------------------------------------
# Persistent state: quota_log.json, last_smoke.json
# ---------------------------------------------------------------------------


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _save_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


def gpu_minutes_used_last_7_days(quota_log: list, now=None) -> float:
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=7)
    total = 0.0
    for entry in quota_log:
        ts = datetime.fromisoformat(entry["timestamp"])
        minutes = entry.get("actual_minutes")
        if minutes is None:
            minutes = entry.get("expected_minutes", 0)
        if ts >= cutoff:
            total += minutes
    return total


def estimate_full_run_minutes(quota_log: list, last_smoke, model_name: str = "tf_efficientnet_b3") -> float:
    """
    Prefer a real, previously observed full-run duration over any
    extrapolation from the smoke run. Confirmed the hard way why this
    matters: the smoke run trains on a ~500-image subset, while a full run
    trains on the whole ~38k-image dataset. The original approach --
    smoke's blended per-epoch time times epoch COUNT alone -- ignores that
    per-epoch time scales with dataset SIZE too, and undershot a real full
    run's actual duration by ~9.7x (32.8min estimated vs 319.77min actual,
    kernel version 32, 2026-09-16 -- see that quota_log.json entry's
    correction_note). Rather than invent a scaling constant from one data
    point (which would just be a different guess -- C6), use real history
    once it exists: it's ground truth for this exact pipeline, not a model.

    model_name matters here too (Phase 3g Baselines): ResNet-50 and
    EfficientNet-B0/B3 have different per-epoch compute costs, so a
    completed run's actual_minutes is only a valid estimate for a FUTURE
    push of the SAME architecture -- reusing e.g. a real ResNet-50 timing
    to estimate an EfficientNet-B0 push would silently misinform the quota
    guardrail. entry.get("model_name", "tf_efficientnet_b3") treats
    history recorded before this field existed as tf_efficientnet_b3,
    which is simply true -- every push in this project was B3 until
    Baselines. label_smoothing is deliberately NOT filtered on: it changes
    the loss computation, not the forward/backward pass cost, so it
    doesn't affect per-epoch duration.
    """
    for entry in reversed(quota_log):
        if (
            not entry.get("smoke")
            and entry.get("status") == "complete"
            and entry.get("actual_minutes") is not None
            and entry.get("model_name", "tf_efficientnet_b3") == model_name
        ):
            # +15% margin on a real measurement -- much smaller than the
            # smoke-extrapolation margin below, since this is observed
            # fact for this exact pipeline, not a guess.
            return entry["actual_minutes"] * 1.15

    # No completed full run recorded yet: this is the necessarily-rough
    # first estimate. Still derived from measurement, never invented (C6),
    # but flagged loudly because a ~500-image smoke subset is not
    # representative of a full run's per-epoch cost.
    if not last_smoke or last_smoke.get("status") != "passed" or not last_smoke.get("epoch_seconds"):
        return None
    total_epochs = config.EPOCHS_PHASE_A + config.EPOCHS_PHASE_B
    rough = (last_smoke["epoch_seconds"] * total_epochs / 60.0) * 1.3
    print(
        "[run] WARNING: no completed full run recorded yet -- this estimate extrapolates from the smoke "
        "run's ~500-image subset and is known to undershoot a full run substantially (confirmed ~9.7x low "
        "on 2026-09-16). Treat this number with real skepticism; it will self-correct after this push "
        "completes once, since future estimates prefer real full-run history over this extrapolation."
    )
    return rough


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def check_quota_guardrail(expected_minutes: float, quota_log: list, budget_min: float) -> None:
    used = gpu_minutes_used_last_7_days(quota_log)
    projected = used + expected_minutes
    print(
        f"[run] GPU quota: used {used:.1f} min in the last 7 days, this push est. {expected_minutes:.1f} min, "
        f"projected {projected:.1f} min (push budget {budget_min:.1f} min, weekly free-tier cap "
        f"{config.KAGGLE_GPU_WEEKLY_BUDGET_MIN} min)"
    )
    if projected > budget_min:
        raise QuotaExceeded(
            f"Refusing push: projected {projected:.1f} min would exceed the {budget_min:.1f} min budget. "
            "Pass --budget-min to override this run, or wait for older usage to roll off the 7-day window."
        )


def check_smoke_gate(
    current_hash: str, last_smoke, model_name: str = "tf_efficientnet_b3", label_smoothing: float = 0.0
) -> None:
    """
    Phase 3g Baselines added --model-name/--label-smoothing as CLI-only
    push args -- they are NOT part of source_hash (that only hashes the
    tracked .py files), so a smoke test of one architecture would
    otherwise silently satisfy the gate for a full push of a DIFFERENT,
    never-smoke-tested architecture as long as the source tree was
    unchanged. Keying the gate on (source_hash, model_name,
    label_smoothing) closes that: each architecture/label-smoothing
    combination needs its own passing smoke run. last_smoke.get(...,
    default) treats a pre-Baselines smoke record (no model_name/
    label_smoothing fields) as tf_efficientnet_b3 / 0.0, which is simply
    true of this project's history before this feature existed.
    """
    if last_smoke is None:
        raise SmokeGateError("Refusing full push: no recorded smoke run yet. Run `push --smoke` first.")
    current = (current_hash, model_name, label_smoothing)
    recorded = (
        last_smoke.get("source_hash"),
        last_smoke.get("model_name", "tf_efficientnet_b3"),
        last_smoke.get("label_smoothing", 0.0),
    )
    if recorded != current:
        raise SmokeGateError(
            f"Refusing full push: last smoke was source={recorded[0]} model_name={recorded[1]} "
            f"label_smoothing={recorded[2]}, current push is source={current[0]} model_name={current[1]} "
            f"label_smoothing={current[2]}. Re-run `push --smoke` with this exact source + "
            "--model-name/--label-smoothing combination before a full push."
        )
    if last_smoke.get("status") != "passed":
        raise SmokeGateError(
            f"Refusing full push: last smoke on this source has status {last_smoke.get('status')!r}, not 'passed'."
        )


def check_username_configured() -> None:
    if config.KAGGLE_USERNAME == "REPLACE_WITH_KAGGLE_USERNAME":
        raise RuntimeError(
            "config.KAGGLE_USERNAME is still the placeholder -- set it to your real Kaggle username before pushing."
        )


# ---------------------------------------------------------------------------
# Staging + push
# ---------------------------------------------------------------------------


def _stage_source_dataset(
    repo_root: Path, stage_dir: Path, smoke: bool, model_name: str = "tf_efficientnet_b3", label_smoothing: float = 0.0
) -> None:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    for p in _existing_source_files(repo_root):
        rel = p.relative_to(repo_root)
        dest = stage_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    smoke_marker = stage_dir / "SMOKE_MODE"  # flat, at dataset root -- see SOURCE_FILES comment above
    smoke_marker.write_text("true" if smoke else "false")
    # Phase 3g Baselines: which architecture/label-smoothing entry.py should
    # forward to train.py on the Kaggle side. Flat at the dataset root, same
    # reasoning as SMOKE_MODE above (kaggle datasets create/version silently
    # skips subfolders). A single JSON file rather than two more flat text
    # files since this is naturally a small, growable config blob (more
    # per-push train.py args may follow) and entry.py only needs one read.
    train_args_path = stage_dir / "TRAIN_ARGS.json"
    train_args_path.write_text(json.dumps({"model_name": model_name, "label_smoothing": label_smoothing}, indent=2))
    dataset_metadata = {
        "title": "dme-oct-src",
        "id": config.KAGGLE_SRC_DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (stage_dir / "dataset-metadata.json").write_text(json.dumps(dataset_metadata, indent=2))


def _dataset_exists(slug: str) -> bool:
    # `kaggle datasets status` has no documented "not found" exit code in
    # the CLI help; treating any nonzero exit as "doesn't exist yet" is
    # safe here because a false negative just routes to `create`, which
    # itself fails loudly (and visibly, via the raised RuntimeError below)
    # if the slug actually already exists.
    result = _run_subprocess(["kaggle", "datasets", "status", slug])
    return result.returncode == 0


# Shared with classify_failure() below, which applies the same substring
# test to a *fetched kernel run's* failure message/log -- this is the same
# judgment call applied one layer earlier, to a `kaggle` CLI subprocess's own
# stderr, before a kernel run even exists to classify.
_INFRA_ERROR_KEYWORDS = ("session", "internal error", "unavailable", "timed out", "timeout", "disconnected")


def _run_kaggle_cli(cmd: list, error_context: str, infra_retry: bool = False) -> str:
    """
    infra_retry=True retries a failing call up to config.KAGGLE_MAX_INFRA_RETRIES
    times, but ONLY when the failure looks infra-side (stderr matches
    _INFRA_ERROR_KEYWORDS) -- anything else (a real code/config error) still
    raises immediately, same as infra_retry=False. Opt-in per call site
    rather than the default for every `kaggle` CLI invocation this function
    makes: most callers (dataset staging, `kernels pull`/`status`) have their
    own polling/readiness logic around them already, where blind retries here
    would just duplicate or race that. See config.KAGGLE_PUSH_INFRA_RETRY_DELAY_SEC
    for why this exists at all -- a real `kernels push` 503 that this
    couldn't have retried around otherwise (2026-09-23).
    """
    max_attempts = (config.KAGGLE_MAX_INFRA_RETRIES if infra_retry else 0) + 1
    for attempt in range(1, max_attempts + 1):
        print(f"[run] {' '.join(cmd)}")
        result = _run_subprocess(cmd)
        if result.stdout:
            print(result.stdout)
        if result.returncode == 0:
            return result.stdout
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        is_infra = infra_retry and any(kw in (result.stderr or "").lower() for kw in _INFRA_ERROR_KEYWORDS)
        if is_infra and attempt < max_attempts:
            print(
                f"[run] {error_context}: infra-looking failure (attempt {attempt}/{max_attempts}), "
                f"retrying in {config.KAGGLE_PUSH_INFRA_RETRY_DELAY_SEC}s"
            )
            time.sleep(config.KAGGLE_PUSH_INFRA_RETRY_DELAY_SEC)
            continue
        raise RuntimeError(f"{error_context} failed (exit {result.returncode})")


def get_dataset_status(slug: str) -> str:
    """
    Uses --format json (the plain-text form of `datasets status` has no
    such flag documented for a reason to parse it further -- json avoids
    the same class of text-parsing fragility that broke kernel status
    parsing). Returns "unknown" on any parse/call failure rather than
    raising, since this is polled in a loop where the caller decides what
    to do about repeated "unknown"s (see wait_for_dataset_ready).
    """
    result = _run_subprocess(["kaggle", "datasets", "status", slug, "--format", "json"])
    if result.returncode != 0:
        return "unknown"
    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line).get("status", "unknown")
            except json.JSONDecodeError:
                continue
    return "unknown"


def wait_for_dataset_ready(
    slug: str, poll_interval_sec: int = None, max_wait_minutes: float = None
) -> None:
    """
    `kaggle datasets create`/`version` return as soon as the upload
    finishes, printing "...is being created. Please check progress at
    ..." -- that message is the CLI's own admission this is async, not a
    guarantee of readiness. Confirmed the hard way: the first real smoke
    push uploaded successfully, the script moved straight on to push the
    kernel, and the kernel failed with FileNotFoundError on
    /kaggle/input/dme-oct-src because Kaggle hadn't finished attaching
    that version yet. This blocks until dataset status reports "ready",
    or raises on "error" or a timeout, before the caller is allowed to
    push a kernel that depends on it.
    """
    poll_interval_sec = config.KAGGLE_DATASET_READY_POLL_SEC if poll_interval_sec is None else poll_interval_sec
    max_wait_minutes = config.KAGGLE_DATASET_READY_TIMEOUT_MIN if max_wait_minutes is None else max_wait_minutes
    start = time.time()
    while True:
        status = get_dataset_status(slug)
        elapsed_min = (time.time() - start) / 60.0
        print(f"[run] dataset {slug} status={status!r} elapsed={elapsed_min:.1f}min")
        if status == "ready":
            return
        if status == "error":
            raise RuntimeError(
                f"Dataset {slug} failed processing (status=error) -- check "
                f"https://www.kaggle.com/datasets/{slug}"
            )
        if elapsed_min >= max_wait_minutes:
            raise RuntimeError(
                f"Dataset {slug} did not report 'ready' within {max_wait_minutes} min "
                f"(last status={status!r}). Check https://www.kaggle.com/datasets/{slug} before pushing again."
            )
        time.sleep(poll_interval_sec)


def push_source_dataset(
    repo_root: Path,
    smoke: bool,
    version_message: str,
    model_name: str = "tf_efficientnet_b3",
    label_smoothing: float = 0.0,
) -> None:
    stage_dir = config.KAGGLE_ARTIFACTS_DIR / "_src_stage"
    _stage_source_dataset(repo_root, stage_dir, smoke, model_name=model_name, label_smoothing=label_smoothing)
    if _dataset_exists(config.KAGGLE_SRC_DATASET_SLUG):
        cmd = ["kaggle", "datasets", "version", "-p", str(stage_dir), "-m", version_message]
        context = "kaggle datasets version"
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(stage_dir)]
        context = "kaggle datasets create"
    _run_kaggle_cli(cmd, context)
    wait_for_dataset_ready(config.KAGGLE_SRC_DATASET_SLUG)


def _stage_kernel_push_folder(stage_dir: Path) -> None:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    entry_src = config.REPO_ROOT / "kaggle" / "entry.py"
    shutil.copy2(entry_src, stage_dir / "entry.py")
    # kernel-metadata.json is regenerated here from config.py rather than
    # copied verbatim from the checked-in kaggle/kernel-metadata.json --
    # config.py is the single source of truth for KAGGLE_USERNAME, so the
    # actual push can never drift out of sync with a stale placeholder in
    # the repo file (which stays purely as a human-readable template).
    metadata = {
        "id": config.KAGGLE_KERNEL_SLUG,
        "title": "dme-oct-train",
        "code_file": "entry.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [config.KAGGLE_SRC_DATASET_SLUG, config.KAGGLE_KERMANY_DATASET_SLUG],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (stage_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))


def push(
    repo_root: Path,
    smoke: bool,
    budget_min: float = None,
    model_name: str = "tf_efficientnet_b3",
    label_smoothing: float = 0.0,
) -> str:
    check_username_configured()
    quota_log = _load_json(config.KAGGLE_QUOTA_LOG, [])
    last_smoke = _load_json(config.KAGGLE_LAST_SMOKE, None)
    src_hash = git_sha_or_source_hash(repo_root)
    budget_min = config.KAGGLE_PUSH_BUDGET_DEFAULT_MIN if budget_min is None else budget_min

    if smoke:
        expected_minutes = float(config.KAGGLE_SMOKE_TIME_BUDGET_MIN)
    else:
        check_smoke_gate(src_hash, last_smoke, model_name=model_name, label_smoothing=label_smoothing)
        expected_minutes = estimate_full_run_minutes(quota_log, last_smoke, model_name=model_name)
        if expected_minutes is None:
            raise RuntimeError(
                "Cannot estimate full-run duration (no per-epoch timing recorded in last_smoke.json) -- "
                "refusing to push without an estimate rather than guessing (C6)."
            )

    check_quota_guardrail(expected_minutes, quota_log, budget_min)

    push_source_dataset(repo_root, smoke, version_message=src_hash, model_name=model_name, label_smoothing=label_smoothing)

    kernel_stage = config.KAGGLE_ARTIFACTS_DIR / "_kernel_stage"
    _stage_kernel_push_folder(kernel_stage)
    _run_kaggle_cli(["kaggle", "kernels", "push", "-p", str(kernel_stage)], "kaggle kernels push", infra_retry=True)

    quota_log.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_hash": src_hash,
            "kernel_slug": config.KAGGLE_KERNEL_SLUG,
            "smoke": smoke,
            "model_name": model_name,
            "label_smoothing": label_smoothing,
            "expected_minutes": expected_minutes,
            "actual_minutes": None,
            "status": "pushed",
        }
    )
    _save_json_atomic(config.KAGGLE_QUOTA_LOG, quota_log)
    return config.KAGGLE_KERNEL_SLUG


# ---------------------------------------------------------------------------
# Poll / fetch / logs
# ---------------------------------------------------------------------------


def _normalize_status(raw_status: str) -> str:
    """
    Confirmed against a real account (2026-09): some `kaggle` CLI/package
    versions print a bare lowercase word ("error", "complete"); others
    print a Python Enum's default str() instead ("KernelWorkerStatus.ERROR").
    The first real smoke push hit the latter, and TERMINAL_SUCCESS /
    TERMINAL_FAILURE membership checks against the raw string silently
    never matched -- poll() treated "KernelWorkerStatus.ERROR" as
    perpetually non-terminal and looped for the full KAGGLE_MAX_POLL_MIN
    instead of recognizing the run had already finished. Normalizing once,
    here, means every downstream comparison only ever sees the short form.
    """
    return raw_status.rsplit(".", 1)[-1].strip().lower()


def get_status(slug: str) -> tuple:
    result = _run_subprocess(["kaggle", "kernels", "status", slug])
    combined = (result.stdout or "") + (result.stderr or "")
    status_m = STATUS_RE.search(combined)
    failure_m = FAILURE_MSG_RE.search(combined)
    raw_status = status_m.group(1) if status_m else "unknown"
    status = _normalize_status(raw_status)
    failure_message = failure_m.group(1) if failure_m else None
    return status, failure_message


def poll(slug: str, interval_sec: int = None, max_minutes: float = None) -> tuple:
    interval_sec = config.KAGGLE_POLL_INTERVAL_SEC if interval_sec is None else interval_sec
    max_minutes = config.KAGGLE_MAX_POLL_MIN if max_minutes is None else max_minutes
    start = time.time()
    while True:
        status, failure_message = get_status(slug)
        elapsed_min = (time.time() - start) / 60.0
        print(f"[run] poll {slug}: status={status!r} elapsed={elapsed_min:.1f}min")
        if status.lower() in TERMINAL_SUCCESS or status.lower() in TERMINAL_FAILURE:
            return status, failure_message
        if elapsed_min >= max_minutes:
            return "timeout", f"polling exceeded {max_minutes} min without reaching a terminal status"
        time.sleep(interval_sec)


def logs(slug: str) -> str:
    result = _run_subprocess(["kaggle", "kernels", "logs", slug])
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.stdout or ""


def fetch(slug: str, out_dir: Path = None) -> Path:
    """
    Download the kernel's output files, then always fetch the run log --
    even if the output download itself fails. Confirmed the hard way: a
    Windows console encoding crash inside the `kaggle kernels output`
    subprocess call once raised uncaught here, which killed this function
    before logs(slug) ever ran and cost us the one artifact we actually
    needed (the failed run's traceback). _run_subprocess's errors="replace"
    should prevent that specific crash now, but this function no longer
    depends on that fix alone -- any failure in the output-download step
    (transient network blip, a different encoding edge case, anything)
    must never again prevent the log fetch.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = out_dir or (config.KAGGLE_ARTIFACTS_DIR / timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        _run_kaggle_cli(["kaggle", "kernels", "output", slug, "-p", str(out_dir), "-o"], "kaggle kernels output")
    except RuntimeError as e:
        print(f"[run] WARNING: {e} -- continuing to fetch logs anyway, since that's the most useful diagnostic")
    # encoding="utf-8" is load-bearing, not decoration: confirmed the hard
    # way on the first fully-successful smoke run -- Path.write_text()
    # without an explicit encoding defaults to locale.getpreferredencoding()
    # (cp1252 on Windows) for the FILE WRITE itself, a completely different
    # code path from the sys.stdout / subprocess encoding fixes above (those
    # don't touch file I/O at all). The real training log contains
    # characters cp1252 can't represent (e.g. an em dash in a C8 warning
    # message), so the write crashed even though everything up to that
    # point -- including printing this same text to the console -- had
    # already succeeded.
    (out_dir / "run.log").write_text(logs(slug), encoding="utf-8")
    return out_dir


GPU_TORCH_ARCH_MISMATCH_MARKERS = (
    "no kernel image is available for execution on the device",
    "is not compatible with the current pytorch installation",
)


def _is_gpu_torch_arch_mismatch(log_text: str) -> bool:
    text = log_text.lower()
    return any(marker in text for marker in GPU_TORCH_ARCH_MISMATCH_MARKERS)


def classify_failure(status: str, failure_message, log_text) -> str:
    """See module docstring: a judgment call, not documented Kaggle behaviour."""
    if status.lower() == "timeout":
        return "infra"
    if log_text and _is_gpu_torch_arch_mismatch(log_text):
        # A genuine Python traceback, but not a bug in OUR code: Kaggle
        # allocated a GPU generation (P100) its own preinstalled torch build
        # doesn't support -- confirmed against Kaggle/docker-python#1546,
        # a real unresolved Kaggle-side bug, not ours. Must be checked
        # before the generic traceback rule below, which would otherwise
        # misclassify this as a code error and refuse to ever retry it.
        # entry.py now works around this proactively (see
        # _ensure_gpu_torch_compat there); this is the safety net in case
        # a future GPU/torch combination slips past that check.
        return "infra"
    if log_text and "Traceback (most recent call last)" in log_text:
        return "code_error"
    text = (failure_message or "").lower()
    if any(kw in text for kw in _INFRA_ERROR_KEYWORDS):
        return "infra"
    return "code_error"  # default to the classification that never auto-retries when unsure


def _parse_epoch_seconds_from_training_log(out_dir: Path, log_filename: str = "training_log.csv"):
    # log_filename is a parameter (not just "training_log.csv" hardcoded) so
    # run_segment.py can reuse this same CSV-parsing logic against segment.py's
    # own log ("segmentation_training_log.csv", written under a different
    # --artifacts-dir) without duplicating it -- both logs share the same
    # epoch_time_sec column by construction (segment.py's _log_epoch_row
    # deliberately matches train.py's csv logging convention).
    log_path = out_dir / "artifacts" / log_filename
    if not log_path.exists():
        print(f"[run] note: {log_path} not found in fetched output, cannot record epoch timing")
        return None
    times = []
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                times.append(float(row["epoch_time_sec"]))
            except (KeyError, ValueError):
                continue
    if not times:
        return None
    return sum(times) / len(times)


def _find_c5_line(log_text: str):
    if not log_text:
        return None
    for line in log_text.splitlines():
        if "[C5]" in line and "PASSED" in line:
            return line
    return None


# ---------------------------------------------------------------------------
# Orchestration: push, poll to completion, fetch, classify, retry
# ---------------------------------------------------------------------------


def _real_duration_minutes_from_log(log_text: str):
    """
    The fetched log is Kaggle's own array of
    {"stream_name": ..., "time": <seconds since kernel start>, "data": ...}
    records. The last record's "time" is the real, ground-truth kernel
    execution duration -- this is what was used to manually correct a real
    quota_log.json entry that a ~9 hour local network/sleep outage during
    polling had inflated to 681.86 "actual" minutes for a push that really
    took 319.77 (kernel version 32, 2026-09-16). Returns None -- never a
    guess -- if the log is empty or not parseable this way (e.g. fetch
    itself failed), so the caller falls back to wall-clock time instead.
    """
    try:
        records = json.loads(log_text)
        times = [r["time"] for r in records if isinstance(r, dict) and "time" in r]
    except (json.JSONDecodeError, TypeError, KeyError):
        return None
    return max(times) / 60.0 if times else None


def push_and_run(
    repo_root: Path,
    smoke: bool,
    budget_min: float = None,
    model_name: str = "tf_efficientnet_b3",
    label_smoothing: float = 0.0,
) -> dict:
    attempt = 0
    while True:
        slug = push(repo_root, smoke, budget_min, model_name=model_name, label_smoothing=label_smoothing)
        poll_start = time.time()
        status, failure_message = poll(slug)
        wall_clock_minutes = (time.time() - poll_start) / 60.0
        out_dir = fetch(slug)
        log_text = (out_dir / "run.log").read_text(encoding="utf-8") if (out_dir / "run.log").exists() else ""

        # actual_minutes was written as None at push() time and never
        # updated afterward (confirmed on a real quota_log.json). Fixing
        # that by recording poll()'s local wall-clock time seemed right at
        # first, but a real ~9 hour local network/sleep outage during
        # polling proved it isn't: that gap got counted as "GPU minutes
        # consumed" even though the kernel may have finished hours earlier
        # and just sat idle while unobservable. The fetched log's own
        # timestamps are ground truth, immune to local outages, and are
        # available whenever the log fetch itself succeeds -- prefer them,
        # and fall back to wall-clock (with a loud warning) only when the
        # log can't be parsed (e.g. fetch also failed).
        real_minutes = _real_duration_minutes_from_log(log_text)
        if real_minutes is not None:
            actual_minutes = real_minutes
        else:
            actual_minutes = wall_clock_minutes
            print(
                f"[run] WARNING: could not read a real duration from the fetched log -- recording poll()'s "
                f"local wall-clock time ({wall_clock_minutes:.1f}min) as actual_minutes instead. This can be "
                "inflated by a local network/sleep gap during polling; verify against `kaggle kernels status` "
                "if this number looks implausible."
            )

        quota_log = _load_json(config.KAGGLE_QUOTA_LOG, [])
        if quota_log:
            quota_log[-1]["status"] = status
            quota_log[-1]["actual_minutes"] = round(actual_minutes, 2)
            _save_json_atomic(config.KAGGLE_QUOTA_LOG, quota_log)

        src_hash = git_sha_or_source_hash(repo_root)

        if status.lower() in TERMINAL_SUCCESS:
            c5_line = _find_c5_line(log_text)
            if smoke:
                _save_json_atomic(
                    config.KAGGLE_LAST_SMOKE,
                    {
                        "source_hash": src_hash,
                        "kernel_slug": slug,
                        "model_name": model_name,
                        "label_smoothing": label_smoothing,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "status": "passed",
                        "c5_passed": c5_line is not None,
                        "epoch_seconds": _parse_epoch_seconds_from_training_log(out_dir),
                        "artifacts_dir": str(out_dir),
                    },
                )
            return {"status": status, "artifacts_dir": out_dir, "c5_line": c5_line, "attempts": attempt + 1}

        failure_type = classify_failure(status, failure_message, log_text)
        print(f"[run] status={status!r} classified as: {failure_type}")

        if smoke:
            _save_json_atomic(
                config.KAGGLE_LAST_SMOKE,
                {
                    "source_hash": src_hash,
                    "kernel_slug": slug,
                    "model_name": model_name,
                    "label_smoothing": label_smoothing,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                    "failure_type": failure_type,
                    "artifacts_dir": str(out_dir),
                },
            )

        if failure_type == "code_error" or attempt >= config.KAGGLE_MAX_INFRA_RETRIES:
            raise RunFailed(
                f"Run failed (status={status}, classified={failure_type}) after {attempt + 1} attempt(s). "
                f"See {out_dir / 'run.log'}. "
                + ("Not retrying: this looks like a code error, not an infra failure."
                   if failure_type == "code_error"
                   else f"Not retrying: infra-retry budget ({config.KAGGLE_MAX_INFRA_RETRIES}) exhausted.")
            )
        attempt += 1
        print(f"[run] infra failure, retrying (attempt {attempt + 1}/{config.KAGGLE_MAX_INFRA_RETRIES + 1})")


# ---------------------------------------------------------------------------
# status subcommand
# ---------------------------------------------------------------------------


def print_status(repo_root: Path) -> None:
    quota_log = _load_json(config.KAGGLE_QUOTA_LOG, [])
    last_smoke = _load_json(config.KAGGLE_LAST_SMOKE, None)
    used = gpu_minutes_used_last_7_days(quota_log)
    print(f"GPU minutes used in the last 7 days: {used:.1f} / {config.KAGGLE_GPU_WEEKLY_BUDGET_MIN} weekly cap")
    print(f"Per-push default budget: {config.KAGGLE_PUSH_BUDGET_DEFAULT_MIN} min")
    if last_smoke:
        print(
            f"Last smoke: source={last_smoke.get('source_hash')} "
            f"model_name={last_smoke.get('model_name', 'tf_efficientnet_b3')} "
            f"label_smoothing={last_smoke.get('label_smoothing', 0.0)} status={last_smoke.get('status')} "
            f"at {last_smoke.get('timestamp')}"
        )
    else:
        print("Last smoke: none recorded yet")
    print(f"Current source hash: {git_sha_or_source_hash(repo_root)}")
    print(f"Kaggle username configured: {config.KAGGLE_USERNAME != 'REPLACE_WITH_KAGGLE_USERNAME'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Phase 2.5: Kaggle push/poll/fetch automation.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_push = sub.add_parser("push", help="Push source + kernel, poll to completion, fetch artifacts.")
    p_push.add_argument("--smoke", action="store_true")
    p_push.add_argument("--budget-min", type=float, default=None)
    p_push.add_argument(
        "--model-name", type=str, default="tf_efficientnet_b3",
        help="Architecture to train on Kaggle (Phase 3g Baselines: tf_efficientnet_b0, resnet50). Not "
             "validated here -- train.py's own argparse (choices=MODEL_CONFIGS.keys()) validates it on the "
             "kernel, seconds into the run, well before any GPU-hour is spent on training. Keeping run.py "
             "torch-free (it only needs network + kaggle.json, per this module's docstring) matters more "
             "than duplicating that choice list here.",
    )
    p_push.add_argument(
        "--label-smoothing", type=float, default=0.0,
        help="CrossEntropyLoss label_smoothing forwarded to train.py (Phase 3b ablation A5; brief specifies 0.05).",
    )

    p_poll = sub.add_parser("poll", help="Poll an already-pushed kernel's status.")
    p_poll.add_argument("slug")

    p_fetch = sub.add_parser("fetch", help="Fetch a kernel's output + log.")
    p_fetch.add_argument("slug")
    p_fetch.add_argument("--out", type=Path, default=None)

    p_logs = sub.add_parser("logs", help="Print a kernel's execution log.")
    p_logs.add_argument("slug")

    sub.add_parser("status", help="Print GPU quota usage and last-smoke status.")

    args = parser.parse_args()
    repo_root = config.REPO_ROOT

    if args.command == "push":
        result = push_and_run(
            repo_root, smoke=args.smoke, budget_min=args.budget_min,
            model_name=args.model_name, label_smoothing=args.label_smoothing,
        )
        print(f"\n[run] DONE: {result}")
    elif args.command == "poll":
        status, failure_message = poll(args.slug)
        print(f"Final status: {status}" + (f" ({failure_message})" if failure_message else ""))
    elif args.command == "fetch":
        out_dir = fetch(args.slug, args.out)
        print(f"Fetched to: {out_dir}")
    elif args.command == "logs":
        logs(args.slug)
    elif args.command == "status":
        print_status(repo_root)


if __name__ == "__main__":
    main()
