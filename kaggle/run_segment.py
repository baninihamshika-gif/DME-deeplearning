"""
Phase 4 (extended, E1): Kaggle push/poll/fetch automation for segment.py.

    python kaggle/run_segment.py push --smoke --classifier-checkpoint <path>
    python kaggle/run_segment.py push --classifier-checkpoint <path>

`poll <slug>`, `fetch <slug>`, `logs <slug>`, and `status` are NOT
duplicated here -- kaggle/run.py's own poll()/fetch()/logs()/print_status()
already take a bare slug and have no classifier-specific assumptions, so
use those directly:

    python kaggle/run.py poll baninihamshika/dme-oct-segment
    python kaggle/run.py fetch baninihamshika/dme-oct-segment
    python kaggle/run.py status   -- shared quota/budget view, both pipelines

Design notes (read once):

  * Same "run this from a real terminal with kaggle.json in place" caveat
    as kaggle/run.py -- see that module's own docstring. This has not been
    exercised against the live Kaggle API either; the first real push is
    what finds what reasoning-without-a-live-account missed (same as
    entry.py's own history -- see that module's docstring).

  * Three datasets get staged/pushed here, not one: the shared source
    dataset (dme-oct-src, now including segment.py -- reuses run.py's own
    push_source_dataset() unmodified), a NEW Duke dataset
    (dme-oct-duke-src, staged fresh from --duke-dir every push, matching
    the existing pattern), and a NEW classifier-checkpoint dataset
    (dme-oct-classifier-ckpt, staged from --classifier-checkpoint). All
    three are private to this account -- see config.py's
    KAGGLE_DUKE_DATASET_SLUG / KAGGLE_CLASSIFIER_CKPT_DATASET_SLUG comment
    for why neither uses an existing public source or Kaggle's
    kernel_sources mount instead.

  * Shares kaggle/run.py's QUOTA_LOG (not a separate one) because the
    7-day/30h GPU cap is one real resource for the whole Kaggle account --
    see config.py's KAGGLE_SEGMENT_LAST_SMOKE comment. Uses its OWN
    last-smoke file (segment_last_smoke.json) because smoke-gating is
    pipeline-specific.

  * estimate_segment_run_minutes() mirrors run.py's
    estimate_full_run_minutes() (prefer real history for THIS kernel_slug
    over any extrapolation), but does not need run.py's ~9.7x-undershoot
    correction: segment.py's --smoke already trains on the full 110-scan
    annotated set (no subsampling), so a smoke epoch's real cost should
    already be representative -- flagged as an unconfirmed "should" until
    a real smoke + real full run both exist to compare (C6).
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as base  # noqa: E402  -- reuse its generic, already-hardened primitives


def _stage_segment_source_dataset(repo_root: Path, stage_dir: Path, smoke: bool) -> None:
    """Stages the SAME dme-oct-src dataset the classifier pipeline uses
    (segment.py rides along in the shared SOURCE_FILES list -- see
    kaggle/run.py's own comment there) in a SINGLE combined staging step,
    rather than calling run.py's push_source_dataset() AND this function
    separately (an earlier draft did that and pushed two dataset versions
    per segmentation push -- wasteful and confusing).

    Writes ALL FOUR marker files the two pipelines' entry scripts read:
      - SEGMENT_SMOKE_MODE: this pipeline's real smoke flag.
      - SMOKE_MODE / TRAIN_ARGS.json: the classifier pipeline's markers,
        written here with train.py's own argparse DEFAULTS (matching
        exactly what entry.py already falls back to when these files are
        absent -- see entry.py's _read_train_args docstring) rather than
        omitted. Omitting them would DELETE them from the shared dataset's
        latest version until the next classifier push re-stages its own
        real values; writing the neutral defaults here is equivalent to
        "as if not present" and avoids that gap. A classifier kernel only
        ever actually reads these when (re)pushed through run.py's own
        push(), which always re-stages fresh first -- so this is a safety
        net for a manual Kaggle-UI re-run of an old kernel version, not
        the normal path.
    """
    import shutil

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    for p in base._existing_source_files(repo_root):
        rel = p.relative_to(repo_root)
        dest = stage_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    (stage_dir / "SEGMENT_SMOKE_MODE").write_text("true" if smoke else "false")
    (stage_dir / "SMOKE_MODE").write_text("false")
    (stage_dir / "TRAIN_ARGS.json").write_text(json.dumps({"model_name": "tf_efficientnet_b3", "label_smoothing": 0.0}, indent=2))
    dataset_metadata = {
        "title": "dme-oct-src",
        "id": config.KAGGLE_SRC_DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (stage_dir / "dataset-metadata.json").write_text(json.dumps(dataset_metadata, indent=2))


def _push_single_file_dataset(
    slug: str, title: str, local_file: Path, stage_dir: Path, version_message: str, dest_filename: str = None
) -> None:
    """Stages one file at the root of a fresh dataset directory and
    creates/versions it. Used for the classifier checkpoint -- doesn't
    need anything fancier than 'this one file, flat, at the dataset root'
    (same 'flat, not nested' lesson as run.py's own SOURCE_FILES comment --
    kaggle datasets create/version silently skips subfolders by default).

    dest_filename normalizes the staged name regardless of the local
    file's own name (e.g. always 'checkpoint_best.pt' on Kaggle even if
    the local file is named after its run timestamp) -- entry_segment.py's
    _find_checkpoint_file() searches by *.pt glob rather than depending on
    this exact name, but normalizing here removes the ambiguity anyway
    rather than relying on the caller's local filename happening to match.
    Caught by this module's own staging test using a differently-named
    local file, which is exactly the case dest_filename exists to cover.
    """
    import shutil

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    shutil.copy2(local_file, stage_dir / (dest_filename or local_file.name))
    dataset_metadata = {"title": title, "id": slug, "licenses": [{"name": "other"}]}
    (stage_dir / "dataset-metadata.json").write_text(json.dumps(dataset_metadata, indent=2))
    if base._dataset_exists(slug):
        cmd = ["kaggle", "datasets", "version", "-p", str(stage_dir), "-m", version_message]
        context = f"kaggle datasets version ({slug})"
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(stage_dir)]
        context = f"kaggle datasets create ({slug})"
    base._run_kaggle_cli(cmd, context)
    base.wait_for_dataset_ready(slug)


def _push_duke_dataset(duke_dir: Path, version_message: str) -> None:
    """Duke ships as 10 separate Subject_XX.mat files, not one archive --
    stages all 10 flat at the dataset root (same reasoning as
    _push_single_file_dataset)."""
    import shutil

    files = sorted(Path(duke_dir).glob("Subject_*.mat"))
    if not files:
        raise FileNotFoundError(f"No Subject_*.mat files found under {duke_dir}.")
    stage_dir = config.KAGGLE_ARTIFACTS_DIR / "_duke_stage"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    for f in files:
        shutil.copy2(f, stage_dir / f.name)
    dataset_metadata = {
        "title": "dme-oct-duke-src",
        "id": config.KAGGLE_DUKE_DATASET_SLUG,
        "licenses": [{"name": "other"}],
    }
    (stage_dir / "dataset-metadata.json").write_text(json.dumps(dataset_metadata, indent=2))
    if base._dataset_exists(config.KAGGLE_DUKE_DATASET_SLUG):
        cmd = ["kaggle", "datasets", "version", "-p", str(stage_dir), "-m", version_message]
        context = "kaggle datasets version (duke)"
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(stage_dir)]
        context = "kaggle datasets create (duke)"
    base._run_kaggle_cli(cmd, context)
    base.wait_for_dataset_ready(config.KAGGLE_DUKE_DATASET_SLUG)


def _stage_segment_kernel_push_folder(stage_dir: Path) -> None:
    import shutil

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    # entry.py is NOT staged here, deliberately -- confirmed on the first
    # real push (2026-09-23): `kaggle kernels push` for a script-type
    # kernel uploads ONLY the code_file, not sibling .py files sitting in
    # the same local push folder. entry_segment.py is self-contained
    # precisely because of this (see its own module docstring) -- do not
    # reintroduce an `import entry` there or add entry.py back here
    # without re-verifying against a real push first.
    entry_segment_src = config.REPO_ROOT / "kaggle" / "entry_segment.py"
    shutil.copy2(entry_segment_src, stage_dir / "entry_segment.py")
    metadata = {
        "id": config.KAGGLE_SEGMENT_KERNEL_SLUG,
        "title": "dme-oct-segment",
        "code_file": "entry_segment.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [
            config.KAGGLE_SRC_DATASET_SLUG,
            config.KAGGLE_DUKE_DATASET_SLUG,
            config.KAGGLE_CLASSIFIER_CKPT_DATASET_SLUG,
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (stage_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))


def estimate_segment_run_minutes(quota_log: list, last_smoke) -> float:
    for entry in reversed(quota_log):
        if (
            not entry.get("smoke")
            and entry.get("status") == "complete"
            and entry.get("actual_minutes") is not None
            and entry.get("kernel_slug") == config.KAGGLE_SEGMENT_KERNEL_SLUG
        ):
            return entry["actual_minutes"] * 1.15
    if not last_smoke or last_smoke.get("status") != "passed" or not last_smoke.get("epoch_seconds"):
        return None
    # EPOCHS_SEG lives in segment.py, not config.py -- a deliberate choice
    # documented in segment.py's own module docstring (its judgment-call
    # constants are kept local rather than added to config.py, since the
    # brief's config list predates the segmentation phase). Imported here,
    # lazily, rather than duplicating the number -- this is the one place
    # run_segment.py needs it, and importing segment.py pulls in cv2/numpy
    # (not torch), which is already a real local dependency for anyone
    # running this from the same machine that runs segment.py itself.
    from segment import EPOCHS_SEG

    rough = (last_smoke["epoch_seconds"] * EPOCHS_SEG / 60.0) * 1.3
    print(
        "[run_segment] WARNING: no completed full segmentation run recorded yet -- extrapolating from the "
        "smoke run's single epoch. Unlike the classifier pipeline, segment.py's --smoke already trains on the "
        "FULL annotated Duke set (no subsampling), so this should be closer to real than the classifier's "
        "first estimate was -- but that's an expectation, not a confirmed fact (C6): treat with real "
        "skepticism until a real full run exists to compare."
    )
    return rough


def check_segment_smoke_gate(current_hash: str, last_smoke) -> None:
    if last_smoke is None:
        raise base.SmokeGateError("Refusing full push: no recorded segmentation smoke run yet. Run `push --smoke` first.")
    if last_smoke.get("source_hash") != current_hash:
        raise base.SmokeGateError(
            f"Refusing full push: last segmentation smoke was source={last_smoke.get('source_hash')}, "
            f"current push is source={current_hash}. Re-run `push --smoke` on this exact source first."
        )
    if last_smoke.get("status") != "passed":
        raise base.SmokeGateError(
            f"Refusing full push: last segmentation smoke has status {last_smoke.get('status')!r}, not 'passed'."
        )


def push_segment(
    repo_root: Path,
    smoke: bool,
    duke_dir: Path,
    classifier_checkpoint: Path,
    budget_min: float = None,
) -> str:
    base.check_username_configured()
    quota_log = base._load_json(config.KAGGLE_QUOTA_LOG, [])
    last_smoke = base._load_json(config.KAGGLE_SEGMENT_LAST_SMOKE, None)
    src_hash = base.git_sha_or_source_hash(repo_root)
    budget_min = config.KAGGLE_PUSH_BUDGET_DEFAULT_MIN if budget_min is None else budget_min

    if smoke:
        expected_minutes = float(config.KAGGLE_SEGMENT_SMOKE_TIME_BUDGET_MIN)
    else:
        check_segment_smoke_gate(src_hash, last_smoke)
        expected_minutes = estimate_segment_run_minutes(quota_log, last_smoke)
        if expected_minutes is None:
            raise RuntimeError(
                "Cannot estimate full segmentation run duration (no per-epoch timing recorded in "
                "segment_last_smoke.json) -- refusing to push without an estimate rather than guessing (C6)."
            )

    base.check_quota_guardrail(expected_minutes, quota_log, budget_min)

    version_message = src_hash
    segment_src_stage = config.KAGGLE_ARTIFACTS_DIR / "_segment_src_stage"
    _stage_segment_source_dataset(repo_root, segment_src_stage, smoke)
    if base._dataset_exists(config.KAGGLE_SRC_DATASET_SLUG):
        cmd = ["kaggle", "datasets", "version", "-p", str(segment_src_stage), "-m", version_message]
        context = "kaggle datasets version (src, +segment.py)"
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(segment_src_stage)]
        context = "kaggle datasets create (src, +segment.py)"
    base._run_kaggle_cli(cmd, context)
    base.wait_for_dataset_ready(config.KAGGLE_SRC_DATASET_SLUG)

    _push_duke_dataset(Path(duke_dir), version_message)
    _push_single_file_dataset(
        config.KAGGLE_CLASSIFIER_CKPT_DATASET_SLUG, "dme-oct-classifier-ckpt", Path(classifier_checkpoint),
        config.KAGGLE_ARTIFACTS_DIR / "_ckpt_stage", version_message, dest_filename="checkpoint_best.pt",
    )

    kernel_stage = config.KAGGLE_ARTIFACTS_DIR / "_segment_kernel_stage"
    _stage_segment_kernel_push_folder(kernel_stage)
    base._run_kaggle_cli(
        ["kaggle", "kernels", "push", "-p", str(kernel_stage)], "kaggle kernels push (segment)", infra_retry=True
    )

    quota_log.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_hash": src_hash,
            "kernel_slug": config.KAGGLE_SEGMENT_KERNEL_SLUG,
            "smoke": smoke,
            "expected_minutes": expected_minutes,
            "actual_minutes": None,
            "status": "pushed",
        }
    )
    base._save_json_atomic(config.KAGGLE_QUOTA_LOG, quota_log)
    return config.KAGGLE_SEGMENT_KERNEL_SLUG


_VERDICT_RE = re.compile(r"Verdict:\s*(ship|future work)")


def _find_phase4_verdict_line(log_text: str):
    """Substring search, not a stricter line.strip().startswith() check --
    same reasoning as run.py's own _find_c5_line(): caught by this
    module's own dry-run test against a fake fetched log, Kaggle's real
    fetched log is the JSON blob _real_duration_minutes_from_log()
    describes (records with a "data" field, possibly all on one compact
    line, possibly pretty-printed -- unconfirmed which without a live
    account), so a printed "Verdict: ..." line does not reliably start its
    own top-level text line. Goes one step further than _find_c5_line,
    though: rather than returning the whole matched (possibly
    JSON-wrapped, possibly compact-blob) line verbatim, extracts just the
    "Verdict: ship"/"Verdict: future work" substring via regex -- these
    are the only two literal strings segment.py ever prints there (see its
    verdict = "ship" if ... else "future work") -- so the recorded value
    stays clean and useful regardless of which raw log shape this turns
    out to be."""
    if not log_text:
        return None
    m = _VERDICT_RE.search(log_text)
    return m.group(0) if m else None


def push_and_run_segment(
    repo_root: Path, smoke: bool, duke_dir: Path, classifier_checkpoint: Path, budget_min: float = None
) -> dict:
    attempt = 0
    while True:
        slug = push_segment(repo_root, smoke, duke_dir, classifier_checkpoint, budget_min)
        poll_start = time.time()
        status, failure_message = base.poll(slug)
        wall_clock_minutes = (time.time() - poll_start) / 60.0
        out_dir = base.fetch(slug)
        log_text = (out_dir / "run.log").read_text(encoding="utf-8") if (out_dir / "run.log").exists() else ""

        real_minutes = base._real_duration_minutes_from_log(log_text)
        actual_minutes = real_minutes if real_minutes is not None else wall_clock_minutes
        if real_minutes is None:
            print(f"[run_segment] WARNING: could not read a real duration from the fetched log -- "
                  f"recording wall-clock time ({wall_clock_minutes:.1f}min) instead.")

        quota_log = base._load_json(config.KAGGLE_QUOTA_LOG, [])
        if quota_log:
            quota_log[-1]["status"] = status
            quota_log[-1]["actual_minutes"] = round(actual_minutes, 2)
            base._save_json_atomic(config.KAGGLE_QUOTA_LOG, quota_log)

        src_hash = base.git_sha_or_source_hash(repo_root)

        if status.lower() in base.TERMINAL_SUCCESS:
            verdict_line = _find_phase4_verdict_line(log_text)
            if smoke:
                base._save_json_atomic(
                    config.KAGGLE_SEGMENT_LAST_SMOKE,
                    {
                        "source_hash": src_hash,
                        "kernel_slug": slug,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "status": "passed",
                        "verdict_line": verdict_line,
                        "epoch_seconds": base._parse_epoch_seconds_from_training_log(
                            out_dir, log_filename="segmentation/segmentation_training_log.csv"
                        ),
                        "artifacts_dir": str(out_dir),
                    },
                )
            return {"status": status, "artifacts_dir": out_dir, "verdict_line": verdict_line, "attempts": attempt + 1}

        failure_type = base.classify_failure(status, failure_message, log_text)
        print(f"[run_segment] status={status!r} classified as: {failure_type}")

        if smoke:
            base._save_json_atomic(
                config.KAGGLE_SEGMENT_LAST_SMOKE,
                {
                    "source_hash": src_hash,
                    "kernel_slug": slug,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                    "failure_type": failure_type,
                    "artifacts_dir": str(out_dir),
                },
            )

        if failure_type == "code_error" or attempt >= config.KAGGLE_MAX_INFRA_RETRIES:
            raise base.RunFailed(
                f"Segmentation run failed (status={status}, classified={failure_type}) after {attempt + 1} "
                f"attempt(s). See {out_dir / 'run.log'}. "
                + ("Not retrying: this looks like a code error, not an infra failure."
                   if failure_type == "code_error"
                   else f"Not retrying: infra-retry budget ({config.KAGGLE_MAX_INFRA_RETRIES}) exhausted.")
            )
        attempt += 1
        print(f"[run_segment] infra failure, retrying (attempt {attempt + 1}/{config.KAGGLE_MAX_INFRA_RETRIES + 1})")


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Kaggle push/run automation for segment.py.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_push = sub.add_parser("push", help="Push src+Duke+checkpoint datasets and the segmentation kernel, poll, fetch.")
    p_push.add_argument("--smoke", action="store_true")
    p_push.add_argument("--budget-min", type=float, default=None)
    p_push.add_argument("--duke-dir", type=Path, default=config.REPO_ROOT / "data" / "raw" / "duke_dme" / "2015_BOE_Chiu")
    p_push.add_argument("--classifier-checkpoint", type=Path, required=True)

    args = parser.parse_args()
    repo_root = config.REPO_ROOT

    if args.command == "push":
        result = push_and_run_segment(
            repo_root, smoke=args.smoke, duke_dir=args.duke_dir,
            classifier_checkpoint=args.classifier_checkpoint, budget_min=args.budget_min,
        )
        print(f"\n[run_segment] DONE: {result}")


if __name__ == "__main__":
    main()
