"""
Reconstructs a PARTIAL training_log.csv from a fetched Kaggle run log
(the JSON-lines format `kaggle kernels logs` / kaggle/run.py's `logs`
subcommand produces), for the one real case this project hit: the real
training_log.csv never fully downloaded (a mid-download connection reset
during `kaggle kernels output` dropped that one file specifically, while
checkpoint_best.pt/checkpoint_last.pt and the JSON log both survived --
see the 2026-09-17 diagnosis), so evaluate.py's --training-log-csv has had
nothing to point at.

IMPORTANT -- what this is and is not:
  - train_loss / val_loss / val_auc / epoch_time_sec / phase / global_epoch
    ARE recovered exactly, because train.py's run_phase() prints every one
    of those to stdout every epoch (see the f-string in run_phase()) and
    that stdout is what's captured in the JSON log. This is the same real
    run's real numbers, just re-extracted from a different place they were
    also written -- not estimated, not fabricated.
  - train_accuracy / val_accuracy / learning_rate / trainable_params are
    NOT recoverable this way: run_phase() only ever wrote those to the CSV
    itself (_log_epoch_row), never printed them to stdout. Those columns
    are simply absent from the output here, not filled with a guess.

Usage:
    python reconstruct_training_log.py full_run_log.txt artifacts\\training_log_reconstructed.csv
"""

import csv
import json
import re
import sys
from pathlib import Path

# Mirrors run_phase()'s print format exactly:
#   f"[{phase_name} epoch {epoch_in_phase + 1}/{epochs_total} | global {global_epoch}] "
#   f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f} "
#   f"{'*best*' if improved else ''} ({epoch_time:.1f}s)"
_EPOCH_LINE = re.compile(
    r"^\[(?P<phase>[AB]) epoch (?P<epoch_in_phase>\d+)/(?P<epochs_total>\d+) \| "
    r"global (?P<global_epoch>\d+)\] train_loss=(?P<train_loss>[\d.]+) "
    r"val_loss=(?P<val_loss>[\d.]+) val_auc=(?P<val_auc>[\d.]+) *(?P<best>\*best\*)? *"
    r"\((?P<epoch_time_sec>[\d.]+)s\)$"
)

CSV_FIELDNAMES = [
    "phase", "global_epoch", "epoch_in_phase", "train_loss", "val_loss",
    "val_auc", "epoch_time_sec", "is_best",
]


def parse_epoch_rows(log_text: str) -> list:
    """
    Extract one row per completed epoch from run.py/run_phase()'s stdout,
    in log order (== epoch order -- the real run never reorders or
    repeats an epoch print). Lines that don't match the exact epoch
    format (e.g. the final "Training complete: ..." summary line, or
    anything else printed during the run) are silently skipped, not an
    error -- this is meant to run against a real, noisy full log.
    """
    rows = []
    for line in log_text.splitlines():
        match = _EPOCH_LINE.match(line.strip())
        if match is None:
            continue
        rows.append(
            {
                "phase": match.group("phase"),
                "global_epoch": int(match.group("global_epoch")),
                "epoch_in_phase": int(match.group("epoch_in_phase")),
                "train_loss": float(match.group("train_loss")),
                "val_loss": float(match.group("val_loss")),
                "val_auc": float(match.group("val_auc")),
                "epoch_time_sec": float(match.group("epoch_time_sec")),
                "is_best": match.group("best") is not None,
            }
        )
    return rows


def load_kaggle_log_text(path) -> str:
    """
    Kaggle's `kernels logs` output, redirected on Windows (`> file.txt`),
    lands on disk encoded as the console's active code page (cp1252 on
    this project's machine), not UTF-8 -- confirmed directly against this
    exact file (2026-09-17: a plain UTF-8 read raised UnicodeDecodeError
    at a smart-quote byte inside a GPU name string). Tries utf-8 first
    (the common case elsewhere), falls back to cp1252, and only as a last
    resort falls back to latin-1 with lossy replacement -- never silently
    drops bytes if a clean decode is available.
    """
    raw = Path(path).read_bytes()
    for encoding in ("utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def extract_stdout_text(text: str) -> str:
    """
    The fetched log is Kaggle's own JSON-lines-style array of
    {"stream_name", "time", "data"} records (see kaggle/run.py's
    _real_duration_minutes_from_log for the same format). Concatenates
    every record's "data" field, in order, into one text blob to scan for
    epoch lines. Falls back to treating `text` as already-plain stdout if
    it isn't valid JSON (e.g. someone points this at a raw piped log
    instead of the `kernels logs` JSON output).
    """
    try:
        records = json.loads(text)
    except json.JSONDecodeError:
        return text
    return "\n".join(r.get("data", "") for r in records if isinstance(r, dict))


def main():
    if len(sys.argv) != 3:
        print("Usage: python reconstruct_training_log.py <run_log.txt> <output.csv>")
        sys.exit(1)
    log_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])

    text = load_kaggle_log_text(log_path)
    log_text = extract_stdout_text(text)

    rows = parse_epoch_rows(log_text)
    if not rows:
        raise SystemExit(f"No epoch lines found in {log_path} -- wrong file, or the log format changed.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[reconstruct_training_log] wrote {len(rows)} epoch rows to {out_path}")
    # Deliberately NOT reporting "the best epoch" here: val_auc is only printed
    # to 4 decimal places, so on this project's real log epochs 19, 20, and 22
    # all tie at the printed 0.9958 -- naively taking the first (or any) max
    # over these rows silently picks the wrong one against the checkpoint's own
    # full-float64-precision record (confirmed: checkpoint_best.pt's actual
    # global_epoch is 20, val_auc=0.9958184899820804). The checkpoint itself
    # (already reported by evaluate.py's "[evaluate] checkpoint: ..." line) is
    # the authoritative source for best epoch/val_auc; this script only recovers
    # the per-epoch curve for plotting, not a substitute ranking of epochs.
    print(
        "[reconstruct_training_log] NOTE: train_accuracy/val_accuracy/learning_rate/trainable_params "
        "were never printed to stdout, so those columns are absent here -- this is a partial "
        "reconstruction (train_loss/val_loss/val_auc/epoch_time only), not a substitute for the "
        "real training_log.csv."
    )


if __name__ == "__main__":
    main()
