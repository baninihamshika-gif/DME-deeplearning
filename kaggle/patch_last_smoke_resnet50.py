"""
One-off: the resnet50 smoke push (2026-09-20) trained fine on Kaggle (C5
passed for real) but every `kaggle kernels output` fetch since has died at
almost exactly the same point -- right after checkpoint_best.pt, before
checkpoint_last.pt even starts -- which points to a fixed connection/size
limit on this machine's network path, not random bad luck. training_log.csv
never comes down, and estimate_full_run_minutes() needs last_smoke.json's
epoch_seconds to size the full ResNet-50 push (no completed full-run history
exists yet for this architecture).

training_log.csv's epoch_time_sec column would just contain the same numbers
train.py already prints to stdout -- "(96.9s)" for the Phase A epoch and
"(9.4s)" for the Phase B epoch in this run's real log. This re-fetches the
log fresh (a small JSON call, not a file download -- reliable every time so
far) and parses those same real numbers out of it, rather than waiting
indefinitely for a CSV download that keeps failing. Nothing here is
invented: these are the exact durations this real run reported for itself.

Also fixes c5_passed (wrongly recorded False because the log fetch failed
mid-push, before the C5 line could be seen) and points artifacts_dir at the
one fetch directory that actually has a complete checkpoint_best.pt.

Delete this file once it's run.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as kaggle_run
import config

SLUG = "baninihamshika/dme-oct-train"
GOOD_ARTIFACTS_DIR = r"C:\Users\banin\DME-DEEPLEARN\artifacts\kaggle\20260920T105605Z"

log_text = kaggle_run.logs(SLUG)
c5_line = kaggle_run._find_c5_line(log_text)
real_minutes = kaggle_run._real_duration_minutes_from_log(log_text)

records = json.loads(log_text)
epoch_pattern = re.compile(r"\| global \d+\].*?\(([\d.]+)s\)")
epoch_times = []
for rec in records:
    if isinstance(rec, dict) and rec.get("stream_name") == "stdout":
        m = epoch_pattern.search(rec.get("data", ""))
        if m:
            epoch_times.append(float(m.group(1)))
epoch_seconds = (sum(epoch_times) / len(epoch_times)) if epoch_times else None

print(f"c5_line: {c5_line!r}")
print(f"real_minutes (from log timestamps): {real_minutes}")
print(f"epoch_times found in stdout: {epoch_times}")
print(f"epoch_seconds (average): {epoch_seconds}")

last_smoke = json.loads(config.KAGGLE_LAST_SMOKE.read_text())
print("\nBefore:", json.dumps(last_smoke, indent=2))

last_smoke["c5_passed"] = c5_line is not None
last_smoke["epoch_seconds"] = epoch_seconds
last_smoke["artifacts_dir"] = GOOD_ARTIFACTS_DIR
config.KAGGLE_LAST_SMOKE.write_text(json.dumps(last_smoke, indent=2))

quota_log = json.loads(config.KAGGLE_QUOTA_LOG.read_text())
for entry in reversed(quota_log):
    if entry.get("model_name") == "resnet50" and entry.get("smoke") and entry.get("status") == "complete":
        print(f"\nCorrecting quota_log actual_minutes: {entry.get('actual_minutes')} -> {round(real_minutes, 2)}")
        entry["actual_minutes"] = round(real_minutes, 2)
        break
else:
    print("\nWARNING: no matching resnet50 smoke quota_log entry found -- check manually.")
config.KAGGLE_QUOTA_LOG.write_text(json.dumps(quota_log, indent=2))

print("\nAfter:", json.dumps(last_smoke, indent=2))
