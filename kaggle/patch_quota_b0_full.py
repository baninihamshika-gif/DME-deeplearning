"""
One-off: this tf_efficientnet_b0 FULL push's poll() wall-clock time
(815.4 min) was inflated by a local network/sleep gap during the long
wait -- the same failure mode already documented in _real_duration_
minutes_from_log()'s docstring (a prior ~9hr outage once inflated 681.86
to a real 319.77 for the original B3 push). The kernel's own log
timestamps are ground truth. This re-fetches just the log text (fast,
no file downloads) and uses the already-tested function to correct
quota_log.json. Delete this file once it's run.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as kaggle_run
import config

SLUG = "baninihamshika/dme-oct-train"

log_text = kaggle_run.logs(SLUG)
real_minutes = kaggle_run._real_duration_minutes_from_log(log_text)
c5_line = kaggle_run._find_c5_line(log_text)
print(f"\nReal duration from log: {real_minutes:.2f} min")
print(f"C5 line: {c5_line!r}")

quota_log = json.loads(config.KAGGLE_QUOTA_LOG.read_text())
for entry in reversed(quota_log):
    if entry.get("model_name") == "tf_efficientnet_b0" and not entry.get("smoke") and entry.get("status") == "pushed":
        print(f"\nCorrecting entry: status 'pushed' -> 'complete', actual_minutes {entry['actual_minutes']} -> {round(real_minutes, 2)}")
        entry["status"] = "complete"
        entry["actual_minutes"] = round(real_minutes, 2)
        break
else:
    print("WARNING: no matching quota_log entry found -- check manually.")

config.KAGGLE_QUOTA_LOG.write_text(json.dumps(quota_log, indent=2))
print("Wrote corrected quota_log.json")