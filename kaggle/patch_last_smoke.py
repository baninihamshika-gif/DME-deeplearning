"""
One-off: the 2026-09-17 tf_efficientnet_b0 smoke push's `kaggle kernels
output` download was cut short by a local network blip, so last_smoke.json
got saved with epoch_seconds=None. The 2026-09-19 re-fetch confirms the run
genuinely passed (C5 PASSED, training completed cleanly) -- this just
recomputes the fields that depended on the interrupted download, using the
same functions push_and_run() itself uses, now that the real data is here.
Nothing here is guessed; delete this file once it's run.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as kaggle_run
import config

OUT_DIR = Path(r"C:\Users\banin\DME-DEEPLEARN\artifacts\kaggle\20260919T183145Z")

log_text = (OUT_DIR / "run.log").read_text(encoding="utf-8")
c5_line = kaggle_run._find_c5_line(log_text)
epoch_seconds = kaggle_run._parse_epoch_seconds_from_training_log(OUT_DIR)
real_minutes = kaggle_run._real_duration_minutes_from_log(log_text)

last_smoke = json.loads(config.KAGGLE_LAST_SMOKE.read_text())
print("Before:", json.dumps(last_smoke, indent=2))

last_smoke["c5_passed"] = c5_line is not None
last_smoke["epoch_seconds"] = epoch_seconds
last_smoke["artifacts_dir"] = str(OUT_DIR)
config.KAGGLE_LAST_SMOKE.write_text(json.dumps(last_smoke, indent=2))

# Bonus precision fix: the quota_log entry for this same push recorded
# poll()'s wall-clock time (18.2 min) as actual_minutes because the log
# fetch failed at push time. The real per-run duration is in the log now.
quota_log = json.loads(config.KAGGLE_QUOTA_LOG.read_text())
for entry in reversed(quota_log):
    if entry.get("model_name") == "tf_efficientnet_b0" and entry.get("smoke") and entry.get("status") == "complete":
        print(f"\nCorrecting quota_log actual_minutes: {entry['actual_minutes']} -> {round(real_minutes, 2)}")
        entry["actual_minutes"] = round(real_minutes, 2)
        break
config.KAGGLE_QUOTA_LOG.write_text(json.dumps(quota_log, indent=2))

print("\nAfter:", json.dumps(last_smoke, indent=2))
print(f"\nc5_line: {c5_line!r}")