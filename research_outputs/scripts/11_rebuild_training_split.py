"""
Rebuild the TRAINING-TIME train/validation split exactly as the Kaggle runs
produced it, and save it as a manifest.

Why: the Kaggle runs used scikit-learn 1.4.2; the local evaluate.py re-ran
the same utils.build_splits() under scikit-learn 1.9.0, whose
StratifiedGroupKFold returned a different patient partition (val 6,047
instead of 5,560). The threshold and temperature were fitted on that wrong
partition (research_audit/leakage_audit.md section 5).

Must run under scikit-learn 1.4.2, i.e. Python <= 3.12, e.g.:
    uv venv --python 3.12 .venv-sk142
    uv pip install --python .venv-sk142 "scikit-learn==1.4.2" "numpy==1.26.*" "pandas==2.2.*"
    .venv-sk142/Scripts/python research_outputs/scripts/11_rebuild_training_split.py

Calls the project's own utils.build_splits() unmodified. utils imports cv2
at module level but build_splits() never uses it, so cv2 is stubbed rather
than installed. Fails loudly unless every count matches the Kaggle log
(full_run_log.txt line 83 and the ResNet-50 / A5 run logs).

Output: research_outputs/data/training_split_manifest.csv
        (path, label, patient, split in {train, val}).
"""

import sys
import types
from pathlib import Path

import sklearn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.modules.setdefault("cv2", types.ModuleType("cv2"))  # unused by build_splits()

from utils import build_splits  # noqa: E402

assert sklearn.__version__ == "1.4.2", f"needs scikit-learn 1.4.2 (Kaggle), got {sklearn.__version__}"

# Logged by every full Kaggle run: "excluded 305 overlapping patient(s) (7503 images:
# train 30666->24600, val 6997->5560)"
EXPECTED = {"excluded_patients": 305, "excluded_images": 7503, "train": 24600, "val": 5560}

r = build_splits(REPO / "data" / "raw" / "OCT2017" / "train", test_dir=REPO / "data" / "raw" / "OCT2017" / "test")
got = {"excluded_patients": r.excluded_test_overlap_patients, "excluded_images": r.excluded_test_overlap_images,
       "train": len(r.train_df), "val": len(r.val_df)}
assert got == EXPECTED, f"split does not match the Kaggle log: {got} != {EXPECTED}"
assert not set(r.train_df.patient) & set(r.val_df.patient)

rows = []
for split, d in (("train", r.train_df), ("val", r.val_df)):
    d = d.copy()
    d["path"] = d.path.map(lambda p: Path(p).resolve().relative_to(REPO).as_posix())
    d["split"] = split
    rows.append(d[["path", "label", "patient", "split"]])
import pandas as pd  # noqa: E402

out = pd.concat(rows, ignore_index=True)
out.to_csv(REPO / "research_outputs" / "data" / "training_split_manifest.csv", index=False)
print(f"MATCHES KAGGLE LOG {got}; val classes {r.val_df.label.value_counts().to_dict()}; train classes {r.train_df.label.value_counts().to_dict()}")
