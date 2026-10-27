"""
Phase B3 -- quantify the white-wedge corner artifact (rotation/registration
fill, pixel value ~255) seen in some Kermany B-scans, by class and subset,
to check whether it could act as a label shortcut.

An image is flagged if any of its four 12x12 corner patches has median
intensity >= 250 (pure-white fill; retinal tissue and background noise never
saturate uniformly over a whole corner patch). Also scores the flag as a
one-feature classifier on the test set.

Output: research_outputs/data/border_artifact_by_class.csv and
border_artifact_summary.json.

Usage: python research_outputs/scripts/06_border_artifact_scan.py
"""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "research_outputs" / "data"
P = 12


def flag(path):
    img = cv2.imread(str(REPO / path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return path, np.nan
    c = [img[:P, :P], img[:P, -P:], img[-P:, :P], img[-P:, -P:]]
    return path, float(any(np.median(x) >= 250 for x in c))


def main():
    h = pd.read_csv(OUT / "image_hashes.csv", dtype={"patient": str})
    test_patients = set(h[h.split == "test"].patient)
    h["subset"] = np.where(h.split == "test", "test", np.where(h.patient.isin(test_patients), "train_excluded", "train_pool"))
    with ProcessPoolExecutor(max_workers=10) as ex:
        res = dict(ex.map(flag, h.path.tolist(), chunksize=256))
    h["white_corner"] = h.path.map(res)
    tab = h.groupby(["subset", "label"]).white_corner.agg(["mean", "sum", "count"]).reset_index()
    tab["label"] = tab.label.map({1: "DME", 0: "Normal"})
    tab.rename(columns={"mean": "fraction_flagged", "sum": "n_flagged", "count": "n"}).to_csv(OUT / "border_artifact_by_class.csv", index=False)
    t = h[h.subset == "test"]
    from sklearn.metrics import roc_auc_score
    summ = {"rule": f"any {P}x{P} corner patch with median >= 250",
            "test_auroc_flag_as_DME_score": float(roc_auc_score(t.label, t.white_corner)),
            "pool_auroc_flag_as_DME_score": float(roc_auc_score(h[h.subset == "train_pool"].label, h[h.subset == "train_pool"].white_corner.fillna(0)))}
    json.dump(summ, open(OUT / "border_artifact_summary.json", "w"), indent=2)
    print(tab.to_string()); print(summ)


if __name__ == "__main__":
    main()
