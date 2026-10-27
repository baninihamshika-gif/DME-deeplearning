"""
Phase B3 -- leakage / shortcut scan over the Kermany OCT2017 DME+NORMAL images.

For every image in train/{DME,NORMAL} and test/{DME,NORMAL}:
  * MD5 of the raw file bytes (exact duplicates)
  * 64-bit difference hash (dHash, 9x8 grayscale) for near-duplicates
  * width, height
Then reports:
  * exact / near duplicate pairs within test, and between test and the
    training pool actually available to the models (train/ minus every image
    of a patient that also appears in test/ -- the exclusion build_splits()
    applies; this union of train+val is identical regardless of how
    StratifiedGroupKFold partitioned it)
  * cross-class duplicates (same pixels, different label)
  * patient-ID overlap test <-> training pool (should be 0 after exclusion)
  * dimension distribution by class and split
  * a geometry-only logistic regression (trained on the pool, scored on test)

Outputs under research_outputs/data/: image_hashes.csv, leakage_summary.json,
near_duplicate_pairs.csv, dimension_by_class.csv.

Usage: python research_outputs/scripts/02_leakage_scan.py
"""

import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from utils import parse_patient_id  # noqa: E402

OUT = REPO / "research_outputs" / "data"
RAW = REPO / "data" / "raw" / "OCT2017"
NEAR_DUP_HAMMING = 4  # <=4 of 64 bits differ


def _hash_one(path: str):
    data = Path(path).read_bytes()
    md5 = hashlib.md5(data).hexdigest()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return path, md5, -1, -1, None
    h, w = img.shape
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    dhash = int("".join("1" if b else "0" for b in bits), 2)
    return path, md5, w, h, dhash


def popcount64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    c = np.zeros_like(x)
    for _ in range(64):
        c += x & np.uint64(1)
        x >>= np.uint64(1)
    return c


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("train", "test"):
        for folder, label in (("DME", 1), ("NORMAL", 0)):
            for p in sorted((RAW / split / folder).iterdir()):
                if p.suffix.lower() in (".jpeg", ".jpg", ".png"):
                    rows.append({"split": split, "label": label, "path": str(p), "filename": p.name, "patient": parse_patient_id(p)})
    df = pd.DataFrame(rows)

    hashes_path = OUT / "image_hashes.csv"
    if hashes_path.exists():
        h = pd.read_csv(hashes_path, dtype={"patient": str})
        df = h
    else:
        with ProcessPoolExecutor(max_workers=8) as ex:
            res = list(ex.map(_hash_one, df["path"].tolist(), chunksize=256))
        r = pd.DataFrame(res, columns=["path", "md5", "width", "height", "dhash"])
        df = df.merge(r, on="path")
        df["path"] = df["path"].map(lambda s: Path(s).relative_to(REPO).as_posix())
        df.to_csv(hashes_path, index=False)

    test = df[df.split == "test"].copy()
    train = df[df.split == "train"].copy()
    test_patients = set(test.patient)
    pool = train[~train.patient.isin(test_patients)].copy()       # what training+validation could contain
    excluded = train[train.patient.isin(test_patients)].copy()     # never seen by any model (if exclusion applied)

    summary = {
        "n_train_folder": len(train), "n_test": len(test),
        "n_training_pool_after_exclusion": len(pool),
        "n_excluded_test_patient_images": len(excluded),
        "n_test_patients": len(test_patients),
        "n_test_patients_also_in_train_folder": len(test_patients & set(train.patient)),
        "patient_overlap_test_vs_pool": len(test_patients & set(pool.patient)),
        "unreadable_images": int((df.width < 0).sum()),
        "test_class_counts": test.label.map({1: "DME", 0: "Normal"}).value_counts().to_dict(),
        "test_images_per_patient": test.groupby("patient").size().describe().to_dict(),
        "test_patients_with_both_labels": int((test.groupby("patient").label.nunique() > 1).sum()),
    }

    # --- exact duplicates ---
    def exact_pairs(a, b, same=False):
        m = a[["path", "md5", "label", "patient"]].merge(b[["path", "md5", "label", "patient"]], on="md5", suffixes=("_a", "_b"))
        if same:
            m = m[m.path_a < m.path_b]
        return m

    ex_tt = exact_pairs(test, test, same=True)
    ex_tp = exact_pairs(test, pool)
    ex_te = exact_pairs(test, excluded)
    ex_pp = exact_pairs(pool, pool, same=True)
    summary["exact_dup_pairs_within_test"] = len(ex_tt)
    summary["exact_dup_pairs_test_vs_pool"] = len(ex_tp)
    summary["exact_dup_pairs_test_vs_excluded"] = len(ex_te)
    summary["exact_dup_pairs_within_pool"] = len(ex_pp)
    summary["exact_dup_pairs_within_pool_cross_label"] = int((ex_pp.label_a != ex_pp.label_b).sum())

    # --- near duplicates (dHash Hamming distance) ---
    valid = df.dhash.notna()
    test_v = test[valid.loc[test.index]]
    pool_v = pool[valid.loc[pool.index]]
    th = test_v.dhash.astype(np.uint64).to_numpy()
    ph = pool_v.dhash.astype(np.uint64).to_numpy()
    pairs = []
    for i, hv in enumerate(th):
        d_pool = popcount64(ph ^ hv)
        for j in np.nonzero(d_pool <= NEAR_DUP_HAMMING)[0]:
            pairs.append(("test-pool", test_v.iloc[i].path, pool_v.iloc[j].path, int(d_pool[j]), int(test_v.iloc[i].label), int(pool_v.iloc[j].label)))
        d_test = popcount64(th ^ hv)
        for j in np.nonzero(d_test <= NEAR_DUP_HAMMING)[0]:
            if j > i:
                pairs.append(("test-test", test_v.iloc[i].path, test_v.iloc[j].path, int(d_test[j]), int(test_v.iloc[i].label), int(test_v.iloc[j].label)))
    nd = pd.DataFrame(pairs, columns=["kind", "path_a", "path_b", "hamming", "label_a", "label_b"])
    nd.to_csv(OUT / "near_duplicate_pairs.csv", index=False)
    summary["near_dup_threshold_hamming"] = NEAR_DUP_HAMMING
    summary["near_dup_pairs_test_vs_pool"] = int((nd.kind == "test-pool").sum())
    summary["near_dup_test_images_with_pool_match"] = int(nd[nd.kind == "test-pool"].path_a.nunique())
    summary["near_dup_pairs_within_test"] = int((nd.kind == "test-test").sum())
    summary["near_dup_pairs_cross_label"] = int((nd.label_a != nd.label_b).sum())

    # --- dimensions by class/split ---
    df["dims"] = df.width.astype(str) + "x" + df.height.astype(str)
    df["subset"] = np.where(df.split == "test", "test", np.where(df.patient.isin(test_patients), "train_excluded", "train_pool"))
    dim_tab = df.groupby(["subset", "label", "dims"]).size().rename("n").reset_index()
    dim_tab["frac_within_subset_label"] = dim_tab.n / dim_tab.groupby(["subset", "label"]).n.transform("sum")
    dim_tab["label"] = dim_tab.label.map({1: "DME", 0: "Normal"})
    dim_tab.sort_values(["subset", "label", "n"], ascending=[True, True, False]).to_csv(OUT / "dimension_by_class.csv", index=False)

    # --- geometry-only shortcut classifier ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    def feats(d):
        w, h = d.width.to_numpy(float), d.height.to_numpy(float)
        scale = 300.0 / np.maximum(w, h)
        return np.c_[w, h, w / h, np.round(w * scale), np.round(h * scale), 300 - np.round(h * scale)]

    sc = StandardScaler().fit(feats(pool))
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(feats(pool)), pool.label)
    s_test = lr.predict_proba(sc.transform(feats(test)))[:, 1]
    summary["geometry_only_logreg_test_auroc"] = float(roc_auc_score(test.label, s_test))
    summary["geometry_only_logreg_test_acc_at_0.5"] = float(((s_test >= 0.5) == test.label).mean())
    s_exc = lr.predict_proba(sc.transform(feats(excluded)))[:, 1]
    summary["geometry_only_logreg_excluded_pool_auroc"] = float(roc_auc_score(excluded.label, s_exc))
    for name, d in (("pool", pool), ("test", test), ("excluded", excluded)):
        for lab, lname in ((1, "DME"), (0, "Normal")):
            sub = d[d.label == lab]
            summary[f"frac_width512_{name}_{lname}"] = float((sub.width == 512).mean())

    with open(OUT / "leakage_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
