"""
Refit the temperature and the sensitivity-targeted threshold on the CORRECT
(training-time) validation set, then re-evaluate the official test set once.

Inputs:
  research_outputs/data/training_split_manifest.csv   (script 11, scikit-learn 1.4.2)
  research_outputs/data/test_predictions_<model>.csv  (script 01; test logits)
  the four unmodified checkpoints (read-only; nothing is written back --
  unlike evaluate.py, which saves threshold/temperature into the checkpoint)

Procedure (same functions and settings as evaluate.py, so only the data
changes): score the 5,560 validation images; temperature =
evaluate.fit_temperature(val logits); threshold = evaluate.sweep_threshold(
calibrated val probabilities, SENSITIVITY_TARGET=0.97); apply both to the
saved test logits. Also checks the rebuilt split against each checkpoint's
own stored val_auc, and measures how much of the old local "validation" set
(scikit-learn 1.9 partition, n=6,047) was actually training data.

Outputs (research_outputs/data/): val_predictions_true_split.csv,
refit_calibration_threshold.csv, val_split_contamination.json.

Usage (project environment): python research_outputs/scripts/12_refit_calibration_threshold.py
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import config  # noqa: E402
from evaluate import _softmax_positive, compute_metrics, expected_calibration_error, fit_temperature, sweep_threshold  # noqa: E402
from utils import PreprocessCache, build_splits  # noqa: E402

DATA = REPO / "research_outputs" / "data"
CACHE = REPO / "research_outputs" / ".cache" / "preprocess"
CHECKPOINTS = {
    "EfficientNet-B3": ("artifacts/kaggle/20260917T035646Z/artifacts/checkpoint_best.pt", "EfficientNet-B3"),
    "EfficientNet-B0": ("artifacts/kaggle/20260920T092347Z/artifacts/checkpoint_best.pt", "EfficientNet-B0"),
    "ResNet-50": ("artifacts/kaggle/20260920T110729Z/artifacts/checkpoint_best.pt", "ResNet-50"),
    "EfficientNet-B3 + LS 0.05 (ablation)": ("artifacts/kaggle/20260922T160023Z/artifacts/checkpoint_best.pt", "EfficientNet-B3_LS005"),
}


def _warm(path):
    PreprocessCache(cache_dir=CACHE, image_size=config.IMAGE_SIZE).get_or_compute(path)
    return path


def main():
    import timm
    import torch
    from sklearn.metrics import roc_auc_score

    man = pd.read_csv(DATA / "training_split_manifest.csv", dtype={"patient": str})
    val = man[man.split == "val"].reset_index(drop=True)
    val["abs_path"] = val.path.map(lambda p: str(REPO / p))
    y_val = (val.label == "DME").astype(int).to_numpy()
    print(f"true validation set: {len(val)} images ({int(y_val.sum())} DME), {val.patient.nunique()} patients")

    # --- contamination of the old local 'validation' set (scikit-learn installed here, 1.9) ---
    import sklearn
    local = build_splits(REPO / "data" / "raw" / "OCT2017" / "train", test_dir=REPO / "data" / "raw" / "OCT2017" / "test").val_df
    local_paths = set(local.path.map(lambda p: Path(p).resolve().relative_to(REPO).as_posix()))
    true_train = set(man[man.split == "train"].path); true_val = set(val.path)
    contam = {"local_sklearn_version": sklearn.__version__, "local_val_n": len(local_paths),
              "local_val_images_in_true_TRAIN": len(local_paths & true_train),
              "local_val_images_in_true_val": len(local_paths & true_val),
              "local_val_fraction_in_true_train": len(local_paths & true_train) / len(local_paths),
              "local_val_patients_in_true_train": len(set(local.patient) & set(man[man.split == "train"].patient)),
              "local_val_patients": local.patient.nunique()}
    json.dump(contam, open(DATA / "val_split_contamination.json", "w"), indent=2)
    print(contam)

    with ProcessPoolExecutor(max_workers=10) as ex:
        list(ex.map(_warm, val.abs_path.tolist(), chunksize=32))
    cache = PreprocessCache(cache_dir=CACHE, image_size=config.IMAGE_SIZE)
    mean = np.array((0.485, 0.456, 0.406), dtype=np.float32); std = np.array((0.229, 0.224, 0.225), dtype=np.float32)

    def batches(bs=64):
        for i in range(0, len(val), bs):
            x = (np.stack([cache.get(p)[0] for p in val.abs_path.iloc[i:i + bs]]).astype(np.float32) / 255.0 - mean) / std
            yield torch.from_numpy(x.transpose(0, 3, 1, 2).copy())

    out_val = val[["path", "patient"]].copy(); out_val["label"] = y_val
    rows = []
    for name, (rel, slug) in CHECKPOINTS.items():
        ckpt = torch.load(REPO / rel, map_location="cpu", weights_only=False)
        model = timm.create_model(ckpt.get("model_name", "tf_efficientnet_b3"), pretrained=False, num_classes=2, drop_rate=config.DROP_RATE)
        model.load_state_dict(ckpt["model_state_dict"]); model.eval()
        with torch.no_grad():
            vlog = np.concatenate([model(x).numpy() for x in batches()])
        out_val[f"logit_normal_{slug}"] = vlog[:, 0]; out_val[f"logit_dme_{slug}"] = vlog[:, 1]

        # same procedure as evaluate.py, on the correct validation set
        t_new = fit_temperature(vlog, y_val)
        thr_new, v_sens, v_spec = sweep_threshold(y_val, _softmax_positive(vlog / t_new), config.SENSITIVITY_TARGET)
        val_auc = roc_auc_score(y_val, _softmax_positive(vlog))

        te = pd.read_csv(DATA / f"test_predictions_{slug}.csv")
        tlog = te[["logit_normal", "logit_dme"]].to_numpy(); y_te = te.label.to_numpy()
        p_raw = _softmax_positive(tlog); p_cal = _softmax_positive(tlog / t_new)
        m_new = compute_metrics(y_te, p_cal, thr_new)
        ece_raw, _ = expected_calibration_error(y_te, p_raw); ece_cal, _ = expected_calibration_error(y_te, p_cal)
        brier_raw = float(np.mean((p_raw - y_te) ** 2)); brier_cal = float(np.mean((p_cal - y_te) ** 2))
        vp_raw = _softmax_positive(vlog); vp_cal = _softmax_positive(vlog / t_new)
        rows.append({
            "model": name, "val_n": len(val), "val_n_dme": int(y_val.sum()),
            "val_auroc_recomputed": val_auc, "val_auroc_stored_in_checkpoint": ckpt.get("val_auc"),
            "temperature_old_contaminated": ckpt.get("temperature"), "temperature_refit": t_new,
            "threshold_old_contaminated": ckpt.get("threshold"), "threshold_refit": thr_new,
            "val_sensitivity_at_refit": v_sens, "val_specificity_at_refit": v_spec,
            "val_ece_raw": expected_calibration_error(y_val, vp_raw)[0], "val_ece_calibrated": expected_calibration_error(y_val, vp_cal)[0],
            "test_TP": m_new["tp"], "test_FP": m_new["fp"], "test_TN": m_new["tn"], "test_FN": m_new["fn"],
            "test_accuracy": m_new["accuracy"], "test_sensitivity": m_new["sensitivity"], "test_specificity": m_new["specificity"],
            "test_precision": m_new["precision"], "test_f1": m_new["f1"], "test_auroc": m_new["auc"],
            "test_ece_raw": ece_raw, "test_ece_refit_calibrated": ece_cal, "test_brier_raw": brier_raw, "test_brier_refit_calibrated": brier_cal,
        })
        print({k: rows[-1][k] for k in ("model", "val_auroc_recomputed", "val_auroc_stored_in_checkpoint", "temperature_refit", "threshold_refit", "test_TP", "test_FN", "test_FP")})

    out_val.to_csv(DATA / "val_predictions_true_split.csv", index=False)
    pd.DataFrame(rows).to_csv(DATA / "refit_calibration_threshold.csv", index=False)


if __name__ == "__main__":
    main()
