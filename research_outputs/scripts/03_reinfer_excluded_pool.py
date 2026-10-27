"""
Phase B3 (supplementary, NEW analysis) -- score the existing checkpoints on the
7,503 train/ images whose patients also appear in the official test set.

build_splits() removes every image of those 305 patients from BOTH train and
validation before training (logged for B3, ResNet-50 and the A5 run; the B0
run's log is empty, so for B0 this is implied by kaggle/entry.py's code path
but not verified from a log). These images were therefore never used to fit or
select any of those models, which makes them a second, ~15x larger unseen
internal set. They are NOT independent of the official test set (same
patients; 114 are byte-identical copies of test images), so they are reported
separately, with patient-clustered bootstrap CIs, and never pooled with it.

No training and no checkpoint writes. Preprocessing = utils.preprocess_image via
PreprocessCache in research_outputs/.cache (never artifacts/). Unreadable JPEGs
are skipped and listed.

Usage: python research_outputs/scripts/03_reinfer_excluded_pool.py
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
from utils import PreprocessCache  # noqa: E402

OUT = REPO / "research_outputs" / "data"
CACHE = REPO / "research_outputs" / ".cache" / "preprocess"

CHECKPOINTS = {
    "EfficientNet-B3": "artifacts/kaggle/20260917T035646Z/artifacts/checkpoint_best.pt",
    "EfficientNet-B0": "artifacts/kaggle/20260920T092347Z/artifacts/checkpoint_best.pt",
    "ResNet-50": "artifacts/kaggle/20260920T110729Z/artifacts/checkpoint_best.pt",
}


def _warm(path):
    try:
        PreprocessCache(cache_dir=CACHE, image_size=config.IMAGE_SIZE).get_or_compute(path)
        return path, None
    except Exception as e:  # unreadable / truncated JPEG
        return path, f"{type(e).__name__}: {e}"


def main():
    import timm
    import torch

    hashes = pd.read_csv(OUT / "image_hashes.csv", dtype={"patient": str})
    test_patients = set(hashes[hashes.split == "test"].patient)
    exc = hashes[(hashes.split == "train") & hashes.patient.isin(test_patients)].copy()
    exc["abs_path"] = exc.path.map(lambda p: str(REPO / p))
    print(f"excluded-patient pool: {len(exc)} images, {exc.patient.nunique()} patients, DME={int(exc.label.sum())}")

    with ProcessPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(_warm, exc.abs_path.tolist(), chunksize=32))
    failed = {p: err for p, err in results if err}
    exc = exc[~exc.abs_path.isin(failed)].reset_index(drop=True)
    print(f"preprocessed; {len(failed)} unreadable skipped")

    cache = PreprocessCache(cache_dir=CACHE, image_size=config.IMAGE_SIZE)
    mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.array((0.229, 0.224, 0.225), dtype=np.float32)

    def batches(bs=64):
        for i in range(0, len(exc), bs):
            imgs = [cache.get(p)[0] for p in exc.abs_path.iloc[i:i + bs]]
            x = (np.stack(imgs).astype(np.float32) / 255.0 - mean) / std  # == A.Normalize(max_pixel_value=255)
            yield torch.from_numpy(x.transpose(0, 3, 1, 2).copy())

    out = exc[["path", "filename", "patient", "label", "md5"]].copy()
    for name, rel in CHECKPOINTS.items():
        ckpt = torch.load(REPO / rel, map_location="cpu", weights_only=False)
        model = timm.create_model(ckpt.get("model_name", "tf_efficientnet_b3"), pretrained=False, num_classes=2, drop_rate=config.DROP_RATE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        logits = []
        with torch.no_grad():
            for x in batches():
                logits.append(model(x).numpy())
        logits = np.concatenate(logits)
        z = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(z)[:, 1] / np.exp(z).sum(axis=1)
        slug = name.replace("-", "").replace("+", "_")
        out[f"logit_normal_{slug}"] = logits[:, 0]
        out[f"logit_dme_{slug}"] = logits[:, 1]
        out[f"p_dme_raw_{slug}"] = p
        print(name, "acc@0.5", float(((p >= 0.5) == out.label).mean()))

    out.to_csv(OUT / "excluded_pool_predictions.csv", index=False)
    with open(OUT / "excluded_pool_unreadable.json", "w") as f:
        json.dump({str(Path(k).relative_to(REPO).as_posix()): v for k, v in failed.items()}, f, indent=2)


if __name__ == "__main__":
    main()
