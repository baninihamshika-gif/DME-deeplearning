"""
Phase B2 -- regenerate per-image test-set predictions from the EXISTING
classifier checkpoints (no training, no checkpoint writes).

The original evaluate.py never saved per-image outputs, only aggregates in
metrics.json, so independent metric reconstruction requires re-running the
forward pass. Preprocessing is the project's own utils.preprocess_image via
PreprocessCache, pointed at a separate cache directory so nothing under
artifacts/ is touched. Models are built with pretrained=False (weights come
entirely from the checkpoint) to avoid any network download.

Output: research_outputs/data/test_predictions_<model>.csv with columns
path, filename, patient, label (0=Normal, 1=DME), logit_normal, logit_dme,
p_dme_raw, p_dme_calibrated (softmax(logits / T_stored)), plus a JSON of
checkpoint metadata.

Usage (from repo root):
    python research_outputs/scripts/01_reinfer_classifiers.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import config  # noqa: E402
from train import OCTDataset, build_class_to_idx  # noqa: E402
from utils import PreprocessCache, load_class_folder_df, set_seed  # noqa: E402

OUT = REPO / "research_outputs" / "data"
CACHE = REPO / "research_outputs" / ".cache" / "preprocess"

CHECKPOINTS = {
    "EfficientNet-B3": "artifacts/kaggle/20260917T035646Z/artifacts/checkpoint_best.pt",
    "EfficientNet-B0": "artifacts/kaggle/20260920T092347Z/artifacts/checkpoint_best.pt",
    "ResNet-50": "artifacts/kaggle/20260920T110729Z/artifacts/checkpoint_best.pt",
    "EfficientNet-B3+LS0.05": "artifacts/kaggle/20260922T160023Z/artifacts/checkpoint_best.pt",
}


def softmax_pos(logits, t=1.0):
    z = logits / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


def main():
    set_seed(config.SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)

    test_df = load_class_folder_df(REPO / "data" / "raw" / "OCT2017" / "test")
    class_to_idx = build_class_to_idx()
    cache = PreprocessCache(cache_dir=CACHE, image_size=config.IMAGE_SIZE)
    loader = torch.utils.data.DataLoader(
        OCTDataset(test_df, cache, class_to_idx, train=False), batch_size=32, shuffle=False, num_workers=0
    )

    meta = {}
    for name, rel in CHECKPOINTS.items():
        ckpt = torch.load(REPO / rel, map_location="cpu", weights_only=False)
        model_name = ckpt.get("model_name", "tf_efficientnet_b3")
        model = timm.create_model(model_name, pretrained=False, num_classes=2, drop_rate=config.DROP_RATE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        logits, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                logits.append(model(x).numpy())
                labels.extend(y.numpy().tolist())
        logits = np.concatenate(logits)
        labels = np.array(labels)
        t = ckpt.get("temperature")

        df = pd.DataFrame({
            "path": test_df["path"].map(lambda p: Path(p).relative_to(REPO).as_posix() if Path(p).is_absolute() else Path(p).as_posix()),
            "filename": test_df["path"].map(lambda p: Path(p).name),
            "patient": test_df["patient"],
            "label": labels,
            "logit_normal": logits[:, 0],
            "logit_dme": logits[:, 1],
            "p_dme_raw": softmax_pos(logits),
            "p_dme_calibrated": softmax_pos(logits, t) if t else np.nan,
        })
        assert (df["label"].values == (test_df["label"] == "DME").astype(int).values).all()
        slug = name.replace("+", "_").replace(".", "")
        df.to_csv(OUT / f"test_predictions_{slug}.csv", index=False)

        meta[name] = {
            "checkpoint": rel,
            "model_name": model_name,
            "phase": ckpt.get("phase"),
            "global_epoch": ckpt.get("global_epoch"),
            "val_auc_at_selection": ckpt.get("val_auc"),
            "best_val_auc": ckpt.get("best_val_auc"),
            "stored_threshold": ckpt.get("threshold"),
            "stored_temperature": t,
            "checkpoint_keys": sorted(k for k in ckpt.keys() if k not in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict")),
            "prediction_file": f"research_outputs/data/test_predictions_{slug}.csv",
        }
        print(name, meta[name]["global_epoch"], "n=", len(df), "acc@0.5=", ((df.p_dme_raw >= 0.5) == df.label).mean())

    meta["_environment"] = {"torch": torch.__version__, "timm": timm.__version__, "numpy": np.__version__, "python": sys.version}
    with open(OUT / "checkpoint_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


if __name__ == "__main__":
    main()
