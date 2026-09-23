"""
Phase 2: EfficientNet-B3 classifier training for DME vs Normal.

Two-phase fine-tuning schedule (Phase A: head-only warmup with the
backbone frozen; Phase B: full fine-tune at a low LR), patient-grouped
data from build_splits(), checkpointing with resumability across a
process crash at ANY point (mid-epoch data loss is acceptable; losing
track of which phase/epoch you're in is not), and a --smoke mode that
exercises the real code path on a small patient-grouped subset.

Per PROJECT_BRIEF.md Section 4, this is written to run on Kaggle (or any
machine with torch+timm+CUDA/CPU) -- it is NOT expected to run on the
team's local machines, which have no GPU and no torch by design.

Usage:
    python train.py --train-dir data/raw/OCT2017/train --test-dir data/raw/OCT2017/test
    python train.py --train-dir ... --smoke
    python train.py --train-dir ... --resume artifacts/checkpoint_last.pt
"""

import argparse
import csv
import time
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

import config
from utils import PreprocessCache, build_splits, set_seed

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_NAME = "tf_efficientnet_b3"
PHASE_A_UNFROZEN_MODULES = ("conv_head", "bn2", "classifier")

# Phase 3g Baselines -- EfficientNet-B0 and ResNet-50 "under identical conditions"
# (PROJECT_BRIEF.md Section 5). "Identical conditions" means the same data
# pipeline, splits, hyperparameters (epochs/LR/batch size/class weights/
# augmentation) and the same two-phase warmup-then-finetune STRATEGY -- it
# cannot mean literally the same Phase A module names, since architectures
# differ: tf_efficientnet_b0 shares EfficientNet's conv_head/bn2/classifier
# head (same tuple as B3, confirmed via timm.create_model introspection),
# but resnet50 has no conv_head/bn2 at all -- its head is just a global
# pool into a single `fc` Linear layer (also confirmed via introspection),
# so "head-only Phase A warmup" for it means unfreezing fc alone, the
# direct architectural analogue of "train only the newly-initialised
# classification head first."
MODEL_CONFIGS = {
    "tf_efficientnet_b3": {"phase_a_unfrozen_modules": ("conv_head", "bn2", "classifier")},
    "tf_efficientnet_b0": {"phase_a_unfrozen_modules": ("conv_head", "bn2", "classifier")},
    "resnet50": {"phase_a_unfrozen_modules": ("fc",)},
}


# ---------------------------------------------------------------------------
# C5 -- class index / class weight alignment
# ---------------------------------------------------------------------------


def build_class_to_idx() -> dict:
    """{class name: integer label}, in CLASS_NAMES order -- CLASS_WEIGHTS[i] must be that class's weight."""
    return {name: i for i, name in enumerate(config.CLASS_NAMES)}


def assert_class_weight_alignment(class_to_idx: dict) -> None:
    """
    C5: print class_to_idx and CLASS_WEIGHTS side by side and assert they
    align, before the first batch. Reversing this trains the model
    against its own objective while the loss curve looks completely
    normal -- the assertion prints on success too, not just fails on
    mismatch, so "it didn't error" and "it ran and passed" are
    distinguishable in the log.
    """
    expected = build_class_to_idx()
    print(f"[C5] class_to_idx        = {class_to_idx}")
    print(f"[C5] CLASS_WEIGHTS (by class) = {dict(zip(config.CLASS_NAMES, config.CLASS_WEIGHTS))}")
    if class_to_idx != expected:
        raise AssertionError(
            f"class_to_idx {class_to_idx} does not match the CLASS_NAMES-derived ordering "
            f"{expected} that CLASS_WEIGHTS is indexed by -- training would silently "
            "optimise against the wrong class weights (C5)."
        )
    print("[C5] class_to_idx vs CLASS_WEIGHTS alignment: PASSED")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _train_augmentation() -> A.Compose:
    # No vertical flip, ever (C4) -- retinal layer order is anatomically fixed.
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.7),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def _eval_transform() -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class OCTDataset(Dataset):
    """
    Wraps a build_splits() dataframe (columns: path, label, patient).

    The deterministic preprocessing (denoise/flatten/CLAHE/letterbox) is
    cached via PreprocessCache; augmentation and normalisation happen
    per __getitem__ call, NOT cached, so they vary every epoch -- baking
    them into the cache was explicitly ruled out in Phase 1.
    """

    def __init__(self, df: pd.DataFrame, cache: PreprocessCache, class_to_idx: dict, train: bool):
        self.df = df.reset_index(drop=True)
        self.cache = cache
        self.class_to_idx = class_to_idx
        self.transform = _train_augmentation() if train else _eval_transform()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rgb, _content_bbox, _fallback = self.cache.get_or_compute(row["path"])
        augmented = self.transform(image=rgb)
        label = self.class_to_idx[row["label"]]
        return augmented["image"], label


def make_smoke_subset(train_df: pd.DataFrame, val_df: pd.DataFrame, n_total: int = 500, seed: int = config.SEED):
    """
    Patient-grouped subset of ~n_total images (split ~(1-VAL_SPLIT)/VAL_SPLIT
    between train/val, matching the real run) for --smoke mode. Picks
    patients round-robin across classes so a small subset still contains
    both classes, rather than relying on chance.
    """
    rng = np.random.RandomState(seed)
    n_train_target = int(round(n_total * (1 - config.VAL_SPLIT)))
    n_val_target = n_total - n_train_target

    def subset(df: pd.DataFrame, n_target: int) -> pd.DataFrame:
        by_class = {cls: df[df["label"] == cls]["patient"].unique().tolist() for cls in config.CLASS_NAMES}
        for plist in by_class.values():
            rng.shuffle(plist)
        idx = {cls: 0 for cls in config.CLASS_NAMES}
        picked, count = [], 0
        while count < n_target and any(idx[c] < len(by_class[c]) for c in config.CLASS_NAMES):
            for c in config.CLASS_NAMES:
                if count >= n_target:
                    break
                if idx[c] < len(by_class[c]):
                    patient = by_class[c][idx[c]]
                    idx[c] += 1
                    picked.append(patient)
                    count += int((df["patient"] == patient).sum())
        return df[df["patient"].isin(picked)].reset_index(drop=True)

    return subset(train_df, n_train_target), subset(val_df, n_val_target)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_model(model_name: str = MODEL_NAME) -> nn.Module:
    return timm.create_model(
        model_name, pretrained=True, num_classes=len(config.CLASS_NAMES), drop_rate=config.DROP_RATE
    )


def set_phase_a_trainable(model: nn.Module, unfrozen_modules=PHASE_A_UNFROZEN_MODULES) -> None:
    """Phase A: only `unfrozen_modules` (default: conv_head + bn2 + classifier, B3's ~0.6M-param head)."""
    for p in model.parameters():
        p.requires_grad = False
    for name in unfrozen_modules:
        for p in getattr(model, name).parameters():
            p.requires_grad = True


def set_phase_b_trainable(model: nn.Module) -> None:
    """Phase B: everything unfrozen."""
    for p in model.parameters():
        p.requires_grad = True


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(path: Path, payload: dict) -> None:
    """
    Atomic write: save to a temp file, then rename over the target. A
    crash mid-write can never leave a corrupt or half-written checkpoint
    in place -- the old file (if any) stays intact until the new one is
    fully on disk, "deleted" only by the atomic rename replacing it.
    """
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(path: Path, map_location=None) -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`. Trains if `optimizer` is given, else evaluates under no_grad."""
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, correct, n = 0.0, 0, 0
    all_probs, all_labels = [], []

    torch.set_grad_enabled(train_mode)
    try:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train_mode:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if train_mode:
                loss.backward()
                optimizer.step()

            batch_n = images.size(0)
            total_loss += loss.item() * batch_n
            n += batch_n
            correct += (logits.argmax(dim=1) == labels).sum().item()
            # index 1 == "DME" (positive class) because class_to_idx follows CLASS_NAMES order (C5).
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
    finally:
        torch.set_grad_enabled(True)

    avg_loss = total_loss / max(1, n)
    accuracy = correct / max(1, n)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    return avg_loss, accuracy, auc


def _log_epoch_row(log_path: Path, row: dict) -> None:
    write_header = not Path(log_path).exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_phase(
    phase_name: str,
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    criterion,
    device,
    epochs_total: int,
    start_epoch_in_phase: int,
    global_epoch: int,
    best_val_auc: float,
    best_epoch: int,
    epochs_no_improve: int,
    last_ckpt_path: Path,
    best_ckpt_path: Path,
    log_path: Path,
    early_stop_patience: int,
    model_name: str = MODEL_NAME,
):
    """
    Run `phase_name` from `start_epoch_in_phase` up to `epochs_total`
    epochs (or until early stopping on val loss fires). Returns updated
    (global_epoch, best_val_auc, best_epoch, epochs_no_improve, phase_complete).

    Checkpoint-after-the-decision (not before): each epoch, the
    continue-vs-stop-this-phase decision is made FIRST (epoch budget
    exhausted, or early-stop patience exhausted), and `phase_complete` is
    set accordingly; only THEN is checkpoint_last written, carrying that
    resolved decision. So a crash-and-resume never re-enters a phase that
    had already been decided complete -- resume logic trusts
    `phase_complete`, not epoch-count guesswork.
    """
    phase_complete = False
    epoch_in_phase = start_epoch_in_phase

    for epoch_in_phase in range(start_epoch_in_phase, epochs_total):
        t0 = time.time()
        train_loss, train_acc, _train_auc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss, val_acc, val_auc = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step(val_loss)
        epoch_time = time.time() - t0

        global_epoch += 1
        improved = (not np.isnan(val_auc)) and val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            best_epoch = global_epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # --- the decision, made before any checkpoint reflecting it is written ---
        epoch_budget_exhausted = (epoch_in_phase + 1) >= epochs_total
        early_stopped = epochs_no_improve >= early_stop_patience
        phase_complete = epoch_budget_exhausted or early_stopped

        _log_epoch_row(
            log_path,
            {
                "phase": phase_name,
                "global_epoch": global_epoch,
                "epoch_in_phase": epoch_in_phase + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_auc": val_auc,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "trainable_params": count_trainable_params(model),
                "epoch_time_sec": epoch_time,
            },
        )

        if improved:
            save_checkpoint(
                best_ckpt_path,
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "phase": phase_name,
                    "epoch_in_phase": epoch_in_phase + 1,
                    "global_epoch": global_epoch,
                    "val_auc": val_auc,
                    "threshold": None,  # set in Phase 3
                    "temperature": None,  # set in Phase 3
                },
            )

        save_checkpoint(
            last_ckpt_path,
            {
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "phase": phase_name,
                "epoch_in_phase": epoch_in_phase + 1,
                "phase_complete": phase_complete,
                "global_epoch": global_epoch,
                "best_val_auc": best_val_auc,
                "best_epoch": best_epoch,
                "epochs_no_improve": epochs_no_improve,
                "threshold": None,
                "temperature": None,
            },
        )

        print(
            f"[{phase_name} epoch {epoch_in_phase + 1}/{epochs_total} | global {global_epoch}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f} "
            f"{'*best*' if improved else ''} ({epoch_time:.1f}s)"
        )

        if phase_complete:
            if early_stopped and not epoch_budget_exhausted:
                # The patience counter above tracks val_auc ("improved = val_auc > best_val_auc"),
                # not val_loss -- this message said "val_loss" for a while and was confirmed wrong
                # against this exact function on 2026-09-17 while reading a real training log.
                print(f"[{phase_name}] early stopping: no val_auc improvement in {early_stop_patience} epochs")
            break

    return global_epoch, best_val_auc, best_epoch, epochs_no_improve, phase_complete


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Phase 2 / 3g: DME/Normal classifier training (default tf_efficientnet_b3; --model-name for Baselines).")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=None, help="PreprocessCache dir (default: <artifacts-dir>/preprocess_cache)")
    parser.add_argument("--resume", type=Path, default=None, help="Path to a checkpoint_last.pt to resume from")
    parser.add_argument("--smoke", action="store_true", help="1 epoch/phase on a ~500-image patient-grouped subset")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--model-name", type=str, default=MODEL_NAME, choices=list(MODEL_CONFIGS.keys()),
        help="Architecture to train (Phase 3g Baselines: tf_efficientnet_b0, resnet50). Phase A's "
             "head-only-unfrozen module set is looked up per architecture from MODEL_CONFIGS.",
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0,
        help="CrossEntropyLoss label_smoothing (Phase 3b ablation A5; brief specifies 0.05). Default 0.0 "
             "(off) reproduces the exact Phase 2 loss unchanged.",
    )
    args = parser.parse_args()

    set_seed(config.SEED)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.artifacts_dir / "preprocess_cache")
    last_ckpt_path = args.artifacts_dir / "checkpoint_last.pt"
    best_ckpt_path = args.artifacts_dir / "checkpoint_best.pt"
    log_path = args.artifacts_dir / "training_log.csv"

    # Never touches the test set (only used here for the C8 exclusion check inside build_splits).
    result = build_splits(args.train_dir, test_dir=args.test_dir)
    train_df, val_df = result.train_df, result.val_df

    epochs_a, epochs_b = config.EPOCHS_PHASE_A, config.EPOCHS_PHASE_B
    if args.smoke:
        train_df, val_df = make_smoke_subset(train_df, val_df, n_total=500, seed=config.SEED)
        epochs_a, epochs_b = 1, 1
        print(f"[smoke] subset: train={len(train_df)} val={len(val_df)} "
              f"train_classes={train_df['label'].value_counts().to_dict()} "
              f"val_classes={val_df['label'].value_counts().to_dict()}")

    class_to_idx = build_class_to_idx()
    cache = PreprocessCache(cache_dir=cache_dir, image_size=config.IMAGE_SIZE)
    train_loader = DataLoader(
        OCTDataset(train_df, cache, class_to_idx, train=True),
        batch_size=config.BATCH_SIZE_CLS, shuffle=True, num_workers=args.num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        OCTDataset(val_df, cache, class_to_idx, train=False),
        batch_size=config.BATCH_SIZE_CLS, shuffle=False, num_workers=args.num_workers, drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[model] {args.model_name}" + (f" (label_smoothing={args.label_smoothing})" if args.label_smoothing else ""))
    model = build_model(model_name=args.model_name).to(device)
    phase_a_unfrozen_modules = MODEL_CONFIGS[args.model_name]["phase_a_unfrozen_modules"]

    # C5 -- must happen before the first batch.
    assert_class_weight_alignment(class_to_idx)
    class_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    # --- resume state (defaults = fresh run, starting Phase A) ---
    start_phase = "A"
    start_epoch_in_phase = 0
    global_epoch = 0
    best_val_auc = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    resumed_ckpt = None

    if args.resume is not None:
        resumed_ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(resumed_ckpt["model_state_dict"])
        global_epoch = resumed_ckpt["global_epoch"]
        best_val_auc = resumed_ckpt["best_val_auc"]
        best_epoch = resumed_ckpt["best_epoch"]
        epochs_no_improve = resumed_ckpt["epochs_no_improve"]
        if resumed_ckpt["phase_complete"]:
            # That phase was already decided complete before the crash -- move on,
            # never re-enter it (see run_phase's docstring / PROJECT_BRIEF.md Phase 2).
            start_phase = "B" if resumed_ckpt["phase"] == "A" else "DONE"
            start_epoch_in_phase = 0
        else:
            start_phase = resumed_ckpt["phase"]
            start_epoch_in_phase = resumed_ckpt["epoch_in_phase"]
        print(f"[resume] {args.resume}: -> phase={start_phase} start_epoch_in_phase={start_epoch_in_phase} "
              f"global_epoch={global_epoch} best_val_auc={best_val_auc:.4f}")

    if start_phase == "A":
        set_phase_a_trainable(model, unfrozen_modules=phase_a_unfrozen_modules)
        trainable_a = count_trainable_params(model)
        print(f"[phase transition] Phase A trainable params: {trainable_a}")
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LR_FROZEN)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=config.PLATEAU_FACTOR, patience=config.PLATEAU_PATIENCE
        )
        if resumed_ckpt is not None and resumed_ckpt["phase"] == "A" and not resumed_ckpt["phase_complete"]:
            optimizer.load_state_dict(resumed_ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(resumed_ckpt["scheduler_state_dict"])

        global_epoch, best_val_auc, best_epoch, epochs_no_improve, _complete = run_phase(
            "A", model, optimizer, scheduler, train_loader, val_loader, criterion, device,
            epochs_total=epochs_a, start_epoch_in_phase=start_epoch_in_phase,
            global_epoch=global_epoch, best_val_auc=best_val_auc, best_epoch=best_epoch,
            epochs_no_improve=epochs_no_improve, last_ckpt_path=last_ckpt_path,
            best_ckpt_path=best_ckpt_path, log_path=log_path,
            early_stop_patience=config.EARLY_STOP_PATIENCE, model_name=args.model_name,
        )
        start_phase = "B"
        start_epoch_in_phase = 0
        # Phase B is a different optimisation problem (all params, 100x lower LR) --
        # it gets its own early-stopping patience budget, not whatever was left of Phase A's.
        epochs_no_improve = 0

    if start_phase == "B":
        set_phase_b_trainable(model)
        trainable_b = count_trainable_params(model)
        print(f"[phase transition] Phase B trainable params: {trainable_b}")
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LR_FINETUNE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=config.PLATEAU_FACTOR, patience=config.PLATEAU_PATIENCE
        )
        if resumed_ckpt is not None and resumed_ckpt["phase"] == "B" and not resumed_ckpt["phase_complete"]:
            optimizer.load_state_dict(resumed_ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(resumed_ckpt["scheduler_state_dict"])

        global_epoch, best_val_auc, best_epoch, epochs_no_improve, _complete = run_phase(
            "B", model, optimizer, scheduler, train_loader, val_loader, criterion, device,
            epochs_total=epochs_b, start_epoch_in_phase=start_epoch_in_phase,
            global_epoch=global_epoch, best_val_auc=best_val_auc, best_epoch=best_epoch,
            epochs_no_improve=epochs_no_improve, last_ckpt_path=last_ckpt_path,
            best_ckpt_path=best_ckpt_path, log_path=log_path,
            early_stop_patience=config.EARLY_STOP_PATIENCE, model_name=args.model_name,
        )

    print(f"\nTraining complete. Best epoch: {best_epoch}, best val_auc: {best_val_auc:.4f}")
    print(f"Total epochs run: {global_epoch}")
    print(f"Checkpoints: {last_ckpt_path} (last), {best_ckpt_path} (best)")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
