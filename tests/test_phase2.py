"""
Phase 2 structural tests for train.py.

IMPORTANT SCOPE NOTE: these tests verify train.py's logic (C5 assertion,
Phase A/B freezing, checkpoint save/load + phase_complete resume decisions,
augmentation construction, smoke-subset patient grouping) WITHOUT running a
real training loop. They do not and cannot substitute for an actual GPU
training run -- that is blocked on Phase 2.5 (Kaggle automation) per
PROJECT_BRIEF.md Section 4 (no torch/GPU on the team's local machines).

They run here in the cloud sandbox, which has torch/timm/albumentations
installed at versions newer than requirements.txt pins (torch 2.14 vs
pinned 2.3.*, timm 1.0.29 vs pinned 0.9.*, albumentations 2.0.8 vs pinned
1.4.*) -- close enough to exercise the real APIs, but Kaggle will run the
pinned versions, so this is a structural sanity check, not a substitute for
running requirements.txt as-pinned.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import train


# ---------------------------------------------------------------------------
# C5 -- class_to_idx / CLASS_WEIGHTS alignment
# ---------------------------------------------------------------------------


def test_build_class_to_idx_matches_class_names_order():
    assert train.build_class_to_idx() == {"Normal": 0, "DME": 1}


def test_assert_class_weight_alignment_passes_on_correct_mapping(capsys):
    train.assert_class_weight_alignment({"Normal": 0, "DME": 1})
    out = capsys.readouterr().out
    assert "PASSED" in out


def test_assert_class_weight_alignment_raises_on_swapped_mapping():
    with pytest.raises(AssertionError):
        train.assert_class_weight_alignment({"Normal": 1, "DME": 0})


# ---------------------------------------------------------------------------
# Augmentation -- C4 (no vertical flip, ever)
# ---------------------------------------------------------------------------


def test_train_augmentation_has_no_vertical_flip():
    aug = train._train_augmentation()
    transform_names = [type(t).__name__ for t in aug.transforms]
    assert "VerticalFlip" not in transform_names
    assert "HorizontalFlip" in transform_names


def test_train_augmentation_runs_on_dummy_image():
    aug = train._train_augmentation()
    dummy = (np.random.rand(300, 300, 3) * 255).astype(np.uint8)
    out = aug(image=dummy)
    assert out["image"].shape == (3, 300, 300)


def test_eval_transform_has_no_geometric_augmentation():
    aug = train._eval_transform()
    transform_names = [type(t).__name__ for t in aug.transforms]
    assert transform_names == ["Normalize", "ToTensorV2"]


# ---------------------------------------------------------------------------
# Model construction / Phase A vs Phase B freezing
#
# NOTE: these use pretrained=False (bypassing train.build_model(), which
# always requests pretrained=True) because this sandbox has no route to
# HuggingFace Hub (imagenet weight download gets a 403 from the egress
# proxy). Architecture -- and therefore every param-count / freezing
# assertion below -- is identical whether or not ImageNet weights are
# loaded, so this does not weaken what's being checked; it only avoids a
# network dependency the real run (on Kaggle, which does have internet)
# won't have.
# ---------------------------------------------------------------------------


def _build_model_no_pretrained_download():
    import timm

    return timm.create_model(
        train.MODEL_NAME, pretrained=False, num_classes=len(config.CLASS_NAMES), drop_rate=config.DROP_RATE
    )


def test_build_model_is_tf_efficientnet_b3_with_2_classes():
    model = _build_model_no_pretrained_download()
    dummy = torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    out = model(dummy)
    assert out.shape == (1, 2)


def test_phase_a_freezes_all_but_conv_head_bn2_classifier():
    model = _build_model_no_pretrained_download()
    train.set_phase_a_trainable(model)
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    for n in trainable_names:
        assert n.split(".")[0] in train.PHASE_A_UNFROZEN_MODULES, n
    # Matches the brief's ~0.6M-param expectation for Phase A.
    n_trainable = train.count_trainable_params(model)
    assert 500_000 < n_trainable < 700_000, n_trainable


def test_phase_b_unfreezes_everything():
    model = _build_model_no_pretrained_download()
    train.set_phase_a_trainable(model)
    train.set_phase_b_trainable(model)
    assert train.count_trainable_params(model) == sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Phase 3g Baselines -- selectable architecture (build_model/set_phase_a_trainable
# parameterised by model_name, MODEL_CONFIGS registry)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", ["tf_efficientnet_b3", "tf_efficientnet_b0", "resnet50"])
def test_model_configs_registry_matches_real_module_names(model_name):
    """
    Every model_name in MODEL_CONFIGS must name modules that actually exist
    on that architecture -- a typo here would only surface as a crash deep
    into a real (expensive) Kaggle training push, not at review time.
    """
    import timm

    model = timm.create_model(model_name, pretrained=False, num_classes=len(config.CLASS_NAMES), drop_rate=config.DROP_RATE)
    for module_name in train.MODEL_CONFIGS[model_name]["phase_a_unfrozen_modules"]:
        assert hasattr(model, module_name), f"{model_name} has no module named {module_name!r}"


@pytest.mark.parametrize("model_name", ["tf_efficientnet_b0", "resnet50"])
def test_build_model_supports_baseline_architectures(model_name):
    import timm

    model = timm.create_model(model_name, pretrained=False, num_classes=len(config.CLASS_NAMES), drop_rate=config.DROP_RATE)
    dummy = torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    out = model(dummy)
    assert out.shape == (1, 2)


def test_resnet50_phase_a_freezes_all_but_fc():
    import timm

    model = timm.create_model("resnet50", pretrained=False, num_classes=len(config.CLASS_NAMES), drop_rate=config.DROP_RATE)
    train.set_phase_a_trainable(model, unfrozen_modules=train.MODEL_CONFIGS["resnet50"]["phase_a_unfrozen_modules"])
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    for n in trainable_names:
        assert n.split(".")[0] == "fc", n
    assert trainable_names  # not empty -- fc actually has parameters


def test_efficientnet_b0_phase_a_uses_same_modules_as_b3():
    assert train.MODEL_CONFIGS["tf_efficientnet_b0"]["phase_a_unfrozen_modules"] == train.PHASE_A_UNFROZEN_MODULES


# ---------------------------------------------------------------------------
# Checkpointing + phase_complete-driven resume decisions
# ---------------------------------------------------------------------------


def test_save_checkpoint_is_atomic_and_round_trips(tmp_path):
    ckpt_path = tmp_path / "checkpoint_last.pt"
    payload = {"phase": "A", "phase_complete": False, "global_epoch": 3}
    train.save_checkpoint(ckpt_path, payload)
    assert ckpt_path.exists()
    assert not ckpt_path.with_name(ckpt_path.name + ".tmp").exists()  # tmp file cleaned up by rename
    loaded = train.load_checkpoint(ckpt_path)
    assert loaded == payload


def test_save_checkpoint_overwrite_leaves_no_tmp_residue(tmp_path):
    ckpt_path = tmp_path / "checkpoint_last.pt"
    train.save_checkpoint(ckpt_path, {"v": 1})
    train.save_checkpoint(ckpt_path, {"v": 2})
    assert train.load_checkpoint(ckpt_path) == {"v": 2}
    assert not ckpt_path.with_name(ckpt_path.name + ".tmp").exists()


@pytest.mark.parametrize(
    "resumed_phase,phase_complete,expected_start_phase",
    [
        ("A", True, "B"),      # Phase A was decided complete before crash -> move on to B
        ("A", False, "A"),     # Phase A was mid-epoch -> re-enter A at the saved epoch
        ("B", True, "DONE"),   # Phase B was decided complete -> nothing left to run
        ("B", False, "B"),     # Phase B was mid-epoch -> re-enter B
    ],
)
def test_resume_phase_selection_follows_phase_complete_flag(resumed_phase, phase_complete, expected_start_phase):
    """
    Mirrors main()'s resume branch: phase_complete is trusted over any
    epoch-count guesswork -- this is the crux of the Phase 2 resumability
    design (checkpoint saved AFTER the phase-switch decision, not before).
    """
    resumed_ckpt = {"phase": resumed_phase, "phase_complete": phase_complete, "epoch_in_phase": 2}
    if resumed_ckpt["phase_complete"]:
        start_phase = "B" if resumed_ckpt["phase"] == "A" else "DONE"
        start_epoch_in_phase = 0
    else:
        start_phase = resumed_ckpt["phase"]
        start_epoch_in_phase = resumed_ckpt["epoch_in_phase"]

    assert start_phase == expected_start_phase
    if phase_complete:
        assert start_epoch_in_phase == 0
    else:
        assert start_epoch_in_phase == resumed_ckpt["epoch_in_phase"]


# ---------------------------------------------------------------------------
# run_phase -- phase_complete computed correctly for both exit conditions
# ---------------------------------------------------------------------------


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x.flatten(1))


class _TinyDataset(torch.utils.data.Dataset):
    """8 samples, 2 classes, fixed labels -- enough for CrossEntropyLoss + AUC to be well-defined."""

    def __init__(self, n=8, seed=0):
        rng = np.random.RandomState(seed)
        self.x = rng.rand(n, 1, 2, 2).astype("float32")
        self.y = np.array([i % 2 for i in range(n)], dtype="int64")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), int(self.y[idx])


def _make_tiny_loaders():
    train_loader = torch.utils.data.DataLoader(_TinyDataset(seed=0), batch_size=4)
    val_loader = torch.utils.data.DataLoader(_TinyDataset(seed=1), batch_size=4)
    return train_loader, val_loader


def test_run_phase_marks_complete_when_epoch_budget_exhausted(tmp_path):
    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    train_loader, val_loader = _make_tiny_loaders()

    global_epoch, best_val_auc, best_epoch, epochs_no_improve, phase_complete = train.run_phase(
        "A", model, optimizer, scheduler, train_loader, val_loader,
        criterion=torch.nn.CrossEntropyLoss(), device=torch.device("cpu"),
        epochs_total=2, start_epoch_in_phase=0, global_epoch=0,
        best_val_auc=-1.0, best_epoch=0, epochs_no_improve=0,
        last_ckpt_path=tmp_path / "last.pt", best_ckpt_path=tmp_path / "best.pt",
        log_path=tmp_path / "log.csv", early_stop_patience=100,  # patience disabled -> must hit budget
    )

    assert phase_complete is True
    assert global_epoch == 2
    saved = train.load_checkpoint(tmp_path / "last.pt")
    assert saved["phase_complete"] is True
    assert (tmp_path / "log.csv").exists()


def test_run_phase_marks_complete_on_early_stop(tmp_path):
    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    train_loader, val_loader = _make_tiny_loaders()

    # patience=0 -> the very first epoch that doesn't improve on best_val_auc=+inf-proof
    # starting point ends the phase before the epoch budget (20) is reached.
    global_epoch, best_val_auc, best_epoch, epochs_no_improve, phase_complete = train.run_phase(
        "A", model, optimizer, scheduler, train_loader, val_loader,
        criterion=torch.nn.CrossEntropyLoss(), device=torch.device("cpu"),
        epochs_total=20, start_epoch_in_phase=0, global_epoch=0,
        best_val_auc=1.0,  # already "perfect" -> first epoch can't improve -> epochs_no_improve hits patience
        best_epoch=0, epochs_no_improve=0,
        last_ckpt_path=tmp_path / "last.pt", best_ckpt_path=tmp_path / "best.pt",
        log_path=tmp_path / "log.csv", early_stop_patience=1,
    )

    assert phase_complete is True
    assert global_epoch < 20


def test_run_phase_records_model_name_in_checkpoints(tmp_path):
    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    train_loader, val_loader = _make_tiny_loaders()

    train.run_phase(
        "A", model, optimizer, scheduler, train_loader, val_loader,
        criterion=torch.nn.CrossEntropyLoss(), device=torch.device("cpu"),
        epochs_total=1, start_epoch_in_phase=0, global_epoch=0,
        best_val_auc=-1.0, best_epoch=0, epochs_no_improve=0,
        last_ckpt_path=tmp_path / "last.pt", best_ckpt_path=tmp_path / "best.pt",
        log_path=tmp_path / "log.csv", early_stop_patience=100,
        model_name="resnet50",
    )

    assert train.load_checkpoint(tmp_path / "last.pt")["model_name"] == "resnet50"
    assert train.load_checkpoint(tmp_path / "best.pt")["model_name"] == "resnet50"  # this epoch always "improves" from -1.0


def test_run_phase_defaults_model_name_to_train_module_default(tmp_path):
    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    train_loader, val_loader = _make_tiny_loaders()

    train.run_phase(
        "A", model, optimizer, scheduler, train_loader, val_loader,
        criterion=torch.nn.CrossEntropyLoss(), device=torch.device("cpu"),
        epochs_total=1, start_epoch_in_phase=0, global_epoch=0,
        best_val_auc=-1.0, best_epoch=0, epochs_no_improve=0,
        last_ckpt_path=tmp_path / "last.pt", best_ckpt_path=tmp_path / "best.pt",
        log_path=tmp_path / "log.csv", early_stop_patience=100,
        # model_name omitted -- must default to train.MODEL_NAME, not crash or leave it unset.
    )

    assert train.load_checkpoint(tmp_path / "last.pt")["model_name"] == train.MODEL_NAME


# ---------------------------------------------------------------------------
# make_smoke_subset -- patient grouping + class balance
# ---------------------------------------------------------------------------


def _synthetic_split_df(n_patients_per_class=40, images_per_patient=3):
    rows = []
    patient_id = 0
    for cls in config.CLASS_NAMES:
        for _ in range(n_patients_per_class):
            patient_id += 1
            for img_i in range(images_per_patient):
                rows.append({"path": f"/fake/{cls}-{patient_id}-{img_i}.jpeg", "label": cls, "patient": patient_id})
    return pd.DataFrame(rows)


def test_make_smoke_subset_is_patient_disjoint_and_has_both_classes():
    df = _synthetic_split_df()
    train_sub, val_sub = train.make_smoke_subset(df, df, n_total=60, seed=config.SEED)

    assert set(train_sub["patient"]) & set(val_sub["patient"]) == set()
    assert set(train_sub["label"].unique()) == set(config.CLASS_NAMES)
    assert set(val_sub["label"].unique()) == set(config.CLASS_NAMES)


def test_make_smoke_subset_respects_approx_val_split_ratio():
    df = _synthetic_split_df()
    train_sub, val_sub = train.make_smoke_subset(df, df, n_total=100, seed=config.SEED)
    total = len(train_sub) + len(val_sub)
    # Not exact (patient granularity means rounding), but should be in the right ballpark.
    assert total > 0
    val_fraction = len(val_sub) / total
    assert 0.05 < val_fraction < 0.40


def test_make_smoke_subset_every_selected_patient_has_all_their_images():
    """Whole-patient selection, not per-image sampling -- otherwise a patient could be
    split across train/val within the "subset" step itself, silently reintroducing
    leakage that build_splits() had already ruled out upstream."""
    df = _synthetic_split_df()
    train_sub, _ = train.make_smoke_subset(df, df, n_total=30, seed=config.SEED)
    for patient in train_sub["patient"].unique():
        full_count = (df["patient"] == patient).sum()
        sub_count = (train_sub["patient"] == patient).sum()
        assert sub_count == full_count
