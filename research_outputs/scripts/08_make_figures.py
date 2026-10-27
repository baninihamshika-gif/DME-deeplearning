"""
Phase E -- publication figures. Every plotted value is read from a CSV in
research_outputs/data/ (or research_audit/literature_extraction.csv); nothing
is typed in. Each figure is saved as 300-dpi PNG and vector PDF, and the exact
plotted numbers are written next to it as <figure>_plotted_values.csv so
the final audit can diff figure <-> table.

Palette: validated categorical slots 1-3 (blue, orange, aqua) from the dataviz
reference palette; aqua is < 3:1 contrast on white, so all bars carry direct
value labels and line series also differ by line style (secondary encoding).

Usage: python research_outputs/scripts/08_make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "research_outputs" / "data"
FIG = REPO / "research_outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

C = {"EfficientNet-B3": "#2a78d6", "EfficientNet-B0": "#eb6834", "ResNet-50": "#1baf7a"}
LS = {"EfficientNet-B3": "-", "EfficientNet-B0": "--", "ResNet-50": ":"}
HATCH = {"EfficientNet-B3": "", "EfficientNet-B0": "//", "ResNet-50": ".."}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODELS = list(C)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name, plotted: pd.DataFrame):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    plotted.to_csv(FIG / f"{name}_plotted_values.csv", index=False)


# --------------------------------------------------------------------------- Figure 1
def figure1():
    perf = pd.read_csv(DATA / "table2_within_project_performance.csv")
    perf = perf[(perf.threshold == 0.5) & perf.model.isin(MODELS)].set_index("model").loc[MODELS]
    exc_path = DATA / "supplementary_excluded_pool_performance.csv"
    exc = pd.read_csv(exc_path).set_index("model").loc[MODELS] if exc_path.exists() else None

    metrics = [("accuracy", "Accuracy"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity"), ("f1", "F1-score")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [2.3, 1], "wspace": 0.45})
    w = 0.26; x = np.arange(len(metrics)); rows = []
    for i, m in enumerate(MODELS):
        vals = [perf.loc[m, k] for k, _ in metrics]
        lo = [perf.loc[m, f"{k}_ci_lo"] for k, _ in metrics]; hi = [perf.loc[m, f"{k}_ci_hi"] for k, _ in metrics]
        pos = x + (i - 1) * w
        ax.bar(pos, vals, w - 0.02, color=C[m], hatch=HATCH[m], edgecolor="white", linewidth=0.6, label=m)
        ax.errorbar(pos, vals, yerr=[np.array(vals) - lo, np.array(hi) - vals], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5)
        for p_, v in zip(pos, vals):
            ax.text(p_, 0.45, f"{v:.3f}", rotation=90, ha="center", va="center", color=INK, fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))
        for (k, lab), v, l_, h_ in zip(metrics, vals, lo, hi):
            rows.append({"panel": "a", "model": m, "metric": k, "value": v, "ci_lo": l_, "ci_hi": h_})
    ax.set_xticks(x, [lab for _, lab in metrics]); ax.set_ylim(0, 1.05); ax.set_ylabel("Value (threshold 0.5)")
    ax.set_title("a  Threshold-dependent metrics, official test set", loc="left")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8)

    # panel b: AUROC
    y = np.arange(len(MODELS))[::-1]
    for yi, m in zip(y, MODELS):
        a = perf.loc[m, "auroc"]
        ax2.plot([a], [yi + (0.15 if exc is not None else 0)], marker="o", ms=7, color=C[m], mec="white", mew=1.2, ls="none")
        ax2.text(0.52, yi + (0.15 if exc is not None else 0), f"test {a:.3f}", va="center", fontsize=7.5, color=INK)
        rows.append({"panel": "b", "model": m, "metric": "auroc_test", "value": a, "ci_lo": perf.loc[m, "auroc_ci_lo"], "ci_hi": perf.loc[m, "auroc_ci_hi"]})
        if exc is not None:
            ea, elo, ehi = exc.loc[m, "auroc"], exc.loc[m, "auroc_cluster_bootstrap_ci_lo"], exc.loc[m, "auroc_cluster_bootstrap_ci_hi"]
            ax2.errorbar([ea], [yi - 0.15], xerr=[[ea - elo], [ehi - ea]], fmt="D", ms=6, color=C[m], mfc="white", mec=C[m], mew=1.5, ecolor=INK, elinewidth=0.9, capsize=2.5)
            ax2.text(0.52, yi - 0.15, f"pool {ea:.3f} ({elo:.3f}–{ehi:.3f})", va="center", fontsize=7.5, color=INK)
            rows.append({"panel": "b", "model": m, "metric": "auroc_excluded_pool", "value": ea, "ci_lo": elo, "ci_hi": ehi})
    ax2.set_yticks(y, MODELS); ax2.set_xlim(0.5, 1.02); ax2.set_xlabel("AUROC (0.5 = chance)")
    ax2.set_title("b  Discrimination (AUROC)", loc="left")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", color=INK2, ls="none", ms=6, label="Official test set, n=484\n(DeLong CI degenerate at AUROC=1)")]
    if exc is not None:
        h.append(Line2D([], [], marker="D", color=INK2, mfc="white", ls="none", ms=6, label=f"Supplementary unseen pool, n={int(exc.n.iloc[0]):,}\n(patient-cluster bootstrap 95% CI)"))
    ax2.legend(handles=h, loc="upper center", bbox_to_anchor=(0.45, -0.17), fontsize=7.5)
    fig.suptitle("Performance of Deep Learning Models for Binary DME Classification on the Held-Out Test Set", fontsize=11, y=1.02)
    fig.text(0.01, -0.3, "Official Kermany OCT2017 test split: 242 DME (positive class) / 242 Normal B-scans, identical for all models. "
             "Error bars in (a): Wilson 95% CI (F1: class-stratified bootstrap, B=2000).", fontsize=7.5, color=INK2)
    save(fig, "figure1_within_project_performance", pd.DataFrame(rows))


# --------------------------------------------------------------------------- Figure 2
def figure2():
    roc = pd.read_csv(DATA / "roc_coordinates_test.csv")
    perf = pd.read_csv(DATA / "table2_within_project_performance.csv")
    exc_roc = DATA / "roc_coordinates_excluded_pool.csv"
    has_exc = exc_roc.exists()
    fig, axes = plt.subplots(1, 2 if has_exc else 1, figsize=(10, 4.6) if has_exc else (5, 4.6), squeeze=False)
    rows = []
    ax = axes[0, 0]
    for m in MODELS:
        r = roc[roc.model == m]; a = perf[(perf.model == m) & (perf.threshold == 0.5)].auroc.iloc[0]
        ax.plot(r.fpr, r.tpr, LS[m], color=C[m], lw=2, label=f"{m}  AUROC = {a:.3f}")
        rows += [{"panel": "a", "model": m, "fpr": f, "tpr": t} for f, t in zip(r.fpr, r.tpr)]
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (2, 2)), label="Random classifier")
    ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01), xlabel="False positive rate (1 − specificity)", ylabel="True positive rate (sensitivity)")
    ax.set_aspect("equal"); ax.legend(loc="lower right", fontsize=7.5)
    ax.set_title("a  Official test set (n = 484; 242 DME / 242 Normal)", loc="left")
    ax.text(0.02, 0.62, "All three curves coincide on the\nleft and top axes: every DME image\nscores above every Normal image.\nDeLong CI degenerate (variance 0).", fontsize=7.5, color=INK2)
    if has_exc:
        e = pd.read_csv(exc_roc); ep = pd.read_csv(DATA / "supplementary_excluded_pool_performance.csv").set_index("model")
        ax = axes[0, 1]
        for m in MODELS:
            r = e[e.model == m]
            ax.plot(r.fpr, r.tpr, LS[m], color=C[m], lw=2,
                    label=f"{m}  AUROC = {ep.loc[m, 'auroc']:.3f} ({ep.loc[m, 'auroc_cluster_bootstrap_ci_lo']:.3f}–{ep.loc[m, 'auroc_cluster_bootstrap_ci_hi']:.3f})")
            rows += [{"panel": "b", "model": m, "fpr": f, "tpr": t} for f, t in zip(r.fpr, r.tpr)]
        ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (2, 2)), label="Random classifier")
        ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01), xlabel="False positive rate (1 − specificity)", ylabel="True positive rate (sensitivity)")
        ax.set_aspect("equal"); ax.legend(loc="lower right", fontsize=7)
        n = int(ep.n.iloc[0]); nd = int(ep.n_dme.iloc[0])
        ax.set_title(f"b  Supplementary unseen pool (n = {n:,}; {nd:,} DME / {n - nd:,} Normal)", loc="left")
        ins = ax.inset_axes([0.42, 0.38, 0.45, 0.32]); ins.set_facecolor("white"); ins.set_zorder(5)
        for m in MODELS:
            r = e[e.model == m]; ins.plot(r.fpr, r.tpr, LS[m], color=C[m], lw=1.5)
        ins.set(xlim=(0, 0.1), ylim=(0.85, 1.0)); ins.tick_params(labelsize=6.5); ins.set_title("detail: FPR 0–0.10, TPR 0.85–1.00", fontsize=6.5)
    fig.suptitle("Receiver Operating Characteristic Curve for DME Classification", fontsize=11, y=1.0)
    fig.text(0.01, -0.03, "Curves from continuous raw (uncalibrated) softmax P(DME); temperature scaling is monotonic and leaves ROC/AUROC unchanged. "
             "Positive class: DME. Panel b CIs: patient-cluster bootstrap, B=2000.", fontsize=7.5, color=INK2)
    save(fig, "figure2_roc_curves", pd.DataFrame(rows))


# --------------------------------------------------------------------------- Figure 3
def figure3():
    lit = pd.read_csv(REPO / "research_audit" / "literature_extraction.csv", dtype=str)
    exc_path = DATA / "supplementary_excluded_pool_performance.csv"
    items = []  # (group, label, auroc, lo, hi, marker_filled)
    proj = lit[lit.study.str.startswith("This project")]
    b3 = proj[proj.model.str.startswith("EfficientNet-B3")].iloc[0]
    items.append(("This project — 2D B-scan, internal held-out test (Kermany)", "EfficientNet-B3, official test set (n=484)", float(b3.auroc), np.nan, np.nan, True))
    if exc_path.exists():
        e = pd.read_csv(exc_path).set_index("model").loc["EfficientNet-B3"]
        items.append(("This project — 2D B-scan, internal held-out test (Kermany)", f"EfficientNet-B3, supplementary unseen pool (n={int(e.n):,})", e.auroc, e.auroc_cluster_bootstrap_ci_lo, e.auroc_cluster_bootstrap_ci_hi, True))
    s1 = lit[lit.source_location.str.startswith("Supplementary Table S1")]
    for _, r in s1.iterrows():
        dev = "SPECTRALIS" if "SPECTRALIS" in r.device else "Triton"
        items.append(("Tang et al. 2021 — 2D ResNet-18, B-scan level (category B)", f"{dev} · {r.dataset} · {'internal' if r.cohort_role.startswith('primary') else 'external'}",
                      float(r.auroc), float(r.auroc_ci_lo), float(r.auroc_ci_hi), True))
    t2 = lit[lit.source_location.str.startswith("Main text Table 2") & (lit.dataset == "Primary")]
    for _, r in t2.iterrows():
        unit = "3D CNN, volume" if r.model.startswith("3D") else "2D CNN aggregated to volume"
        items.append(("Context only (category C) — volume-level evaluation", f"Tang 2021 · {r.device} · primary · {unit}", float(r.auroc), float(r.auroc_ci_lo), float(r.auroc_ci_hi), False))
    z = lit[lit.study.str.startswith("Zhu 2025") & lit.task.str.startswith("non-DME vs DME") & lit.dataset.isin(["Testing", "External validation Total"])]
    for _, r in z.iterrows():
        items.append(("Context only (category C) — volume-level evaluation", f"Zhu 2025 · 3D CNN · {'hold-out test' if r.dataset == 'Testing' else 'external validation (total)'}",
                      float(r.auroc), float(r.auroc_ci_lo), float(r.auroc_ci_hi), False))

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    yy, ylab, rows, cur, groups = [], [], [], 0, []
    for g in dict.fromkeys(i[0] for i in items):
        groups.append((g, cur)); cur += 1
        for it in [i for i in items if i[0] == g]:
            yy.append(cur); ylab.append(it[1]); cur += 1
            col = "#2a78d6" if g.startswith("This") else ("#eb6834" if "category B" in g else INK2)
            if not np.isnan(it[3]):
                ax.errorbar([it[2]], [cur - 1], xerr=[[max(0, it[2] - it[3])], [max(0, it[4] - it[2])]], fmt="none", ecolor=col, elinewidth=1.2, capsize=3)
            ax.plot([it[2]], [cur - 1], "o" if it[5] else "s", ms=7, color=col if it[5] else "white", mec=col, mew=1.5)
            ax.text(1.009, cur - 1, f"{it[2]:.3f}" + ("" if np.isnan(it[3]) else f" ({it[3]:.3f}–{it[4]:.3f})"), va="center", fontsize=7.5, color=INK)
            rows.append({"group": g, "label": it[1], "auroc": it[2], "ci_lo": it[3], "ci_hi": it[4]})
        cur += 0.4
    for g, gy in groups:
        ax.text(0.80, gy, g, fontsize=8.5, fontweight="bold", color=INK, va="center")
    ax.set_yticks(yy, ylab, fontsize=7.8); ax.invert_yaxis(); ax.set_xlim(0.80, 1.006); fig.subplots_adjust(left=0.36, right=0.80); ax.set_xlabel("Reported AUROC (95% CI as published or computed)")
    ax.grid(axis="y", visible=False)
    fig.suptitle("Reported Performance of Published DME Classification Systems and the Present 2D OCT Model", fontsize=11, x=0.42, y=0.995)
    fig.text(0.02, -0.02, "Not a head-to-head comparison: every row uses a different dataset, population, device, label definition and evaluation protocol. "
             "Tang 2021 'DME' includes retinal thickening and hard exudates in diabetic eyes;\nthis project's 'DME vs Normal' uses Kermany folder labels. "
             "Square markers = volume-level results (different evaluation unit; context only). The x-axis starts at 0.80 to resolve the published intervals; "
             "no statistical test across studies was performed.\nVolume-level rows show primary datasets only; all external results (both papers) are listed in Table 3. "
             "This project's official-test AUROC has no meaningful CI (perfect separation).", fontsize=7, color=INK2)
    save(fig, "figure3_literature_context", pd.DataFrame(rows))


# --------------------------------------------------------------------------- Figure 4
def figure4():
    b3 = pd.read_csv(REPO / "artifacts" / "training_log_reconstructed.csv")
    rn = pd.read_csv(REPO / "artifacts" / "kaggle" / "20260920T110729Z" / "artifacts" / "training_log.csv")
    meta = json.loads((DATA / "checkpoint_metadata.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw={"wspace": 0.32}); rows = []
    ax = axes[0]
    ax.plot(b3.global_epoch, b3.train_loss, "-", color=C["EfficientNet-B3"], lw=2, marker="o", ms=3.5, label="Training loss")
    ax.plot(b3.global_epoch, b3.val_loss, "--", color=C["EfficientNet-B0"], lw=2, marker="s", ms=3.5, label="Validation loss")
    sel = meta["EfficientNet-B3"]["global_epoch"]
    ax.axvline(sel, color=INK2, lw=1, ls=":"); ax.text(sel + 0.3, ax.get_ylim()[1] * 0.92, f"selected\nepoch {sel}", fontsize=7, color=INK2)
    ax.axvline(5.5, color=GRID, lw=3); ax.text(5.9, 0.55, "phase A → B\n(full unfreeze)", fontsize=7, color=INK2)
    ax.set(xlabel="Epoch", ylabel="Weighted cross-entropy loss"); ax.set_title("a  EfficientNet-B3 loss", loc="left")
    ax.legend(fontsize=7.5)
    ax = axes[1]
    ax.plot(b3.global_epoch, b3.val_auc, "-", color=C["EfficientNet-B3"], lw=2, marker="o", ms=3.5, label="EfficientNet-B3 validation AUROC")
    ax.plot(rn.global_epoch, rn.val_auc, ":", color=C["ResNet-50"], lw=2, marker="^", ms=3.5, label="ResNet-50 validation AUROC")
    ax.set(xlabel="Epoch", ylabel="Validation AUROC", ylim=(0.85, 1.0)); ax.set_title("b  Validation AUROC (selection metric)", loc="left"); ax.legend(fontsize=7.5, loc="lower right")
    ax = axes[2]
    ax.plot(rn.global_epoch, rn.train_loss, "-", color=C["EfficientNet-B3"], lw=2, marker="o", ms=3.5, label="Training loss")
    ax.plot(rn.global_epoch, rn.val_loss, "--", color=C["EfficientNet-B0"], lw=2, marker="s", ms=3.5, label="Validation loss")
    sel_r = meta["ResNet-50"]["global_epoch"]
    ax.axvline(sel_r, color=INK2, lw=1, ls=":"); ax.text(sel_r - 0.4, ax.get_ylim()[1] * 0.92, f"selected\nepoch {sel_r}", fontsize=7, color=INK2, ha="right")
    ax.set(xlabel="Epoch", ylabel="Weighted cross-entropy loss"); ax.set_title("c  ResNet-50 loss", loc="left"); ax.legend(fontsize=7.5)
    fig.suptitle("Training and Validation Performance Across Training Epochs", fontsize=11, y=1.03)
    fig.text(0.0, -0.08, "B3 per-epoch values were parsed from the Kaggle console log (artifacts/training_log_reconstructed.csv; the run's training_log.csv was lost); accuracy was not logged for that run. "
             "ResNet-50 values from its training_log.csv.\nEfficientNet-B0 and the label-smoothing run have no surviving per-epoch log and are not shown. Training loss is computed on augmented images with class weights; "
             "validation = patient-disjoint split (n=5,560 after exclusion). Test data never used.", fontsize=7, color=INK2)
    for _, r in b3.iterrows():
        rows.append({"model": "EfficientNet-B3", "epoch": r.global_epoch, "train_loss": r.train_loss, "val_loss": r.val_loss, "val_auc": r.val_auc})
    for _, r in rn.iterrows():
        rows.append({"model": "ResNet-50", "epoch": r.global_epoch, "train_loss": r.train_loss, "val_loss": r.val_loss, "val_auc": r.val_auc, "train_accuracy": r.train_accuracy, "val_accuracy": r.val_accuracy})
    save(fig, "figure4_learning_curves", pd.DataFrame(rows))


# --------------------------------------------------------------------------- Figure 5
def figure5():
    bins = pd.read_csv(DATA / "reliability_bins.csv"); cal = pd.read_csv(DATA / "calibration_summary.csv").set_index("model")
    show = MODELS + ["EfficientNet-B3 + LS 0.05 (ablation)"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.7), sharey=True); rows = []
    for ax, m in zip(axes, show):
        for kind, col, mk, off in (("raw", "#2a78d6", "o", -0.012), ("calibrated", "#eb6834", "s", 0.012)):
            b = bins[(bins.model == m) & (bins.probabilities == kind) & (bins["count"] > 0)]
            ax.plot(b.mean_pred + off, b.frac_pos, mk, color=col, ms=6, mec="white", mew=0.8, label=f"{kind}  ECE {cal.loc[m, 'ece10_' + kind]:.4f}, Brier {cal.loc[m, 'brier_' + kind]:.4f}")
            for _, r in b.iterrows():
                rows.append({"model": m, "probabilities": kind, "bin": r.bin, "mean_pred": r.mean_pred, "frac_pos": r.frac_pos, "count": r["count"]})
        braw = bins[(bins.model == m) & (bins.probabilities == "raw")]
        for _, r in braw.iterrows():
            ax.text((r.lo + r.hi) / 2, -0.11, str(int(r["count"])), ha="center", fontsize=6.3, color=INK2)
        ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (2, 2)))
        ax.set(xlim=(0, 1), ylim=(-0.15, 1.02), xlabel="Mean predicted P(DME) in bin"); ax.set_title(m.replace(" (ablation)", "\n(ablation)"), loc="left", fontsize=9)
        ax.legend(fontsize=6.3, loc="center left", bbox_to_anchor=(0.0, 0.62), frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    axes[0].set_ylabel("Observed fraction DME"); axes[0].text(-0.02, -0.11, "n/bin\n(raw)", fontsize=6.3, color=INK2, ha="right", va="center")
    fig.suptitle("Calibration on the Official Test Set: Raw vs Temperature-Scaled Probabilities", fontsize=11, y=1.04)
    fig.text(0.0, -0.1, "10 equal-width bins (same as evaluate.py); empty bins omitted; numbers under each axis = images per bin (raw). Almost all predictions fall in the outermost bins, so intermediate-bin "
             "estimates rest on 0–2 images.\nTemperatures were fitted by the original evaluate.py on a re-generated 'validation' split that does not match the training-time split and overlaps the training data "
             "(see leakage_audit.md §5); the test set was not used for fitting.", fontsize=7, color=INK2)
    save(fig, "figure5_reliability", pd.DataFrame(rows))


# --------------------------------------------------------------------------- Audit figure A1 (segmentation)
def figure_a1():
    seg = pd.read_csv(DATA / "segmentation_per_image.csv", dtype={"subject": str})
    npz_path = REPO / "research_outputs" / ".cache" / "segmentation_examples.npz"
    t = seg[seg.split == "test"]
    fig = plt.figure(figsize=(12, 6.2)); gs = fig.add_gridspec(2, 5, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1, 1.3], wspace=0.3, hspace=0.4)
    rows = []
    if npz_path.exists():
        z = np.load(npz_path)
        keys = sorted({k.split("__")[0] for k in z.files})
        pos_key = [k for k in keys if t[(t.subject + "_" + t.scan_idx.astype(str)) == k].gt_fluid_pixels.iloc[0] > 0]
        neg_key = [k for k in keys if t[(t.subject + "_" + t.scan_idx.astype(str)) == k].gt_fluid_pixels.iloc[0] == 0]
        picks = [pos_key[0], neg_key[0]]  # first fluid-positive and first fluid-free test scan in subject/scan order (not cherry-picked)
        for r, k in enumerate(picks):
            img, gt, pred, prob, graded = (z[f"{k}__{i}"] for i in range(5))
            sub = t[(t.subject + "_" + t.scan_idx.astype(str)) == k].iloc[0]
            panels = [(img, "gray", "Input (320×320 letterbox)"), (gt, "gray", f"Ground truth, grader 1\n{int(gt.sum())} fluid px (white)"),
                      (pred, "gray", f"Prediction p>0.5 (white = fluid)\n{pred.mean() * 100:.3f}% of frame"), (prob, "magma", f"Probability, mean {prob.mean():.2f}\n(black 0 – yellow 1)")]
            for c, (a, cm, ttl) in enumerate(panels):
                ax = fig.add_subplot(gs[r, c]); ax.imshow(a, cmap=cm, vmin=0, vmax=1 if a.dtype != np.uint8 or a.max() <= 1 else 255)
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_title(ttl, fontsize=7, loc="left")
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_color(INK2); sp.set_linewidth(0.8)
                if c == 0:
                    ax.set_ylabel(f"Subject {sub.subject}, scan {sub.scan_idx}\nDice {sub.dice_pipeline:.4f}", fontsize=7.5)
            rows.append({"panel": "example", "key": k, "dice": sub.dice_pipeline, "pred_frac_of_frame": sub.pred_frac_of_frame, "gt_fluid_pixels": sub.gt_fluid_pixels})
    ax = fig.add_subplot(gs[:, 4]); ax.set_position(ax.get_position().translated(0.04, 0))
    for s, col, mk in (("val", "#2a78d6", "o"), ("test", "#eb6834", "s")):
        d = seg[seg.split == s]
        ax.plot(d.gt_frac_of_graded * 100, d.pred_frac_of_frame * 100, mk, color=col, ms=6, mec="white", mew=0.8, label=f"{s} (n={len(d)})")
        rows += [{"panel": "scatter", "split": s, "gt_pct_graded": a * 100, "pred_pct_frame": b * 100} for a, b in zip(d.gt_frac_of_graded, d.pred_frac_of_frame)]
    ax.set(xlim=(-0.5, 10), ylim=(0, 101), xlabel="Ground-truth fluid, % of graded region", ylabel="Predicted fluid, % of whole frame")
    ax.set_title("Predicted vs true fluid fraction", loc="left", fontsize=9); ax.legend(fontsize=7.5, loc="center right")
    fig.suptitle("Audit Figure A1 — U-Net Fluid Segmentation Collapsed to All-Foreground (Not a Performance Figure)", fontsize=11, y=1.0)
    fig.text(0.0, -0.04, "Duke DME (Chiu et al. 2015), held-out subjects 06 and 07 (22 annotated B-scans; 8 with no fluid). Target: any manualFluid1 region; ungraded pixels counted as background by segment.py. "
             "Checkpoint from epoch 1 of 6.\nThe model labels 99.99% of every frame (including zero padding) as fluid; mean test Dice 0.0090 / IoU 0.0045 reflects this failure, not partial success. "
             "Examples = first fluid-positive and first fluid-free test scan in subject/scan order.", fontsize=7, color=INK2)
    save(fig, "figureA1_segmentation_audit", pd.DataFrame(rows))


if __name__ == "__main__":
    figure1(); figure2(); figure3(); figure4(); figure5(); figure_a1()
    print("figures written to", FIG)
