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
import re
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

# The five published papers (Phase C). Colours are distinct from the three project models;
# DMENet uses fundus photographs (different modality), so it is grey.
PAPERS = ["Tang 2021", "Zhu 2025", "Li F 2019", "Rodríguez-Miguel 2024", "Singh & Gorantla 2020"]
PC = {"Tang 2021": "#8e5bd0", "Zhu 2025": "#c98a00", "Li F 2019": "#c2418f", "Rodríguez-Miguel 2024": "#7a5230", "Singh & Gorantla 2020": "#8a8984"}
PMK = {"Tang 2021": "P", "Zhu 2025": "s", "Li F 2019": "^", "Rodríguez-Miguel 2024": "D", "Singh & Gorantla 2020": "X"}
PNOTE = {"Tang 2021": "OCT, B-scan + volume", "Zhu 2025": "OCT, volume", "Li F 2019": "OCT, B-scan", "Rodríguez-Miguel 2024": "OCT, cube",
         "Singh & Gorantla 2020": "fundus photo, different modality"}
NOT_REPORTED = ("Comparison with the five papers: Tang 2021, Zhu 2025, Li F 2019, Rodríguez-Miguel 2024 and Singh & Gorantla 2020 "
                "report no {what} (full text searched; NR), so no published values can be placed on this figure.")


def per_paper():
    return pd.read_csv(DATA / "per_paper_comparison.csv")


def paper_label(p):
    return f"{p} ({PNOTE[p]})"

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
    """One graph: the three project models (official test) and the five papers' headline rows, metric by metric.
    Bars = headline value; dots = every other DME-presence row of that source; bracket = min-max over all rows."""
    perf = pd.read_csv(DATA / "table2_within_project_performance.csv")
    perf = perf[(perf.threshold == 0.5) & perf.model.isin(MODELS)].set_index("model").loc[MODELS]
    pp = per_paper()
    metrics = [("accuracy", "Accuracy"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity"), ("f1", "F1-score"), ("auroc", "AUROC")]
    series = MODELS + PAPERS
    fig, ax = plt.subplots(figsize=(14, 5.6))
    w = 0.86 / len(series); x = np.arange(len(metrics)); rows = []
    rng = np.random.default_rng(0)  # horizontal jitter of the all-rows dots only
    for i, s in enumerate(series):
        pos = x - 0.43 + w * (i + 0.5)
        if s in MODELS:
            vals = np.array([perf.loc[s, k] for k, _ in metrics])
            lo = np.array([perf.loc[s, f"{k}_ci_lo"] for k, _ in metrics]); hi = np.array([perf.loc[s, f"{k}_ci_hi"] for k, _ in metrics])
            ax.bar(pos, vals, w * 0.92, color=C[s], hatch=HATCH[s], edgecolor="white", linewidth=0.5, label=f"This project · {s} (official test, n=484)")
            ok = ~np.isnan(lo)
            ax.errorbar(pos[ok], vals[ok], yerr=[vals[ok] - lo[ok], hi[ok] - vals[ok]], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
            for (k, _), v, l_, h_ in zip(metrics, vals, lo, hi):
                rows.append({"panel": "a", "source": "This project", "model": s, "metric": k, "value": v, "ci_lo": l_, "ci_hi": h_, "headline": True})
            others = pp[(pp.paper == "This project") & (pp.model == s) & ~pp.headline]
        else:
            d = pp[pp.paper == s]; h = d[d.headline].iloc[0]
            vals = np.array([h[k] for k, _ in metrics])
            ax.bar(pos, np.nan_to_num(vals), w * 0.92, color=PC[s], edgecolor="white", linewidth=0.5, alpha=0.95, label=paper_label(s))
            for (k, _), p_, v in zip(metrics, pos, vals):
                rows.append({"panel": "a", "source": s, "model": h.model, "metric": k, "value": v, "ci_lo": h.auroc_ci_lo if k == "auroc" else np.nan,
                             "ci_hi": h.auroc_ci_hi if k == "auroc" else np.nan, "headline": True})
                if np.isnan(v):
                    ax.text(p_, 0.42, "NR", rotation=90, ha="center", va="bottom", fontsize=6.5, color=INK2)
                allv = d[k].dropna()
                if len(allv) > 1:  # min-max bracket over all rows of this paper
                    ax.plot([p_, p_], [allv.min(), allv.max()], color=INK, lw=0.9, alpha=0.8)
                    ax.plot([p_ - w * 0.25, p_ + w * 0.25], [allv.min()] * 2, color=INK, lw=0.9); ax.plot([p_ - w * 0.25, p_ + w * 0.25], [allv.max()] * 2, color=INK, lw=0.9)
            others = d[~d.headline]
        for (k, _), p_ in zip(metrics, pos):
            v = others[k].dropna()
            ax.plot(p_ + rng.uniform(-w * 0.2, w * 0.2, len(v)), v, "o", ms=2.6, color="white", mec=INK, mew=0.6, ls="none", zorder=4)
            rows += [{"panel": "a_other_rows", "source": "This project" if s in MODELS else s, "model": s, "metric": k, "value": z, "headline": False} for z in v]
    for xi in x[:-1]:
        ax.axvline(xi + 0.5, color=GRID, lw=1)
    ax.set_xticks(x, [lab for _, lab in metrics]); ax.set_ylim(0.4, 1.04); ax.set_ylabel("Reported value"); ax.grid(axis="x", visible=False)
    from matplotlib.lines import Line2D
    hl, lb = ax.get_legend_handles_labels()
    hl += [Line2D([], [], marker="o", color="white", mec=INK, mew=0.6, ls="none", ms=4), Line2D([], [], color=INK, lw=0.9)]
    lb += ["Other rows of the same source (other cohorts, devices, units, models)", "Min–max over all rows of the paper"]
    ax.legend(hl, lb, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.07), fontsize=7.6)
    fig.suptitle("Performance of This Project's Models and the Five Published DME Classifiers, Metric by Metric", fontsize=11, y=0.98)
    fig.text(0.01, -0.2, "Bars: this project = official Kermany test split (242 DME / 242 Normal, threshold 0.5; Wilson 95% CI, F1 class-stratified bootstrap); each paper = its headline row "
             "(rule in 14_per_paper_comparison.py: internal B-scan/image-level result if reported, otherwise the main-text primary result).\n"
             "Project dots = the same model on the unseen pool, OCTDL and Duke 2014 (volume). Not a head-to-head comparison: datasets, populations, devices, label definitions and evaluation units "
             "differ for every source; no cross-study statistical test.\nZhu and Rodríguez-Miguel headlines are volume-level; Singh & Gorantla use fundus photographs. NR = not reported. "
             "The project's official-test values have no meaningful CI (perfect separation). Y-axis starts at 0.40.", fontsize=7.3, color=INK2)
    save(fig, "figure1_within_project_performance", pd.DataFrame(rows))


# --------------------------------------------------------------------------- Figure 2
def figure2():
    """One ROC graph: project curves (official test and unseen pool) plus each paper's published operating points;
    panel b is the same graph zoomed. The papers publish operating points, not ROC coordinates."""
    roc = pd.read_csv(DATA / "roc_coordinates_test.csv")
    perf = pd.read_csv(DATA / "table2_within_project_performance.csv")
    e = pd.read_csv(DATA / "roc_coordinates_excluded_pool.csv"); ep = pd.read_csv(DATA / "supplementary_excluded_pool_performance.csv").set_index("model")
    pp = per_paper()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0), gridspec_kw={"wspace": 0.18}); rows = []
    SET_MK = {"official_test": "o", "unseen_pool": "D", "octdl": "^", "duke2014": "s"}
    SET_LAB = {"official_test": "official test", "unseen_pool": "unseen pool", "octdl": "OCTDL, external", "duke2014": "Duke 2014 volume, external"}
    for k, ax in enumerate(axes):
        lab = k == 0
        for m in MODELS:
            r = roc[roc.model == m]; a = perf[(perf.model == m) & (perf.threshold == 0.5)].auroc.iloc[0]
            ax.plot(r.fpr, r.tpr, LS[m], color=C[m], lw=1.2, alpha=0.55, label=f"This project · {m} · official test ROC, AUROC {a:.3f}" if lab else None)
            r2 = e[e.model == m]
            ax.plot(r2.fpr, r2.tpr, LS[m], color=C[m], lw=2,
                    label=f"This project · {m} · unseen pool ROC, AUROC {ep.loc[m, 'auroc']:.3f}" if lab else None)
            if k == 0:
                rows += [{"panel": "a", "model": m, "fpr": f, "tpr": t_} for f, t_ in zip(r.fpr, r.tpr)]
                rows += [{"panel": "a_pool", "model": m, "fpr": f, "tpr": t_} for f, t_ in zip(r2.fpr, r2.tpr)]
        b3 = pp[(pp.paper == "This project") & (pp.model == "EfficientNet-B3")]
        for _, q in b3.iterrows():
            ax.plot([1 - q.specificity], [q.sensitivity], SET_MK[q.setting_key], ms=8, color=C["EfficientNet-B3"] if q.unit == "image" else "white",
                    mec=C["EfficientNet-B3"], mew=1.8, ls="none", zorder=6, label=f"This project · EfficientNet-B3 at 0.5 · {SET_LAB[q.setting_key]}" if lab else None)
            if k == 0:
                rows.append({"panel": "a_points", "model": "This project EfficientNet-B3", "setting": q.setting_key, "fpr": 1 - q.specificity, "tpr": q.sensitivity})
        pub = pp[(pp.paper != "This project") & pp.sensitivity.notna() & pp.specificity.notna()]
        for paper in PAPERS:
            d = pub[pub.paper == paper]
            for j, (_, q) in enumerate(d.iterrows()):
                big = bool(q.headline)
                ax.plot([1 - q.specificity], [q.sensitivity], PMK[paper], ms=9 if big else 6, color=PC[paper] if q.unit == "image" else "white",
                        mec=INK if big else PC[paper], mew=1.6 if big else 1.3, ls="none", zorder=5,
                        label=None)
                if k == 0:
                    rows.append({"panel": "a_points", "model": paper, "setting": q.setting, "fpr": 1 - q.specificity, "tpr": q.sensitivity, "headline": big})
        ax.set(xlabel="False positive rate (1 − specificity)", ylabel="True positive rate (sensitivity)")
        if k == 0:
            ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (2, 2)))
            ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01)); ax.set_aspect("equal")
            ax.set_title("a  ROC: this project and the five papers", loc="left")
        else:
            ax.set(xlim=(-0.005, 0.56), ylim=(0.43, 1.005))
            ax.set_title("b  Same graph, zoomed (1 − specificity ≤ 0.56, sensitivity ≥ 0.43)", loc="left")
    from matplotlib.lines import Line2D
    hl, lb = axes[0].get_legend_handles_labels()
    for paper in PAPERS:
        n_pts = int((pub.paper == paper).sum())
        hl.append(Line2D([], [], marker=PMK[paper], color=PC[paper], mec=PC[paper], ls="none", ms=7)); lb.append(f"{paper_label(paper)} · {n_pts} points")
    hl += [Line2D([], [], marker="o", color="white", mec=INK2, mew=1.3, ls="none", ms=7), Line2D([], [], marker="o", color=INK2, mec=INK, mew=1.6, ls="none", ms=9)]
    lb += ["Hollow = volume-level result", "Large, black edge = the paper's headline row"]
    axes[0].legend(hl, lb, loc="upper center", bbox_to_anchor=(1.05, -0.12), ncol=2, fontsize=7.2)
    fig.suptitle("Receiver Operating Characteristic: This Project's Curves and the Five Papers' Published Operating Points", fontsize=11, y=0.98)
    fig.text(0.01, -0.3, "Project curves from continuous raw softmax P(DME) (temperature scaling is monotonic and leaves ROC unchanged): thin = official test (n=484; all three curves lie on the left and top axes, "
             "AUROC 1.000), thick = unseen pool (n=7,503).\nPublished markers: one per DME-presence row with both sensitivity and specificity (per_paper_comparison.csv); filled = image level, "
             "hollow = volume level; large black-edged marker = the paper's headline row. Li 2019, Rodríguez-Miguel 2024 and\nSingh & Gorantla 2020 print ROC curves as images only; "
             "curves were not digitised. Rodríguez-Miguel internal-test rows report no sensitivity/specificity and are not plotted. Not a head-to-head comparison; no cross-study test.", fontsize=7.3, color=INK2)
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
    pp = per_paper()
    for _, q in pp[(pp.paper == "This project") & (pp.model == "EfficientNet-B3") & (pp.role == "external")].iterrows():
        unit = "volume, max rule" if q.unit == "volume" else "B-scan"
        items.append(("This project — external test sets", f"EfficientNet-B3, {q.setting.split(' (')[0]} ({unit}, n={q.n})",
                      q.auroc, q.auroc_ci_lo, q.auroc_ci_hi, q.unit == "image"))
    s1 = lit[lit.source_location.str.startswith("Supplementary Table S1")]
    for _, r in s1.iterrows():
        dev = "SPECTRALIS" if "SPECTRALIS" in r.device else "Triton"
        items.append(("Tang et al. 2021 — 2D ResNet-18, B-scan level (category B)", f"{dev} · {r.dataset} · {'internal' if r.cohort_role.startswith('primary') else 'external'}",
                      float(r.auroc), float(r.auroc_ci_lo), float(r.auroc_ci_hi), True))
    li = lit[lit.study.str.startswith("Li F 2019") & (lit.comparability_category == "B")]
    for _, r in li.iterrows():
        items.append(("Li et al. 2019 — ResNet50 ensemble, B-scan level (category B)", "SPECTRALIS · Shanghai · internal test · DME vs NORMAL",
                      float(r.auroc), float(r.auroc_ci_lo), float(r.auroc_ci_hi), True))
    t2 = lit[lit.source_location.str.startswith("Main text Table 2") & (lit.dataset == "Primary")]
    for _, r in t2.iterrows():
        unit = "3D CNN, volume" if r.model.startswith("3D") else "2D CNN aggregated to volume"
        items.append(("Context only (category C) — volume-level evaluation", f"Tang 2021 · {r.device} · primary · {unit}", float(r.auroc), float(r.auroc_ci_lo), float(r.auroc_ci_hi), False))
    z = lit[lit.study.str.startswith("Zhu 2025") & lit.task.str.startswith("non-DME vs DME") & lit.dataset.isin(["Testing", "External validation Total"])]
    for _, r in z.iterrows():
        items.append(("Context only (category C) — volume-level evaluation", f"Zhu 2025 · 3D CNN · {'hold-out test' if r.dataset == 'Testing' else 'external validation (total)'}",
                      float(r.auroc), float(r.auroc_ci_lo), float(r.auroc_ci_hi), False))
    rm = lit[lit.study.str.startswith("Rodríguez-Miguel 2024") & lit.source_location.str.startswith("Table 2") & (lit.auroc != "NR")]
    for _, r in rm.iterrows():
        items.append(("Context only (category C) — volume-level evaluation", "Rodríguez-Miguel 2024 · CNN-RNN (model 2) · screening cohort",
                      float(r.auroc), float(r.auroc_ci_lo), float(r.auroc_ci_hi), False))
    sg = lit[lit.study.str.startswith("Singh & Gorantla 2020") & (lit.auroc != "NR")]
    for _, r in sg.iterrows():
        items.append(("Different modality (category D) — fundus photograph", "Singh & Gorantla 2020 · DMENet · IDRiD + MESSIDOR",
                      float(r.auroc), np.nan, np.nan, False))

    fig, ax = plt.subplots(figsize=(9.5, 8.3))
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
             "\nTang 2021 'DME' includes retinal thickening and hard exudates in diabetic eyes; this project's 'DME vs Normal' uses Kermany folder labels. "
             "Square markers = volume-level results (different evaluation unit; context only).\nThe x-axis starts at 0.80 to resolve the published intervals; "
             "no statistical test\nacross studies was performed. Volume-level rows show primary datasets only; all external results are listed in Table 3.\nLi 2019 CIs are printed values, narrower than binomial for its n (Table 3 notes); DMENet (Singh & Gorantla 2020) uses fundus photographs and is shown as a different-modality reference only.\n"
             "This project's official-test AUROC has no meaningful CI (perfect separation).", fontsize=7, color=INK2)
    save(fig, "figure3_literature_context", pd.DataFrame(rows))


# --------------------------------------------------------------------------- Figure 4
EPOCH_RE = re.compile(r"\[[AB] epoch \d+/\d+ \| global (\d+)\] train_loss=([\d.]+) val_loss=([\d.]+) val_auc=([\d.]+)")


def parse_console_log(path):
    """Per-epoch values from a Kaggle run.log (JSON-lines console capture)."""
    rows = {}
    for line in path.read_text(encoding="utf8").splitlines():
        m = EPOCH_RE.search(line)
        if m:
            g, tl, vl, va = m.groups()
            rows[int(g)] = {"global_epoch": int(g), "train_loss": float(tl), "val_loss": float(vl), "val_auc": float(va)}
    return pd.DataFrame(sorted(rows.values(), key=lambda r: r["global_epoch"]))


def figure4():
    b3 = pd.read_csv(REPO / "artifacts" / "training_log_reconstructed.csv")
    rn = pd.read_csv(REPO / "artifacts" / "kaggle" / "20260920T110729Z" / "artifacts" / "training_log.csv")
    b0 = parse_console_log(REPO / "artifacts" / "kaggle" / "20260920T095427Z" / "run.log")
    ls = parse_console_log(REPO / "artifacts" / "kaggle" / "20260922T160023Z" / "run.log")
    meta = json.loads((DATA / "checkpoint_metadata.json").read_text())
    LS_COL = "#8a5cc2"
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), gridspec_kw={"wspace": 0.25, "hspace": 0.42}); rows = []
    ax = axes[0, 0]
    ax.plot(b3.global_epoch, b3.train_loss, "-", color=C["EfficientNet-B3"], lw=2, marker="o", ms=3.5, label="Training loss")
    ax.plot(b3.global_epoch, b3.val_loss, "--", color=C["EfficientNet-B0"], lw=2, marker="s", ms=3.5, label="Validation loss")
    sel = meta["EfficientNet-B3"]["global_epoch"]
    ax.axvline(sel, color=INK2, lw=1, ls=":"); ax.text(sel + 0.3, ax.get_ylim()[1] * 0.92, f"selected\nepoch {sel}", fontsize=7, color=INK2)
    ax.axvline(5.5, color=GRID, lw=3); ax.text(5.9, 0.55, "phase A → B\n(full unfreeze)", fontsize=7, color=INK2)
    ax.set(xlabel="Epoch", ylabel="Weighted cross-entropy loss"); ax.set_title("a  EfficientNet-B3 loss", loc="left")
    ax.legend(fontsize=7.5)
    ax = axes[0, 1]
    ax.plot(b3.global_epoch, b3.val_auc, "-", color=C["EfficientNet-B3"], lw=2, marker="o", ms=3.5, label="EfficientNet-B3")
    ax.plot(b0.global_epoch, b0.val_auc, "--", color=C["EfficientNet-B0"], lw=2, marker="s", ms=3.5, label="EfficientNet-B0")
    ax.plot(rn.global_epoch, rn.val_auc, ":", color=C["ResNet-50"], lw=2, marker="^", ms=3.5, label="ResNet-50")
    ax.plot(ls.global_epoch, ls.val_auc, "-.", color=LS_COL, lw=2, marker="D", ms=3, label="B3 + label smoothing 0.05")
    ax.set(xlabel="Epoch", ylabel="Validation AUROC", ylim=(0.85, 1.0)); ax.set_title("b  Validation AUROC (selection metric)", loc="left"); ax.legend(fontsize=7.5, loc="lower right")
    ax = axes[1, 0]
    ax.plot(rn.global_epoch, rn.train_loss, "-", color=C["EfficientNet-B3"], lw=2, marker="o", ms=3.5, label="Training loss")
    ax.plot(rn.global_epoch, rn.val_loss, "--", color=C["EfficientNet-B0"], lw=2, marker="s", ms=3.5, label="Validation loss")
    sel_r = meta["ResNet-50"]["global_epoch"]
    ax.axvline(sel_r, color=INK2, lw=1, ls=":"); ax.text(sel_r - 0.4, ax.get_ylim()[1] * 0.92, f"selected\nepoch {sel_r}", fontsize=7, color=INK2, ha="right")
    ax.set(xlabel="Epoch", ylabel="Weighted cross-entropy loss"); ax.set_title("c  ResNet-50 loss", loc="left"); ax.legend(fontsize=7.5)
    ax = axes[1, 1]
    ax.plot(b0.global_epoch, b0.train_loss, "-", color=C["EfficientNet-B0"], lw=2, marker="s", ms=3.5, label="B0 training")
    ax.plot(b0.global_epoch, b0.val_loss, "--", color=C["EfficientNet-B0"], lw=2, marker="s", ms=3.5, mfc="white", label="B0 validation")
    ax.plot(ls.global_epoch, ls.train_loss, "-", color=LS_COL, lw=2, marker="D", ms=3, label="B3 + LS training")
    ax.plot(ls.global_epoch, ls.val_loss, "--", color=LS_COL, lw=2, marker="D", ms=3, mfc="white", label="B3 + LS validation")
    for key, col, dx, ha in (("EfficientNet-B0", C["EfficientNet-B0"], 0.3, "left"), ("EfficientNet-B3+LS0.05", LS_COL, -0.3, "right")):
        e = meta[key]["global_epoch"]; ax.axvline(e, color=col, lw=1, ls=":")
        ax.text(e + dx, ax.get_ylim()[1] * 0.92, f"selected\nepoch {e}", fontsize=7, color=col, ha=ha)
    ax.set(xlabel="Epoch", ylabel="Loss (LS run: label-smoothed, not comparable)"); ax.set_title("d  EfficientNet-B0 and B3 + label-smoothing loss", loc="left"); ax.legend(fontsize=7.5)
    fig.suptitle("Training and Validation Performance Across Training Epochs", fontsize=11, y=0.97)
    fig.text(0.0, -0.01, "B3, B0 and label-smoothing (LS) per-epoch values were parsed from the Kaggle console logs (B3: artifacts/training_log_reconstructed.csv; B0: kaggle/20260920T095427Z/run.log; "
             "LS: kaggle/20260922T160023Z/run.log); accuracy was not logged for those runs.\nResNet-50 values from its training_log.csv. Both early-stopped runs end before 30 epochs. "
             "The LS loss includes the smoothing target, so its floor is higher and it is not on the same scale as the other losses. Training loss is computed on augmented images with class weights;\n"
             "validation = patient-disjoint split (n=5,560 after exclusion). Test data never used.\n" + NOT_REPORTED.format(what="training or validation learning curves"), fontsize=7, color=INK2)
    for name, d in (("EfficientNet-B3", b3), ("EfficientNet-B0", b0), ("EfficientNet-B3+LS0.05", ls)):
        for _, r in d.iterrows():
            rows.append({"model": name, "epoch": r.global_epoch, "train_loss": r.train_loss, "val_loss": r.val_loss, "val_auc": r.val_auc})
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
             "estimates rest on 0–2 images.\nTemperatures were refit with evaluate.fit_temperature on the training-time validation split (n=5,560, rebuilt with scikit-learn 1.4.2; "
             "scripts 11–12); the test set was not used for fitting.\n" + NOT_REPORTED.format(what="calibration analysis (reliability diagram, ECE or Brier score)"), fontsize=7, color=INK2)
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
    fig.text(0.0, -0.04, "Duke DME (Chiu et al. 2015), held-out subjects 06 and 07 (22 annotated B-scans; 8 with no fluid). Target: any manualFluid1 region; ungraded side margins (stored as 0) counted as background by segment.py; graded region = columns with manual layer annotations. "
             "Checkpoint from epoch 1 of 6.\nThe model labels 99.99% of every frame (including zero padding) as fluid; mean test Dice 0.0090 / IoU 0.0045 reflects this failure, not partial success. "
             "Examples = first fluid-positive and first fluid-free test scan in subject/scan order.\n"
             "Comparison with the five papers: all five are classification-only (Tang 2021 explicitly segmentation-free); none reports a fluid-segmentation Dice, so no published values can be placed here.", fontsize=7, color=INK2)
    save(fig, "figureA1_segmentation_audit", pd.DataFrame(rows))


if __name__ == "__main__":
    figure1(); figure2(); figure3(); figure4(); figure5(); figure_a1()
    print("figures written to", FIG)
