"""
Phase C -- comparison figures F1-F6. Reads ONLY research_outputs/data/literature_comparison.csv.

Conventions (see research_audit/figure_index.md):
  * colour = source (one colour per source in every figure); this project = teal family, comparators = muted neutrals.
  * HATCHED bar = value NOT verified against a primary source (verification_status starts with 'unverified').
    Solid bar = verified. (F4a and F4b are checklists, not values; see their notes.)
  * Category A/B sources share axes in F1, F3, F5. Category C sources appear only in F6 (contextual), and in F2/F4,
    which show test-set composition and reporting practice, not performance.
  * No ranking, winner or significance annotations. No p-values.
Usage: python research_outputs/scripts/21_plot_comparison_figures.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "research_outputs" / "data" / "literature_comparison.csv"
FIG = REPO / "research_outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 9, "hatch.linewidth": 0.9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})

D = pd.read_csv(CSV, na_values=["null"]).set_index("row_id")

# colour per source (project = teal family; comparators = muted neutrals) and hatch used only when unverified
COLOR = {"proj_b3_050": "#0b4f4a", "proj_b3_refit": "#2a9d8f", "proj_b0_050": "#7cc6bc", "proj_r50_050": "#cdeeea",
         "P1": "#8a8a8a", "P1_int_comp": "#8a8a8a", "P1_ext": "#8a8a8a", "P2": "#b9ad94", "P3": "#b5c4d3", "P4": "#e6d9d9", "P4_comp": "#e6d9d9"}
HATCH = {"P1": "////", "P1_int_comp": "////", "P1_ext": "////", "P4_comp": "xxxx", "P2": "\\\\\\\\", "P4": "xxxx", "P3": "....",
         "proj_b3_050": "", "proj_b3_refit": "", "proj_b0_050": "", "proj_r50_050": ""}
SHORT = {"proj_b3_050": "This project B3 @0.5", "proj_b3_refit": "This project B3 @0.181", "proj_b0_050": "This project B0 @0.5",
         "proj_r50_050": "This project ResNet-50 @0.5", "P1": "P1 Ai 2022", "P2": "P2 He 2023", "P3": "P3 Takinami 2025 (preprint)",
         "P4": "P4 Kermany 2018", "P1_int_comp": "P1 internal", "P1_ext": "P1 external", "P4_comp": "P4 Kermany 2018"}
UNVER_LABEL = "hatched = not verified against primary source"
ANY_UNVER = any(str(v).startswith("unverified") for v in D["verification_status"])
UNVER_H = [Patch(facecolor="white", edgecolor="#1a1a1a", hatch="////", label=UNVER_LABEL)] if ANY_UNVER else []


def unverified(rid):
    return str(D.loc[rid, "verification_status"]).startswith("unverified")


def bar_kw(rid):
    return dict(facecolor=COLOR[rid], edgecolor="#1a1a1a", linewidth=0.8, hatch=HATCH[rid] if unverified(rid) else "")


def n_text(rid):
    r = D.loc[rid]
    if pd.isna(r.n_test):
        return "n not reported"
    return f"n={int(r.n_test)}, {int(r.n_dme)} DME"


def legend_handles(rids, with_unver=True):
    h = [Patch(label=f"{SHORT[r]} ({n_text(r)})", **bar_kw(r)) for r in rids]
    if with_unver:
        h += UNVER_H
    return h


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


METRICS = [("accuracy", "acc", "Accuracy\n(all images)"), ("sensitivity", "sens", "Sensitivity\n(DME images)"),
           ("specificity", "spec", "Specificity\n(non-DME\nimages)"), ("auroc", None, "AUROC\n(all images)")]


def grouped_bars(ax, rids, zoom=False, width=0.82):
    nb = len(rids)
    w = width / nb
    for j, rid in enumerate(rids):
        for i, (col, ci, _) in enumerate(METRICS):
            x = i - width / 2 + w * (j + 0.5)
            v = D.loc[rid, col]
            if pd.isna(v):
                if not zoom:
                    ax.text(x, 0.012, "n/r", rotation=90, ha="center", va="bottom", fontsize=9)
                continue
            ax.bar(x, v, w * 0.9, **bar_kw(rid))
            if ci:
                lo, hi = D.loc[rid, f"{ci}_ci_lo"], D.loc[rid, f"{ci}_ci_hi"]
                if not (pd.isna(lo) or pd.isna(hi)):
                    ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor="#111", elinewidth=1.1, capsize=2.5, zorder=5)
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([m[2] for m in METRICS], fontsize=9)


# ======================================================================== F1
def fig1():
    rids = ["proj_b3_050", "proj_b3_refit", "P2", "P4"]
    assert set(D.loc[rids, "comparability_category"]) <= {"A", "B"}
    fig = plt.figure(figsize=(14, 6.6))
    ax = fig.add_axes([0.06, 0.30, 0.44, 0.58])
    grouped_bars(ax, rids)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Metric value (0-1); error bars = Wilson 95% CI\nwhere n is known")
    ax.set_title("F1 - Comparable results, category A/B sources only (y-axis starts at 0)", loc="left")
    ax.yaxis.grid(True, color="#e3e3e3", lw=0.7); ax.set_axisbelow(True)
    ins = ax.inset_axes([1.16, 0.0, 0.84, 1.0])
    grouped_bars(ins, rids, zoom=True)
    ins.set_ylim(0.95, 1.002)
    ins.set_title("zoomed: 0.95-1.00", fontsize=10)
    ins.yaxis.grid(True, color="#e3e3e3", lw=0.7); ins.set_axisbelow(True)
    ident = str(D.loc["P2", "test_set_identity"])
    if "set equality = True" in ident:
        ax.text(0.5, 1.115, "This project and P2 share the same 242 DME test images\n(official Kermany test split: this project's 484 = the DME + Normal folders of P2's 968)",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="#f4f4f4", ec="#9a9a9a"))
    fig.legend(handles=legend_handles(rids), loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.06, 0.215, "n/r = metric not reported for that source (P2 reports DME recall and AUC only; its 4-class accuracy is not a DME-vs-Normal quantity).  "
             "No CI where a source's n is not reported (P4).  AUROC bars carry no CI (this project's AUROC CI is degenerate: perfect separation).",
             fontsize=9, ha="left", va="top", wrap=True)
    save(fig, "F1_comparable_results")


# ======================================================================== F2
def fig2():
    rids = ["proj_b3_050", "P1_int_comp", "P1_ext", "P2", "P3", "P4_comp"]
    desc = {"proj_b3_050": "official Kermany split\ninternal [A]", "P1_int_comp": "internal Kermany test\n(release not matched) [C]",
            "P1_ext": "external: Beijing\nCirrus set [C]", "P2": "official 968-image split\n(stated in paper) [B]", "P3": "Kermany, split\nnot stated [C]",
            "P4_comp": "Cell test set (binary n\nnot stated) [B]"}
    fig, ax = plt.subplots(figsize=(14.0, 7.2))
    w = 0.36
    for j, rid in enumerate(rids):
        r = D.loc[rid]
        for col, off, alpha in [("n_test", -w / 2, 1.0), ("n_dme", w / 2, 0.55)]:
            ax.bar(j + off, r[col], w * 0.95, alpha=alpha, **bar_kw(rid))
            ax.text(j + off, r[col] + 14, f"{int(r[col]):,}", ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(range(len(rids)))
    ax.set_xticklabels([f"{SHORT[r]}\n{desc[r]}" for r in rids])
    ax.set_ylabel("Number of test images\n(left bar of each pair = n test; right, paler bar = n DME)")
    ax.set_ylim(0, 1450)
    ax.yaxis.grid(True, color="#e3e3e3", lw=0.7); ax.set_axisbelow(True)
    ax.set_title("F2 - Test set composition (sizes only, not performance; all categories shown, category in brackets)", loc="left")
    p3 = D.loc["P3"]; pr = D.loc["proj_b3_050"]
    j3 = rids.index("P3")
    ax.annotate(f"P3 discrepancy: {int(p3.n_dme)} DME vs {int(pr.n_dme)} per class in the official split ({int(p3.n_dme - pr.n_dme):+d}); "
                f"{int(p3.n_test):,} total vs {int(D.loc['P2','n_test']):,} ({int(p3.n_test - D.loc['P2','n_test']):+d}).\n"
                "Arithmetic from the papers (not P3's stated cause): He 2023 gives 242 test + 8 validation images per class = 250 per class,\n"
                "and Kermany 2018 gives a 1,000-image test set (250 per class). P3 does not state its split. Cause not established.",
                xy=(j3 - w / 2, p3.n_test + 70), xytext=(j3 - 4.05, 1230), fontsize=9.3, ha="left",
                arrowprops=dict(arrowstyle="->", color="#333"), bbox=dict(boxstyle="round,pad=0.3", fc="#f4f4f4", ec="#9a9a9a"))
    if UNVER_H:
        fig.legend(handles=UNVER_H, loc="lower center", frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.01, 0.075, "No bar (n not reported): P1's 1.000 results (test set not stated in the paper).   P4 bar = whole Cell test set; its DME-vs-Normal n is not stated.   "
             "Official Kermany split: this project, P2.   External set: P1 Beijing only.", fontsize=9.5, ha="left", va="top")
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    save(fig, "F2_test_set_composition")


# ======================================================================== F3
def fig3():
    rids = ["proj_b3_050", "proj_b3_refit", "P2", "P4"]
    assert set(D.loc[rids, "comparability_category"]) <= {"A", "B"}
    order = sorted(rids, key=lambda r: D.loc[r, "sensitivity"])  # ascending -> highest estimate drawn on top
    ref = D.loc["proj_b3_050", "sensitivity"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    for ax, (lo_lim, hi_lim, ttl) in zip(axes, [(0.0, 1.0, "(a) zero-based axis"), (0.95, 1.002, "(b) zoomed: 0.95-1.00")]):
        for yi, rid in enumerate(order):
            r = D.loc[rid]
            if pd.isna(r.sens_ci_lo):
                ax.text(lo_lim + 0.02 * (hi_lim - lo_lim), yi, "n not reported - no interval", va="center", fontsize=9.5, style="italic")
                continue
            ax.barh(yi, r.sens_ci_hi - r.sens_ci_lo, left=r.sens_ci_lo, height=0.5, **bar_kw(rid))
            ax.plot(r.sensitivity, yi, marker="D", ms=6, mfc="white", mec="#111", zorder=6)
        ax.axvline(ref, color="#111", ls="--", lw=1.1)
        ax.set_xlim(lo_lim, hi_lim)
        ax.set_ylim(-0.6, len(order) - 0.4 + 0.45)
        ax.set_title(ttl, loc="left")
        ax.xaxis.grid(True, color="#e3e3e3", lw=0.7); ax.set_axisbelow(True)
        ax.set_xlabel("Sensitivity = DME recall (point = estimate; bar = Wilson 95% CI on n DME images)")
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([f"{SHORT[r]}\n({n_text(r)}; est. {D.loc[r,'sensitivity']:.3f})" for r in order])
    fig.suptitle("F3 - Sensitivity confidence intervals, category A/B sources (sorted by point estimate)", x=0.01, ha="left", fontsize=11)
    fig.legend(handles=UNVER_H + [plt.Line2D([], [], color="#111", ls="--", label="this project B3 @0.5 point estimate"),
                        plt.Line2D([], [], marker="D", ls="none", mfc="white", mec="#111", label="point estimate")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.045))
    fig.text(0.01, 0.01, "Sources have different data and test sets; intervals are shown side by side, not tested against each other.", fontsize=9)
    fig.tight_layout(rect=[0, 0.13, 1, 0.94])
    save(fig, "F3_ci_overlap_sensitivity")


# ======================================================================== F4
CRIT = [("rep_patient_split", "PS", "patient-level\nsplit stated"), ("rep_external", "EV", "external\nvalidation"),
        ("rep_ci", "CI", "CIs\nreported"), ("rep_leakage", "LK", "leakage\ninvestigated"),
        ("rep_calibration", "CA", "calibration\nreported"), ("rep_explain_validated", "XV", "explanation\nvalidated"),
        ("rep_peer", "PR", "peer\nreviewed")]


def fig4():
    rids = ["proj_b3_050", "P1", "P2", "P3", "P4"]
    name = {"proj_b3_050": "This project", "P1": "P1 Ai 2022", "P2": "P2 He 2023", "P3": "P3 Takinami\n2025", "P4": "P4 Kermany\n2018"}
    readlab = {"full": "full read", "partial": "partial read", "not read": "not read"}
    read = {r: readlab[str(D.loc[r, "rep_read_status"]).split(" (")[0].replace(" this session", "")] for r in rids}
    fig, (a, b) = plt.subplots(1, 2, figsize=(15.5, 6.6), gridspec_kw=dict(width_ratios=[1.0, 1.5]))
    for j, rid in enumerate(rids):
        base = 0
        for col, code, _ in CRIT:
            if D.loc[rid, col] == 1:
                a.bar(j, 1, 0.62, bottom=base, facecolor=COLOR[rid], edgecolor="white", linewidth=1.5)
                a.text(j, base + 0.5, code, ha="center", va="center", fontsize=9.5, color="white" if rid == "proj_b3_050" else "#111", fontweight="bold")
                base += 1
        a.text(j, base + 0.12, f"{base} of 7", ha="center", va="bottom", fontsize=9.5)
    a.set_ylim(0, 8); a.set_yticks(range(0, 8))
    a.set_xticks(range(len(rids)))
    a.set_xticklabels([f"{name[r]}\n[{read[r]}]" for r in rids], fontsize=9)
    a.set_ylabel("Criteria confirmed as reported (count, of 7)")
    a.set_title("(a) Reporting criteria confirmed as reported", loc="left")
    a.yaxis.grid(True, color="#e3e3e3", lw=0.7); a.set_axisbelow(True)
    nb = len(rids); w = 0.8 / nb
    for j, rid in enumerate(rids):
        for i, (col, code, _) in enumerate(CRIT):
            x = i - 0.4 + w * (j + 0.5)
            v = D.loc[rid, col]
            b.bar(x, v, w * 0.9, facecolor=COLOR[rid], edgecolor="#1a1a1a", linewidth=0.8)
            if v == 0:
                b.text(x, 0.02, "n/r", rotation=90, ha="center", va="bottom", fontsize=8.5)
    b.set_xticks(range(len(CRIT)))
    b.set_xticklabels([f"{c[1]}\n{c[2]}" for c in CRIT], fontsize=9)
    b.set_ylim(0, 1.3); b.set_yticks([0, 1]); b.set_yticklabels(["0  not reported", "1  reported"])
    b.set_title("(b) Per criterion (supplementary)", loc="left")
    b.legend(handles=[Patch(facecolor=COLOR[r], edgecolor="#1a1a1a", label=name[r].replace("\n", " ")) for r in rids], loc="upper center", ncol=5, frameon=False, fontsize=8.8)
    fig.suptitle("F4 - Methodological reporting checklist (scored only on what each text reports)", x=0.01, ha="left", fontsize=11)
    fig.text(0.01, 0.02, "n/r = not reported. Each source's text was searched for all seven criteria (full read of the PDFs, 2026-10-31); 'not reported' means the text does not state it.\n"
             "The checklist scores reporting only, not study quality.  XV = a check of the explanation was reported (showing heatmaps alone scores 0).\n"
             "CI for P4 = 95% CIs on likelihood ratios vs experts only (Fig. S3), not on accuracy/sensitivity/specificity.  XV for P1 = Grad-CAM maps checked by clinicians (qualitative).",
             fontsize=8.8, ha="left", va="bottom")
    fig.tight_layout(rect=[0, 0.12, 1, 0.94])
    save(fig, "F4_reporting_checklist")


# ======================================================================== F5
def fig5():
    rids = ["proj_b3_050", "proj_b0_050", "proj_r50_050"]
    fig = plt.figure(figsize=(14, 6.4))
    ax = fig.add_axes([0.06, 0.30, 0.44, 0.56])
    grouped_bars(ax, rids)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Metric value at threshold 0.5 (0-1); error bars = Wilson 95% CI")
    ax.set_title("F5 - This project's three architectures, official test set (y-axis starts at 0)", loc="left")
    ax.yaxis.grid(True, color="#e3e3e3", lw=0.7); ax.set_axisbelow(True)
    ins = ax.inset_axes([1.16, 0.0, 0.84, 1.0])
    grouped_bars(ins, rids, zoom=True)
    ins.set_ylim(0.95, 1.002); ins.set_title("zoomed: 0.95-1.00", fontsize=10)
    ins.yaxis.grid(True, color="#e3e3e3", lw=0.7); ins.set_axisbelow(True)
    r = D.loc["proj_b3_050"]
    ax.text(0.5, 1.115, f"{int(r.discordant_images_any_pair)} of {int(r.n_test)} test images are classified differently by any pair of the three models\n"
            f"({int(r.discordant_pairwise_total)} pairwise discordances in total); no difference is detectable on this test set.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5, bbox=dict(boxstyle="round,pad=0.3", fc="#f4f4f4", ec="#9a9a9a"))
    fig.legend(handles=legend_handles(rids, with_unver=False), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.03))
    fig.text(0.06, 0.20, "All three: TP 241, FP 0, TN 242, FN 1 at threshold 0.5. AUROC bars carry no CI (degenerate: perfect separation).",
             fontsize=9, ha="left", va="top")
    save(fig, "F5_architecture_comparison")


# ======================================================================== F6
def fig6():
    rids = ["proj_b3_050", "P1", "P1_ext", "P3"]
    reason = {"proj_b3_050": "This project B3 @0.5\nDME vs Normal, official test,\nn=484 (reference, category A)",
              "P1": "P1 Ai 2022 [C] - Table 5\nDME vs Normal; trained on\n1,000 + 1,000 images; test set\nnot stated; n not reported",
              "P1_ext": "P1 Ai 2022 [C] - Table 8\nexternal Beijing Cirrus set\n(n=277, 107 DME); 4-class model,\nDME one-vs-rest; different\npopulation and device",
              "P3": "P3 Takinami 2025 preprint [C]\npooled negative class:\nCNV + drusen + normal"}
    fig, ax = plt.subplots(figsize=(15.0, 7.6))
    w = 0.19
    xt, xl = [], []
    for g, rid in enumerate(rids):
        for i, (col, ci, lab) in enumerate(METRICS):
            x = g * 1.15 + (i - 1.5) * w * 1.08
            v = D.loc[rid, col]
            xt.append(x); xl.append(lab.split("\n")[0])
            if pd.isna(v):
                ax.text(x, 0.40, "n/r", ha="center", va="center", fontsize=9, rotation=90, zorder=6)
                continue
            ax.bar(x, v, w, **bar_kw(rid))
            if ci:
                lo, hi = D.loc[rid, f"{ci}_ci_lo"], D.loc[rid, f"{ci}_ci_hi"]
                if not (pd.isna(lo) or pd.isna(hi)):
                    ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor="#111", elinewidth=1.1, capsize=2.5, zorder=5)
            ax.text(x, 0.40, f"{v:.3f}", ha="center", va="center", fontsize=9, rotation=90, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    ax.set_xticks(xt); ax.set_xticklabels(xl, rotation=40, ha="right", fontsize=9)
    for g, rid in enumerate(rids):
        ax.text(g * 1.15, -0.2, reason[rid], ha="center", va="top", fontsize=9.3, transform=ax.get_xaxis_transform(), fontweight="bold" if g == 0 else "normal")
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Metric value (0-1); error bars = Wilson 95% CI where n is stated")
    ax.set_title("F6 - Contextual - not directly comparable", loc="left", fontsize=12)
    ax.yaxis.grid(True, color="#e3e3e3", lw=0.7); ax.set_axisbelow(True)
    ax.text(0.5, 0.905, "Different tasks or populations. These bars are not a ranking.", transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#555"))
    fig.text(0.01, 0.02, "n/r = metric not reported for that source (Table 8 gives per-class one-vs-rest values only: sensitivity and AUROC are the DME-class recall and AUC).", fontsize=9)
    if UNVER_H:
        fig.legend(handles=UNVER_H, loc="lower center", frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.3, top=0.93)
    save(fig, "F6_contextual_noncomparable")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
