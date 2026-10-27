"""
Phase D/E -- per-paper comparison data: this project and each of the five
papers side by side, as one long table that every comparison figure and
Table 6 read from. Nothing is typed in; all values come from:
  research_audit/literature_extraction.csv                       published rows
  research_outputs/data/table2_within_project_performance.csv    official test
  research_outputs/data/supplementary_excluded_pool_performance.csv  unseen pool
  research_outputs/data/external_validation_performance.csv      OCTDL, Duke 2014

Project settings (threshold 0.5, raw softmax):
  official_test  Kermany official test split, B-scan, internal (n=484)
  unseen_pool    Kermany unseen pool, B-scan, internal (n=7,503)
  octdl          OCTDL (Optovue), B-scan, external (n=479)
  duke2014       Duke 2014 (Spectralis), volume = max over B-scans, external (n=30)

Published rows: every DME-presence row of the five papers. Tasks whose
positive class is not DME presence (Zhu CI-DME, Li 4-class abnormal) stay in
Table 3 only.

Headline row per paper (fixed rule, chosen before plotting): the internal
held-out result at the B-scan/image unit when the paper reports one (Tang
Suppl. S1 Spectralis, Li Table 6, Singh Table 3); otherwise the paper's
main-text primary result (Zhu Table 3 Testing, Rodriguez-Miguel Table 2
Model 2). All other rows are kept and shown as individual points and as a
min-max range, so no row is dropped.

Not a head-to-head comparison: data, populations, devices, labels and units
differ in every pair; no cross-study statistical test is computed.

Output: research_outputs/data/per_paper_comparison.csv
Usage: python research_outputs/scripts/14_per_paper_comparison.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "research_outputs" / "data"
AUDIT = REPO / "research_audit"
MODELS = ["EfficientNet-B3", "EfficientNet-B0", "ResNet-50"]
PAPERS = ["Tang 2021", "Zhu 2025", "Li F 2019", "Rodríguez-Miguel 2024", "Singh & Gorantla 2020"]
METRICS = ["accuracy", "sensitivity", "specificity", "f1", "auroc"]
SETTING_LABEL = {"official_test": "Kermany official test (B-scan, internal, n=484)",
                 "unseen_pool": "Kermany unseen pool (B-scan, internal, n=7,503)",
                 "octdl": "OCTDL Optovue (B-scan, external, n=479)",
                 "duke2014": "Duke 2014 Spectralis (volume, max rule, external, n=30)"}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def headline_mask(lit):
    s = lit.source_location
    rules = {
        "Tang 2021": s.str.startswith("Supplementary Table S1") & lit.device.str.contains("SPECTRALIS") & lit.cohort_role.str.startswith("primary"),
        "Zhu 2025": (s == "Table 3") & (lit.dataset == "Testing") & (lit.positive_class == "DME"),
        "Li F 2019": s.str.startswith("Table 6"),
        "Rodríguez-Miguel 2024": s.str.startswith("Table 2") & lit.model.str.contains("Model 2"),
        "Singh & Gorantla 2020": s.str.startswith("Table 3"),
    }
    out = pd.Series(False, index=lit.index)
    for paper, m in rules.items():
        m = m & lit.study.str.startswith(paper)
        assert m.sum() == 1, (paper, int(m.sum()))
        out |= m
    return out


def project_rows():
    t2 = pd.read_csv(DATA / "table2_within_project_performance.csv")
    pool = pd.read_csv(DATA / "supplementary_excluded_pool_performance.csv")
    ext = pd.read_csv(DATA / "external_validation_performance.csv")
    sets = {
        "official_test": (t2[(t2.threshold == 0.5) & t2.model.isin(MODELS)], "auroc_ci_lo", "auroc_ci_hi", "table2_within_project_performance.csv"),
        "unseen_pool": (pool[pool.model.isin(MODELS)], "auroc_cluster_bootstrap_ci_lo", "auroc_cluster_bootstrap_ci_hi", "supplementary_excluded_pool_performance.csv"),
        "octdl": (ext[ext.dataset.str.startswith("OCTDL") & (ext.threshold == 0.5) & ext.model.isin(MODELS)], "auroc_ci_lo", "auroc_ci_hi",
                  "external_validation_performance.csv (OCTDL)"),
        "duke2014": (ext[ext.dataset.str.startswith("Duke 2014") & (ext.threshold == 0.5) & ext.threshold_name.str.contains("max over") & ext.model.isin(MODELS)],
                     "auroc_ci_lo", "auroc_ci_hi", "external_validation_performance.csv (Duke 2014, max rule)"),
    }
    out = []
    for key, (d, lo, hi, src) in sets.items():
        assert len(d) == len(MODELS), (key, len(d))
        for _, r in d.iterrows():
            row = {"paper": "This project", "model": r.model, "setting_key": key, "setting": SETTING_LABEL[key],
                   "modality": "OCT", "device": "Spectralis" if key != "octdl" else "Optovue",
                   "unit": "volume" if key == "duke2014" else "image", "role": "external" if key in ("octdl", "duke2014") else "internal",
                   "category": "A", "n": str(int(r.n)), "source": src,
                   "headline": key == "official_test" and r.model == "EfficientNet-B3",
                   "auroc_ci_lo": num(r[lo]), "auroc_ci_hi": num(r[hi])}
            row.update({m: num(r[m]) for m in METRICS})
            out.append(row)
    return out


def published_rows():
    lit = pd.read_csv(AUDIT / "literature_extraction.csv", dtype=str)
    head = headline_mask(lit)
    # comparability category as assigned in Table 3 (same row order; literature_extraction leaves it NR for Tang/Zhu)
    cat = pd.read_csv(REPO / "research_outputs" / "tables" / "table3_literature_comparison.csv", dtype=str)["Comparability category"]
    assert len(cat) == len(lit)
    out = []
    for i, r in lit.iterrows():
        paper = next((p for p in PAPERS if r.study.startswith(p)), None)
        if paper is None or not (r.positive_class.startswith("DME") or r.positive_class.startswith("any DME")):
            continue
        unit_raw = r.evaluation_unit.lower()
        role = r.cohort_role.lower()
        row = {"paper": paper, "model": r.model, "setting_key": "published", "setting": f"{r.dataset} [{r.source_location}]",
               "modality": "fundus photograph" if "fundus" in r.input_representation else "OCT", "device": r.device,
               "unit": "volume" if ("volume" in unit_raw or "cube" in unit_raw) else "image",
               "role": "external" if "external" in role else ("separate cohort" if "screening cohort" in role else "internal"),
               "category": cat[i].split()[0], "n": r.n_evaluated, "source": r.source_location, "headline": bool(head[i]),
               "auroc_ci_lo": num(r.auroc_ci_lo), "auroc_ci_hi": num(r.auroc_ci_hi)}
        row.update({m: num(r[m]) for m in METRICS})
        out.append(row)
    return out


def main():
    df = pd.DataFrame(project_rows() + published_rows())
    df.to_csv(DATA / "per_paper_comparison.csv", index=False)
    print(df.groupby("paper", sort=False).agg(rows=("model", "size"), headline=("headline", "sum")).to_string())


if __name__ == "__main__":
    main()
