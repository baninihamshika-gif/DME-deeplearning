"""
External validation of the four frozen classifiers on data from outside the
Kermany OCT2017 release. Nothing is fitted here: checkpoints, preprocessing
(utils.PreprocessCache), the 0.5 threshold and the refit temperature/threshold
(script 12, fitted on the Kermany training-time validation split) are all used
unchanged. Protocol fixed before any external prediction was looked at.

Datasets (not redistributed; downloaded into the git-ignored data/external/):
  1. OCTDL (Kulyabin et al., Sci Data 2024;11:365, doi:10.1038/s41597-024-03182-7;
     Mendeley Data sncdhf53xc v4, CC BY 4.0). Optovue Avanti RTVue XR.
     DME (147 B-scans, 107 patients) vs NO = normal (332 B-scans, 110 patients);
     the other OCTDL classes are excluded (task is DME vs Normal, as in training).
     Expected at data/external/octdl/{OCTDL_labels.csv, OCTDL/<class>/*.jpg}.
  2. Duke Srinivasan 2014 (Srinivasan et al., BOE 2014;5(10):3568-77;
     people.duke.edu/~sf59; research/education use, no redistribution).
     Spectralis; 15 DME vs 15 normal volumes (one per subject; the 15 AMD
     volumes are excluded). Labels are per volume, and DME is not visible on
     every B-scan, so the primary analysis is volume level: volume score = max
     P(DME) over its B-scans (presence rule, as in Tang 2021); mean P(DME) is a
     secondary rule. Expected at data/external/duke2014/ (unzipped archive).
  3. Duke DME (Chiu et al., BOE 2015), already local for the segmentation
     audit. Spectralis; 10 DME subjects, 11 fluid-graded B-scans each. No normal
     eyes, so only sensitivity is estimable: positive B-scan = grader-1 fluid
     present (manualFluid1). Fluid-free B-scans of DME eyes are reported
     descriptively, never as negatives.

Analyses: image-level metrics at (a) raw P(DME) >= 0.5 and (b) the refit
temperature-scaled threshold, with patient-cluster bootstrap CIs (B=2000,
seed 42; images within a patient are correlated); AUROC with DeLong
(image-level, ignores clustering) and cluster-bootstrap CIs; patient-level
OCTDL analysis with patient score = max P(DME) over that patient's images.

Outputs (research_outputs/data/): external_predictions_octdl.csv,
external_predictions_duke2015.csv, external_validation_performance.csv,
external_validation_patient_level.csv, external_validation_duke2015.csv.

Usage (project environment): python research_outputs/scripts/13_external_validation.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import config  # noqa: E402
from utils import PreprocessCache  # noqa: E402

DATA = REPO / "research_outputs" / "data"
CACHE = REPO / "research_outputs" / ".cache" / "preprocess_external"
DUKE_PNG = REPO / "research_outputs" / ".cache" / "external_duke2015_png"
OCTDL = REPO / "data" / "external" / "octdl"
DUKE = REPO / "data" / "raw" / "duke_dme" / "2015_BOE_Chiu"
DUKE14 = REPO / "data" / "external" / "duke2014"
CHECKPOINTS = {
    "EfficientNet-B3": "artifacts/kaggle/20260917T035646Z/artifacts/checkpoint_best.pt",
    "EfficientNet-B0": "artifacts/kaggle/20260920T092347Z/artifacts/checkpoint_best.pt",
    "ResNet-50": "artifacts/kaggle/20260920T110729Z/artifacts/checkpoint_best.pt",
    "EfficientNet-B3 + LS 0.05 (ablation)": "artifacts/kaggle/20260922T160023Z/artifacts/checkpoint_best.pt",
}
spec = importlib.util.spec_from_file_location("cm", REPO / "research_outputs" / "scripts" / "04_compute_metrics.py")
cm = importlib.util.module_from_spec(spec); spec.loader.exec_module(cm)


def softmax_pos(logits, t=1.0):
    z = logits / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


def octdl_frame():
    lab = pd.read_csv(OCTDL / "OCTDL_labels.csv", dtype={"patient_id": str})
    lab = lab[lab.disease.isin(["DME", "NO"])].copy()
    lab["path"] = lab.apply(lambda r: str(OCTDL / "OCTDL" / r.disease / f"{r.file_name}.jpg"), axis=1)
    missing = [p for p in lab.path if not Path(p).exists()]
    assert not missing, f"{len(missing)} OCTDL images missing, e.g. {missing[:2]}"
    assert not set(lab[lab.disease == "DME"].patient_id) & set(lab[lab.disease == "NO"].patient_id)
    lab["label"] = (lab.disease == "DME").astype(int)
    lab["patient"] = lab.patient_id
    assert (lab.label == 1).sum() == 147 and (lab.label == 0).sum() == 332
    return lab[["path", "file_name", "patient", "label", "condition", "subcategory"]].reset_index(drop=True)


def duke2014_frame():
    import re
    rows = []
    for d in sorted(p for p in DUKE14.rglob("*") if p.is_dir() and re.fullmatch(r"(DME|NORMAL)\d+", p.name)):
        files = sorted(f for f in d.rglob("*") if f.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg"))
        assert files, f"no images under {d}"
        cls, num = re.fullmatch(r"(DME|NORMAL)(\d+)", d.name).groups()
        for f in files:
            rows.append({"path": str(f), "file_name": f.relative_to(DUKE14).as_posix(), "volume": d.name, "patient": d.name, "label": int(cls == "DME")})
    df = pd.DataFrame(rows)
    nvol = df.groupby("label").volume.nunique().to_dict()
    assert nvol == {0: 15, 1: 15}, f"expected 15 DME + 15 NORMAL volumes, got {nvol}"
    return df


def duke_frame():
    import cv2
    import segment as S
    DUKE_PNG.mkdir(parents=True, exist_ok=True)
    rows, subjects = [], {}
    for e in S.build_duke_index(DUKE):
        out = DUKE_PNG / f"{e['subject']}_scan{e['scan_idx']:02d}.png"
        if e["mat_path"] not in subjects:
            subjects.clear(); subjects[e["mat_path"]] = S.load_duke_subject(e["mat_path"])
        sub = subjects[e["mat_path"]]
        img = np.asarray(sub["images"][e["pos"]])
        if not out.exists():
            assert img.dtype == np.uint8, img.dtype
            cv2.imwrite(str(out), img)
        f1 = sub["fluid1"][e["pos"]]
        rows.append({"path": str(out), "file_name": out.name, "subject": e["subject"], "patient": e["subject"], "scan_idx": e["scan_idx"],
                     "fluid_present_grader1": int(S.binarize_fluid(f1).sum() > 0)})
    return pd.DataFrame(rows)


def _warm(path):
    PreprocessCache(cache_dir=CACHE, image_size=config.IMAGE_SIZE).get_or_compute(path)
    return path


def infer(df, bs=64):
    from concurrent.futures import ProcessPoolExecutor

    import timm
    import torch
    with ProcessPoolExecutor(max_workers=10) as ex:
        list(ex.map(_warm, df.path.tolist(), chunksize=16))
    cache = PreprocessCache(cache_dir=CACHE, image_size=config.IMAGE_SIZE)
    mean = np.array((0.485, 0.456, 0.406), dtype=np.float32); std = np.array((0.229, 0.224, 0.225), dtype=np.float32)
    paths = df.path.tolist()

    def batches():
        for i in range(0, len(paths), bs):
            x = (np.stack([cache.get(p)[0] for p in paths[i:i + bs]]).astype(np.float32) / 255.0 - mean) / std
            yield torch.from_numpy(x.transpose(0, 3, 1, 2).copy())

    out = {}
    for name, rel in CHECKPOINTS.items():
        ckpt = torch.load(REPO / rel, map_location="cpu", weights_only=False)
        model = timm.create_model(ckpt.get("model_name", "tf_efficientnet_b3"), pretrained=False, num_classes=2, drop_rate=config.DROP_RATE)
        model.load_state_dict(ckpt["model_state_dict"]); model.eval()
        with torch.no_grad():
            out[name] = np.concatenate([model(x).numpy() for x in batches()])
        print(f"  {name}: {len(out[name])} images scored", flush=True)
    return out


def main():
    refit = pd.read_csv(DATA / "refit_calibration_threshold.csv").set_index("model")
    perf, pat_rows, duke_rows = [], [], []

    # ---------------- OCTDL: DME vs Normal ----------------
    od = octdl_frame()
    print(f"OCTDL: {len(od)} images ({int(od.label.sum())} DME / {int((od.label == 0).sum())} normal), {od.patient.nunique()} patients")
    logits = infer(od)
    pred = od[["file_name", "patient", "label", "condition", "subcategory"]].copy()
    for name, lg in logits.items():
        t, thr = refit.loc[name, "temperature_refit"], refit.loc[name, "threshold_refit"]
        y = od.label.to_numpy(); g = od.patient.to_numpy()
        p_raw = softmax_pos(lg); p_cal = softmax_pos(lg, t)
        pred[f"p_dme_raw__{name}"] = p_raw; pred[f"p_dme_refit__{name}"] = p_cal
        src = "research_outputs/data/external_predictions_octdl.csv"
        for r in cm.evaluate_set(name, y, p_raw, [("default 0.5 (raw softmax)", 0.5)], groups=g, source=src):
            perf.append({"dataset": "OCTDL (Optovue), image level", **r})
        for r in cm.evaluate_set(name, y, p_cal, [(f"refit tuned {thr} (temperature {t:.4f}; fitted on Kermany validation)", thr)], groups=g, source=src):
            perf.append({"dataset": "OCTDL (Optovue), image level", **r})
        # patient level: max P(DME) over the patient's images
        pp = pd.DataFrame({"patient": g, "y": y, "p": p_raw}).groupby("patient").agg(y=("y", "max"), p=("p", "max"))
        for r in cm.evaluate_set(name, pp.y.to_numpy(), pp.p.to_numpy(), [("default 0.5 (raw softmax), patient = max over images", 0.5)], source=src):
            pat_rows.append({"dataset": "OCTDL (Optovue), patient level", **r})
    pred.to_csv(DATA / "external_predictions_octdl.csv", index=False)

    # ---------------- Duke Srinivasan 2014: DME vs Normal volumes ----------------
    d14 = duke2014_frame()
    print(f"Duke 2014: {len(d14)} B-scans in {d14.volume.nunique()} volumes ({d14[d14.label == 1].volume.nunique()} DME / {d14[d14.label == 0].volume.nunique()} normal)")
    logits = infer(d14)
    p14 = d14[["file_name", "volume", "label"]].copy()
    src = "research_outputs/data/external_predictions_duke2014.csv"
    for name, lg in logits.items():
        t, thr = refit.loc[name, "temperature_refit"], refit.loc[name, "threshold_refit"]
        p_raw = softmax_pos(lg); p_cal = softmax_pos(lg, t)
        p14[f"p_dme_raw__{name}"] = p_raw; p14[f"p_dme_refit__{name}"] = p_cal
        v = pd.DataFrame({"volume": d14.volume, "y": d14.label, "raw": p_raw, "cal": p_cal}).groupby("volume").agg(
            y=("y", "first"), raw_max=("raw", "max"), cal_max=("cal", "max"), raw_mean=("raw", "mean"), cal_mean=("cal", "mean"))
        y = v.y.to_numpy()
        for rule, col_raw, col_cal in (("max over B-scans (primary)", "raw_max", "cal_max"), ("mean over B-scans (secondary)", "raw_mean", "cal_mean")):
            for r in cm.evaluate_set(name, y, v[col_raw].to_numpy(), [(f"default 0.5 (raw softmax), volume = {rule}", 0.5)], source=src):
                perf.append({"dataset": "Duke 2014 (Spectralis), volume level", **r})
            for r in cm.evaluate_set(name, y, v[col_cal].to_numpy(), [(f"refit tuned {thr} (temperature {t:.4f}), volume = {rule}", thr)], source=src):
                perf.append({"dataset": "Duke 2014 (Spectralis), volume level", **r})
    p14.to_csv(DATA / "external_predictions_duke2014.csv", index=False)

    # ---------------- Duke 2015: sensitivity only ----------------
    dk = duke_frame()
    print(f"Duke 2015: {len(dk)} graded B-scans, {dk.subject.nunique()} DME subjects, fluid present in {int(dk.fluid_present_grader1.sum())}")
    logits = infer(dk)
    dpred = dk[["file_name", "subject", "scan_idx", "fluid_present_grader1"]].copy()
    for name, lg in logits.items():
        t, thr = refit.loc[name, "temperature_refit"], refit.loc[name, "threshold_refit"]
        p_raw = softmax_pos(lg); p_cal = softmax_pos(lg, t)
        dpred[f"p_dme_raw__{name}"] = p_raw; dpred[f"p_dme_refit__{name}"] = p_cal
        pos = dk.fluid_present_grader1.to_numpy() == 1
        for op, p, th in (("default 0.5 (raw softmax)", p_raw, 0.5), (f"refit tuned {thr}", p_cal, thr)):
            k = int((p[pos] >= th).sum()); n = int(pos.sum())
            subj = pd.DataFrame({"s": dk.subject, "p": p}).groupby("s").p.max()
            ks = int((subj >= th).sum()); ns = len(subj)
            neg_rate = float((p[~pos] >= th).mean()) if (~pos).any() else np.nan
            duke_rows.append({"dataset": "Duke DME 2015 (Spectralis), DME eyes only", "model": name, "threshold_name": op, "threshold": th,
                              "fluid_bscans_n": n, "fluid_bscans_called_dme": k, "sensitivity_fluid_bscans": k / n,
                              "sensitivity_ci_lo": cm.wilson(k, n)[0], "sensitivity_ci_hi": cm.wilson(k, n)[1],
                              "subjects_n": ns, "subjects_called_dme_max_rule": ks, "subject_sensitivity": ks / ns,
                              "subject_sensitivity_ci_lo": cm.wilson(ks, ns)[0], "subject_sensitivity_ci_hi": cm.wilson(ks, ns)[1],
                              "fluid_free_bscans_n": int((~pos).sum()), "fluid_free_bscans_called_dme_fraction": neg_rate,
                              "note": "No normal eyes: specificity not estimable. Fluid-free B-scans come from DME eyes and are descriptive only. Wilson CIs ignore within-subject clustering for the B-scan figure."})
    dpred.to_csv(DATA / "external_predictions_duke2015.csv", index=False)

    pd.DataFrame(perf).to_csv(DATA / "external_validation_performance.csv", index=False)
    pd.DataFrame(pat_rows).to_csv(DATA / "external_validation_patient_level.csv", index=False)
    pd.DataFrame(duke_rows).to_csv(DATA / "external_validation_duke2015.csv", index=False)
    cols = ["dataset", "model", "threshold", "TP", "FP", "TN", "FN", "sensitivity", "specificity", "accuracy", "auroc", "auroc_cluster_bootstrap_ci_lo", "auroc_cluster_bootstrap_ci_hi"]
    pd.set_option("display.width", 250)
    print(pd.DataFrame(perf)[cols].to_string(index=False))
    print(pd.DataFrame(pat_rows)[["model", "TP", "FP", "TN", "FN", "sensitivity", "specificity", "auroc", "auroc_ci_lo", "auroc_ci_hi"]].to_string(index=False))
    print(pd.DataFrame(duke_rows)[["model", "threshold", "fluid_bscans_called_dme", "fluid_bscans_n", "sensitivity_fluid_bscans", "subjects_called_dme_max_rule", "fluid_free_bscans_called_dme_fraction"]].to_string(index=False))


def posthoc_duke2014():
    """POST HOC (added after the primary Duke 2014 results were seen): the max rule
    flagged normal volumes because off-centre B-scans score high, and Kermany
    trains only on foveal B-scans. Re-scores each volume by its central B-scan
    (index (n+1)//2, a proxy for the fovea) and summarises B-scan-level rates.
    Reads the saved predictions; no new inference."""
    from sklearn.metrics import roc_auc_score
    d = pd.read_csv(DATA / "external_predictions_duke2014.csv")
    d["idx"] = d.file_name.str.extract(r"/(\d+)\.tif", expand=False).astype(int)
    d["n"] = d.groupby("volume").idx.transform("max")
    c = d[d.idx == (d.n + 1) // 2]
    assert len(c) == 30
    rows = []
    for col in [x for x in d.columns if x.startswith("p_dme_raw__")]:
        y, p = c.label.to_numpy(), c[col].to_numpy()
        tp, fp = int(((p >= 0.5) & (y == 1)).sum()), int(((p >= 0.5) & (y == 0)).sum())
        rows.append({"model": col.split("__")[1], "analysis": "POST HOC: central B-scan per volume, raw 0.5",
                     "TP": tp, "FN": 15 - tp, "FP": fp, "TN": 15 - fp, "sensitivity": tp / 15, "specificity": (15 - fp) / 15,
                     "sensitivity_ci_lo": cm.wilson(tp, 15)[0], "sensitivity_ci_hi": cm.wilson(tp, 15)[1],
                     "specificity_ci_lo": cm.wilson(15 - fp, 15)[0], "specificity_ci_hi": cm.wilson(15 - fp, 15)[1],
                     "auroc": roc_auc_score(y, p),
                     "all_bscans_normal_called_dme_fraction": float((d.loc[d.label == 0, col] >= 0.5).mean()),
                     "all_bscans_dme_volume_called_dme_fraction": float((d.loc[d.label == 1, col] >= 0.5).mean())})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "external_validation_duke2014_posthoc.csv", index=False)
    print(out[["model", "TP", "FP", "sensitivity", "specificity", "auroc", "all_bscans_normal_called_dme_fraction", "all_bscans_dme_volume_called_dme_fraction"]].to_string(index=False))


if __name__ == "__main__":
    if "--posthoc-only" not in sys.argv:
        main()
    posthoc_duke2014()
