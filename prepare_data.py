"""
Phase 1 orchestration for the DME OCT Detection project.

Builds the patient-grouped train/validation split (C1), curates
data/sample/ for local development, and renders fig_preprocessing_stages.png.
Contains no algorithms of its own -- everything here calls into utils.py /
config.py, per PROJECT_BRIEF.md Section 7 ("notebooks and entry.py contain
no logic. All logic lives in importable modules under version control.").

Usage:
    python prepare_data.py --train-dir data/raw/OCT2017/train \
                            --test-dir  data/raw/OCT2017/test

Cannot produce real output until the Kermany OCT2017 dataset (the
83,484-train-image release -- see PROJECT_BRIEF.md Section 5) is present
under --train-dir / --test-dir.
"""

import argparse
import csv
import random
import time
from pathlib import Path

import cv2
import matplotlib
from PIL import Image

matplotlib.use("Agg")  # headless -- this script writes files, never shows plots
import matplotlib.pyplot as plt

import config
from utils import build_splits, flatten_retina, preprocess_image, set_seed


def _image_size(path: Path):
    """
    (width, height) of an image on disk, read from the file header only
    (PIL.Image.open() is lazy -- it doesn't decode pixel data until you
    access them). Scanning ~38k images for the widest/narrowest with a
    full cv2.imread decode each time was the dominant cost in this script;
    this is the same information for a fraction of the I/O.
    """
    with Image.open(path) as img:
        return img.size  # (width, height)


def _load_dims_cache(cache_path: Path) -> dict:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return {}
    dims = {}
    with open(cache_path, newline="") as f:
        for row in csv.DictReader(f):
            dims[row["path"]] = (int(row["width"]), int(row["height"]))
    return dims


def _load_corrupt_set(corrupt_path: Path) -> set:
    corrupt_path = Path(corrupt_path)
    if not corrupt_path.exists():
        return set()
    return {line.strip() for line in corrupt_path.read_text().splitlines() if line.strip()}


def scan_dims(paths, cache_path, time_budget_seconds: float = 150.0) -> dict:
    """
    Return {path: (width, height)} for every path in `paths`, persisting
    results to `cache_path` (a CSV) as it goes.

    Reading image dimensions one file at a time over a slow (e.g.
    network-mounted) filesystem for ~38k images does not fit in one
    invocation. This stops after `time_budget_seconds` -- whatever wasn't
    scanned yet stays unscanned, not guessed -- and can simply be called
    again (e.g. by re-running this script) to pick up where it left off,
    since already-cached paths are skipped.

    Also doubles as an integrity pass: a file PIL can't open (e.g. a
    zero-byte file left over from an interrupted copy) is logged to
    "<cache_path>.corrupt.txt" and skipped, rather than crashing the whole
    scan -- one bad file out of ~38k shouldn't block Phase 1. Corrupt
    paths already logged are skipped on subsequent calls too (they're not
    counted as "done" until fixed and re-scanned, so they keep surfacing
    in the incomplete-scan count rather than silently vanishing).
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path = cache_path.with_suffix(cache_path.suffix + ".corrupt.txt")
    dims = _load_dims_cache(cache_path)
    known_corrupt = _load_corrupt_set(corrupt_path)
    remaining = [p for p in paths if p not in dims and p not in known_corrupt]

    newly_corrupt = []
    if remaining:
        start = time.time()
        write_header = not cache_path.exists() or cache_path.stat().st_size == 0
        with open(cache_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["path", "width", "height"])
            for p in remaining:
                if time.time() - start > time_budget_seconds:
                    break
                try:
                    with Image.open(p) as img:
                        w, h = img.size
                except Exception as exc:
                    newly_corrupt.append(p)
                    print(f"[scan_dims] CORRUPT (skipped): {p} ({exc})")
                    continue
                writer.writerow([p, w, h])
                dims[p] = (w, h)

    if newly_corrupt:
        with open(corrupt_path, "a") as f:
            for p in newly_corrupt:
                f.write(p + "\n")
        known_corrupt |= set(newly_corrupt)

    # Counted against the current `paths` list, not the raw cache size --
    # the cache file can hold entries for a larger/different path set from
    # an earlier call (e.g. before a C8 exclusion shrank the pool), and
    # "412/300 scanned" would be a confusing thing to print.
    done = sum(1 for p in paths if p in dims)
    total = len(paths)
    if known_corrupt:
        print(f"[scan_dims] {len(known_corrupt)} corrupt file(s) logged to {corrupt_path} -- "
              "re-extract these from the source archive, then re-run to pick them up.")
    if done + len(known_corrupt) < total:
        print(f"[scan_dims] {done}/{total} image dimensions cached so far "
              f"({len(known_corrupt)} corrupt) -- stopped after {time_budget_seconds:.0f}s, re-run to continue.")
    else:
        print(f"[scan_dims] {done}/{total} image dimensions cached, {len(known_corrupt)} corrupt (scan complete).")
    return dims


def curate_sample(
    train_df,
    sample_dir,
    per_class: int = 15,
    n_pathology: int = 8,
    min_multi_scan_patients: int = 3,
    max_pathology_candidates_scanned: int = 4000,
    seed: int = config.SEED,
    dims_cache=None,
) -> list:
    """
    Curate a local-dev sample from `train_df`:
      - `per_class` images per class (DME, Normal), randomly sampled
      - the widest and narrowest image in each class
      - up to `n_pathology` images where flatten_retina's fallback triggers
        -- a concrete, automatic proxy for "visible pathology" that the
        brief calls for (an image the flattening heuristic can't handle
        cleanly) -- found by scanning at most `max_pathology_candidates_scanned`
        remaining images (each requires a real decode + flatten attempt,
        unlike the header-only widest/narrowest scan, so this is bounded)
      - extra scans added, if needed, so at least `min_multi_scan_patients`
        distinct patients contribute more than one image to the sample

    Copies selected files into `sample_dir` and returns the manifest rows
    (path, class, patient, width, height, reason); the caller writes
    manifest.csv. NOTE: the brief's "~60 curated images" is an approximate
    target reached through these overlapping criteria (a "widest" pick can
    already be one of the `per_class` random picks), not a hard total --
    the actual count is reported honestly, not padded to hit 60.

    `dims_cache`, if given, is a {path: (width, height)} dict (e.g. from
    scan_dims()) consulted before falling back to a direct read -- avoids
    re-reading every image in `train_df` just to find the widest/narrowest
    when the caller has already scanned them once.
    """
    rng = random.Random(seed)
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    dims_cache = dims_cache or {}

    def size_of(path) -> tuple:
        path = str(path)
        if path in dims_cache:
            return dims_cache[path]
        return _image_size(Path(path))

    selected = {}  # path -> manifest row

    def add(row, reason):
        path = row["path"]
        if path in selected:
            selected[path]["reason"] += f"+{reason}"
        else:
            w, h = size_of(path)
            selected[path] = {
                "path": path,
                "class": row["label"],
                "patient": row["patient"],
                "width": w,
                "height": h,
                "reason": reason,
            }

    for label in config.CLASS_NAMES:
        class_rows = train_df[train_df["label"] == label].to_dict("records")
        if not class_rows:
            continue

        for row in rng.sample(class_rows, k=min(per_class, len(class_rows))):
            add(row, "base")

        dims = [(size_of(r["path"]), r) for r in class_rows]
        _, widest_row = max(dims, key=lambda item: item[0][0])
        _, narrowest_row = min(dims, key=lambda item: item[0][0])
        add(widest_row, "widest")
        add(narrowest_row, "narrowest")

    pathology_candidates = [r for r in train_df.to_dict("records") if r["path"] not in selected]
    rng.shuffle(pathology_candidates)
    pathology_found = 0
    scanned = 0
    for row in pathology_candidates:
        if pathology_found >= n_pathology or scanned >= max_pathology_candidates_scanned:
            break
        scanned += 1
        gray = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        _, fallback = flatten_retina(gray, enabled=True)
        if fallback:
            add(row, "pathology_fallback")
            pathology_found += 1
    if pathology_found < n_pathology:
        print(
            f"[curate_sample] found only {pathology_found}/{n_pathology} flatten-fallback "
            f"(\"pathology\") examples after scanning {scanned} candidates -- "
            "increase max_pathology_candidates_scanned to look harder."
        )

    patient_counts = {}
    for row in selected.values():
        patient_counts[row["patient"]] = patient_counts.get(row["patient"], 0) + 1
    multi_scan_patients = {p for p, n in patient_counts.items() if n > 1}

    if len(multi_scan_patients) < min_multi_scan_patients:
        by_patient = {}
        for row in train_df.to_dict("records"):
            by_patient.setdefault(row["patient"], []).append(row)

        # Patients who *could* become multi-scan in the sample: they have
        # more than one image in the full dataset. Prefer ones already
        # partially selected (costs one extra image) before pulling in a
        # patient with none of their images selected yet (costs two).
        partially_selected = [
            p for p, imgs in by_patient.items()
            if len(imgs) > 1 and p in patient_counts and p not in multi_scan_patients
        ]
        not_yet_selected = [
            p for p, imgs in by_patient.items()
            if len(imgs) > 1 and p not in patient_counts
        ]
        rng.shuffle(partially_selected)
        rng.shuffle(not_yet_selected)

        for patient in partially_selected + not_yet_selected:
            if len(multi_scan_patients) >= min_multi_scan_patients:
                break
            imgs = by_patient[patient]
            if patient in patient_counts:
                extra = [r for r in imgs if r["path"] not in selected]
                if not extra:
                    continue
                add(extra[0], "multi_scan")
            else:
                for r in imgs[:2]:
                    add(r, "multi_scan")
            multi_scan_patients.add(patient)

    for row in selected.values():
        dest = sample_dir / Path(row["path"]).name
        if not dest.exists():
            dest.write_bytes(Path(row["path"]).read_bytes())

    return sorted(selected.values(), key=lambda r: r["path"])


def write_manifest(rows, sample_dir) -> Path:
    manifest_path = Path(sample_dir) / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "class", "patient", "width", "height", "reason"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def generate_preprocessing_figure(example_path, out_path, dpi: int = 150) -> Path:
    """One scan through every preprocessing stage, saved as a labelled row of panels."""
    original = cv2.imread(str(example_path), cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"Could not read image: {example_path}")

    denoised = cv2.fastNlMeansDenoising(original, h=10)
    flattened, fallback = flatten_retina(denoised, enabled=True)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(flattened)
    rgb, _content_bbox, _ = preprocess_image(original, flatten=True)

    stages = [
        ("Original", original),
        ("Denoised", denoised),
        (f"Flattened{' (fallback)' if fallback else ''}", flattened),
        ("CLAHE", enhanced),
        ("Letterboxed", rgb),
    ]

    fig, axes = plt.subplots(1, len(stages), figsize=(4 * len(stages), 4))
    for ax, (title, img) in zip(axes, stages):
        ax.imshow(img, cmap=None if img.ndim == 3 else "gray")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: build splits, curate data/sample/, render the preprocessing figure."
    )
    parser.add_argument("--train-dir", type=Path, required=True,
                         help="Kermany OCT2017 train/ folder (contains DME/, NORMAL/, ...)")
    parser.add_argument("--test-dir", type=Path, default=None,
                         help="Kermany OCT2017 test/ folder (enables the C8 overlap check)")
    parser.add_argument("--sample-dir", type=Path, default=config.SAMPLE_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    parser.add_argument("--dims-cache", type=Path, default=None,
                         help="CSV to persist image-dimension scan progress (default: <artifacts-dir>/image_dims_cache.csv)")
    parser.add_argument("--time-budget", type=float, default=150.0,
                         help="Seconds to spend scanning image dimensions before stopping (re-run to continue)")
    args = parser.parse_args()

    set_seed(config.SEED)

    result = build_splits(args.train_dir, test_dir=args.test_dir)

    print(f"\nSplit sizes: train={len(result.train_df)} val={len(result.val_df)}")
    print(f"Train patients: {result.train_df['patient'].nunique()}  "
          f"Val patients: {result.val_df['patient'].nunique()}")
    print(f"Train class balance: {result.train_df['label'].value_counts().to_dict()}")
    print(f"Val class balance: {result.val_df['label'].value_counts().to_dict()}")
    print(f"Patient overlap train<->val: {result.train_val_patient_overlap}")
    print(f"Patient overlap train/val<->test: {result.test_patient_overlap}")

    dims_cache_path = args.dims_cache or (args.artifacts_dir / "image_dims_cache.csv")
    all_train_paths = result.train_df["path"].tolist()
    dims_cache = scan_dims(all_train_paths, dims_cache_path, time_budget_seconds=args.time_budget)

    corrupt_path = dims_cache_path.with_suffix(dims_cache_path.suffix + ".corrupt.txt")
    known_corrupt = _load_corrupt_set(corrupt_path)
    if len(dims_cache) + len(known_corrupt) < len(all_train_paths):
        print("\nImage-dimension scan incomplete -- re-run this exact command to continue "
              "(already-scanned images are cached and skipped). Stopping before curate_sample.")
        return

    if known_corrupt:
        print(f"\n{len(known_corrupt)} corrupt file(s) excluded from curation: {sorted(known_corrupt)[:10]}")
        clean_train_df = result.train_df[~result.train_df["path"].isin(known_corrupt)].reset_index(drop=True)
    else:
        clean_train_df = result.train_df

    rows = curate_sample(clean_train_df, args.sample_dir, dims_cache=dims_cache)
    manifest_path = write_manifest(rows, args.sample_dir)
    print(f"\nCurated {len(rows)} images into {args.sample_dir} ({manifest_path.name} written)")

    fallback_count = sum(1 for r in rows if "pathology_fallback" in r["reason"])
    print(f"Flattening fallback rate in sample: {fallback_count}/{len(rows)}")

    if rows:
        fig_path = generate_preprocessing_figure(
            Path(rows[0]["path"]), args.artifacts_dir / "fig_preprocessing_stages.png"
        )
        print(f"Preprocessing figure written to {fig_path}")


if __name__ == "__main__":
    main()
