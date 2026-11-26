"""
Standalone, read-only diagnostic: record (path, width, height, label) for
every image in data/raw/OCT2017/test/{DME,NORMAL}, the same way
prepare_data.py's scan_dims() does for train (PIL.Image.open().size reads
just the file header, no full decode -- fast and 100% safe, no writes to
any image file).

Run from the DME-DEEPLEARN repo root:
    python scan_test_dims.py

Writes artifacts/test_dims_cache.csv (path,width,height,label), analogous
to the existing artifacts/image_dims_cache.csv but for the test split.
Nothing else is touched.
"""
import csv
from pathlib import Path

from PIL import Image

TEST_DIR = Path("data/raw/OCT2017/test")
OUT_PATH = Path("artifacts/test_dims_cache.csv")

FOLDER_TO_LABEL = {"DME": "DME", "NORMAL": "Normal"}
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


def main():
    rows = []
    for folder_name, label in FOLDER_TO_LABEL.items():
        class_dir = TEST_DIR / folder_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class folder not found: {class_dir}")
        paths = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        for p in paths:
            with Image.open(p) as img:
                w, h = img.size
            rows.append({"path": str(p).replace("\\", "/"), "width": w, "height": h, "label": label})
        print(f"[scan_test_dims] {folder_name}: {len(paths)} images")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "width", "height", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[scan_test_dims] wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
