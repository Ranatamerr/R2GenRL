"""
Convert MIMIC-CXR parquet files into the directory structure expected by R2GenRL:
  data/mimic_cxr/images/<id>.jpg
  data/mimic_cxr/annotation.json  (keys: train / val / test)

Each annotation entry:
  { "id": str, "image_path": [str], "report": str }

Split: 70% train / 15% val / 15% test  (reproducible with fixed seed)
Report text: findings + impression (both present), or whichever is non-empty.
"""

import os
import json
import random
import pandas as pd
from pathlib import Path

PARQUET_FILES = [
    r"D:\Sem8\Bachelor\Data Sets\archive\train-00000-of-00002.parquet",
    r"D:\Sem8\Bachelor\Data Sets\archive\train-00001-of-00002.parquet",
]
OUT_IMG_DIR = Path("data/mimic_cxr/images")
OUT_ANN_PATH = Path("data/mimic_cxr/annotation.json")
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test gets the remainder


def build_report(findings: str, impression: str) -> str:
    parts = []
    if isinstance(findings, str) and findings.strip():
        parts.append(findings.strip())
    if isinstance(impression, str) and impression.strip():
        parts.append(impression.strip())
    return " ".join(parts)


def main():
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading parquet files...")
    dfs = [pd.read_parquet(p) for p in PARQUET_FILES]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(df)}")

    # Drop rows with no usable report
    df["report"] = df.apply(lambda r: build_report(r["findings"], r["impression"]), axis=1)
    before = len(df)
    df = df[df["report"].str.strip() != ""].reset_index(drop=True)
    print(f"Rows with non-empty report: {len(df)}  (dropped {before - len(df)})")

    # Shuffle and split
    indices = list(range(len(df)))
    random.seed(SEED)
    random.shuffle(indices)

    n_train = int(len(indices) * TRAIN_RATIO)
    n_val   = int(len(indices) * VAL_RATIO)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]
    print(f"Split — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    annotation = {"train": [], "val": [], "test": []}

    def process_split(idxs, split_name):
        print(f"Extracting {split_name} images...")
        for pos, i in enumerate(idxs):
            row = df.iloc[i]
            img_name = f"{i:06d}.jpg"
            img_path = OUT_IMG_DIR / img_name
            if not img_path.exists():
                img_path.write_bytes(row["image"])
            annotation[split_name].append({
                "id": str(i),
                "image_path": [img_name],
                "report": row["report"],
            })
            if (pos + 1) % 2000 == 0:
                print(f"  {split_name}: {pos + 1}/{len(idxs)}")

    process_split(train_idx, "train")
    process_split(val_idx,   "val")
    process_split(test_idx,  "test")

    OUT_ANN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_ANN_PATH, "w") as f:
        json.dump(annotation, f)
    print(f"Saved annotation to {OUT_ANN_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
