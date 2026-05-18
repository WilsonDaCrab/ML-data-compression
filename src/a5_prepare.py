"""
A5: build stratified train/val/test splits.

Merges features with the registry (to get the type_group strata), splits
70/15/15 stratified by type_group with random_state=123, and writes one NPZ
per split plus per-split benchmark CSVs.

Usage:
    python src/a5_prepare.py
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 123


def main():
    parser = argparse.ArgumentParser(description="A5: prepare ML dataset splits.")
    parser.add_argument("--registry", default="data/file_registry.csv")
    parser.add_argument("--features", default="data/features.csv")
    parser.add_argument("--benchmarks", default="data/benchmarks.csv")
    parser.add_argument("--out", default="data/splits")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    features_path = Path(args.features)
    benchmarks_path = Path(args.benchmarks)
    out_dir = Path(args.out)

    for path in [registry_path, features_path, benchmarks_path]:
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    print("Loading registry ...")
    reg = pd.read_csv(registry_path, dtype={"sha256": str, "type_group": str})

    print("Loading features ...")
    feat = pd.read_csv(features_path, dtype={"sha256": str})

    print("Loading benchmarks ...")
    bench = pd.read_csv(benchmarks_path, dtype={"sha256": str, "algo_id": str,
                                                "cr": float, "comp_mbps": float,
                                                "decomp_mbps": float})

    print(f"  Registry   : {len(reg):,} files")
    print(f"  Features   : {len(feat):,} rows")
    print(f"  Benchmarks : {len(bench):,} rows")

    print("\nMerging features with registry ...")
    df = feat.merge(reg[["sha256", "type_group"]], on="sha256", how="inner")

    if len(df) != len(feat):
        print(f"WARNING: {len(feat) - len(df)} feature rows dropped (no registry match)",
              file=sys.stderr)

    feature_cols = [c for c in df.columns if c not in ("sha256", "type_group")]

    print(f"  Merged rows    : {len(df):,}")
    print(f"  Feature columns: {len(feature_cols)}")

    print("\nSplitting dataset (stratified by type_group, random_state=123) ...")

    sha256s = df["sha256"].values
    stratum = df["type_group"].values

    # 70% train / 30% rest
    idx_train, idx_rest = train_test_split(
        range(len(df)),
        test_size=0.30,
        stratify=stratum,
        random_state=RANDOM_STATE,
    )

    # split the rest 50/50 -> val/test
    stratum_rest = stratum[idx_rest]
    idx_val, idx_test = train_test_split(
        idx_rest,
        test_size=0.50,
        stratify=stratum_rest,
        random_state=RANDOM_STATE,
    )

    splits = {"train": idx_train, "val": idx_val, "test": idx_test}

    print(f"  Train : {len(idx_train):,}  ({len(idx_train)/len(df)*100:.1f}%)")
    print(f"  Val   : {len(idx_val):,}  ({len(idx_val)/len(df)*100:.1f}%)")
    print(f"  Test  : {len(idx_test):,}  ({len(idx_test)/len(df)*100:.1f}%)")

    print("\n  Type group distribution across splits:")
    header = f"  {'type_group':<22}  {'total':>6}  {'train':>6}  {'val':>6}  {'test':>6}"
    print(header)
    for tg in sorted(set(stratum)):
        total = (stratum == tg).sum()
        tr = sum(1 for i in idx_train if stratum[i] == tg)
        va = sum(1 for i in idx_val if stratum[i] == tg)
        te = sum(1 for i in idx_test if stratum[i] == tg)
        print(f"  {tg:<22}  {total:>6}  {tr:>6}  {va:>6}  {te:>6}")

    print("\nBuilding feature matrix ...")
    X = df[feature_cols].astype(np.float32).values
    print(f"  X shape: {X.shape}  dtype: {X.dtype}")

    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, idx in splits.items():
        idx_arr = list(idx)
        X_split = X[idx_arr]
        sha_split = sha256s[idx_arr]
        out_path = out_dir / f"{split_name}.npz"
        np.savez_compressed(out_path, X=X_split, sha256=sha_split)
        print(f"  Saved {out_path}  shape={X_split.shape}")

    feat_names_path = out_dir / "feature_names.txt"
    feat_names_path.write_text("\n".join(feature_cols) + "\n", encoding="utf-8")
    print(f"  Saved {feat_names_path}  ({len(feature_cols)} features)")

    split_label = np.empty(len(df), dtype=object)
    for split_name, idx in splits.items():
        for i in idx:
            split_label[i] = split_name

    index_path = out_dir / "splits_index.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sha256", "split", "type_group"])
        for i in range(len(df)):
            writer.writerow([sha256s[i], split_label[i], stratum[i]])
    print(f"  Saved {index_path}")

    all_indices = {split_name: set(sha256s[list(idx)]) for split_name, idx in splits.items()}
    for split_name, sha_set in all_indices.items():
        split_bench = bench[bench["sha256"].isin(sha_set)]
        out_path = out_dir / f"benchmarks_{split_name}.csv"
        split_bench.to_csv(out_path, index=False)
        print(f"  Saved {out_path}  ({len(split_bench):,} rows)")

    print(f"\nOutput directory: {out_dir}")
    print(f"Files: {len(df):,}  (train={len(idx_train):,}  val={len(idx_val):,}  test={len(idx_test):,})")
    print(f"Feature columns: {len(feature_cols)}")


if __name__ == "__main__":
    main()
