"""
SKIP threshold analysis.

For each training file, compute max CR across all 20 algorithms and print the
distribution to help select a data-driven SKIP cutoff. Validation and test
sets are excluded — the threshold must be chosen on training data only.

Usage:
    python src/analyze_skip_threshold.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", default="data/benchmarks.csv")
    parser.add_argument("--registry", default="data/file_registry.csv")
    parser.add_argument("--splits", default="data/splits/splits_index.csv")
    args = parser.parse_args()

    print("Loading benchmarks ...")
    bench = pd.read_csv(args.benchmarks,
                        dtype={"sha256": str, "algo_id": str,
                               "cr": float, "comp_mbps": float, "decomp_mbps": float})
    reg = pd.read_csv(args.registry, dtype={"sha256": str, "type_group": str})
    splits = pd.read_csv(args.splits, dtype={"sha256": str, "split": str})

    # threshold must not see val/test
    train_sha = splits.loc[splits["split"] == "train", "sha256"]
    bench = bench[bench["sha256"].isin(train_sha)]

    max_cr = bench.groupby("sha256")["cr"].max().reset_index()
    max_cr.columns = ["sha256", "max_cr"]
    max_cr = max_cr.merge(reg[["sha256", "type_group"]], on="sha256", how="left")

    print(f"Training files: {len(max_cr):,}\n")

    print("=" * 55)
    print("  max_cr percentiles (training files)")
    print("=" * 55)
    pcts = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
    vals = np.percentile(max_cr["max_cr"], pcts)
    for p, v in zip(pcts, vals):
        print(f"  p{p:>3}  {v:.4f}")

    print()
    print("=" * 55)
    print("  Files below threshold (would be labelled SKIP)")
    print("=" * 55)
    print(f"  {'Threshold':>10}  {'SKIP count':>10}  {'SKIP %':>8}")
    for t in [1.01, 1.02, 1.03, 1.05, 1.08, 1.10, 1.15, 1.20, 1.25]:
        n = (max_cr["max_cr"] < t).sum()
        print(f"  {t:>10.2f}  {n:>10,}  {n/len(max_cr)*100:>7.1f}%")

    print()
    print("=" * 55)
    print("  SKIP rate per type_group (threshold = 1.05)")
    print("=" * 55)
    threshold = 1.05
    max_cr["is_skip"] = max_cr["max_cr"] < threshold
    grp = max_cr.groupby("type_group").agg(
        total=("sha256", "count"),
        skip=("is_skip", "sum"),
    ).reset_index()
    grp["skip_pct"] = grp["skip"] / grp["total"] * 100
    grp = grp.sort_values("skip_pct", ascending=False)
    print(f"  {'type_group':<22}  {'total':>6}  {'skip':>6}  {'skip%':>7}")
    for _, row in grp.iterrows():
        print(f"  {row['type_group']:<22}  {row['total']:>6}  "
              f"{row['skip']:>6}  {row['skip_pct']:>6.1f}%")

    print()
    print("=" * 55)
    print("  max_cr distribution (bins near 1.0)")
    print("=" * 55)
    bins = [0.0, 0.90, 0.95, 0.98, 1.00, 1.01, 1.02, 1.03, 1.05,
            1.10, 1.20, 1.50, 2.00, 5.00, float("inf")]
    labels = ["<0.90", "0.90-0.95", "0.95-0.98", "0.98-1.00",
              "1.00-1.01", "1.01-1.02", "1.02-1.03", "1.03-1.05",
              "1.05-1.10", "1.10-1.20", "1.20-1.50", "1.50-2.00",
              "2.00-5.00", ">5.00"]
    cuts = pd.cut(max_cr["max_cr"], bins=bins, labels=labels, right=False)
    counts = cuts.value_counts().reindex(labels)
    for label, count in counts.items():
        bar = "#" * int(count / len(max_cr) * 60)
        print(f"  {label:>12}  {count:>6,}  {bar}")


if __name__ == "__main__":
    main()
