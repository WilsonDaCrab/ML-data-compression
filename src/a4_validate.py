"""
A4: validate registry, benchmarks and features CSVs before A5.

Runs 21 integrity checks across the three CSVs (expected columns, no
duplicates, value ranges, cross-file consistency) and prints PASS/FAIL for
each. Exits 1 if anything fails.

Usage:
    python src/a4_validate.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

EXPECTED_TYPE_GROUPS = {
    "text_code", "binary_docs", "images", "compressed",
    "encrypted_random", "audio_video", "executables", "other",
}

ALGO_IDS = [
    "gzip_1",   "gzip_4",   "gzip_6",   "gzip_9",
    "lz4_0",    "lz4_4",    "lz4_9",    "lz4_12",
    "zstd_1",   "zstd_5",   "zstd_12",  "zstd_19",
    "brotli_1", "brotli_5", "brotli_9", "brotli_11",
    "lzma_1",   "lzma_4",   "lzma_6",   "lzma_9",
]
N_ALGOS = len(ALGO_IDS)

BYTE_COLS = [f"byte_{i:02x}" for i in range(256)]
STAT_COLS = [
    "entropy", "file_size_log2", "zero_byte_ratio",
    "ascii_ratio", "unique_bytes", "longest_run",
]

failures = 0


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def check(label, passed, detail=""):
    global failures
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f"  -- {detail}"
    print(msg)
    if not passed:
        failures += 1
    return passed


def main():
    global failures

    parser = argparse.ArgumentParser(description="A4: validate pipeline data.")
    parser.add_argument("--registry", default="data/file_registry.csv")
    parser.add_argument("--benchmarks", default="data/benchmarks.csv")
    parser.add_argument("--features", default="data/features.csv")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    benchmarks_path = Path(args.benchmarks)
    features_path = Path(args.features)

    section("Loading CSVs")

    for path in [registry_path, benchmarks_path, features_path]:
        if not path.exists():
            print(f"  [FAIL] File not found: {path}", file=sys.stderr)
            sys.exit(1)

    print(f"  Loading {registry_path} ...")
    reg = pd.read_csv(registry_path,
                      dtype={"sha256": str, "rel_path": str,
                             "size_bytes": "int64", "type_group": str})

    print(f"  Loading {benchmarks_path} ...")
    bench = pd.read_csv(benchmarks_path,
                        dtype={"sha256": str, "algo_id": str,
                               "cr": float, "comp_mbps": float, "decomp_mbps": float})

    print(f"  Loading {features_path} ...")
    feat = pd.read_csv(features_path, dtype=str)

    print(f"\n  Registry   : {len(reg):,} rows")
    print(f"  Benchmarks : {len(bench):,} rows")
    print(f"  Features   : {len(feat):,} rows")

    section("Registry Checks")

    expected_reg_cols = {"sha256", "rel_path", "size_bytes", "type_group"}
    missing_cols = expected_reg_cols - set(reg.columns)
    check("Expected columns present", not missing_cols,
          f"missing: {missing_cols}" if missing_cols else "")

    dups = reg["sha256"].duplicated().sum()
    check("No duplicate sha256", dups == 0, f"{dups} duplicates" if dups else "")

    bad_size = (reg["size_bytes"] <= 0).sum()
    check("All size_bytes > 0", bad_size == 0,
          f"{bad_size} zero/negative rows" if bad_size else "")

    unknown_tg = set(reg["type_group"].unique()) - EXPECTED_TYPE_GROUPS
    check("All type_groups valid", not unknown_tg,
          f"unknown: {unknown_tg}" if unknown_tg else "")

    print("\n  Type group distribution:")
    for tg, count in reg["type_group"].value_counts().items():
        print(f"    {tg:<22} {count:>6}")

    section("Benchmarks Checks")

    expected_bench_cols = {"sha256", "algo_id", "cr", "comp_mbps", "decomp_mbps"}
    missing_cols = expected_bench_cols - set(bench.columns)
    check("Expected columns present", not missing_cols,
          f"missing: {missing_cols}" if missing_cols else "")

    unknown_algos = set(bench["algo_id"].unique()) - set(ALGO_IDS)
    check("All algo_ids valid", not unknown_algos,
          f"unknown: {unknown_algos}" if unknown_algos else "")

    algo_counts = bench.groupby("sha256")["algo_id"].count()
    wrong_count = (algo_counts != N_ALGOS).sum()
    check(f"Each file has exactly {N_ALGOS} algo rows", wrong_count == 0,
          f"{wrong_count} files with wrong count" if wrong_count else "")
    if wrong_count:
        bad = algo_counts[algo_counts != N_ALGOS]
        print(f"    Examples: {bad.head(5).to_dict()}")

    dups = bench.duplicated(subset=["sha256", "algo_id"]).sum()
    check("No duplicate (sha256, algo_id)", dups == 0,
          f"{dups} duplicates" if dups else "")

    bad_cr = (bench["cr"] < 0).sum()
    check("No failed benchmark rows (cr >= 0)", bad_cr == 0,
          f"{bad_cr} rows with cr < 0" if bad_cr else "")

    bad_comp = (bench["comp_mbps"] <= 0).sum()
    bad_decomp = (bench["decomp_mbps"] <= 0).sum()
    check("comp_mbps > 0", bad_comp == 0,
          f"{bad_comp} bad rows" if bad_comp else "")
    check("decomp_mbps > 0", bad_decomp == 0,
          f"{bad_decomp} bad rows" if bad_decomp else "")

    print(f"\n  CR statistics:")
    cr = bench["cr"]
    print(f"    min={cr.min():.4f}  max={cr.max():.4f}  "
          f"mean={cr.mean():.4f}  median={cr.median():.4f}")
    print(f"    CR > 2.0 : {(cr > 2.0).sum():,} rows")
    print(f"    CR < 1.0 : {(cr < 1.0).sum():,} rows")

    section("Features Checks")

    has_lz4 = "lz4_cr" in feat.columns
    print(f"  lz4_cr column present: {has_lz4}")

    expected_feat_cols = {"sha256"} | set(BYTE_COLS) | set(STAT_COLS)
    missing_cols = expected_feat_cols - set(feat.columns)
    check("Expected columns present", not missing_cols,
          f"missing: {missing_cols}" if missing_cols else "")

    dups = feat["sha256"].duplicated().sum()
    check("No duplicate sha256", dups == 0, f"{dups} duplicates" if dups else "")

    numeric_cols = BYTE_COLS + STAT_COLS + (["lz4_cr"] if has_lz4 else [])
    feat_num = feat[["sha256"] + numeric_cols].copy()
    for col in numeric_cols:
        feat_num[col] = pd.to_numeric(feat_num[col], errors="coerce")

    total_nan = feat_num[numeric_cols].isna().sum().sum()
    check("No NaN in numeric columns", total_nan == 0,
          f"{total_nan} NaN values (parse errors)" if total_nan else "")

    byte_data = feat_num[BYTE_COLS]
    bad_byte = ((byte_data < 0) | (byte_data > 1)).any(axis=1).sum()
    check("byte_xx values in [0, 1]", bad_byte == 0,
          f"{bad_byte} rows with out-of-range values" if bad_byte else "")

    # tolerance for float precision
    hist_sums = byte_data.sum(axis=1)
    bad_sum = ((hist_sums - 1.0).abs() > 1e-4).sum()
    check("Byte histogram sums ~= 1.0", bad_sum == 0,
          f"{bad_sum} rows with sum != 1" if bad_sum else "")
    if bad_sum:
        print(f"    Sum range: {hist_sums.min():.6f} - {hist_sums.max():.6f}")

    ent = feat_num["entropy"]
    bad_ent = ((ent < 0) | (ent > 8)).sum()
    check("entropy in [0, 8]", bad_ent == 0,
          f"{bad_ent} bad rows" if bad_ent else "")

    for col in ["zero_byte_ratio", "ascii_ratio"]:
        v = feat_num[col]
        bad = ((v < 0) | (v > 1)).sum()
        check(f"{col} in [0, 1]", bad == 0, f"{bad} bad rows" if bad else "")

    bad_ub = ((feat_num["unique_bytes"] < 0) | (feat_num["unique_bytes"] > 256)).sum()
    check("unique_bytes in [0, 256]", bad_ub == 0,
          f"{bad_ub} bad rows" if bad_ub else "")

    bad_lr = (feat_num["longest_run"] < 1).sum()
    check("longest_run >= 1", bad_lr == 0,
          f"{bad_lr} bad rows" if bad_lr else "")

    if has_lz4:
        bad_lz4 = (feat_num["lz4_cr"] <= 0).sum()
        check("lz4_cr > 0", bad_lz4 == 0,
              f"{bad_lz4} bad rows" if bad_lz4 else "")

    print(f"\n  Feature statistics:")
    for col in ["entropy", "zero_byte_ratio", "ascii_ratio", "unique_bytes"]:
        s = feat_num[col]
        print(f"    {col:<22}  min={s.min():.3f}  max={s.max():.3f}  mean={s.mean():.3f}")

    section("Cross-File Consistency")

    reg_hashes = set(reg["sha256"])
    bench_hashes = set(bench["sha256"])
    feat_hashes = set(feat["sha256"])

    orphan_bench = bench_hashes - reg_hashes
    check("All benchmark sha256s in registry", not orphan_bench,
          f"{len(orphan_bench)} orphan hashes" if orphan_bench else "")

    missing_bench = reg_hashes - bench_hashes
    check("All registry sha256s in benchmarks", not missing_bench,
          f"{len(missing_bench)} files missing from benchmarks" if missing_bench else "")

    orphan_feat = feat_hashes - reg_hashes
    check("All feature sha256s in registry", not orphan_feat,
          f"{len(orphan_feat)} orphan hashes" if orphan_feat else "")

    missing_feat = reg_hashes - feat_hashes
    check("All registry sha256s in features", not missing_feat,
          f"{len(missing_feat)} files missing from features" if missing_feat else "")

    section("Summary")

    print(f"  Registry   : {len(reg):,} files")
    print(f"  Benchmarks : {len(bench):,} rows  "
          f"({len(bench_hashes):,} unique files x {N_ALGOS} algos)")
    print(f"  Features   : {len(feat):,} rows  "
          f"({len(feat_hashes):,} unique files, lz4_cr: {'yes' if has_lz4 else 'no'})")

    if failures == 0:
        print("\n  ALL CHECKS PASSED")
    else:
        print(f"\n  {failures} CHECK(S) FAILED")

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
