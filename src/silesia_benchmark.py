"""
Run all 20 algorithm configurations on the 12 Silesia corpus files and
produce a per-family min-max summary table.

Inputs files are expected in data/raw/silesia/. Outputs go to
data/analysis/silesia_benchmark.csv (per-file raw results) and
data/analysis/silesia_summary.csv (per-family aggregated ranges).

Usage:
    python src/silesia_benchmark.py
"""

import csv
import gzip
import io
import lzma
import time
from pathlib import Path

import brotli
import lz4.frame
import zstandard as zstd

# same as a2_benchmark.py
ALGOS = [
    ("gzip",   1), ("gzip",   4), ("gzip",   6), ("gzip",   9),
    ("lz4",    0), ("lz4",    4), ("lz4",    9), ("lz4",   12),
    ("zstd",   1), ("zstd",   5), ("zstd",  12), ("zstd",  19),
    ("brotli", 1), ("brotli", 5), ("brotli", 9), ("brotli", 11),
    ("lzma",   1), ("lzma",   4), ("lzma",   6), ("lzma",   9),
]

FAMILY_NAMES = {
    "gzip": "Gzip",
    "lz4": "LZ4",
    "zstd": "Zstd",
    "brotli": "Brotli",
    "lzma": "LZMA2",
}

FAMILY_ORDER = ["lz4", "gzip", "zstd", "brotli", "lzma"]


def reps_for_size(size):
    if size < 100_000:
        return 5
    if size < 1_000_000:
        return 3
    return 1


def bench_gzip(data, level, n):
    comp_times, decomp_times = [], []
    compressed = b""
    for _ in range(n):
        t0 = time.perf_counter()
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=level) as gz:
            gz.write(data)
        compressed = buf.getvalue()
        comp_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as gz:
            gz.read()
        decomp_times.append(time.perf_counter() - t0)

    return compressed, sum(comp_times) / n, sum(decomp_times) / n


def bench_lz4(data, level, n):
    comp_times, decomp_times = [], []
    compressed = b""
    for _ in range(n):
        t0 = time.perf_counter()
        compressed = lz4.frame.compress(data, compression_level=level)
        comp_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        lz4.frame.decompress(compressed)
        decomp_times.append(time.perf_counter() - t0)

    return compressed, sum(comp_times) / n, sum(decomp_times) / n


def bench_zstd(data, level, n):
    comp_times, decomp_times = [], []
    compressed = b""
    cctx = zstd.ZstdCompressor(level=level)
    dctx = zstd.ZstdDecompressor()
    for _ in range(n):
        t0 = time.perf_counter()
        compressed = cctx.compress(data)
        comp_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        dctx.decompress(compressed)
        decomp_times.append(time.perf_counter() - t0)

    return compressed, sum(comp_times) / n, sum(decomp_times) / n


def bench_brotli(data, level, n):
    comp_times, decomp_times = [], []
    compressed = b""
    for _ in range(n):
        t0 = time.perf_counter()
        compressed = brotli.compress(data, quality=level)
        comp_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        brotli.decompress(compressed)
        decomp_times.append(time.perf_counter() - t0)

    return compressed, sum(comp_times) / n, sum(decomp_times) / n


def bench_lzma(data, level, n):
    comp_times, decomp_times = [], []
    compressed = b""
    filters = [{"id": lzma.FILTER_LZMA2, "preset": level}]
    for _ in range(n):
        t0 = time.perf_counter()
        compressed = lzma.compress(data, format=lzma.FORMAT_XZ, filters=filters)
        comp_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        lzma.decompress(compressed)
        decomp_times.append(time.perf_counter() - t0)

    return compressed, sum(comp_times) / n, sum(decomp_times) / n


BENCH_FN = {
    "gzip": bench_gzip,
    "lz4": bench_lz4,
    "zstd": bench_zstd,
    "brotli": bench_brotli,
    "lzma": bench_lzma,
}


def main():
    silesia_dir = Path("data/raw/silesia")
    out_dir = Path("data/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(silesia_dir.iterdir())
    if not files:
        print(f"ERROR: No files found in {silesia_dir}")
        return

    print(f"Found {len(files)} Silesia files: {[f.name for f in files]}\n")

    raw_results = {f"{name}_{level}": [] for name, level in ALGOS}
    raw_rows = []

    for fpath in files:
        data = fpath.read_bytes()
        size_mb = len(data) / 1e6
        n = reps_for_size(len(data))
        print(f"  Benchmarking {fpath.name} ({size_mb:.1f} MB, {n} rep(s))...")

        for name, level in ALGOS:
            algo_id = f"{name}_{level}"
            fn = BENCH_FN[name]
            try:
                compressed, comp_sec, decomp_sec = fn(data, level, n)
                cr = len(data) / len(compressed) if len(compressed) > 0 else 0.0
                comp_mbps = (len(data) / 1e6) / comp_sec if comp_sec > 0 else 0.0
                decomp_mbps = (len(data) / 1e6) / decomp_sec if decomp_sec > 0 else 0.0
            except Exception as e:
                print(f"    WARNING: {algo_id} failed on {fpath.name}: {e}")
                cr, comp_mbps, decomp_mbps = -1.0, -1.0, -1.0

            raw_results[algo_id].append((cr, comp_mbps, decomp_mbps))
            raw_rows.append({
                "file": fpath.name,
                "algo_id": algo_id,
                "cr": f"{cr:.4f}",
                "comp_mbps": f"{comp_mbps:.2f}",
                "decomp_mbps": f"{decomp_mbps:.2f}",
            })

    raw_csv = out_dir / "silesia_benchmark.csv"
    with open(raw_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "algo_id", "cr", "comp_mbps", "decomp_mbps"])
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"\nRaw results saved to {raw_csv}")

    # per algo_id averages across all files
    algo_avg = {}
    for algo_id, records in raw_results.items():
        valid = [(cr, c, d) for cr, c, d in records if cr > 0]
        if not valid:
            algo_avg[algo_id] = (0.0, 0.0, 0.0)
        else:
            avg_cr = sum(r[0] for r in valid) / len(valid)
            avg_comp = sum(r[1] for r in valid) / len(valid)
            avg_decomp = sum(r[2] for r in valid) / len(valid)
            algo_avg[algo_id] = (avg_cr, avg_comp, avg_decomp)

    # per family min/max across the levels used
    summary_rows = []
    for family in FAMILY_ORDER:
        family_levels = [(name, level) for name, level in ALGOS if name == family]
        algo_ids = [f"{name}_{level}" for name, level in family_levels]
        level_nums = [level for _, level in family_levels]

        crs = [algo_avg[a][0] for a in algo_ids]
        comps = [algo_avg[a][1] for a in algo_ids]
        decomps = [algo_avg[a][2] for a in algo_ids]

        levels_str = ", ".join(str(l) for l in level_nums)

        summary_rows.append({
            "algorithm": FAMILY_NAMES[family],
            "levels": levels_str,
            "cr_min": f"{min(crs):.2f}",
            "cr_max": f"{max(crs):.2f}",
            "comp_mbps_min": f"{min(comps):.0f}",
            "comp_mbps_max": f"{max(comps):.0f}",
            "decomp_mbps_min": f"{min(decomps):.0f}",
            "decomp_mbps_max": f"{max(decomps):.0f}",
        })

    summary_csv = out_dir / "silesia_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary saved to {summary_csv}\n")

    print("=" * 75)
    print(f"{'Algorithm':<10} {'Levels':<16} {'CR':^12} {'Comp (MB/s)':^16} {'Decomp (MB/s)':^16}")
    print(f"{'':10} {'':16} {'min - max':^12} {'min - max':^16} {'min - max':^16}")
    print("-" * 75)
    for r in summary_rows:
        cr_range = f"{r['cr_min']} - {r['cr_max']}"
        comp_range = f"{r['comp_mbps_min']} - {r['comp_mbps_max']}"
        decomp_range = f"{r['decomp_mbps_min']} - {r['decomp_mbps_max']}"
        print(f"{r['algorithm']:<10} {r['levels']:<16} {cr_range:^12} {comp_range:^16} {decomp_range:^16}")
    print("=" * 75)
    print("Note: averages across all Silesia corpus files.")
    print("      CR = original_size / compressed_size (higher = better).")


if __name__ == "__main__":
    main()
