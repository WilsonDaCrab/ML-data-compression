"""
A2: run all 20 algorithm configurations on every file in the registry.

For each file, records the compression ratio and compression/decompression
speeds in MB/s. Small files are timed over multiple repetitions to reduce
noise:
    size <  100 KB : 5 reps
    size <    1 MB : 3 reps
    size >= 1  MB  : 1 rep

Usage:
    python src/a2_benchmark.py [--resume] [--max-size BYTES]
"""

import argparse
import csv
import gzip
import io
import lzma
import sys
import time
from pathlib import Path

import brotli
import lz4.frame
import zstandard as zstd

ALGOS = [
    ("gzip",   1), ("gzip",   4), ("gzip",   6), ("gzip",   9),
    ("lz4",    0), ("lz4",    4), ("lz4",    9), ("lz4",   12),
    ("zstd",   1), ("zstd",   5), ("zstd",  12), ("zstd",  19),
    ("brotli", 1), ("brotli", 5), ("brotli", 9), ("brotli", 11),
    ("lzma",   1), ("lzma",   4), ("lzma",   6), ("lzma",   9),
]

ALGO_IDS = [f"{name}_{level}" for name, level in ALGOS]


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
    "gzip":   bench_gzip,
    "lz4":    bench_lz4,
    "zstd":   bench_zstd,
    "brotli": bench_brotli,
    "lzma":   bench_lzma,
}


def benchmark_file(data):
    size = len(data)
    n = reps_for_size(size)
    results = []
    for name, level in ALGOS:
        algo_id = f"{name}_{level}"
        try:
            compressed, comp_sec, decomp_sec = BENCH_FN[name](data, level, n)
            cr = size / len(compressed) if len(compressed) > 0 else 0.0
            comp_mbps = (size / 1e6) / comp_sec if comp_sec > 0 else 0.0
            decomp_mbps = (size / 1e6) / decomp_sec if decomp_sec > 0 else 0.0
        except Exception as e:
            print(f"    WARNING: {algo_id} failed: {e}", file=sys.stderr)
            cr, comp_mbps, decomp_mbps = -1.0, -1.0, -1.0
        results.append((algo_id, cr, comp_mbps, decomp_mbps))
    return results


def load_registry(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_done(path):
    """Return set of sha256 hashes that already have all 20 algos recorded."""
    if not path.exists():
        return set()
    counts = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h = row["sha256"]
            counts[h] = counts.get(h, 0) + 1
    n_algos = len(ALGOS)
    return {h for h, c in counts.items() if c >= n_algos}


def main():
    parser = argparse.ArgumentParser(description="A2: benchmark compression algorithms.")
    parser.add_argument("--registry", default="data/file_registry.csv")
    parser.add_argument("--clean", default="data/clean")
    parser.add_argument("--out", default="data/benchmarks.csv")
    parser.add_argument("--resume", action="store_true",
                        help="Skip files already fully benchmarked")
    parser.add_argument("--max-size", type=int, default=0,
                        help="Skip files larger than this many bytes (0 = no limit)")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    clean_dir = Path(args.clean)
    out_path = Path(args.out)

    if not registry_path.exists():
        print(f"ERROR: registry not found: {registry_path}", file=sys.stderr)
        sys.exit(1)

    registry = load_registry(registry_path)

    done = set()
    if args.resume:
        done = load_done(out_path)
        print(f"Resume mode: {len(done)} files already fully benchmarked, skipping.")

    todo = [r for r in registry if r["sha256"] not in done]
    if args.max_size > 0:
        skipped_size = [r for r in todo if int(r["size_bytes"]) > args.max_size]
        todo = [r for r in todo if int(r["size_bytes"]) <= args.max_size]
        if skipped_size:
            print(f"Skipping {len(skipped_size)} files above {args.max_size:,} bytes.")

    total = len(todo)
    print(f"Files to benchmark: {total} ({len(ALGOS)} algos each = {total * len(ALGOS):,} runs)")
    print(f"Output: {out_path}")
    print()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (args.resume and out_path.exists())

    with open(out_path, "a" if args.resume else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["sha256", "algo_id", "cr", "comp_mbps", "decomp_mbps"])

        for i, row in enumerate(todo, 1):
            sha = row["sha256"]
            rel = row["rel_path"]
            size = int(row["size_bytes"])
            tg = row["type_group"]
            fpath = clean_dir / rel

            if not fpath.exists():
                print(f"  [{i}/{total}] MISSING: {rel}", file=sys.stderr)
                continue

            size_str = f"{size/1e6:.1f} MB" if size >= 1e6 else f"{size/1e3:.1f} KB"
            print(f"  [{i}/{total}] {rel}  ({size_str}, {tg})", flush=True)

            data = fpath.read_bytes()
            results = benchmark_file(data)

            for algo_id, cr, comp_mbps, decomp_mbps in results:
                writer.writerow([sha, algo_id,
                                 f"{cr:.6f}",
                                 f"{comp_mbps:.3f}",
                                 f"{decomp_mbps:.3f}"])
            f.flush()

    print()
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
