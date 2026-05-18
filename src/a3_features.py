"""
A3: extract byte-level features for every file in the registry.

For each file, computes:
    byte_00..byte_ff  256 normalised byte frequencies
    entropy           Shannon entropy in bits
    file_size_log2    log2(file size)
    zero_byte_ratio   fraction of 0x00 bytes
    ascii_ratio       fraction of printable ASCII bytes (0x20-0x7E)
    unique_bytes      number of distinct byte values
    longest_run       longest consecutive run of the same byte
    lz4_cr            LZ4 default-level compression ratio (omitted with --no-lz4)

Usage:
    python src/a3_features.py [--resume] [--no-lz4]
"""

import argparse
import csv
import math
import sys
from collections import Counter
from itertools import groupby
from pathlib import Path

import lz4.frame

ASCII_LO = 0x20
ASCII_HI = 0x7E

BYTE_COLS = [f"byte_{i:02x}" for i in range(256)]
STAT_COLS = [
    "entropy",
    "file_size_log2",
    "zero_byte_ratio",
    "ascii_ratio",
    "unique_bytes",
    "longest_run",
]
LZ4_COL = ["lz4_cr"]


def extract_features(data, include_lz4):
    size = len(data)

    freq = Counter(data)
    counts = [freq.get(i, 0) for i in range(256)]
    hist = [c / size if size > 0 else 0.0 for c in counts]

    entropy = 0.0
    if size > 0:
        for c in counts:
            if c > 0:
                p = c / size
                entropy -= p * math.log2(p)

    file_size_log2 = math.log2(size) if size > 0 else 0.0
    zero_byte_ratio = counts[0] / size if size > 0 else 0.0
    ascii_ratio = sum(counts[ASCII_LO:ASCII_HI + 1]) / size if size > 0 else 0.0
    unique_bytes = sum(1 for c in counts if c > 0)

    if size == 0:
        longest_run = 0
    else:
        longest_run = max(sum(1 for _ in grp) for _, grp in groupby(data))

    row = {}
    for i, h in enumerate(hist):
        row[BYTE_COLS[i]] = f"{h:.8f}"
    row["entropy"] = f"{entropy:.6f}"
    row["file_size_log2"] = f"{file_size_log2:.6f}"
    row["zero_byte_ratio"] = f"{zero_byte_ratio:.8f}"
    row["ascii_ratio"] = f"{ascii_ratio:.8f}"
    row["unique_bytes"] = unique_bytes
    row["longest_run"] = longest_run

    if include_lz4:
        if size > 0:
            compressed = lz4.frame.compress(data, compression_level=0)
            lz4_cr = size / len(compressed)
        else:
            lz4_cr = 0.0
        row["lz4_cr"] = f"{lz4_cr:.6f}"

    return row


def load_registry(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_done(path):
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {row["sha256"] for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(description="A3: extract byte-level features.")
    parser.add_argument("--registry", default="data/file_registry.csv")
    parser.add_argument("--clean", default="data/clean")
    parser.add_argument("--out", default="data/features.csv")
    parser.add_argument("--resume", action="store_true",
                        help="Skip files already present in output CSV")
    parser.add_argument("--no-lz4", action="store_true",
                        help="Omit the lz4_cr feature column")
    args = parser.parse_args()

    include_lz4 = not args.no_lz4
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
        print(f"Resume mode: {len(done)} files already extracted, skipping.")

    todo = [r for r in registry if r["sha256"] not in done]
    total = len(todo)

    fieldnames = ["sha256"] + BYTE_COLS + STAT_COLS + (LZ4_COL if include_lz4 else [])

    print(f"Files to process: {total} (lz4_cr: {'yes' if include_lz4 else 'no'})")
    print(f"Output: {out_path}")
    print()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (args.resume and out_path.exists())

    with open(out_path, "a" if args.resume else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, reg_row in enumerate(todo, 1):
            sha = reg_row["sha256"]
            rel = reg_row["rel_path"]
            size = int(reg_row["size_bytes"])
            fpath = clean_dir / rel

            if not fpath.exists():
                print(f"  [{i}/{total}] MISSING: {rel}", file=sys.stderr)
                continue

            size_str = f"{size/1e6:.1f} MB" if size >= 1e6 else f"{size/1e3:.1f} KB"
            print(f"  [{i}/{total}] {rel}  ({size_str})", flush=True)

            data = fpath.read_bytes()
            feat = extract_features(data, include_lz4)
            feat["sha256"] = sha
            writer.writerow(feat)
            f.flush()

    print()
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
