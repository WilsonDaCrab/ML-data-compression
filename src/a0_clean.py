"""
A0: deduplicate raw files by SHA256.

Reads everything under data/raw/, copies unique files to data/clean/ keeping
the relative path. Duplicates are skipped and logged to
data/clean/duplicates_log.csv.

Usage:
    python src/a0_clean.py [--raw data/raw] [--clean data/clean]
"""

import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path

CHUNK = 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="A0: deduplicate raw files.")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--clean", default="data/clean")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    clean_dir = Path(args.clean)

    if not raw_dir.exists():
        print(f"ERROR: raw directory not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    clean_dir.mkdir(parents=True, exist_ok=True)

    seen = {}
    duplicates = []

    all_files = sorted(raw_dir.rglob("*"))
    all_files = [p for p in all_files if p.is_file()]

    total = len(all_files)
    print(f"Scanning {total} files in {raw_dir} ...")

    copied = 0
    skipped = 0

    for i, src in enumerate(all_files, 1):
        if i % 500 == 0 or i == total:
            print(f"  {i}/{total} ...", flush=True)

        h = sha256(src)
        rel = src.relative_to(raw_dir)
        dst = clean_dir / rel

        if h in seen:
            duplicates.append((h, src, seen[h]))
            skipped += 1
            continue

        seen[h] = src
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    log_path = clean_dir / "duplicates_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sha256", "duplicate_path", "kept_path"])
        for h, dup, kept in duplicates:
            writer.writerow([h, str(dup), str(kept)])

    print()
    print(f"Total scanned: {total}")
    print(f"Unique (kept): {copied}")
    print(f"Duplicates:    {skipped}")
    print(f"Output: {clean_dir}")
    print(f"Duplicate log: {log_path}")


if __name__ == "__main__":
    main()
