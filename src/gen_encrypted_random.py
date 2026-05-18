"""
Generate encrypted_random files with os.urandom() bytes.

Sizes are sampled from the existing registry size distribution so the
generated files match the real dataset size profile. Files are written to
both data/raw/ and data/clean/ and appended to data/file_registry.csv.
"""

import argparse
import csv
import hashlib
import os
import random
import sys
from pathlib import Path


def sha256_of_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_existing_sizes(registry_path):
    sizes = []
    with open(registry_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sizes.append(int(row["size_bytes"]))
    return sizes


def load_existing_hashes(registry_path):
    hashes = set()
    with open(registry_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hashes.add(row["sha256"])
    return hashes


def main():
    parser = argparse.ArgumentParser(description="Generate encrypted_random files.")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--registry", default="data/file_registry.csv")
    parser.add_argument("--raw", default="data/raw/encrypted_random")
    parser.add_argument("--clean", default="data/clean/encrypted_random")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    raw_dir = Path(args.raw)
    clean_dir = Path(args.clean)

    if not registry_path.exists():
        print(f"ERROR: registry not found: {registry_path}", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)

    print("Loading existing registry ...")
    existing_sizes = load_existing_sizes(registry_path)
    existing_hashes = load_existing_hashes(registry_path)
    print(f"  {len(existing_sizes)} existing files, {len(existing_hashes)} unique hashes")

    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    new_rows = []
    skipped_collision = 0

    print(f"Generating {args.count} encrypted_random files ...")

    MAX_SIZE = 10 * 1024 * 1024  # cap at 10 MB
    MIN_SIZE = 64

    for i in range(1, args.count + 1):
        if i % 100 == 0 or i == args.count:
            print(f"  {i}/{args.count} ...", flush=True)

        size = random.choice(existing_sizes)
        size = min(size, MAX_SIZE)
        size = max(size, MIN_SIZE)

        data = os.urandom(size)
        h = sha256_of_bytes(data)

        if h in existing_hashes:
            # extremely unlikely with os.urandom, but check anyway
            skipped_collision += 1
            continue

        filename = f"rnd_{i:06d}.bin"
        raw_path = raw_dir / filename
        clean_path = clean_dir / filename

        with open(raw_path, "wb") as f:
            f.write(data)
        with open(clean_path, "wb") as f:
            f.write(data)

        existing_hashes.add(h)
        rel = f"encrypted_random/{filename}"
        new_rows.append((h, rel, size, "encrypted_random"))

    with open(registry_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

    print()
    print(f"Generated: {len(new_rows)} files")
    if skipped_collision:
        print(f"Hash collisions: {skipped_collision} (skipped)")
    print(f"Raw output: {raw_dir}")
    print(f"Clean output: {clean_dir}")
    print(f"Registry updated: {registry_path}")
    print()

    type_counts = {}
    with open(registry_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tg = row["type_group"]
            type_counts[tg] = type_counts.get(tg, 0) + 1

    print("Type group breakdown:")
    for tg, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {tg:<20} {count:>6}")


if __name__ == "__main__":
    main()
