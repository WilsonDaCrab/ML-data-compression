"""
A1: build the file registry.

Walks data/clean/ and writes one row per file to data/file_registry.csv with
SHA256, relative path, size, and a coarse type group inferred from extension
(with a MIME-based fallback).

Usage:
    python src/a1_registry.py [--clean data/clean] [--out data/file_registry.csv]
"""

import argparse
import csv
import hashlib
import mimetypes
import sys
from pathlib import Path

CHUNK = 1024 * 1024

# extension -> type_group (lowercase, no leading dot)
EXT_MAP = {
    # text_code
    "txt": "text_code", "text": "text_code", "md": "text_code",
    "rst": "text_code", "csv": "text_code", "tsv": "text_code",
    "json": "text_code", "xml": "text_code", "yaml": "text_code",
    "yml": "text_code", "toml": "text_code", "ini": "text_code",
    "cfg": "text_code", "conf": "text_code", "log": "text_code",
    "html": "text_code", "htm": "text_code", "css": "text_code",
    "js": "text_code", "ts": "text_code", "jsx": "text_code",
    "tsx": "text_code", "py": "text_code", "pyw": "text_code",
    "c": "text_code", "h": "text_code", "cpp": "text_code",
    "cc": "text_code", "cxx": "text_code", "hpp": "text_code",
    "hxx": "text_code", "java": "text_code", "cs": "text_code",
    "go": "text_code", "rs": "text_code", "rb": "text_code",
    "php": "text_code", "pl": "text_code", "pm": "text_code",
    "sh": "text_code", "bash": "text_code", "zsh": "text_code",
    "bat": "text_code", "ps1": "text_code", "r": "text_code",
    "lua": "text_code", "sql": "text_code", "swift": "text_code",
    "kt": "text_code", "kts": "text_code", "scala": "text_code",
    "m": "text_code", "tex": "text_code", "bib": "text_code",
    "makefile": "text_code", "cmake": "text_code", "lsp": "text_code",
    "lisp": "text_code", "el": "text_code", "vim": "text_code",
    "1": "text_code",  # man pages

    # binary_docs
    "pdf": "binary_docs",
    "doc": "binary_docs", "docx": "binary_docs",
    "xls": "binary_docs", "xlsx": "binary_docs",
    "ppt": "binary_docs", "pptx": "binary_docs",
    "odt": "binary_docs", "ods": "binary_docs", "odp": "binary_docs",
    "rtf": "binary_docs", "epub": "binary_docs",

    # images
    "jpg": "images", "jpeg": "images", "png": "images",
    "gif": "images", "bmp": "images", "tiff": "images", "tif": "images",
    "webp": "images", "ico": "images", "svg": "images",
    "psd": "images", "raw": "images", "cr2": "images", "nef": "images",
    "heic": "images", "heif": "images", "avif": "images",
    "ppm": "images", "pgm": "images", "pbm": "images",

    # compressed
    "gz": "compressed", "bz2": "compressed", "xz": "compressed",
    "zst": "compressed", "lz4": "compressed", "lzma": "compressed",
    "zip": "compressed", "7z": "compressed", "rar": "compressed",
    "tar": "compressed", "tgz": "compressed", "tbz2": "compressed",
    "txz": "compressed", "tzst": "compressed",
    "br": "compressed", "zlib": "compressed", "z": "compressed",
    "cab": "compressed", "arj": "compressed", "lzh": "compressed",
    "apk": "compressed", "jar": "compressed", "war": "compressed",
    "ear": "compressed", "whl": "compressed", "egg": "compressed",
    "dmg": "compressed", "iso": "compressed",

    # audio_video
    "mp3": "audio_video", "wav": "audio_video", "flac": "audio_video",
    "ogg": "audio_video", "aac": "audio_video", "m4a": "audio_video",
    "wma": "audio_video", "opus": "audio_video", "aiff": "audio_video",
    "mp4": "audio_video", "mkv": "audio_video", "avi": "audio_video",
    "mov": "audio_video", "wmv": "audio_video", "flv": "audio_video",
    "webm": "audio_video", "m4v": "audio_video", "3gp": "audio_video",
    "mpg": "audio_video", "mpeg": "audio_video",

    # executables
    "exe": "executables", "dll": "executables", "so": "executables",
    "dylib": "executables", "elf": "executables", "o": "executables",
    "a": "executables", "lib": "executables", "obj": "executables",
    "sys": "executables", "ko": "executables", "class": "executables",
    "pyc": "executables", "pyd": "executables",
    "wasm": "executables",

    # other / databases
    "db": "other", "sqlite": "other", "sqlite3": "other",
    "mdb": "other", "accdb": "other",
}

MIME_MAP = {
    "text/": "text_code",
    "image/": "images",
    "audio/": "audio_video",
    "video/": "audio_video",
    "application/pdf": "binary_docs",
    "application/msword": "binary_docs",
    "application/zip": "compressed",
    "application/x-gzip": "compressed",
    "application/gzip": "compressed",
    "application/x-bzip2": "compressed",
    "application/x-xz": "compressed",
    "application/x-7z-compressed": "compressed",
    "application/x-rar-compressed": "compressed",
    "application/x-tar": "compressed",
    "application/x-executable": "executables",
    "application/x-sharedlib": "executables",
}


def get_type_group(path):
    ext = path.suffix.lstrip(".").lower()
    if not ext:
        name_lower = path.name.lower()
        if name_lower in ("makefile", "dockerfile", "gemfile", "rakefile",
                          "procfile", "brewfile"):
            return "text_code"

    if ext in EXT_MAP:
        return EXT_MAP[ext]

    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        if mime.startswith("text/"):
            return "text_code"
        if mime.startswith("image/"):
            return "images"
        if mime.startswith("audio/") or mime.startswith("video/"):
            return "audio_video"
        for prefix, group in MIME_MAP.items():
            if mime.startswith(prefix):
                return group

    return "other"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="A1: build file registry.")
    parser.add_argument("--clean", default="data/clean")
    parser.add_argument("--out", default="data/file_registry.csv")
    args = parser.parse_args()

    clean_dir = Path(args.clean)
    out_path = Path(args.out)

    if not clean_dir.exists():
        print(f"ERROR: clean directory not found: {clean_dir}", file=sys.stderr)
        sys.exit(1)

    all_files = sorted(
        p for p in clean_dir.rglob("*")
        if p.is_file() and p.name != "duplicates_log.csv"
    )
    total = len(all_files)
    print(f"Building registry for {total} files in {clean_dir} ...")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    type_counts = {}
    rows = []

    for i, path in enumerate(all_files, 1):
        if i % 500 == 0 or i == total:
            print(f"  {i}/{total} ...", flush=True)

        h = sha256(path)
        rel = path.relative_to(clean_dir).as_posix()
        size = path.stat().st_size
        tg = get_type_group(path)

        type_counts[tg] = type_counts.get(tg, 0) + 1
        rows.append((h, rel, size, tg))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sha256", "rel_path", "size_bytes", "type_group"])
        writer.writerows(rows)

    print()
    print(f"Files registered: {total}")
    print(f"Output: {out_path}")
    print()
    print("Type group breakdown:")
    for tg, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {tg:<20} {count:>6}")


if __name__ == "__main__":
    main()
