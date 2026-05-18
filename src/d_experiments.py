"""
D: extended experiments. Three CatBoost ablations evaluated on the validation
set with t=0 (no threshold), with results written as CSVs to data/analysis/:

  1. Feature subsets   full (263) vs no_lz4 (262) vs stats_only (6)
  2. Sampling          features from file prefix: 16KB / 64KB / 256KB / full
  3. Rule-based        entropy/lz4_cr decision rules vs ML (test set)

Caches:
    data/d_features/sampled_{size}.npz   per-sample-size feature matrices
    data/d_models/{size|subset}_{profile}.cbm   per-variant CatBoost models

Plots for these CSVs are generated separately by e_plots.py.

Usage:
    python src/d_experiments.py [--force]
"""

import argparse
import math
import time
from collections import Counter
from itertools import groupby
from pathlib import Path

import lz4.frame
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

SKIP_THRESHOLD = 1.05

PROFILES = {
    "FAST":     {"alpha": 0.1, "beta": 0.5},
    "BALANCED": {"alpha": 0.5, "beta": 0.5},
    "COMPRESS": {"alpha": 0.9, "beta": 0.5},
    "ARCHIVE":  {"alpha": 0.9, "beta": 0.8},
    "WEB":      {"alpha": 0.6, "beta": 0.1},
}

ALGO_IDS = [
    "gzip_1",   "gzip_4",   "gzip_6",   "gzip_9",
    "lz4_0",    "lz4_4",    "lz4_9",    "lz4_12",
    "zstd_1",   "zstd_5",   "zstd_12",  "zstd_19",
    "brotli_1", "brotli_5", "brotli_9", "brotli_11",
    "lzma_1",   "lzma_4",   "lzma_6",   "lzma_9",
]

# 0-255: byte histogram, 256-261: stats, 262: lz4_cr
FEAT_SUBSETS = {
    "full":       (0,   263),
    "no_lz4":     (0,   262),
    "stats_only": (256, 262),
}

SAMPLE_SIZES = {
    "16KB":  16_384,
    "64KB":  65_536,
    "256KB": 262_144,
    "full":  None,
}

# feature positions within the 263-feature vector
IDX_ENTROPY = 256
IDX_ASCII_RATIO = 259
IDX_LZ4_CR = 262


# duplicated from b1 — scripts are standalone
def compute_utility(bench_df, alpha, beta):
    cr = bench_df.pivot(index="sha256", columns="algo_id", values="cr")[ALGO_IDS]
    comp = bench_df.pivot(index="sha256", columns="algo_id", values="comp_mbps")[ALGO_IDS]
    decomp = bench_df.pivot(index="sha256", columns="algo_id", values="decomp_mbps")[ALGO_IDS]
    skip_mask = cr.max(axis=1) < SKIP_THRESHOLD

    def mm(df):
        mn = df.min(axis=1)
        mx = df.max(axis=1)
        return df.subtract(mn, axis=0).divide((mx - mn).replace(0.0, 1.0), axis=0)

    utility = alpha * mm(cr) + (1 - alpha) * (beta * mm(comp) + (1 - beta) * mm(decomp))
    return utility, skip_mask


def gain_pct(utility_df, skip_mask, predictions, sha256, stat_algo):
    non_skip = ~skip_mask.loc[sha256].values
    if non_skip.sum() == 0:
        return float("nan")
    sha_ns = sha256[non_skip]
    pred_ns = predictions[non_skip]
    util_ns = utility_df.loc[sha_ns]
    util_arr = util_ns.values
    col_idx = {c: i for i, c in enumerate(util_ns.columns)}
    U_oracle = util_arr.max(axis=1)
    U_static = util_arr[:, col_idx[stat_algo]]
    pred_ci = np.array([col_idx.get(p, -1) for p in pred_ns])
    U_model = np.where(pred_ci >= 0,
                       util_arr[np.arange(len(sha_ns)), np.maximum(pred_ci, 0)], 0.0)
    denom = U_oracle - U_static
    has_signal = denom > 1e-9
    if has_signal.sum() == 0:
        return float("nan")
    return float(np.mean((U_model[has_signal] - U_static[has_signal]) / denom[has_signal]) * 100)


# replicated from a3
ASCII_LO, ASCII_HI = 0x20, 0x7E


def extract_features(data):
    size = len(data)
    freq = Counter(data)
    counts = [freq.get(i, 0) for i in range(256)]
    hist = [c / size if size > 0 else 0.0 for c in counts]
    entropy = sum(-(c/size) * math.log2(c/size) for c in counts if c > 0) if size > 0 else 0.0
    size_log2 = math.log2(size) if size > 0 else 0.0
    zero_ratio = counts[0] / size if size > 0 else 0.0
    ascii_ratio = sum(counts[ASCII_LO:ASCII_HI+1]) / size if size > 0 else 0.0
    unique_b = sum(1 for c in counts if c > 0)
    longest_run = max((sum(1 for _ in g) for _, g in groupby(data)), default=0)
    lz4_comp = lz4.frame.compress(data)
    lz4_cr = size / len(lz4_comp) if len(lz4_comp) > 0 else 1.0
    return hist + [entropy, size_log2, zero_ratio, ascii_ratio, unique_b, longest_run, lz4_cr]


def load_or_extract_sampled(registry_df, clean_dir, max_bytes, cache_path, force=False):
    if cache_path.exists() and not force:
        print(f"    Loading cached features from {cache_path.name} ...", flush=True)
        npz = np.load(cache_path, allow_pickle=True)
        sha_arr = npz["sha256"]
        X_arr = npz["X"]
        return dict(zip(sha_arr, X_arr))

    print(f"    Extracting features (max_bytes={max_bytes}) for {len(registry_df)} files ...",
          flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features = {}
    times_ms = []
    skipped = 0

    for _, row in registry_df.iterrows():
        fpath = clean_dir / row["rel_path"]
        if not fpath.exists():
            skipped += 1
            continue
        data = fpath.read_bytes()
        if max_bytes is not None:
            data = data[:max_bytes]
        t0 = time.perf_counter()
        features[row["sha256"]] = extract_features(data)
        times_ms.append((time.perf_counter() - t0) * 1000)

    if times_ms:
        print(f"    Done: {len(features)} files, "
              f"mean {np.mean(times_ms):.1f}ms, "
              f"median {np.median(times_ms):.1f}ms/file "
              f"(skipped={skipped})", flush=True)

    sha_arr = np.array(list(features.keys()), dtype=str)
    X_arr = np.array(list(features.values()), dtype=np.float32)
    np.savez_compressed(cache_path, sha256=sha_arr, X=X_arr)
    return features


def build_split_matrix(features_dict, sha256_arr):
    return np.array([features_dict[h] for h in sha256_arr], dtype=np.float32)


def train_catboost(X_tr, y_tr):
    m = CatBoostClassifier(auto_class_weights="Balanced", random_seed=123, verbose=0)
    m.fit(X_tr, y_tr)
    return m


def load_or_train(model_path, X_tr, y_tr, force=False):
    if model_path.exists() and not force:
        m = CatBoostClassifier()
        m.load_model(str(model_path))
        return m
    m = train_catboost(X_tr, y_tr)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    m.save_model(str(model_path))
    return m


def ablation_feature_subsets(models_dir_d, bench_val,
                             sha256_train, sha256_val,
                             X_train_full, X_val_full,
                             labels_train_df, labels_val_df,
                             static_by_profile, out_dir, force):
    print("\n--- 1. Feature Subset Ablation ---")
    rows = []

    for profile, params in PROFILES.items():
        alpha, beta = params["alpha"], params["beta"]
        stat_algo = static_by_profile[profile]
        util_val, skip_val = compute_utility(bench_val, alpha, beta)
        y_train = labels_train_df.loc[sha256_train, profile].values

        for subset_name, (lo, hi) in FEAT_SUBSETS.items():
            X_tr = X_train_full[:, lo:hi]
            X_va = X_val_full[:, lo:hi]

            model_path = models_dir_d / f"subset_{subset_name}_{profile}.cbm"
            t0 = time.perf_counter()
            model = load_or_train(model_path, X_tr, y_train, force=force)
            train_s = time.perf_counter() - t0

            preds = np.array(model.predict(X_va), dtype=str).ravel()
            g = gain_pct(util_val, skip_val, preds, sha256_val, stat_algo)

            g_str = f"{g:.1f}%" if not np.isnan(g) else "N/A"
            print(f"  {profile:<10} {subset_name:<12} n_feat={hi-lo:>3}  "
                  f"gain={g_str:>7}  train={train_s:.0f}s")

            rows.append({
                "profile": profile,
                "subset": subset_name,
                "n_features": hi - lo,
                "gain_pct": round(g, 2) if not np.isnan(g) else None,
                "train_s": round(train_s, 1),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "d_feature_subsets.csv", index=False)
    print(f"  Saved: d_feature_subsets.csv")
    return df


def ablation_sampling(models_dir_d, feat_dir,
                     registry_df, clean_dir,
                     sha256_train, sha256_val,
                     bench_val, labels_train_df, labels_val_df,
                     static_by_profile, out_dir, force):
    print("\n--- 2. Sampling Ablation ---")
    rows = []

    for size_label, max_bytes in SAMPLE_SIZES.items():
        print(f"\n  Sample size: {size_label}")
        cache_path = feat_dir / f"sampled_{size_label}.npz"
        feat_dict = load_or_extract_sampled(
            registry_df, clean_dir, max_bytes, cache_path, force=force)

        # extraction time on a small sample for reporting
        sample_hashes = list(feat_dict.keys())[:200]
        times_ms = []
        for h in sample_hashes:
            row = registry_df[registry_df["sha256"] == h]
            if row.empty:
                continue
            fpath = clean_dir / row.iloc[0]["rel_path"]
            if not fpath.exists():
                continue
            data = fpath.read_bytes()
            if max_bytes:
                data = data[:max_bytes]
            t0 = time.perf_counter()
            extract_features(data)
            times_ms.append((time.perf_counter() - t0) * 1000)

        mean_ms = float(np.mean(times_ms)) if times_ms else float("nan")

        X_tr = build_split_matrix(feat_dict, sha256_train)
        X_va = build_split_matrix(feat_dict, sha256_val)

        for profile, params in PROFILES.items():
            alpha, beta = params["alpha"], params["beta"]
            stat_algo = static_by_profile[profile]
            util_val, skip_val = compute_utility(bench_val, alpha, beta)
            y_train = labels_train_df.loc[sha256_train, profile].values

            model_path = models_dir_d / f"sample_{size_label}_{profile}.cbm"
            t0 = time.perf_counter()
            model = load_or_train(model_path, X_tr, y_train, force=force)
            train_s = time.perf_counter() - t0

            preds = np.array(model.predict(X_va), dtype=str).ravel()
            g = gain_pct(util_val, skip_val, preds, sha256_val, stat_algo)

            g_str = f"{g:.1f}%" if not np.isnan(g) else "N/A"
            print(f"    {profile:<10} gain={g_str:>7}  train={train_s:.0f}s  "
                  f"extract={mean_ms:.1f}ms/file")

            rows.append({
                "sample_size": size_label,
                "max_bytes": max_bytes if max_bytes else -1,
                "profile": profile,
                "gain_pct": round(g, 2) if not np.isnan(g) else None,
                "train_s": round(train_s, 1),
                "extract_ms": round(mean_ms, 1),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "d_sampling.csv", index=False)
    print(f"  Saved: d_sampling.csv")
    return df


RULES = {
    "FAST": lambda e, a: "lz4_0",
    "BALANCED": lambda e, a: (
        "brotli_11" if a > 0.6 or e < 5.0
        else "zstd_1" if e < 7.0
        else "lz4_0"
    ),
    "COMPRESS": lambda e, a: (
        "brotli_11" if a > 0.4 or e < 6.5
        else "zstd_19" if e < 7.5
        else "lz4_0"
    ),
    "ARCHIVE": lambda e, a: (
        "brotli_11" if a > 0.4 or e < 6.0
        else "lzma_6" if e < 7.5
        else "zstd_1"
    ),
    "WEB": lambda e, a: (
        "brotli_11" if a > 0.5 or e < 5.5
        else "zstd_19" if e < 7.0
        else "lz4_0"
    ),
}


def ablation_rule_based(X_test, sha256_test, bench_test,
                        labels_test_df, static_by_profile, out_dir):
    print("\n--- 3. Rule-Based Baseline (test set) ---")
    rows = []

    for profile, params in PROFILES.items():
        alpha, beta = params["alpha"], params["beta"]
        stat_algo = static_by_profile[profile]
        util_test, skip_test = compute_utility(bench_test, alpha, beta)
        y_true = labels_test_df.loc[sha256_test, profile].values

        entropy = X_test[:, IDX_ENTROPY]
        ascii_ratio = X_test[:, IDX_ASCII_RATIO]
        lz4_cr_vals = X_test[:, IDX_LZ4_CR]
        rule_fn = RULES[profile]

        preds = np.array([
            "SKIP" if lz4_cr_vals[i] < SKIP_THRESHOLD
            else rule_fn(float(entropy[i]), float(ascii_ratio[i]))
            for i in range(len(sha256_test))
        ], dtype=str)

        g = gain_pct(util_test, skip_test, preds, sha256_test, stat_algo)
        acc = float((preds == y_true).mean())
        g_str = f"{g:.1f}%" if not np.isnan(g) else "N/A"
        print(f"  {profile:<10} gain={g_str:>7}  acc={acc:.4f}")

        rows.append({
            "profile": profile,
            "gain_pct": round(g, 2) if not np.isnan(g) else None,
            "accuracy": round(acc, 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "d_rule_based.csv", index=False)
    print(f"  Saved: d_rule_based.csv")
    return df


def main():
    parser = argparse.ArgumentParser(description="D: extended experiments (calculations only).")
    parser.add_argument("--splits", default="data/splits")
    parser.add_argument("--clean", default="data/clean")
    parser.add_argument("--out", default="data/analysis")
    parser.add_argument("--force", action="store_true",
                        help="Ignore caches and re-extract/re-train everything")
    args = parser.parse_args()

    splits_dir = Path(args.splits)
    clean_dir = Path(args.clean)
    out_dir = Path(args.out)
    models_dir_d = Path("data/d_models")
    feat_dir = Path("data/d_features")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading shared data ...")
    train_npz = np.load(splits_dir / "train.npz", allow_pickle=True)
    val_npz = np.load(splits_dir / "val.npz", allow_pickle=True)
    test_npz = np.load(splits_dir / "test.npz", allow_pickle=True)

    X_train_full = train_npz["X"].astype(np.float32)
    sha256_train = train_npz["sha256"]
    X_val_full = val_npz["X"].astype(np.float32)
    sha256_val = val_npz["sha256"]
    X_test = test_npz["X"].astype(np.float32)
    sha256_test = test_npz["sha256"]

    bench_val = pd.read_csv(splits_dir / "benchmarks_val.csv",
                            dtype={"sha256": str, "algo_id": str,
                                   "cr": float, "comp_mbps": float, "decomp_mbps": float})
    bench_test = pd.read_csv(splits_dir / "benchmarks_test.csv",
                             dtype={"sha256": str, "algo_id": str,
                                    "cr": float, "comp_mbps": float, "decomp_mbps": float})

    labels_train_df = pd.read_csv(splits_dir / "labels_train.csv", dtype=str).set_index("sha256")
    labels_val_df = pd.read_csv(splits_dir / "labels_val.csv", dtype=str).set_index("sha256")
    labels_test_df = pd.read_csv(splits_dir / "labels_test.csv", dtype=str).set_index("sha256")

    b1_df = pd.read_csv(splits_dir / "b1_results.csv")
    static_by_profile = b1_df.groupby("profile")["static_algo"].first().to_dict()

    registry_df = pd.read_csv("data/file_registry.csv", dtype=str)

    print(f"  Train: {X_train_full.shape}  Val: {X_val_full.shape}  Test: {X_test.shape}")

    # B1 reference (CatBoost, full features, val, t=0)
    b1_ref = b1_df[b1_df["model"] == "CatBoost"].set_index("profile")["gain_pct"].to_dict()
    print("\n  B1 reference (CatBoost val Gain%, t=0):")
    for p, g in b1_ref.items():
        print(f"    {p:<10} {g:>6.1f}%")

    ablation_feature_subsets(
        models_dir_d, bench_val,
        sha256_train, sha256_val,
        X_train_full, X_val_full,
        labels_train_df, labels_val_df,
        static_by_profile, out_dir, args.force,
    )

    ablation_sampling(
        models_dir_d, feat_dir,
        registry_df, clean_dir,
        sha256_train, sha256_val,
        bench_val, labels_train_df, labels_val_df,
        static_by_profile, out_dir, args.force,
    )

    ablation_rule_based(
        X_test, sha256_test, bench_test,
        labels_test_df, static_by_profile, out_dir,
    )

    print(f"\nOutputs in {out_dir}/")


if __name__ == "__main__":
    main()
