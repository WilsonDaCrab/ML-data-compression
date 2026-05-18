"""
C: analysis phase. Computes five sets of analysis results on B3 outputs and
writes them as CSVs to data/analysis/.

  1. Feature importance   (CatBoost & RF top features per profile)
  2. Type-group breakdown (CatBoost Gain%/accuracy per file type per profile)
  3. Confusion matrices   (CatBoost predicted vs true class per profile)
  4. Error analysis       (per type_group error rate, top confused pairs)
  5. Prediction timing    (feature extraction and inference latency)

Plots for these CSVs are generated separately by e_plots.py.

Usage:
    python src/c_analysis.py [--timing-n 100]
"""

import argparse
import math
import time
from collections import Counter
from itertools import groupby
from pathlib import Path

import joblib
import lz4.frame
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

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

FEAT_GROUPS = {
    "histogram": [f"byte_{i:02x}" for i in range(256)],
    "stats": ["entropy", "file_size_log2", "zero_byte_ratio",
              "ascii_ratio", "unique_bytes", "longest_run"],
    "lz4_cr": ["lz4_cr"],
}


# duplicated from b1 — scripts are standalone
def compute_utility(bench_df, alpha, beta):
    cr = bench_df.pivot(index="sha256", columns="algo_id", values="cr")[ALGO_IDS]
    comp = bench_df.pivot(index="sha256", columns="algo_id", values="comp_mbps")[ALGO_IDS]
    decomp = bench_df.pivot(index="sha256", columns="algo_id", values="decomp_mbps")[ALGO_IDS]
    skip_mask = cr.max(axis=1) < SKIP_THRESHOLD

    def minmax_norm(df):
        mn = df.min(axis=1)
        mx = df.max(axis=1)
        rng = (mx - mn).replace(0.0, 1.0)
        return df.subtract(mn, axis=0).divide(rng, axis=0)

    utility = (alpha * minmax_norm(cr)
               + (1 - alpha) * (beta * minmax_norm(comp)
                                + (1 - beta) * minmax_norm(decomp)))
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


def load_catboost(profile, models_dir):
    m = CatBoostClassifier()
    m.load_model(str(models_dir / f"{profile}_catboost.cbm"))
    return m


def load_rf(profile, models_dir):
    return joblib.load(models_dir / f"{profile}_rf.joblib")


def predict_with_threshold(model, X, t_star, stat_algo):
    if t_star == 0.0:
        return np.array(model.predict(X), dtype=str).ravel()
    proba = model.predict_proba(X)
    classes = np.array(model.classes_, dtype=str)
    top_conf = proba.max(axis=1)
    top_pred = classes[proba.argmax(axis=1)]
    return np.where(top_conf >= t_star, top_pred, stat_algo)


# replicated from a3 for the timing analysis
ASCII_LO, ASCII_HI = 0x20, 0x7E


def extract_features_bytes(data):
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
    size_log2 = math.log2(size) if size > 0 else 0.0
    zero_ratio = counts[0] / size if size > 0 else 0.0
    ascii_ratio = sum(counts[ASCII_LO:ASCII_HI + 1]) / size if size > 0 else 0.0
    unique_bytes = sum(1 for c in counts if c > 0)
    longest_run = max((sum(1 for _ in g) for _, g in groupby(data)), default=0)
    lz4_compressed = lz4.frame.compress(data)
    lz4_cr = size / len(lz4_compressed) if len(lz4_compressed) > 0 else 1.0
    return hist + [entropy, size_log2, zero_ratio, ascii_ratio, unique_bytes, longest_run, lz4_cr]


def analysis_feature_importance(profiles, feat_names, models_dir, out_dir):
    print("\n--- 1. Feature Importance ---")
    rows = []

    feat_to_group = {}
    for grp, feats in FEAT_GROUPS.items():
        for f in feats:
            feat_to_group[f] = grp

    for profile in profiles:
        for model_name, loader in [("CatBoost", load_catboost), ("RandomForest", load_rf)]:
            model = loader(profile, models_dir)

            if model_name == "CatBoost":
                importances = model.get_feature_importance()
            else:
                importances = model.feature_importances_

            for fname, imp in zip(feat_names, importances):
                rows.append({
                    "profile": profile,
                    "model": model_name,
                    "feature": fname,
                    "group": feat_to_group.get(fname, "other"),
                    "importance": round(float(imp), 6),
                })

            top_idx = np.argsort(importances)[-5:][::-1]
            top_feat = [feat_names[i] for i in top_idx]
            top_imp = importances[top_idx]
            print(f"  {profile} {model_name}: top 5 = "
                  + ", ".join(f"{top_feat[i]}({top_imp[i]:.4f})" for i in range(5)))

    imp_df = pd.DataFrame(rows)
    imp_df.to_csv(out_dir / "feature_importance.csv", index=False)
    print(f"  Saved: feature_importance.csv")
    return imp_df


def analysis_type_group(profiles, X_test, sha256_test, bench_test,
                        labels_test_df, thresh_df, splits_index,
                        models_dir, out_dir):
    print("\n--- 2. Per-Type-Group Breakdown (CatBoost) ---")

    tg_map = splits_index.set_index("sha256")["type_group"].to_dict()
    tg_arr = np.array([tg_map.get(h, "unknown") for h in sha256_test])
    type_groups = sorted(splits_index["type_group"].unique())

    rows = []

    for profile, params in profiles.items():
        alpha, beta = params["alpha"], params["beta"]
        t_row = thresh_df[(thresh_df["profile"] == profile) &
                          (thresh_df["model"] == "CatBoost")].iloc[0]
        t_star = float(t_row["t_star"])
        stat_algo = str(t_row["static_algo"])

        model = load_catboost(profile, models_dir)
        preds = predict_with_threshold(model, X_test, t_star, stat_algo)
        y_true = labels_test_df.loc[sha256_test, profile].values

        util_test, skip_test = compute_utility(bench_test, alpha, beta)

        for tg in type_groups:
            mask = tg_arr == tg
            if mask.sum() == 0:
                continue
            sha_tg = sha256_test[mask]
            pred_tg = preds[mask]
            ytrue_tg = y_true[mask]
            n = int(mask.sum())
            acc = float(accuracy_score(ytrue_tg, pred_tg))
            g = gain_pct(util_test, skip_test, pred_tg, sha_tg, stat_algo)

            if np.isnan(g):
                non_skip_n = int((~skip_test.loc[sha_tg].values).sum())
                nan_reason = "all_skip" if non_skip_n == 0 else "optimal"
            else:
                nan_reason = None

            rows.append({
                "profile": profile,
                "type_group": tg,
                "n_files": n,
                "accuracy": round(acc, 4),
                "gain_pct": round(g, 2) if not np.isnan(g) else None,
                "nan_reason": nan_reason,
            })

            g_str = f"{g:.1f}%" if not np.isnan(g) else f"  {nan_reason}"
            print(f"  {profile:<10} {tg:<20} n={n:>4}  acc={acc:.3f}  gain={g_str}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "type_group_breakdown.csv", index=False)
    print(f"  Saved: type_group_breakdown.csv")
    return df


def analysis_confusion(profiles, X_test, sha256_test, labels_test_df,
                       thresh_df, models_dir, out_dir):
    print("\n--- 3. Confusion Matrices (CatBoost, non-SKIP files) ---")
    rows = []

    for profile in profiles:
        t_row = thresh_df[(thresh_df["profile"] == profile) &
                          (thresh_df["model"] == "CatBoost")].iloc[0]
        t_star = float(t_row["t_star"])
        stat_algo = str(t_row["static_algo"])

        model = load_catboost(profile, models_dir)
        preds = predict_with_threshold(model, X_test, t_star, stat_algo)
        y_true = labels_test_df.loc[sha256_test, profile].values

        non_skip_mask = y_true != "SKIP"
        y_ns = y_true[non_skip_mask]
        p_ns = preds[non_skip_mask]

        classes = sorted(set(y_ns) | set(p_ns))
        cm = confusion_matrix(y_ns, p_ns, labels=classes)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        cm_df.to_csv(out_dir / f"confusion_{profile}.csv")

        for i, true_cls in enumerate(classes):
            for j, pred_cls in enumerate(classes):
                if cm[i, j] > 0:
                    rows.append({
                        "profile": profile,
                        "true": true_cls,
                        "pred": pred_cls,
                        "count": int(cm[i, j]),
                        "rate": round(float(cm_norm[i, j]), 4),
                    })

        diag_acc = np.diag(cm_norm).mean()
        print(f"  {profile}: {len(classes)} classes, mean per-class recall = {diag_acc:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "confusion_all.csv", index=False)
    print(f"  Saved: confusion_{{PROFILE}}.csv, confusion_all.csv")
    return df


def analysis_errors(profiles, X_test, sha256_test, labels_test_df,
                    thresh_df, splits_index, models_dir, out_dir):
    print("\n--- 4. Error Analysis (CatBoost) ---")

    tg_map = splits_index.set_index("sha256")["type_group"].to_dict()
    tg_arr = np.array([tg_map.get(h, "unknown") for h in sha256_test])

    rows = []

    for profile in profiles:
        t_row = thresh_df[(thresh_df["profile"] == profile) &
                          (thresh_df["model"] == "CatBoost")].iloc[0]
        t_star = float(t_row["t_star"])
        stat_algo = str(t_row["static_algo"])

        model = load_catboost(profile, models_dir)
        preds = predict_with_threshold(model, X_test, t_star, stat_algo)
        y_true = labels_test_df.loc[sha256_test, profile].values

        non_skip = y_true != "SKIP"
        y_ns = y_true[non_skip]
        p_ns = preds[non_skip]
        tg_ns = tg_arr[non_skip]
        correct = y_ns == p_ns

        for tg in sorted(set(tg_ns)):
            m = tg_ns == tg
            if m.sum() == 0:
                continue
            err = 1.0 - float(correct[m].mean())
            rows.append({
                "profile": profile,
                "type_group": tg,
                "n_non_skip": int(m.sum()),
                "n_correct": int(correct[m].sum()),
                "n_wrong": int((~correct[m]).sum()),
                "error_rate": round(err, 4),
            })

        wrong_mask = ~correct
        if wrong_mask.sum() > 0:
            pairs = Counter(zip(y_ns[wrong_mask], p_ns[wrong_mask]))
            top3 = pairs.most_common(3)
            print(f"  {profile}: error rate={1-correct.mean():.3f}, "
                  f"top confused pairs: "
                  + ", ".join(f"{t}->{p}({c})" for (t, p), c in top3))
        else:
            print(f"  {profile}: no errors on non-SKIP files")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "error_analysis.csv", index=False)
    print(f"  Saved: error_analysis.csv")
    return df


def analysis_timing(profiles, X_test, sha256_test, thresh_df,
                    registry_df, clean_dir, models_dir, out_dir, timing_n=100):
    print("\n--- 5. Prediction Timing ---")
    rows = []

    print("  Inference timing (predict_proba on test set):")
    for profile in profiles:
        model = load_catboost(profile, models_dir)
        # warm-up
        _ = model.predict_proba(X_test[:10])
        t0 = time.perf_counter()
        _ = model.predict_proba(X_test)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_file = elapsed_ms / len(X_test)
        print(f"    {profile:<10}: {elapsed_ms:.1f} ms total, "
              f"{per_file:.3f} ms/file  (n={len(X_test)})")
        rows.append({
            "stage": "inference",
            "profile": profile,
            "n_files": len(X_test),
            "total_ms": round(elapsed_ms, 2),
            "per_file_ms": round(per_file, 4),
        })

    print(f"  Feature extraction timing (n={timing_n} files):")
    test_hashes = set(sha256_test)
    test_reg = registry_df[registry_df["sha256"].isin(test_hashes)].copy()
    sample = test_reg.sample(min(timing_n, len(test_reg)), random_state=42)

    extract_times = []
    sizes_mb = []
    skipped = 0
    for _, row in sample.iterrows():
        fpath = clean_dir / row["rel_path"]
        if not fpath.exists():
            skipped += 1
            continue
        data = fpath.read_bytes()
        t0 = time.perf_counter()
        _ = extract_features_bytes(data)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        extract_times.append(elapsed_ms)
        sizes_mb.append(len(data) / 1_048_576)

    if extract_times:
        mean_ms = float(np.mean(extract_times))
        median_ms = float(np.median(extract_times))
        mean_mb = float(np.mean(sizes_mb))
        ms_per_mb = mean_ms / mean_mb if mean_mb > 0 else float("nan")
        print(f"    Mean: {mean_ms:.1f} ms/file (median {median_ms:.1f} ms)")
        print(f"    Mean file size: {mean_mb:.2f} MB")
        if ms_per_mb > 0:
            print(f"    Throughput: {ms_per_mb:.1f} ms/MB ({1000/ms_per_mb:.0f} MB/s)")
        print(f"    Files skipped (not found): {skipped}/{len(sample)}")
        rows.append({
            "stage": "feature_extraction",
            "profile": "all",
            "n_files": len(extract_times),
            "total_ms": round(sum(extract_times), 2),
            "per_file_ms": round(mean_ms, 2),
            "median_ms": round(median_ms, 2),
            "mean_size_mb": round(mean_mb, 3),
            "ms_per_mb": round(ms_per_mb, 2) if not np.isnan(ms_per_mb) else None,
        })
    else:
        print(f"    No files found (skipped={skipped}) -- check --clean path")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "timing.csv", index=False)
    print(f"  Saved: timing.csv")
    return df


def main():
    parser = argparse.ArgumentParser(description="C: analysis phase (calculations only).")
    parser.add_argument("--splits", default="data/splits")
    parser.add_argument("--models", default="data/models")
    parser.add_argument("--clean", default="data/clean")
    parser.add_argument("--out", default="data/analysis")
    parser.add_argument("--timing-n", type=int, default=100,
                        help="Number of files for feature extraction timing")
    args = parser.parse_args()

    splits_dir = Path(args.splits)
    models_dir = Path(args.models)
    clean_dir = Path(args.clean)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    test_npz = np.load(splits_dir / "test.npz", allow_pickle=True)
    X_test = test_npz["X"].astype(np.float32)
    sha256_test = test_npz["sha256"]

    bench_test = pd.read_csv(splits_dir / "benchmarks_test.csv",
                             dtype={"sha256": str, "algo_id": str,
                                    "cr": float, "comp_mbps": float,
                                    "decomp_mbps": float})
    labels_test = pd.read_csv(splits_dir / "labels_test.csv",
                              dtype=str).set_index("sha256")
    thresh_df = pd.read_csv(splits_dir / "b2_thresholds.csv")
    splits_index = pd.read_csv(splits_dir / "splits_index.csv", dtype=str)
    splits_index = splits_index[splits_index["split"] == "test"]
    registry_df = pd.read_csv("data/file_registry.csv", dtype=str)

    feat_names = (splits_dir / "feature_names.txt").read_text().splitlines()

    print(f"  X_test: {X_test.shape}  features: {len(feat_names)}")

    analysis_feature_importance(PROFILES, feat_names, models_dir, out_dir)

    analysis_type_group(PROFILES, X_test, sha256_test, bench_test,
                        labels_test, thresh_df, splits_index,
                        models_dir, out_dir)

    analysis_confusion(PROFILES, X_test, sha256_test, labels_test,
                       thresh_df, models_dir, out_dir)

    analysis_errors(PROFILES, X_test, sha256_test, labels_test,
                    thresh_df, splits_index, models_dir, out_dir)

    analysis_timing(PROFILES, X_test, sha256_test, thresh_df,
                    registry_df, clean_dir, models_dir, out_dir,
                    timing_n=args.timing_n)

    print(f"\nOutputs in {out_dir}/")


if __name__ == "__main__":
    main()
