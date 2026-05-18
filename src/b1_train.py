"""
B1: generate labels per utility profile, train 4 classifiers per profile and
evaluate on the validation set.

Labels: best algorithm by utility, or SKIP if max CR across all algos is below
1.05 (threshold chosen from training-set max CR distribution; see
analyze_skip_threshold.py).

Trained models are written to data/models/. Gain% on val is computed on
non-SKIP files only and used later by B2 to tune the confidence threshold.

Usage:
    python src/b1_train.py
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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


def make_labels(utility, skip_mask):
    labels = utility.idxmax(axis=1)
    labels[skip_mask] = "SKIP"
    return labels


def best_static_algo(utility_train, skip_mask_train):
    # best single algorithm by mean utility on non-SKIP training files
    return utility_train[~skip_mask_train].mean(axis=0).idxmax()


def compute_gain_pct(utility_val, skip_mask_val, predictions, sha256_val, stat_algo):
    # Gain% on non-SKIP files; if model predicts SKIP for non-SKIP, U_model = 0
    non_skip = ~skip_mask_val.loc[sha256_val].values
    if non_skip.sum() == 0:
        return float("nan")

    sha_ns = sha256_val[non_skip]
    pred_ns = predictions[non_skip]

    util_ns = utility_val.loc[sha_ns]
    util_arr = util_ns.values
    col_idx = {c: i for i, c in enumerate(util_ns.columns)}

    U_oracle = util_arr.max(axis=1)
    U_static = util_arr[:, col_idx[stat_algo]]

    pred_ci = np.array([col_idx.get(p, -1) for p in pred_ns])
    U_model = np.where(
        pred_ci >= 0,
        util_arr[np.arange(len(sha_ns)), np.maximum(pred_ci, 0)],
        0.0,
    )

    denom = U_oracle - U_static
    has_signal = denom > 1e-9
    if has_signal.sum() == 0:
        return float("nan")

    return float(np.mean(
        (U_model[has_signal] - U_static[has_signal]) / denom[has_signal]
    ) * 100)


def compute_skip_acc(skip_mask_val, predictions, sha256_val):
    is_skip = skip_mask_val.loc[sha256_val].values
    if is_skip.sum() == 0:
        return float("nan")
    return float((predictions[is_skip] == "SKIP").mean())


def main():
    parser = argparse.ArgumentParser(description="B1: train classifiers per profile.")
    parser.add_argument("--splits", default="data/splits")
    parser.add_argument("--models", default="data/models")
    args = parser.parse_args()

    splits_dir = Path(args.splits)
    models_dir = Path(args.models)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading feature matrices ...")
    train_npz = np.load(splits_dir / "train.npz", allow_pickle=True)
    val_npz = np.load(splits_dir / "val.npz", allow_pickle=True)

    X_train = train_npz["X"].astype(np.float32)
    sha256_train = train_npz["sha256"]
    X_val = val_npz["X"].astype(np.float32)
    sha256_val = val_npz["sha256"]

    print(f"  X_train: {X_train.shape}   X_val: {X_val.shape}")

    print("Loading benchmarks ...")
    bench_train = pd.read_csv(splits_dir / "benchmarks_train.csv",
                              dtype={"sha256": str, "algo_id": str,
                                     "cr": float, "comp_mbps": float,
                                     "decomp_mbps": float})
    bench_val = pd.read_csv(splits_dir / "benchmarks_val.csv",
                            dtype={"sha256": str, "algo_id": str,
                                   "cr": float, "comp_mbps": float,
                                   "decomp_mbps": float})

    all_labels_train = {"sha256": sha256_train}
    all_labels_val = {"sha256": sha256_val}
    summary_rows = []

    for profile, params in PROFILES.items():
        alpha, beta = params["alpha"], params["beta"]
        print(f"\n{'=' * 65}")
        print(f"  Profile: {profile}  (alpha={alpha}, beta={beta})")
        print("=" * 65)

        util_train, skip_train = compute_utility(bench_train, alpha, beta)
        util_val, skip_val = compute_utility(bench_val, alpha, beta)

        labels_train = make_labels(util_train, skip_train)
        labels_val = make_labels(util_val, skip_val)

        y_train = labels_train.loc[sha256_train].values
        y_val = labels_val.loc[sha256_val].values

        all_labels_train[profile] = y_train
        all_labels_val[profile] = y_val

        def print_dist(y, name):
            unique, counts = np.unique(y, return_counts=True)
            order = np.argsort(-counts)
            print(f"\n  Label distribution ({name}):")
            for cls, cnt in zip(unique[order], counts[order]):
                print(f"    {cls:<15}  {cnt:>5}  ({cnt / len(y) * 100:.1f}%)")
            print(f"    Total classes: {len(unique)}")

        print_dist(y_train, "train")
        print_dist(y_val, "val")

        stat = best_static_algo(util_train, skip_train)
        print(f"\n  Static baseline algo: {stat}")

        model_defs = {
            "DecisionTree": DecisionTreeClassifier(
                class_weight="balanced", random_state=123),
            "RandomForest": RandomForestClassifier(
                n_estimators=100, class_weight="balanced",
                random_state=123, n_jobs=-1),
            "CatBoost": CatBoostClassifier(
                auto_class_weights="Balanced",
                random_seed=123, verbose=0),
            "1-NN": KNeighborsClassifier(n_neighbors=1, n_jobs=-1),
        }

        print(f"\n  {'Model':<15}  {'Acc':>6}  {'F1-mac':>7}  "
              f"{'F1-wgt':>7}  {'Gain%':>7}  {'SKIP-acc':>9}  {'Time':>6}")
        print("  " + "-" * 65)

        for model_name, model in model_defs.items():
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            preds = np.array(model.predict(X_val), dtype=str).ravel()
            elapsed = time.perf_counter() - t0

            acc = accuracy_score(y_val, preds)
            f1_mac = f1_score(y_val, preds, average="macro", zero_division=0)
            f1_wgt = f1_score(y_val, preds, average="weighted", zero_division=0)
            gain = compute_gain_pct(util_val, skip_val, preds, sha256_val, stat)
            skip_acc = compute_skip_acc(skip_val, preds, sha256_val)

            gain_str = f"{gain:>6.1f}%" if not np.isnan(gain) else "    N/A"
            skip_str = f"{skip_acc * 100:>7.1f}%" if not np.isnan(skip_acc) else "    N/A"

            print(f"  {model_name:<15}  {acc:>6.4f}  {f1_mac:>7.4f}  "
                  f"{f1_wgt:>7.4f}  {gain_str}  {skip_str}  {elapsed:>5.1f}s")

            summary_rows.append({
                "profile": profile,
                "model": model_name,
                "accuracy": round(acc, 4),
                "f1_macro": round(f1_mac, 4),
                "f1_weighted": round(f1_wgt, 4),
                "gain_pct": round(gain, 2) if not np.isnan(gain) else None,
                "skip_acc": round(skip_acc, 4) if not np.isnan(skip_acc) else None,
                "static_algo": stat,
                "n_classes": len(np.unique(y_train)),
            })

            if model_name == "CatBoost":
                model.save_model(str(models_dir / f"{profile}_catboost.cbm"))
            else:
                key = {"DecisionTree": "dt", "RandomForest": "rf", "1-NN": "knn"}[model_name]
                joblib.dump(model, models_dir / f"{profile}_{key}.joblib")

        # static baseline row for reference
        static_preds = np.full(len(y_val), stat)
        s_acc = accuracy_score(y_val, static_preds)
        s_f1mac = f1_score(y_val, static_preds, average="macro", zero_division=0)
        s_f1wgt = f1_score(y_val, static_preds, average="weighted", zero_division=0)
        print(f"  {'-- Static':<15}  {s_acc:>6.4f}  {s_f1mac:>7.4f}  "
              f"{s_f1wgt:>7.4f}  {'0.0%':>7}  {'N/A':>9}  {'':>6}")

    pd.DataFrame(all_labels_train).to_csv(splits_dir / "labels_train.csv", index=False)
    pd.DataFrame(all_labels_val).to_csv(splits_dir / "labels_val.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(splits_dir / "b1_results.csv", index=False)

    print(f"\nSaved: {splits_dir}/labels_train.csv, labels_val.csv")
    print(f"Saved: {splits_dir}/b1_results.csv")
    print(f"Saved: {models_dir}/  ({len(PROFILES) * 4} model files)")

    print("\nGain% on non-SKIP val files:")
    print(f"  {'Profile':<10}  {'Model':<15}  {'Acc':>6}  {'Gain%':>7}  {'SKIP-acc':>9}")
    print("  " + "-" * 55)
    for row in summary_rows:
        gain_str = f"{row['gain_pct']:>6.1f}%" if row["gain_pct"] is not None else "    N/A"
        skip_str = (f"{row['skip_acc']*100:>7.1f}%"
                    if row["skip_acc"] is not None else "    N/A")
        print(f"  {row['profile']:<10}  {row['model']:<15}  "
              f"{row['accuracy']:>6.4f}  {gain_str}  {skip_str}")


if __name__ == "__main__":
    main()
