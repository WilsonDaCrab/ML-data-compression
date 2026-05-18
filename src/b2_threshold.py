"""
B2: confidence threshold tuning on the validation set.

For each profile and each model that supports predict_proba (DT, RF,
CatBoost), sweeps t in [0, 1] with step 0.005 and finds the t* that maximises
Gain% on the validation set. Predictions below the threshold fall back to the
static baseline for that profile.

1-NN is excluded because its predict_proba is 0/1 (no useful soft scores).

Usage:
    python src/b2_threshold.py
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

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

# 1-NN has no useful predict_proba
TUNED_MODELS = ["DecisionTree", "RandomForest", "CatBoost"]

THRESHOLD_STEPS = np.linspace(0.0, 1.0, 201)


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


def compute_gain_pct(utility_val, skip_mask_val, predictions, sha256_val, stat_algo):
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


def load_model(profile, model_name, models_dir):
    if model_name == "CatBoost":
        m = CatBoostClassifier()
        m.load_model(str(models_dir / f"{profile}_catboost.cbm"))
        return m
    key = {"DecisionTree": "dt", "RandomForest": "rf"}[model_name]
    return joblib.load(models_dir / f"{profile}_{key}.joblib")


def main():
    parser = argparse.ArgumentParser(description="B2: confidence threshold tuning.")
    parser.add_argument("--splits", default="data/splits")
    parser.add_argument("--models", default="data/models")
    args = parser.parse_args()

    splits_dir = Path(args.splits)
    models_dir = Path(args.models)

    print("Loading validation split ...")
    val_npz = np.load(splits_dir / "val.npz", allow_pickle=True)
    X_val = val_npz["X"].astype(np.float32)
    sha256_val = val_npz["sha256"]
    print(f"  X_val: {X_val.shape}")

    print("Loading validation benchmarks ...")
    bench_val = pd.read_csv(
        splits_dir / "benchmarks_val.csv",
        dtype={"sha256": str, "algo_id": str,
               "cr": float, "comp_mbps": float, "decomp_mbps": float},
    )

    print("Loading B1 labels and results ...")
    labels_val_df = pd.read_csv(splits_dir / "labels_val.csv", dtype=str)
    labels_val_df = labels_val_df.set_index("sha256")

    b1_df = pd.read_csv(splits_dir / "b1_results.csv")
    static_by_profile = b1_df.groupby("profile")["static_algo"].first().to_dict()
    b1_gain = b1_df.set_index(["profile", "model"])["gain_pct"].to_dict()

    sweep_rows = []
    threshold_rows = []

    for profile, params in PROFILES.items():
        alpha, beta = params["alpha"], params["beta"]
        stat_algo = static_by_profile[profile]
        y_val = labels_val_df.loc[sha256_val, profile].values

        print(f"\n{'=' * 65}")
        print(f"  Profile: {profile}  (alpha={alpha}, beta={beta})")
        print(f"  Static baseline: {stat_algo}")
        print("=" * 65)

        util_val, skip_val = compute_utility(bench_val, alpha, beta)

        print(f"\n  {'Model':<15}  {'B1 Gain%':>9}  {'t*':>6}  "
              f"{'B2 Gain%':>9}  {'Fallback%':>10}  {'Acc@t*':>8}  {'Time':>6}")
        print("  " + "-" * 75)

        for model_name in TUNED_MODELS:
            t0 = time.perf_counter()

            model = load_model(profile, model_name, models_dir)
            proba = model.predict_proba(X_val)
            classes = np.array(model.classes_, dtype=str)
            top_idx = proba.argmax(axis=1)
            top_conf = proba.max(axis=1)
            top_pred = classes[top_idx]

            best_t = 0.0
            best_gain = -np.inf
            best_acc = 0.0
            best_fall = 0.0

            for t in THRESHOLD_STEPS:
                preds = np.where(top_conf >= t, top_pred, stat_algo)
                gain = compute_gain_pct(
                    util_val, skip_val, preds, sha256_val, stat_algo
                )
                if np.isnan(gain):
                    continue

                fallback_rate = float((top_conf < t).mean())
                acc = accuracy_score(y_val, preds)

                sweep_rows.append({
                    "profile": profile,
                    "model": model_name,
                    "threshold": round(float(t), 4),
                    "gain_pct": round(gain, 4),
                    "fallback_rate": round(fallback_rate, 4),
                    "accuracy": round(acc, 4),
                })

                if gain > best_gain:
                    best_gain = gain
                    best_t = float(t)
                    best_fall = fallback_rate
                    best_acc = acc

            elapsed = time.perf_counter() - t0

            b1_g = b1_gain.get((profile, model_name))
            b1_str = f"{b1_g:>8.1f}%" if b1_g is not None else "      N/A"
            b2_str = (f"{best_gain:>8.1f}%"
                      if not np.isinf(best_gain) else "      N/A")

            print(f"  {model_name:<15}  {b1_str}  {best_t:>5.3f}  "
                  f"  {b2_str}  {best_fall * 100:>9.1f}%  "
                  f"{best_acc:>7.4f}  {elapsed:>5.1f}s")

            threshold_rows.append({
                "profile": profile,
                "model": model_name,
                "static_algo": stat_algo,
                "t_star": round(best_t, 4),
                "gain_pct_b1": round(b1_g, 2) if b1_g is not None else None,
                "gain_pct_b2": round(best_gain, 2) if not np.isinf(best_gain) else None,
                "fallback_rate": round(best_fall, 4),
                "accuracy": round(best_acc, 4),
            })

    sweep_df = pd.DataFrame(sweep_rows)
    thresh_df = pd.DataFrame(threshold_rows)

    sweep_df.to_csv(splits_dir / "b2_sweep.csv", index=False)
    thresh_df.to_csv(splits_dir / "b2_thresholds.csv", index=False)

    print(f"\nSaved: {splits_dir}/b2_sweep.csv  ({len(sweep_df)} rows)")
    print(f"Saved: {splits_dir}/b2_thresholds.csv  ({len(thresh_df)} rows)")

    print("\nOptimal thresholds:")
    print(f"  {'Profile':<10}  {'Model':<15}  {'t*':>6}  "
          f"{'B1 Gain%':>9}  {'B2 Gain%':>9}  {'Fallback%':>10}  {'Acc@t*':>8}")
    print("  " + "-" * 78)
    for row in threshold_rows:
        b1_str = f"{row['gain_pct_b1']:>8.1f}%" if row["gain_pct_b1"] is not None else "      N/A"
        b2_str = f"{row['gain_pct_b2']:>8.1f}%" if row["gain_pct_b2"] is not None else "      N/A"
        print(f"  {row['profile']:<10}  {row['model']:<15}  {row['t_star']:>5.3f}  "
              f"  {b1_str}    {b2_str}  {row['fallback_rate'] * 100:>9.1f}%  "
              f"{row['accuracy']:>7.4f}")


if __name__ == "__main__":
    main()
