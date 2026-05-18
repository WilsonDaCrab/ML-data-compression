"""
B3: final evaluation on the held-out test set.

Applies each trained model with its B2-tuned confidence threshold (1-NN runs
without a threshold). Reports accuracy, F1, Gain%, SKIP accuracy and fallback
rate per profile and per model.

Statistical significance is tested with the Wilcoxon signed-rank test
(one-sided, alternative='greater') on per-file (U_model - U_static) for
non-SKIP files. Applied to CatBoost and the best-Gain model per profile.

Usage:
    python src/b3_test.py
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score

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

ALL_MODELS = ["DecisionTree", "RandomForest", "CatBoost", "1-NN"]


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


def make_labels(utility, skip_mask):
    labels = utility.idxmax(axis=1)
    labels[skip_mask] = "SKIP"
    return labels


def compute_gain_pct(utility_df, skip_mask, predictions, sha256, stat_algo):
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


def compute_skip_acc(skip_mask, predictions, sha256):
    is_skip = skip_mask.loc[sha256].values
    if is_skip.sum() == 0:
        return float("nan")
    return float((predictions[is_skip] == "SKIP").mean())


def per_file_utility_diff(utility_df, skip_mask, predictions, sha256, stat_algo):
    # (U_model - U_static) for non-SKIP files only
    non_skip = ~skip_mask.loc[sha256].values
    sha_ns = sha256[non_skip]
    pred_ns = predictions[non_skip]

    util_ns = utility_df.loc[sha_ns]
    util_arr = util_ns.values
    col_idx = {c: i for i, c in enumerate(util_ns.columns)}

    U_static = util_arr[:, col_idx[stat_algo]]
    pred_ci = np.array([col_idx.get(p, -1) for p in pred_ns])
    U_model = np.where(
        pred_ci >= 0,
        util_arr[np.arange(len(sha_ns)), np.maximum(pred_ci, 0)],
        0.0,
    )
    return U_model - U_static


def load_model(profile, model_name, models_dir):
    if model_name == "CatBoost":
        m = CatBoostClassifier()
        m.load_model(str(models_dir / f"{profile}_catboost.cbm"))
        return m
    key = {"DecisionTree": "dt", "RandomForest": "rf", "1-NN": "knn"}[model_name]
    return joblib.load(models_dir / f"{profile}_{key}.joblib")


def predict_with_threshold(model, X, t_star, stat_algo, model_name):
    if model_name == "1-NN" or t_star == 0.0:
        return np.array(model.predict(X), dtype=str).ravel()

    proba = model.predict_proba(X)
    classes = np.array(model.classes_, dtype=str)
    top_conf = proba.max(axis=1)
    top_pred = classes[proba.argmax(axis=1)]
    return np.where(top_conf >= t_star, top_pred, stat_algo)


def main():
    parser = argparse.ArgumentParser(description="B3: final test set evaluation.")
    parser.add_argument("--splits", default="data/splits")
    parser.add_argument("--models", default="data/models")
    args = parser.parse_args()

    splits_dir = Path(args.splits)
    models_dir = Path(args.models)

    print("Loading test split ...")
    test_npz = np.load(splits_dir / "test.npz", allow_pickle=True)
    X_test = test_npz["X"].astype(np.float32)
    sha256_test = test_npz["sha256"]
    print(f"  X_test: {X_test.shape}")

    print("Loading test benchmarks ...")
    bench_test = pd.read_csv(
        splits_dir / "benchmarks_test.csv",
        dtype={"sha256": str, "algo_id": str,
               "cr": float, "comp_mbps": float, "decomp_mbps": float},
    )

    print("Loading B2 thresholds and B1 static algos ...")
    thresh_df = pd.read_csv(splits_dir / "b2_thresholds.csv")
    t_star_map = thresh_df.set_index(["profile", "model"])["t_star"].to_dict()
    static_map = thresh_df.set_index(["profile", "model"])["static_algo"].to_dict()

    # 1-NN is not in b2_thresholds; get its static from b1_results
    b1_df = pd.read_csv(splits_dir / "b1_results.csv")
    static_by_prof = b1_df.groupby("profile")["static_algo"].first().to_dict()

    all_labels_test = {"sha256": sha256_test}
    summary_rows = []
    sig_rows = []

    for profile, params in PROFILES.items():
        alpha, beta = params["alpha"], params["beta"]
        stat_algo = static_by_prof[profile]

        print(f"\n{'=' * 70}")
        print(f"  Profile: {profile}  (alpha={alpha}, beta={beta})")
        print(f"  Static baseline: {stat_algo}")
        print("=" * 70)

        util_test, skip_test = compute_utility(bench_test, alpha, beta)
        labels_test = make_labels(util_test, skip_test)
        y_test = labels_test.loc[sha256_test].values
        all_labels_test[profile] = y_test

        static_preds = np.full(len(y_test), stat_algo)
        s_acc = accuracy_score(y_test, static_preds)
        s_f1mac = f1_score(y_test, static_preds, average="macro", zero_division=0)
        s_f1wgt = f1_score(y_test, static_preds, average="weighted", zero_division=0)

        print(f"\n  {'Model':<15}  {'t*':>5}  {'Acc':>6}  {'F1-mac':>7}  "
              f"{'F1-wgt':>7}  {'Gain%':>7}  {'SKIP-acc':>9}  {'Fall%':>6}  {'Time':>6}")
        print("  " + "-" * 80)

        best_gain_profile = -np.inf
        best_model_profile = None

        for model_name in ALL_MODELS:
            t0 = time.perf_counter()

            t_star = t_star_map.get((profile, model_name), 0.0)
            stat = static_map.get((profile, model_name), stat_algo)

            model = load_model(profile, model_name, models_dir)
            preds = predict_with_threshold(model, X_test, t_star, stat, model_name)

            if model_name != "1-NN" and t_star > 0.0:
                proba = model.predict_proba(X_test)
                top_conf = proba.max(axis=1)
                fall_pct = float((top_conf < t_star).mean()) * 100
            else:
                fall_pct = 0.0

            acc = accuracy_score(y_test, preds)
            f1_mac = f1_score(y_test, preds, average="macro", zero_division=0)
            f1_wgt = f1_score(y_test, preds, average="weighted", zero_division=0)
            gain = compute_gain_pct(util_test, skip_test, preds, sha256_test, stat)
            skip_acc = compute_skip_acc(skip_test, preds, sha256_test)
            elapsed = time.perf_counter() - t0

            gain_str = f"{gain:>6.1f}%" if not np.isnan(gain) else "    N/A"
            skip_str = f"{skip_acc * 100:>7.1f}%" if not np.isnan(skip_acc) else "    N/A"

            print(f"  {model_name:<15}  {t_star:>4.3f}  {acc:>6.4f}  {f1_mac:>7.4f}  "
                  f"{f1_wgt:>7.4f}  {gain_str}  {skip_str}  {fall_pct:>5.1f}%  {elapsed:>5.1f}s")

            summary_rows.append({
                "profile": profile,
                "model": model_name,
                "t_star": t_star,
                "accuracy": round(acc, 4),
                "f1_macro": round(f1_mac, 4),
                "f1_weighted": round(f1_wgt, 4),
                "gain_pct": round(gain, 2) if not np.isnan(gain) else None,
                "skip_acc": round(skip_acc, 4) if not np.isnan(skip_acc) else None,
                "fallback_rate": round(fall_pct / 100, 4),
                "static_algo": stat,
            })

            if not np.isnan(gain) and gain > best_gain_profile:
                best_gain_profile = gain
                best_model_profile = model_name

        print(f"  {'-- Static':<15}  {'--':>5}  {s_acc:>6.4f}  {s_f1mac:>7.4f}  "
              f"{s_f1wgt:>7.4f}  {'0.0%':>7}  {'N/A':>9}  {'--':>6}  ")

        summary_rows.append({
            "profile": profile, "model": "Static",
            "t_star": None, "accuracy": round(s_acc, 4),
            "f1_macro": round(s_f1mac, 4), "f1_weighted": round(s_f1wgt, 4),
            "gain_pct": 0.0, "skip_acc": None, "fallback_rate": None,
            "static_algo": stat_algo,
        })

        # Wilcoxon signed-rank: CatBoost and best-Gain model vs static
        seen_sig = set()
        for test_model in ["CatBoost", best_model_profile]:
            if test_model is None or test_model in seen_sig:
                continue
            seen_sig.add(test_model)

            t_star_m = t_star_map.get((profile, test_model), 0.0)
            stat_m = static_map.get((profile, test_model), stat_algo)
            model_m = load_model(profile, test_model, models_dir)
            preds_m = predict_with_threshold(
                model_m, X_test, t_star_m, stat_m, test_model
            )

            diffs = per_file_utility_diff(
                util_test, skip_test, preds_m, sha256_test, stat_m
            )

            if len(diffs) > 0 and not np.all(diffs == 0):
                try:
                    stat_w, p_val = wilcoxon(diffs, alternative="greater")
                except ValueError:
                    stat_w, p_val = float("nan"), float("nan")
            else:
                stat_w, p_val = float("nan"), float("nan")

            mean_diff = float(np.mean(diffs)) if len(diffs) > 0 else float("nan")
            n_better = int((diffs > 0).sum())
            n_worse = int((diffs < 0).sum())
            n_same = int((diffs == 0).sum())

            sig_rows.append({
                "profile": profile,
                "model": test_model,
                "n_non_skip": len(diffs),
                "n_better": n_better,
                "n_worse": n_worse,
                "n_same": n_same,
                "mean_u_diff": round(mean_diff, 6),
                "wilcoxon_stat": round(stat_w, 2) if not np.isnan(stat_w) else None,
                "p_value": round(p_val, 6) if not np.isnan(p_val) else None,
                "significant": bool(p_val < 0.05) if not np.isnan(p_val) else None,
            })

            if np.isnan(p_val):
                sig = "ns"
            elif p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            else:
                sig = "ns"
            print(f"\n  Wilcoxon ({test_model} vs Static): "
                  f"p={p_val:.4f} {sig}  "
                  f"mean dU={mean_diff:+.4f}  "
                  f"better={n_better} worse={n_worse} same={n_same}")

    pd.DataFrame(all_labels_test).to_csv(splits_dir / "labels_test.csv", index=False)

    results_df = pd.DataFrame(summary_rows)
    results_df.to_csv(splits_dir / "b3_results.csv", index=False)

    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(splits_dir / "b3_significance.csv", index=False)

    print(f"\nSaved: {splits_dir}/labels_test.csv")
    print(f"Saved: {splits_dir}/b3_results.csv  ({len(results_df)} rows)")
    print(f"Saved: {splits_dir}/b3_significance.csv  ({len(sig_df)} rows)")

    print("\nGain% on test set:")
    print(f"  {'Profile':<10}  {'Model':<15}  {'t*':>5}  "
          f"{'Acc':>6}  {'Gain%':>7}  {'Fall%':>6}")
    print("  " + "-" * 60)
    for row in summary_rows:
        if row["model"] == "Static":
            continue
        gain_str = f"{row['gain_pct']:>6.1f}%" if row["gain_pct"] is not None else "    N/A"
        t_str = f"{row['t_star']:>4.3f}" if row["t_star"] is not None else "  -- "
        fall_str = f"{row['fallback_rate'] * 100:>5.1f}%" if row["fallback_rate"] is not None else "    --"
        print(f"  {row['profile']:<10}  {row['model']:<15}  {t_str}  "
              f"{row['accuracy']:>6.4f}  {gain_str}  {fall_str}")

    print("\nWilcoxon (one-sided):")
    print(f"  {'Profile':<10}  {'Model':<15}  {'p-value':>9}  "
          f"{'Sig':>4}  {'mean dU':>9}  {'Better':>7}  {'Worse':>6}")
    print("  " + "-" * 68)
    for row in sig_rows:
        p_str = f"{row['p_value']:.4f}" if row["p_value"] is not None else "     N/A"
        diff_str = f"{row['mean_u_diff']:+.4f}" if row["mean_u_diff"] is not None else "     N/A"
        if row["p_value"] is None:
            sig = "ns"
        elif row["p_value"] < 0.001:
            sig = "***"
        elif row["p_value"] < 0.01:
            sig = "**"
        elif row["p_value"] < 0.05:
            sig = "*"
        else:
            sig = "ns"
        print(f"  {row['profile']:<10}  {row['model']:<15}  {p_str:>9}  "
              f"{sig:>4}  {diff_str:>9}  {row['n_better']:>7}  {row['n_worse']:>6}")


if __name__ == "__main__":
    main()
