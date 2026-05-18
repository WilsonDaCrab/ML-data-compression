#!/usr/bin/env python3
"""
Benchmark analyzer using the utility function:
    P(f,a) = a*CR'(f,a) + (1-a)*(b*S_comp'(f,a) + (1-b)*S_decomp'(f,a))

Usage:
    python analyze_benchmarks.py data/benchmarks.csv
    python analyze_benchmarks.py data/benchmarks.csv --alpha 0.8 --beta 0.5
    python analyze_benchmarks.py data/benchmarks.csv --sweep
"""
import csv, sys, argparse
from collections import defaultdict, Counter

def parse_csv(path):
    files = defaultdict(dict)
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 5 or row[0].lower() in ("sha256","hash","file","filename"):
                continue
            sha, algo = row[0], row[1]
            files[sha][algo] = (float(row[2]), float(row[3]), float(row[4]))
    return files

def compute_utility(files, alpha, beta):
    utilities, winners = {}, {}
    for sha, algos in files.items():
        crs = [v[0] for v in algos.values()]
        cs = [v[1] for v in algos.values()]
        ds = [v[2] for v in algos.values()]
        cr_min, cr_max = min(crs), max(crs)
        cs_min, cs_max = min(cs), max(cs)
        ds_min, ds_max = min(ds), max(ds)
        cr_r = (cr_max - cr_min) or 1.0
        cs_r = (cs_max - cs_min) or 1.0
        ds_r = (ds_max - ds_min) or 1.0
        fu = {}
        for algo, (cr, c, d) in algos.items():
            cr_n = (cr_max - cr) / cr_r
            cs_n = (c - cs_min) / cs_r
            ds_n = (d - ds_min) / ds_r
            fu[algo] = alpha * cr_n + (1-alpha) * (beta * cs_n + (1-beta) * ds_n)
        utilities[sha] = fu
        best = max(fu, key=fu.get)
        winners[sha] = (best, fu[best])
    return utilities, winners

def compute_static(utilities):
    totals = Counter()
    for fu in utilities.values():
        for a, u in fu.items():
            totals[a] += u
    best = totals.most_common(1)[0]
    return best[0], best[1]

def analyze(files, alpha, beta, verbose=True):
    utilities, winners = compute_utility(files, alpha, beta)
    n = len(files)
    u_oracle = sum(w[1] for w in winners.values())
    static_algo, _ = compute_static(utilities)
    u_static = sum(utilities[sha].get(static_algo, 0) for sha in utilities)
    avg_oracle = u_oracle / n
    avg_static = u_static / n
    headroom = 1.0 - (avg_static / avg_oracle) if avg_oracle > 0 else 0
    static_optimal = sum(1 for sha,(w,_) in winners.items() if w == static_algo)

    family_wins = Counter()
    algo_wins = Counter()
    margins = []
    for sha, (best, best_u) in winners.items():
        algo_wins[best] += 1
        family_wins[best.rsplit("_",1)[0]] += 1
        sv = sorted(utilities[sha].values(), reverse=True)
        if len(sv) > 1:
            margins.append(sv[0] - sv[1])

    if not verbose:
        return {"headroom": headroom, "static_algo": static_algo,
                "avg_oracle": avg_oracle, "avg_static": avg_static,
                "family_wins": family_wins, "algo_wins": algo_wins}

    if alpha >= 0.9:
        desc = "MAX COMPRESSION"
    elif alpha <= 0.1:
        desc = "MAX SPEED"
    elif 0.4 <= alpha <= 0.6:
        desc = "BALANCED"
    else:
        desc = "CUSTOM"

    sep = "=" * 70
    print(f"\n{sep}")
    print(f" SCENARIO: alpha={alpha:.1f} beta={beta:.1f}  --  {desc}")
    print(sep)
    print(f" P(f,a) = {alpha}*CR' + {1-alpha:.1f}*({beta}*S_comp' + {1-beta:.1f}*S_decomp')")
    print(f" Files: {n:,}")

    print(f"\n-- ORACLE vs STATIC BASELINE")
    print(f" Oracle (perfect per-file):    avg P = {avg_oracle:.4f}")
    print(f" Static best = {static_algo:<14} avg P = {avg_static:.4f}")
    print(f" Headroom for ML:              {headroom:.2%}")
    print(f" Static already optimal for:   {static_optimal:,}/{n:,} ({static_optimal/n:.1%})")
    if headroom < 0.01:
        print(f"\n [!] VERY LOW HEADROOM -- static baseline is near-optimal here")
    elif headroom < 0.05:
        print(f"\n [!] LOW HEADROOM -- only {headroom:.2%} room for ML improvement")
    else:
        print(f"\n [ok] Meaningful headroom -- ML model can add real value")

    print(f"\n-- BEST FAMILY (by utility)")
    print(f" {'Family':<12} {'Wins':>7} {'%':>7}  Bar")
    for fam, count in family_wins.most_common():
        pct = count/n
        print(f" {fam:<12} {count:>7,} {pct:>6.1%}  {'#'*int(pct*40)}")

    print(f"\n-- BEST ALGO+LEVEL (top 10)")
    print(f" {'Algo':<15} {'Wins':>7} {'%':>7}  Bar")
    for algo, count in algo_wins.most_common(10):
        pct = count/n
        print(f" {algo:<15} {count:>7,} {pct:>6.1%}  {'#'*int(pct*40)}")

    if margins:
        avg_m = sum(margins)/len(margins)
        tight = sum(1 for m in margins if m < 0.01)
        clear = sum(1 for m in margins if m > 0.10)
        print(f"\n-- WINNER MARGINS")
        print(f" Avg margin (1st vs 2nd):      {avg_m:.4f}")
        print(f" Tight races (<0.01):          {tight:>6,} ({tight/n:.1%})")
        print(f" Clear wins  (>0.10):          {clear:>6,} ({clear/n:.1%})")

    top_fam, top_count = family_wins.most_common(1)[0]
    top_pct = top_count/n
    print(f"\n-- ML FEASIBILITY")
    if top_pct > 0.7:
        print(f" [!] SEVERE IMBALANCE: {top_fam} wins {top_pct:.0%} -> dummy baseline = {top_pct:.1%}")
    elif top_pct > 0.5:
        print(f" [!] MODERATE IMBALANCE: {top_fam} wins {top_pct:.0%} -> dummy baseline = {top_pct:.1%}")
    else:
        print(f" [ok] Reasonable balance -- top: {top_fam} ({top_pct:.0%})")

    return {"headroom": headroom, "static_algo": static_algo,
            "avg_oracle": avg_oracle, "avg_static": avg_static,
            "family_wins": family_wins, "algo_wins": algo_wins}

def sweep(files):
    n = len(files)
    families = ["gzip","lz4","zstd","brotli","lzma"]
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    betas = [0.0, 0.5, 1.0]

    sep = "=" * 70
    print(f"\n{sep}")
    print(f" SCENARIO SWEEP -- alpha x beta grid")
    print(sep)
    print(f" Files: {n:,}\n")

    print(f" {'a':>4} {'b':>4} | {'Static':>14} {'Headroom':>9} | " +
          " ".join(f"{f:>8}" for f in families) + " | Top")

    results = []
    for alpha in alphas:
        for beta in betas:
            r = analyze(files, alpha, beta, verbose=False)
            fd = r["family_wins"]
            top_fam, top_count = fd.most_common(1)[0]
            vals = " ".join(f"{fd.get(f,0)/n:>7.1%}" for f in families)
            print(f" {alpha:>4.1f} {beta:>4.1f} | {r['static_algo']:>14} {r['headroom']:>8.2%} | {vals} | {top_fam} {top_count/n:.0%}")
            results.append({"alpha":alpha,"beta":beta,**r,"top_fam":top_fam,"top_pct":top_count/n})

    print(f"\n-- KEY FINDINGS")
    best_h = max(results, key=lambda x: x["headroom"])
    best_b = min(results, key=lambda x: x["top_pct"])
    worst_b = max(results, key=lambda x: x["top_pct"])
    print(f" Most ML headroom:    a={best_h['alpha']:.1f} b={best_h['beta']:.1f} ({best_h['headroom']:.2%})")
    print(f" Most balanced:       a={best_b['alpha']:.1f} b={best_b['beta']:.1f} (top={best_b['top_pct']:.1%})")
    print(f" Most imbalanced:     a={worst_b['alpha']:.1f} b={worst_b['beta']:.1f} (top={worst_b['top_pct']:.1%})")

    # files with no meaningful spread (incompressible)
    utilities_r, _ = compute_utility(files, 1.0, 0.5)
    flat = sum(1 for fu in utilities_r.values() if max(fu.values())-min(fu.values()) < 0.001)
    print(f"\n Files with ~zero utility spread: {flat:,}/{n:,} ({flat/n:.1%})")
    print(f" These have no meaningful 'best' algo -- label is noise")
    print(f" Effective training set: ~{n-flat:,} files ({(n-flat)/n:.1%})")

    return results

def main():
    p = argparse.ArgumentParser(description="Analyze compression benchmarks with utility function")
    p.add_argument("csv_path")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--sweep", action="store_true", help="Run across alpha x beta grid")
    p.add_argument("--threshold", type=float, default=0.05)
    args = p.parse_args()

    print(f"\n Loading {args.csv_path}...")
    files = parse_csv(args.csv_path)
    n_algos = len(next(iter(files.values())))
    print(f" Loaded {len(files):,} files x {n_algos} algorithms")

    if args.sweep:
        sweep(files)
    else:
        analyze(files, args.alpha, args.beta)

    n = len(files)
    comp = sum(1 for algos in files.values() if (1-min(v[0] for v in algos.values())) >= args.threshold)
    print(f"\n-- COMPRESSIBILITY ({args.threshold:.0%} threshold)")
    print(f" Compressible:     {comp:>7,} ({comp/n:.1%})")
    print(f" Incompressible:   {n-comp:>7,} ({(n-comp)/n:.1%})\n")

if __name__ == "__main__":
    main()
