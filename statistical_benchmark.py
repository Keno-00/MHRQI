"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Statistical Treatment: Wilcoxon Signed-Rank Test                            ║
║  Benchmark MHRQI vs State-of-the-Art on Medical OCT Images                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Folder Structure:
benchmark/
└── YYYYMMDD_HHMMSS/
    ├── cnv1/
    │   ├── original.png
    │   ├── denoised_bm3d.png
    │   ├── denoised_nlmeans.png
    │   ├── denoised_srad.png
    │   ├── denoised_proposed.png
    │   └── report_*.png
    ├── cnv2/
    │   └── ...
    └── metrics/
        ├── raw_results.json
        ├── summary.json
        ├── speckle_consistency_metrics.png
        └── no-reference_quality.png

SOTA Reference: BM3D (for Full-Reference metrics)
"""

import json
import os
from datetime import datetime

import cv2
import numpy as np
from scipy import stats

import compare_to
import main
import plots

# Medical image paths. Only use DR images for this case. Never use Images in Folders
MEDICAL_IMAGES = [
    "resources/cnv1.jpeg", # 8 CNV
    "resources/cnv2.jpeg",
    "resources/cnv3.jpeg",
    "resources/cnv4.jpeg",
    "resources/cnv5.jpeg",
    "resources/cnv6.jpeg",
    "resources/cnv7.jpeg",
    "resources/cnv8.jpeg",
    
    "resources/dme1.jpeg", # 8 DME
    "resources/dme2.jpeg",
    "resources/dme3.jpeg",
    "resources/dme4.jpeg",
    "resources/dme5.jpeg",
    "resources/dme6.jpeg",
    "resources/dme7.jpeg",
    "resources/dme8.jpeg",
    
    "resources/drusen1.jpeg", # 8 Drusen
    "resources/drusen2.jpeg",
    "resources/drusen3.jpeg",
    "resources/drusen4.jpeg",
    "resources/drusen5.jpeg",
    "resources/drusen6.jpeg",
    "resources/drusen7.jpeg",
    "resources/drusen8.jpeg",
    
    "resources/normal1.jpeg",   # 8 Normal
    "resources/normal2.jpeg",
    "resources/normal3.jpeg",
    "resources/normal4.jpeg",
    "resources/normal5.jpeg",
    "resources/normal6.jpeg",
    "resources/normal7.jpeg",
    "resources/normal8.jpeg",
]

#total 32 images

# Metrics for comparison (No synthetic clean reference available)
SPECKLE_METRICS_LOWER = ["SSI", "SMPI"]  # Lower is better
SPECKLE_METRICS_HIGHER = ["ENL", "CNR", "NSF"]  # Higher is better
STRUCTURAL_METRICS = ["EPF", "EPI", "OMQDI"]  # Higher is better
# NIQE computed but not reported (biased for medical images)
ALL_SPECKLE_METRICS = SPECKLE_METRICS_LOWER + SPECKLE_METRICS_HIGHER

# Methods to compare
METHODS = ["bm3d", "nlmeans", "srad", "proposed"]


def run_benchmark(n=64, strength=1.65):
    """
    Run benchmark on all medical images and collect metrics.
    
    Folder structure:
    - benchmark/timestamp/imagename/ for each image
    - benchmark/timestamp/metrics/ for aggregated stats
    
    Args:
        n: Image size
        strength: Denoiser strength parameter
    
    Returns:
        results: Dict mapping image -> method -> metrics
        base_dir: Path to benchmark output directory
    """
    # Create base directory: benchmark/timestamp/ (matching runs/ format)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')  # Same as runs/ folder
    base_dir = os.path.join("benchmark", timestamp)
    metrics_dir = os.path.join(base_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    all_results = {}

    for img_path in MEDICAL_IMAGES:
        img_name = os.path.basename(img_path).replace(".jpeg", "")
        img_dir = os.path.join(base_dir, img_name)
        os.makedirs(img_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Processing: {img_name}")
        print(f"Output: {img_dir}")
        print(f"{'='*60}")

        # Run MHRQI pipeline
        orig, recon, run_dir = main.main(
            shots=1000,
            n=n,
            d=2,
            denoise=True,
            use_shots=False,
            fast=True,
            verbose_plots=False,
            img_path=img_path,
            run_comparison=False
        )

        # Save original
        cv2.imwrite(os.path.join(img_dir, "original.png"), orig)

        # Run comparison with BM3D as SOTA reference for FR metrics
        noisy_img = compare_to.to_float01(orig)

        # Run comparison (no synthetic clean reference)
        comparison_results = compare_to.compare_to(
            noisy_img,
            proposed_img=compare_to.to_float01(recon),
            methods="all",
            plot=True,  # Generate report plots
            save=True,
            save_prefix="denoised",
            save_dir=img_dir,
            reference_image=None  # No synthetic reference
        )

        # Extract metrics for each method
        img_results = {}
        for r in comparison_results:
            method_name = r["name"]
            img_results[method_name] = r["metrics"]

        all_results[img_name] = img_results

    # Save raw results to metrics folder
    with open(os.path.join(metrics_dir, "raw_results.json"), "w") as f:
        serializable = {}
        for img, methods in all_results.items():
            serializable[img] = {}
            for method, metrics in methods.items():
                serializable[img][method] = {k: float(v) if not np.isnan(v) else None
                                              for k, v in metrics.items()}
        json.dump(serializable, f, indent=2)

    return all_results, base_dir, metrics_dir


def wilcoxon_test(all_results, method1, method2, metric):
    """
    Perform Wilcoxon signed-rank test between two methods on a specific metric.
    """
    diffs = []
    for img_name, methods in all_results.items():
        if method1 in methods and method2 in methods:
            v1 = methods[method1].get(metric, float('nan'))
            v2 = methods[method2].get(metric, float('nan'))
            if not np.isnan(v1) and not np.isnan(v2):
                diffs.append(v1 - v2)

    if len(diffs) < 3:
        return None, None, diffs

    try:
        stat, p_value = stats.wilcoxon(diffs)
        return stat, p_value, diffs
    except Exception as e:
        print(f"Wilcoxon test failed: {e}")
        return None, None, diffs


def compute_bootstrap_ci(values, confidence_level=0.95, n_resamples=10000, seed=20260818):
    """Compute 95% BCa bootstrap confidence interval for the mean."""
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 2 or np.all(data == data[0]):
        m = float(np.mean(data)) if len(data) > 0 else 0.0
        return [m, m]
    try:
        res = stats.bootstrap(
            (data,),
            np.mean,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method="BCa",
            random_state=np.random.default_rng(seed)
        )
        return [float(res.confidence_interval.low), float(res.confidence_interval.high)]
    except Exception:
        try:
            res = stats.bootstrap(
                (data,),
                np.mean,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                method="percentile",
                random_state=np.random.default_rng(seed)
            )
            return [float(res.confidence_interval.low), float(res.confidence_interval.high)]
        except Exception:
            m = float(np.mean(data))
            s = float(np.std(data, ddof=1))
            se = s / np.sqrt(len(data))
            return [m - 1.96 * se, m + 1.96 * se]


def paired_signed_rank_details(proposed_vals, baseline_vals):
    """Compute exact paired Wilcoxon signed-rank test and rank-biserial effect size."""
    proposed = np.asarray(proposed_vals, dtype=float)
    baseline = np.asarray(baseline_vals, dtype=float)
    mask = np.isfinite(proposed) & np.isfinite(baseline)
    p_sub = proposed[mask]
    b_sub = baseline[mask]

    diffs = p_sub - b_sub
    nonzero = diffs[diffs != 0]

    if len(nonzero) == 0:
        return {
            "n_pairs": int(len(diffs)),
            "n_nonzero": 0,
            "w_plus": 0.0,
            "w_minus": 0.0,
            "W_stat": 0.0,
            "p_unadjusted": 1.0,
            "rank_biserial": 0.0,
            "mean_diff": 0.0,
            "median_diff": 0.0,
        }

    ranks = stats.rankdata(np.abs(nonzero), method="average")
    w_plus = float(np.sum(ranks[nonzero > 0]))
    w_minus = float(np.sum(ranks[nonzero < 0]))
    denom = w_plus + w_minus
    rank_biserial = (w_plus - w_minus) / denom if denom else 0.0

    try:
        res = stats.wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
        w_stat = float(res.statistic)
        p_unadj = float(res.pvalue)
    except Exception:
        w_stat = float(min(w_plus, w_minus))
        p_unadj = 1.0

    return {
        "n_pairs": int(len(diffs)),
        "n_nonzero": int(len(nonzero)),
        "w_plus": w_plus,
        "w_minus": w_minus,
        "W_stat": w_stat,
        "p_unadjusted": p_unadj,
        "rank_biserial": float(rank_biserial),
        "mean_diff": float(np.mean(diffs)),
        "median_diff": float(np.median(diffs)),
    }


def apply_holm_adjustment(records):
    """Apply Holm-Bonferroni multiple testing adjustment across comparisons."""
    order = sorted(range(len(records)), key=lambda i: records[i]["p_unadjusted"])
    running = 0.0
    count = len(records)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (count - rank) * records[index]["p_unadjusted"])
        running = max(running, adjusted)
        records[index]["p_holm"] = running
        records[index]["significant_holm"] = (running < 0.05)


def create_results_table(all_results, metrics_dir):
    """
    Create full summary tables and paired statistical benchmarks for all metrics.
    Reports: N, mean, sample SD (ddof=1), median, 95% bootstrap CI (BCa),
    Wilcoxon W, unadjusted p, Holm-adjusted p, and rank-biserial effect size.
    """
    all_metrics_list = ALL_SPECKLE_METRICS + STRUCTURAL_METRICS
    method_metrics = {m: {k: [] for k in all_metrics_list} for m in METHODS}

    for img_name, methods in all_results.items():
        for method in METHODS:
            if method in methods:
                for metric in all_metrics_list:
                    val = methods[method].get(metric, float('nan'))
                    if not np.isnan(val):
                        method_metrics[method][metric].append(val)

    summary = {}
    for method in METHODS:
        summary[method] = {}
        for metric in all_metrics_list:
            vals = method_metrics[method][metric]
            if vals:
                n = len(vals)
                m_val = float(np.mean(vals))
                s_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
                med_val = float(np.median(vals))
                ci95 = compute_bootstrap_ci(vals)
                summary[method][metric] = {
                    "n": n,
                    "mean": m_val,
                    "sample_sd": s_val,
                    "median": med_val,
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "ci95_bca": ci95,
                }
            else:
                summary[method][metric] = {
                    "n": 0, "mean": float('nan'), "sample_sd": float('nan'),
                    "median": float('nan'), "min": float('nan'), "max": float('nan'),
                    "ci95_bca": [float('nan'), float('nan')]
                }

    print("\n" + "="*110)
    print("DESCRIPTIVE BENCHMARK STATISTICS (N, Mean ± Sample SD, Median, 95% BCa Bootstrap CI)")
    print("="*110)
    print(f"{'Method':<12} {'Metric':<10} {'N':<5} {'Mean ± Sample SD':<24} {'Median':<12} {'95% BCa Bootstrap CI':<24}")
    print("-" * 110)

    for method in METHODS:
        for metric in all_metrics_list:
            s = summary[method][metric]
            if s["n"] > 0:
                mean_sd_str = f"{s['mean']:.4f} ± {s['sample_sd']:.4f}"
                ci_str = f"[{s['ci95_bca'][0]:.4f}, {s['ci95_bca'][1]:.4f}]"
                print(f"{method:<12} {metric:<10} {s['n']:<5} {mean_sd_str:<24} {s['median']:<12.4f} {ci_str:<24}")

    # ==========================================================================
    # PAIRED STATISTICAL SIGNIFICANCE (Wilcoxon Signed-Rank Test & Holm Adjustment)
    # ==========================================================================
    metric_categories = [
        {
            "name": "Speckle Reduction (Lower Better)",
            "metrics": SPECKLE_METRICS_LOWER,
            "higher_better": False,
        },
        {
            "name": "Speckle Reduction (Higher Better)",
            "metrics": SPECKLE_METRICS_HIGHER,
            "higher_better": True,
        },
        {
            "name": "Structural Similarity",
            "metrics": STRUCTURAL_METRICS,
            "higher_better": True,
        },
    ]

    stat_results = []
    image_names = sorted(all_results.keys())

    for category in metric_categories:
        for other_method in ["bm3d", "nlmeans", "srad"]:
            for metric in category["metrics"]:
                proposed_vals = [all_results[name]["proposed"][metric] for name in image_names if "proposed" in all_results[name] and metric in all_results[name]["proposed"]]
                baseline_vals = [all_results[name][other_method][metric] for name in image_names if other_method in all_results[name] and metric in all_results[name][other_method]]

                details = paired_signed_rank_details(proposed_vals, baseline_vals)
                details.update({
                    "category": category["name"],
                    "competitor": other_method,
                    "metric": metric,
                    "higher_better": category["higher_better"],
                })
                stat_results.append(details)

    # Apply Holm-Bonferroni adjustment across all paired comparisons
    apply_holm_adjustment(stat_results)

    print("\n" + "="*125)
    print("PAIRED STATISTICAL TEST RESULTS (Proposed MHRQI vs Baselines)")
    print("Reports: N, Mean Δ, Median Δ, Wilcoxon W, Unadjusted p, Holm-Adjusted p, Rank-Biserial r_rb")
    print("="*125)
    print(f"{'Comparison':<18} {'Metric':<8} {'N':<4} {'Mean Δ':<10} {'Median Δ':<10} {'Wilcoxon W':<12} {'p (unadj)':<12} {'p (Holm)':<12} {'Rank-Biserial r_rb':<18} {'Decision'}")
    print("-" * 125)

    for r in stat_results:
        higher_better = r["higher_better"]
        mean_diff = r["mean_diff"]
        p_holm = r["p_holm"]

        if higher_better:
            mhrqi_better = mean_diff > 0
        else:
            mhrqi_better = mean_diff < 0

        if p_holm < 0.05:
            decision = "MHRQI Significantly Better" if mhrqi_better else f"{r['competitor'].upper()} Significantly Better"
        else:
            decision = "No Significant Diff (n.s.)"

        comp_str = f"MHRQI vs {r['competitor'].upper()}"
        print(f"{comp_str:<18} {r['metric']:<8} {r['n_pairs']:<4} {r['mean_diff']:<+10.4f} {r['median_diff']:<+10.4f} {r['W_stat']:<12.1f} {r['p_unadjusted']:<12.4e} {r['p_holm']:<12.4e} {r['rank_biserial']:<+18.4f} {decision}")

        r["interpretation"] = decision

    # Save statistical results
    with open(os.path.join(metrics_dir, "statistical_results.json"), "w") as f:
        json.dump(stat_results, f, indent=2, default=lambda x: int(x) if isinstance(x, (bool, np.bool_)) else float(x) if isinstance(x, np.floating) else x)

    # Save summary
    with open(os.path.join(metrics_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    return summary, stat_results


def create_visualization(all_results, metrics_dir):
    """
    Create separate visualizations for each metric category.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    methods = METHODS

    metric_groups = [
        ("Speckle Reduction (Lower Better)", SPECKLE_METRICS_LOWER, False),
        ("Speckle Reduction (Higher Better)", SPECKLE_METRICS_HIGHER, True),
        ("Structural Similarity Metrics", STRUCTURAL_METRICS, True),
        # Naturalness removed (NIQE computed but not reported)
    ]

    for group_name, metrics, higher_better in metric_groups:
        method_means = {m: [] for m in methods}
        method_stds = {m: [] for m in methods}

        for metric in metrics:
            for method in methods:
                vals = []
                for img_results in all_results.values():
                    if method in img_results:
                        v = img_results[method].get(metric, float('nan'))
                        if not np.isnan(v):
                            vals.append(v)
                if vals:
                    # Clip to 10000 for visualization if needed
                    m_val = np.mean(vals)
                    s_val = np.std(vals)
                    if m_val > 10000: m_val = 10000
                    if s_val > 10000: s_val = 10000
                    method_means[method].append(m_val)
                    method_stds[method].append(s_val)
                else:
                    method_means[method].append(0)
                    method_stds[method].append(0)

        # Skip if no data
        if all(all(v == 0 for v in method_means[m]) for m in methods):
            continue

        x = np.arange(len(methods))
        width = 0.6

        fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 4 * len(metrics)), sharex=False)
        if len(metrics) == 1:
            axes = [axes]

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        labels = ['BM3D', 'NL-Means', 'SRAD', 'MHRQI (Ours)']

        for idx, (ax, metric) in enumerate(zip(axes, metrics)):
            means = [method_means[m][idx] for m in methods]
            stds = [method_stds[m][idx] for m in methods]
            
            # Use bars instead of grouped bars since we have subplots
            bars = ax.bar(methods, means, width, yerr=stds, capsize=5, color=colors)
            
            ax.set_title(f"Metric: {metric}", fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            
            # Use labels for methods
            ax.set_xticks(np.arange(len(methods)))
            ax.set_xticklabels(labels)

            direction_note = '↑ Higher is better' if higher_better else '↓ Lower is better'
            ax.annotate(direction_note, xy=(0.98, 0.02), xycoords='axes fraction',
                       ha='right', fontsize=9, style='italic', color='gray')

        fig.suptitle(group_name, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = group_name.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("=", "") + ".png"
        plt.savefig(os.path.join(metrics_dir, filename), dpi=150)
        plt.close()

        print(f"  Saved: {filename}")

    # Generate metric boxplots and error bar plots across all benchmarks
    plots.plot_metric_boxplots(all_results, run_dir=metrics_dir)
    plots.plot_metric_errorbars(all_results, run_dir=metrics_dir)

    print(f"\nVisualizations saved to: {metrics_dir}")


def create_summary_heatmap(stat_results, metrics_dir):
    """
    Create a WIN/TIE/LOSS heatmap from pre-computed statistical results.

    Args:
        stat_results: List of dicts from create_results_table() (same structure
                      as statistical_results.json).
        metrics_dir:  Directory where the PNG is saved.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    if not stat_results:
        print("  No statistical results to plot; skipping heatmap.")
        return

    competitors = sorted(set(d["competitor"] for d in stat_results))
    metrics = sorted(set(d["metric"] for d in stat_results))

    # Preserve category grouping
    categories = {d["metric"]: d["category"] for d in stat_results}
    metrics.sort(key=lambda x: categories[x])

    # Build matrix: 2 = WIN, 1 = TIE, 0 = LOSS
    heatmap_data = np.zeros((len(competitors), len(metrics)))
    for d in stat_results:
        row = competitors.index(d["competitor"])
        col = metrics.index(d["metric"])
        interp = d["interpretation"].lower()
        if "mhrqi significantly better" in interp:
            heatmap_data[row, col] = 2
        elif "comparable" in interp:
            heatmap_data[row, col] = 1
        # else 0 (competitor better)

    fig, ax = plt.subplots(figsize=(14, 7))

    cmap = mcolors.ListedColormap(["#FF8A8A", "#FFFBD1", "#A3EBB1"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(heatmap_data, cmap=cmap, norm=norm, aspect="auto")

    # Grid lines
    ax.set_xticks(np.arange(len(metrics) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(competitors) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Axis labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(competitors)))
    ax.set_xticklabels(metrics, fontsize=11, fontweight="bold", rotation=0)
    ax.set_yticklabels([c.upper() for c in competitors], fontsize=11, fontweight="bold")

    # Cell annotations
    for i in range(len(competitors)):
        for j in range(len(metrics)):
            val = heatmap_data[i, j]
            label = "WIN" if val == 2 else "TIE" if val == 1 else "LOSS"
            ax.text(j, i, label, ha="center", va="center",
                    color="#333333", fontsize=11, fontweight="bold")

    # Category span headers
    unique_cats, cat_starts, current_cat = [], [], None
    for i, m in enumerate(metrics):
        if categories[m] != current_cat:
            unique_cats.append(categories[m])
            cat_starts.append(i)
            current_cat = categories[m]

    for i, cat in enumerate(unique_cats):
        start = cat_starts[i]
        end = cat_starts[i + 1] if i + 1 < len(cat_starts) else len(metrics)
        mid = (start + end - 1) / 2
        ax.annotate("", xy=(start - 0.4, -0.6), xytext=(end - 0.6, -0.6),
                    xycoords="data", textcoords="data",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=1.5))
        ax.text(mid, -0.8, cat, ha="center", va="bottom",
                fontsize=10, fontweight="bold", style="italic", color="#555555")

    ax.spines[:].set_visible(False)
    ax.set_title(
        "MHRQI Performance Benchmark Summary\n(Statistical Significance vs SOTA)",
        fontsize=15, fontweight="bold", pad=50
    )

    legend_elements = [
        Patch(facecolor="#A3EBB1", label="MHRQI Significantly Better (p < 0.05)"),
        Patch(facecolor="#FFFBD1", label="Competitive / No Significant Difference"),
        Patch(facecolor="#FF8A8A", label="Competitor Significantly Better (p < 0.05)"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(metrics_dir, "benchmark_summary_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: benchmark_summary_heatmap.png")


if __name__ == "__main__":
    print("="*60)
    print("MHRQI Statistical Benchmark")
    print("(No synthetic clean reference - degraded reference only)")
    print("="*60)

    # Run benchmark
    all_results, base_dir, metrics_dir = run_benchmark(n=16, strength=1.65)

    # Create results table and statistical tests
    summary, stat_results = create_results_table(all_results, metrics_dir)

    # Create visualizations
    create_visualization(all_results, metrics_dir)
    create_summary_heatmap(stat_results, metrics_dir)

    print(f"\n{'='*60}")
    print(f"All results saved to: {base_dir}")
    print(f"Metrics and charts: {metrics_dir}")
    print("="*60)
