"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Statistical Treatment: Wilcoxon Signed-Rank Test                           ║
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

import os
import cv2
import numpy as np
from scipy import stats
import main
import compare_to
import plots
from datetime import datetime
import json

# Medical image paths
MEDICAL_IMAGES = [
    "resources/cnv1.jpeg",
    "resources/cnv2.jpeg",
    "resources/dme1.jpeg",
    "resources/dme2.jpeg",
    "resources/drusen1.jpeg",
    "resources/drusen2.jpeg",
    "resources/normal1.jpeg",
    "resources/normal2.jpeg",
    "resources/normal3.jpeg",
]

# Metrics for comparison
FULL_REF_METRICS = ["FSIM", "SSIM"]
SPECKLE_METRICS = ["SSI", "SMPI"]
NO_REF_METRICS = ["PIQE", "BRISQUE"]

# Methods to compare
METHODS = ["bm3d", "nlmeans", "srad", "proposed"]

# SOTA for synthetic reference
SOTA_METHOD = "bm3d"


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
        
        # First generate BM3D output as SOTA reference
        bm3d_output = compare_to.denoise_bm3d(noisy_img)
        
        # Run full comparison with BM3D as clean reference
        comparison_results = compare_to.compare_to(
            noisy_img,
            proposed_img=compare_to.to_float01(recon),
            methods="all",
            plot=True,  # Generate report plots
            save=True,
            save_prefix=f"denoised",
            save_dir=img_dir,
            reference_image=bm3d_output  # BM3D as SOTA reference for FR metrics
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


def create_results_table(all_results, metrics_dir):
    """
    Create summary tables for all metrics.
    """
    # Aggregate metrics across all images
    method_metrics = {m: {k: [] for k in FULL_REF_METRICS + SPECKLE_METRICS + NO_REF_METRICS} 
                      for m in METHODS}
    
    for img_name, methods in all_results.items():
        for method in METHODS:
            if method in methods:
                for metric in FULL_REF_METRICS + SPECKLE_METRICS + NO_REF_METRICS:
                    val = methods[method].get(metric, float('nan'))
                    if not np.isnan(val):
                        method_metrics[method][metric].append(val)
    
    # Calculate means and stds
    summary = {}
    for method in METHODS:
        summary[method] = {}
        for metric in FULL_REF_METRICS + SPECKLE_METRICS + NO_REF_METRICS:
            vals = method_metrics[method][metric]
            if vals:
                summary[method][metric] = {
                    "mean": np.mean(vals),
                    "std": np.std(vals),
                    "n": len(vals)
                }
            else:
                summary[method][metric] = {"mean": float('nan'), "std": float('nan'), "n": 0}
    
    # Print table
    print("\n" + "="*80)
    print("BENCHMARK RESULTS (Mean ± Std across all images)")
    print(f"SOTA Reference for FR Metrics: {SOTA_METHOD.upper()}")
    print("="*80)
    
    all_metrics = FULL_REF_METRICS + SPECKLE_METRICS + NO_REF_METRICS
    header = f"{'Method':<12}" + "".join(f"{m:<15}" for m in all_metrics)
    print(header)
    print("-"*80)
    
    for method in METHODS:
        row = f"{method:<12}"
        for metric in all_metrics:
            s = summary[method][metric]
            if s["n"] > 0:
                row += f"{s['mean']:.4f}±{s['std']:.4f}  "
            else:
                row += "N/A           "
        print(row)
    
    # ==========================================================================
    # STATISTICAL SIGNIFICANCE (Wilcoxon Signed-Rank Test)
    # All metrics are paired samples (same images, different methods)
    # ==========================================================================
    
    print("\n" + "="*90)
    print("STATISTICAL SIGNIFICANCE (Wilcoxon Signed-Rank Test)")
    print("="*90)
    
    # Define metric categories with hypotheses
    metric_categories = [
        {
            "name": "Structural Similarity",
            "metrics": FULL_REF_METRICS,
            "higher_better": True,
            "H0": "MHRQI achieves the same structural preservation as [competitor]",
            "H1": "MHRQI achieves different structural preservation than [competitor]",
        },
        {
            "name": "Speckle Reduction",
            "metrics": SPECKLE_METRICS,
            "higher_better": True,
            "H0": "MHRQI achieves the same speckle reduction as [competitor]",
            "H1": "MHRQI achieves different speckle reduction than [competitor]",
        },
        {
            "name": "Perceptual Quality (No-Reference)",
            "metrics": NO_REF_METRICS,
            "higher_better": False,  # Lower is better for PIQE, BRISQUE, NIQE
            "H0": "MHRQI produces images of the same perceptual quality as [competitor]",
            "H1": "MHRQI produces images of different perceptual quality than [competitor]",
        },
    ]
    
    # Store results for summary
    stat_results = []
    
    for category in metric_categories:
        print(f"\n{'─'*90}")
        print(f"Category: {category['name']}")
        print(f"  H₀: {category['H0']}")
        print(f"  H₁: {category['H1']}")
        print(f"  Direction: {'Higher' if category['higher_better'] else 'Lower'} is better")
        print(f"{'─'*90}")
        
        for other_method in ["bm3d", "nlmeans", "srad"]:
            print(f"\n  {other_method.upper()} vs MHRQI:")
            
            for metric in category["metrics"]:
                stat, p_val, diffs = wilcoxon_test(all_results, "proposed", other_method, metric)
                
                if p_val is not None:
                    mean_diff = np.mean(diffs)
                    
                    # Interpret based on direction
                    if category["higher_better"]:
                        mhrqi_better = mean_diff > 0
                    else:
                        mhrqi_better = mean_diff < 0  # Lower is better
                    
                    # Determine significance and language
                    if p_val < 0.05:
                        decision = "Reject H₀"
                        if mhrqi_better:
                            interpretation = "MHRQI significantly better"
                        else:
                            interpretation = f"{other_method} significantly better"
                    else:
                        decision = "Fail to reject H₀"
                        interpretation = "Comparable (no significant difference)"
                    
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
                    
                    print(f"    {metric:<10}: p={p_val:.4f} ({sig:<4}) → {decision}")
                    print(f"               Mean Δ={mean_diff:+.4f} → {interpretation}")
                    
                    stat_results.append({
                        "category": category["name"],
                        "competitor": other_method,
                        "metric": metric,
                        "p_value": p_val,
                        "mean_diff": mean_diff,
                        "interpretation": interpretation,
                        "significant": p_val < 0.05
                    })
                else:
                    print(f"    {metric:<10}: insufficient data (need ≥3 paired samples)")
    
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
    
    methods = METHODS
    
    metric_groups = [
        ("Full Reference Metrics (vs BM3D)", FULL_REF_METRICS, True),
        ("Speckle/Consistency Metrics", SPECKLE_METRICS, True),
        ("No-Reference Quality", NO_REF_METRICS, False),
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
                    method_means[method].append(np.mean(vals))
                    method_stds[method].append(np.std(vals))
                else:
                    method_means[method].append(0)
                    method_stds[method].append(0)
        
        # Skip if no data
        if all(all(v == 0 for v in method_means[m]) for m in methods):
            continue
        
        x = np.arange(len(metrics))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        labels = ['BM3D', 'NL-Means', 'SRAD', 'MHRQI (Ours)']
        
        for i, method in enumerate(methods):
            offset = (i - len(methods)/2 + 0.5) * width
            ax.bar(x + offset, method_means[method], width, label=labels[i],
                   color=colors[i], yerr=method_stds[method], capsize=3)
        
        ax.set_xlabel('Metric', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(group_name, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend(loc='best')
        ax.grid(axis='y', alpha=0.3)
        
        direction_note = '↑ Higher is better' if higher_better else '↓ Lower is better'
        ax.annotate(direction_note, xy=(0.98, 0.02), xycoords='axes fraction',
                   ha='right', fontsize=9, style='italic', color='gray')
        
        plt.tight_layout()
        filename = group_name.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("=", "") + ".png"
        plt.savefig(os.path.join(metrics_dir, filename), dpi=150)
        plt.close()
        
        print(f"  Saved: {filename}")
    
    print(f"\nVisualizations saved to: {metrics_dir}")


if __name__ == "__main__":
    print("="*60)
    print("MHRQI Statistical Benchmark")
    print(f"SOTA Reference: {SOTA_METHOD.upper()}")
    print("="*60)
    
    # Run benchmark
    all_results, base_dir, metrics_dir = run_benchmark(n=64, strength=1.65)
    
    # Create results table and statistical tests
    summary, stat_results = create_results_table(all_results, metrics_dir)
    
    # Create visualization
    create_visualization(all_results, metrics_dir)
    
    print(f"\n{'='*60}")
    print(f"All results saved to: {base_dir}")
    print(f"Metrics and charts: {metrics_dir}")
    print("="*60)
