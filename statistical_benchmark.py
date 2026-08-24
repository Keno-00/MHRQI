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

import configparser
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def load_config():
    configs_dir = os.path.join(os.path.dirname(__file__), "configs")
    ini_path = os.path.join(configs_dir, "paper.ini")
    json_path = os.path.join(configs_dir, "paper.json")

    if os.path.exists(ini_path):
        cp = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
        cp.read(ini_path, encoding="utf-8")

        shots_raw = cp.get("execution", "shots", fallback="none").strip().lower()
        shots_val = int(shots_raw) if shots_raw.isdigit() else None

        return {
            "image_size": cp.getint("image", "image_size", fallback=128),
            "resize_interpolation": cp.get("image", "resize_interpolation", fallback="opencv_INTER_AREA"),
            "subdivision_factor": cp.getint("image", "subdivision_factor", fallback=2),
            "bit_depth": cp.getint("image", "bit_depth", fallback=8),
            "benchmark_mode": cp.get("execution", "benchmark_mode", fallback="statevector"),
            "shots": shots_val,
            "denoise": cp.getboolean("execution", "denoise", fallback=True),
            "fast": cp.getboolean("execution", "fast", fallback=False),
            "verbose_plots": cp.getboolean("execution", "verbose_plots", fallback=False),
            "seeds": {
                "python": cp.getint("seeds", "python", fallback=20260818),
                "numpy": cp.getint("seeds", "numpy", fallback=20260818),
                "transpiler": cp.getint("seeds", "transpiler", fallback=20260818),
                "simulator": cp.getint("seeds", "simulator", fallback=20260818),
                "bootstrap": cp.getint("seeds", "bootstrap", fallback=20260818),
            },
            "simulator": {
                "backend": cp.get("simulator", "backend", fallback="AerSimulator"),
                "method": cp.get("simulator", "method", fallback="statevector"),
                "device": cp.get("simulator", "device", fallback="CPU"),
                "allow_device_fallback": cp.getboolean("simulator", "allow_device_fallback", fallback=False),
            },
            "statistics": {
                "alpha": cp.getfloat("statistics", "alpha", fallback=0.05),
                "wilcoxon_alternative": cp.get("statistics", "wilcoxon_alternative", fallback="two-sided"),
                "wilcoxon_zero_method": cp.get("statistics", "wilcoxon_zero_method", fallback="wilcox"),
                "wilcoxon_method": cp.get("statistics", "wilcoxon_method", fallback="exact"),
                "bootstrap_method": cp.get("statistics", "bootstrap_method", fallback="BCa"),
                "bootstrap_resamples": cp.getint("statistics", "bootstrap_resamples", fallback=10000),
                "multiple_testing": cp.get("statistics", "multiple_testing", fallback="Holm"),
            },
            "baselines": {
                "bm3d": {
                    "sigma": cp.getfloat("baseline.bm3d", "sigma", fallback=0.05),
                    "stage": cp.get("baseline.bm3d", "stage", fallback="ALL_STAGES"),
                },
                "nlmeans": {
                    "h": cp.getfloat("baseline.nlmeans", "h", fallback=10.0),
                    "template_window": cp.getint("baseline.nlmeans", "template_window", fallback=7),
                    "search_window": cp.getint("baseline.nlmeans", "search_window", fallback=21),
                },
                "srad": {
                    "iterations": cp.getint("baseline.srad", "iterations", fallback=400),
                    "dt": cp.getfloat("baseline.srad", "dt", fallback=0.65),
                    "decay": cp.getfloat("baseline.srad", "decay", fallback=0.8),
                },
                "siamesegan": {
                    "enabled": cp.getboolean("baseline.siamesegan", "enabled", fallback=False),
                    "repository": cp.get("baseline.siamesegan", "repository", fallback=""),
                    "python_executable": cp.get("baseline.siamesegan", "python_executable", fallback=""),
                    "timeout_seconds": (
                        cp.getint("baseline.siamesegan", "timeout_seconds")
                        if cp.has_option("baseline.siamesegan", "timeout_seconds")
                        else None
                    ),
                },
            },
        }
    elif os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

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


def _json_default(value):
    """Convert NumPy values into JSON-compatible scalar values."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path, data):
    """Write a UTF-8 JSON research record with stable indentation."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=_json_default)


def write_csv(path, rows, fieldnames):
    """Write a complete CSV snapshot so partial runs remain inspectable."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_results(all_results):
    """Convert nested metric results to a JSON-safe representation."""
    serializable = {}
    for image_name, methods in all_results.items():
        serializable[image_name] = {}
        for method, metrics in methods.items():
            serializable[image_name][method] = {
                metric: float(value) if np.isfinite(value) else None
                for metric, value in metrics.items()
            }
    return serializable


def pathology_from_image_name(image_name):
    """Return the prespecified OCT pathology stratum encoded by the filename."""
    for pathology in ("cnv", "dme", "drusen", "normal"):
        if image_name.lower().startswith(pathology):
            return pathology.upper()
    return "UNSPECIFIED"


def installed_versions():
    """Record package versions needed to reproduce this benchmark environment."""
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for package in ("numpy", "scipy", "opencv-python", "qiskit", "qiskit-aer", "bm3d", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def git_revision():
    """Capture the source revision when Git metadata is available."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def write_metric_csv(all_results, metrics_dir):
    """Save every method--image--metric observation in a tidy CSV table."""
    rows = []
    for image_name in sorted(all_results):
        for method, metrics in sorted(all_results[image_name].items()):
            for metric, value in sorted(metrics.items()):
                rows.append({
                    "image_id": image_name,
                    "pathology": pathology_from_image_name(image_name),
                    "method": method,
                    "metric": metric,
                    "value": float(value) if np.isfinite(value) else "",
                })
    write_csv(
        os.path.join(metrics_dir, "per_image_metrics_long.csv"),
        rows,
        ["image_id", "pathology", "method", "metric", "value"],
    )


def save_comparison_panel(original, comparison_results, image_path):
    """Save one visual panel containing the input and every evaluated method."""
    import matplotlib.pyplot as plt

    plots.apply_ecti_plot_style()
    ordered = ["Original", "bm3d", "nlmeans", "srad", "siamesegan", "proposed"]
    result_images = {result["name"]: result["image"] for result in comparison_results}
    visible = [name for name in ordered if name in result_images]
    if not visible:
        return

    labels = {
        "Original": "Input B-scan",
        "bm3d": "BM3D",
        "nlmeans": "NLM",
        "srad": "SRAD",
        "siamesegan": "SiameseGAN",
        "proposed": "MHRQI",
    }
    figure, axes = plt.subplots(1, len(visible), figsize=(plots.ECTI_DOUBLE_COLUMN_IN, 1.35))
    if len(visible) == 1:
        axes = [axes]
    for axis, method in zip(axes, visible):
        image = original if method == "Original" else result_images[method]
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(labels.get(method, method), fontsize=7, pad=3)
        axis.axis("off")
    figure.tight_layout(pad=0.15, w_pad=0.25)
    figure.savefig(image_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def write_artifact_manifest(base_dir):
    """Index all retained evidence files without duplicating image data."""
    rows = []
    for path in sorted(Path(base_dir).rglob("*")):
        if path.is_file():
            suffix = path.suffix.lower()
            rows.append({
                "relative_path": path.relative_to(base_dir).as_posix(),
                "artifact_type": "figure" if suffix in {".png", ".pdf", ".svg"} else "data",
                "bytes": path.stat().st_size,
            })
    write_csv(
        os.path.join(base_dir, "artifact_manifest.csv"),
        rows,
        ["relative_path", "artifact_type", "bytes"],
    )


def benchmark_methods(all_results):
    """Return observed methods in the fixed paper-comparison order."""
    observed = {method for image_results in all_results.values() for method in image_results}
    paper_order = ["bm3d", "nlmeans", "srad", "siamesegan", "proposed"]
    ordered = [method for method in paper_order if method in observed]
    return ordered + sorted(observed.difference(ordered).difference({"Original"}))


def run_benchmark(n=None, strength=1.65, baseline_config=None, config=None):
    """
    Run benchmark on all medical images and collect metrics.
    
    Folder structure:
    - benchmark/timestamp/imagename/ for each image
    - benchmark/timestamp/metrics/ for aggregated stats
    
    Args:
        n: Optional image size override. If omitted, uses paper.ini.
        strength: Denoiser strength parameter
        baseline_config: Optional comparator configuration passed to compare_to.
        config: Optional parsed configuration. If omitted, reads paper.ini.
    
    Returns:
        results: Dict mapping image -> method -> metrics
        base_dir: Path to benchmark output directory
    """
    config = config or load_config()
    n = config.get("image_size", 128) if n is None else n
    baseline_config = config.get("baselines", {}) if baseline_config is None else baseline_config
    benchmark_mode = config.get("benchmark_mode", "statevector").strip().lower()
    use_shots = benchmark_mode in {"shots", "qasm"}
    configured_shots = config.get("shots")
    if use_shots and configured_shots is None:
        raise ValueError("paper.ini selects shot-based benchmarking but does not set execution.shots")

    seeds = config.get("seeds", {})
    simulator = config.get("simulator", {})

    # Create base directory: benchmark/timestamp/ (matching runs/ format)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join("benchmark", timestamp)
    metrics_dir = os.path.join(base_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    plots.copy_config_to_run_dir(base_dir)

    all_results = {}
    image_records = []
    run_manifest = {
        "record_schema": "mhrqi-benchmark-v1",
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "source_revision": git_revision(),
        "environment": installed_versions(),
        "resolved_configuration": config,
        "benchmark": {
            "image_size": n,
            "benchmark_mode": benchmark_mode,
            "use_shots": use_shots,
            "shots": configured_shots if use_shots else None,
            "expected_samples_per_address": (
                configured_shots / float(n * n) if use_shots else None
            ),
            "images_requested": len(MEDICAL_IMAGES),
            "input_images": MEDICAL_IMAGES,
        },
    }
    write_json(os.path.join(base_dir, "run_manifest.json"), run_manifest)

    def checkpoint():
        """Persist completed observations immediately during a long benchmark."""
        write_json(os.path.join(metrics_dir, "raw_results.json"), json_results(all_results))
        write_metric_csv(all_results, metrics_dir)
        write_csv(
            os.path.join(metrics_dir, "image_run_log.csv"),
            image_records,
            [
                "image_id", "pathology", "source_path", "benchmark_directory",
                "mhrqi_run_directory", "source_width", "source_height", "image_width",
                "image_height", "status",
                "mhrqi_seconds", "comparison_seconds", "total_seconds", "error",
            ],
        )

    for img_path in MEDICAL_IMAGES:
        img_name = os.path.basename(img_path).replace(".jpeg", "")
        img_dir = os.path.join(base_dir, img_name)
        os.makedirs(img_dir, exist_ok=True)
        image_started = time.perf_counter()
        source_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if source_image is None:
            raise FileNotFoundError(f"Could not read benchmark input image: {img_path}")

        print(f"\n{'='*60}")
        print(f"Processing: {img_name}")
        print(f"Output: {img_dir}")
        print(f"{'='*60}")

        # Run MHRQI pipeline
        try:
            mhrqi_started = time.perf_counter()
            orig, recon, run_dir = main.main(
                shots=configured_shots if use_shots else 1000,
                n=n,
                d=config.get("subdivision_factor", 2),
                denoise=config.get("denoise", True),
                use_shots=use_shots,
                fast=config.get("fast", False),
                verbose_plots=config.get("verbose_plots", False),
                img_path=img_path,
                run_comparison=False,
                simulation_device=simulator.get("device", "CPU"),
                seed_python=seeds.get("python", 20260818),
                seed_numpy=seeds.get("numpy", 20260818),
                seed_simulator=seeds.get("simulator", 20260818),
                seed_transpiler=seeds.get("transpiler", 20260818),
                bit_depth=config.get("bit_depth", 8),
                resize_interpolation=config.get("resize_interpolation", "opencv_INTER_AREA"),
                baseline_config=baseline_config,
                evidence_dir=img_dir,
            )
            mhrqi_seconds = time.perf_counter() - mhrqi_started

            cv2.imwrite(os.path.join(img_dir, "original.png"), orig)
            comparison_started = time.perf_counter()
            comparison_results = compare_to.compare_to(
                compare_to.to_float01(orig),
                proposed_img=compare_to.to_float01(recon),
                methods="all",
                plot=True,
                save=True,
                save_prefix="denoised",
                save_dir=img_dir,
                reference_image=None,
                baseline_config=baseline_config,
            )
            comparison_seconds = time.perf_counter() - comparison_started
            save_comparison_panel(
                orig,
                comparison_results,
                os.path.join(img_dir, "comparison_panel.png"),
            )

            all_results[img_name] = {
                result["name"]: result["metrics"] for result in comparison_results
            }
            image_records.append({
                "image_id": img_name,
                "pathology": pathology_from_image_name(img_name),
                "source_path": img_path,
                "benchmark_directory": img_dir,
                "mhrqi_run_directory": run_dir,
                "source_width": int(source_image.shape[1]),
                "source_height": int(source_image.shape[0]),
                "image_width": int(orig.shape[1]),
                "image_height": int(orig.shape[0]),
                "status": "completed",
                "mhrqi_seconds": round(mhrqi_seconds, 6),
                "comparison_seconds": round(comparison_seconds, 6),
                "total_seconds": round(time.perf_counter() - image_started, 6),
                "error": "",
            })
            checkpoint()
        except Exception as error:
            image_records.append({
                "image_id": img_name,
                "pathology": pathology_from_image_name(img_name),
                "source_path": img_path,
                "benchmark_directory": img_dir,
                "mhrqi_run_directory": "",
                "source_width": int(source_image.shape[1]),
                "source_height": int(source_image.shape[0]),
                "image_width": "",
                "image_height": "",
                "status": "failed",
                "mhrqi_seconds": "",
                "comparison_seconds": "",
                "total_seconds": round(time.perf_counter() - image_started, 6),
                "error": str(error),
            })
            run_manifest["status"] = "failed"
            run_manifest["failed_image"] = img_name
            run_manifest["error"] = str(error)
            run_manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
            checkpoint()
            write_json(os.path.join(base_dir, "run_manifest.json"), run_manifest)
            write_artifact_manifest(base_dir)
            raise

    run_manifest["status"] = "completed"
    run_manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    run_manifest["images_completed"] = len(all_results)
    checkpoint()
    write_json(os.path.join(base_dir, "run_manifest.json"), run_manifest)

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


def compute_bootstrap_ci(values, confidence_level=0.95, n_resamples=10000,
                         seed=20260818, method="BCa"):
    """Compute the configured bootstrap confidence interval for the mean."""
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
            method=method,
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


def paired_signed_rank_details(proposed_vals, baseline_vals,
                               alternative="two-sided", zero_method="wilcox",
                               method="exact"):
    """Compute configured paired Wilcoxon statistics and rank-biserial effect size."""
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
        res = stats.wilcoxon(
            nonzero,
            alternative=alternative,
            zero_method=zero_method,
            method=method,
        )
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


def apply_holm_adjustment(records, alpha=0.05):
    """Apply Holm-Bonferroni multiple testing adjustment across comparisons."""
    order = sorted(range(len(records)), key=lambda i: records[i]["p_unadjusted"])
    running = 0.0
    count = len(records)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (count - rank) * records[index]["p_unadjusted"])
        running = max(running, adjusted)
        records[index]["p_holm"] = running
        records[index]["significant_holm"] = (running < alpha)


def create_results_table(all_results, metrics_dir, statistics_config=None, bootstrap_seed=20260818):
    """
    Create full summary tables and paired statistical benchmarks for all metrics.
    Reports: N, mean, sample SD (ddof=1), median, 95% bootstrap CI (BCa),
    Wilcoxon W, unadjusted p, Holm-adjusted p, and rank-biserial effect size.
    """
    statistics_config = statistics_config or {}
    alpha = statistics_config.get("alpha", 0.05)
    bootstrap_method = statistics_config.get("bootstrap_method", "BCa")
    bootstrap_resamples = statistics_config.get("bootstrap_resamples", 10000)
    wilcoxon_alternative = statistics_config.get("wilcoxon_alternative", "two-sided")
    wilcoxon_zero_method = statistics_config.get("wilcoxon_zero_method", "wilcox")
    wilcoxon_method = statistics_config.get("wilcoxon_method", "exact")

    all_metrics_list = ALL_SPECKLE_METRICS + STRUCTURAL_METRICS
    methods = benchmark_methods(all_results)
    method_metrics = {m: {k: [] for k in all_metrics_list} for m in methods}

    for img_name, image_methods in all_results.items():
        for method in method_metrics:
            if method in image_methods:
                for metric in all_metrics_list:
                    val = image_methods[method].get(metric, float('nan'))
                    if not np.isnan(val):
                        method_metrics[method][metric].append(val)

    summary = {}
    for method in method_metrics:
        summary[method] = {}
        for metric in all_metrics_list:
            vals = method_metrics[method][metric]
            if vals:
                n = len(vals)
                m_val = float(np.mean(vals))
                s_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
                med_val = float(np.median(vals))
                ci95 = compute_bootstrap_ci(
                    vals,
                    confidence_level=1.0 - alpha,
                    n_resamples=bootstrap_resamples,
                    seed=bootstrap_seed,
                    method=bootstrap_method,
                )
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

    for method in method_metrics:
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
        for other_method in [method for method in methods if method != "proposed"]:
            for metric in category["metrics"]:
                paired_names = [
                    name for name in image_names
                    if "proposed" in all_results[name]
                    and other_method in all_results[name]
                    and metric in all_results[name]["proposed"]
                    and metric in all_results[name][other_method]
                ]
                proposed_vals = [all_results[name]["proposed"][metric] for name in paired_names]
                baseline_vals = [all_results[name][other_method][metric] for name in paired_names]

                details = paired_signed_rank_details(
                    proposed_vals,
                    baseline_vals,
                    alternative=wilcoxon_alternative,
                    zero_method=wilcoxon_zero_method,
                    method=wilcoxon_method,
                )
                details.update({
                    "category": category["name"],
                    "competitor": other_method,
                    "metric": metric,
                    "higher_better": category["higher_better"],
                })
                stat_results.append(details)

    # Apply Holm-Bonferroni adjustment across all paired comparisons
    apply_holm_adjustment(stat_results, alpha=alpha)

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

        if p_holm < alpha:
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
    plots.apply_ecti_plot_style()

    methods = benchmark_methods(all_results)

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

        fig, axes = plt.subplots(
            len(metrics), 1, figsize=(plots.ECTI_SINGLE_COLUMN_IN, 1.65 * len(metrics)), sharex=False
        )
        if len(metrics) == 1:
            axes = [axes]

        colors = [plots.ecti_method_style(method)[0] for method in methods]
        labels = [plots.ecti_method_style(method)[1] for method in methods]

        for idx, (ax, metric) in enumerate(zip(axes, metrics)):
            means = [method_means[m][idx] for m in methods]
            stds = [method_stds[m][idx] for m in methods]
            
            # Use bars instead of grouped bars since we have subplots
            ax.bar(
                methods, means, width, yerr=stds, capsize=2.5, color=colors,
                edgecolor="#333333", linewidth=0.45, error_kw={"elinewidth": 0.65, "capthick": 0.65},
                zorder=2,
            )
            
            ax.set_title(metric, pad=4)
            ax.set_ylabel('Value')
            plots.ecti_axis(ax)
            
            # Use labels for methods
            ax.set_xticks(np.arange(len(methods)))
            ax.set_xticklabels(labels)

            direction_note = 'Higher is better' if higher_better else 'Lower is better'
            ax.annotate(direction_note, xy=(0.98, 0.02), xycoords='axes fraction',
                       ha='right', fontsize=6.5, style='italic', color='#555555')

        fig.suptitle(group_name, fontsize=9.5, y=0.99)
        fig.tight_layout(rect=[0, 0.02, 1, 0.94], pad=0.55, h_pad=0.8)
        filename = group_name.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("=", "") + ".png"
        fig.savefig(os.path.join(metrics_dir, filename), dpi=300, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)

        print(f"  Saved: {filename}")

    # Generate metric boxplots and error bar plots across all benchmarks
    plots.plot_metric_boxplots(all_results, run_dir=metrics_dir)
    plots.plot_metric_errorbars(all_results, run_dir=metrics_dir)

    print(f"\nVisualizations saved to: {metrics_dir}")


def write_summary_csv(summary, stat_results, metrics_dir):
    """Save aggregate estimates and paired tests in publication-ready CSV tables."""
    summary_rows = []
    for method in sorted(summary):
        for metric, values in sorted(summary[method].items()):
            ci_lower, ci_upper = values["ci95_bca"]
            summary_rows.append({
                "method": method,
                "metric": metric,
                "n": values["n"],
                "mean": values["mean"],
                "sample_sd": values["sample_sd"],
                "median": values["median"],
                "minimum": values["min"],
                "maximum": values["max"],
                "ci95_bca_lower": ci_lower,
                "ci95_bca_upper": ci_upper,
            })
    write_csv(
        os.path.join(metrics_dir, "metric_summary.csv"),
        summary_rows,
        [
            "method", "metric", "n", "mean", "sample_sd", "median", "minimum",
            "maximum", "ci95_bca_lower", "ci95_bca_upper",
        ],
    )

    statistic_fields = [
        "category", "competitor", "metric", "higher_better", "n_pairs", "mean_diff",
        "median_diff", "W_stat", "p_unadjusted", "p_holm", "significant_holm",
        "rank_biserial", "interpretation",
    ]
    write_csv(
        os.path.join(metrics_dir, "paired_statistics.csv"),
        stat_results,
        statistic_fields,
    )


def create_bootstrap_ci_plot(summary, all_results, metrics_dir):
    """Save one readable 95% BCa confidence-interval figure for each metric."""
    import matplotlib.pyplot as plt

    plots.apply_ecti_plot_style()
    methods = benchmark_methods(all_results)
    metrics = ALL_SPECKLE_METRICS + STRUCTURAL_METRICS
    figures_dir = os.path.join(metrics_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    output_paths = []
    for metric in metrics:
        positions, means, lower_errors, upper_errors, labels, colors = [], [], [], [], [], []
        for position, method in enumerate(methods):
            values = summary.get(method, {}).get(metric, {})
            mean = values.get("mean", float("nan"))
            ci = values.get("ci95_bca", [float("nan"), float("nan")])
            if not (np.isfinite(mean) and np.isfinite(ci[0]) and np.isfinite(ci[1])):
                continue
            positions.append(position)
            means.append(mean)
            lower_errors.append(max(0.0, mean - ci[0]))
            upper_errors.append(max(0.0, ci[1] - mean))
            color, label = plots.ecti_method_style(method)
            colors.append(color)
            labels.append(label)
        figure, axis = plt.subplots(figsize=(plots.ECTI_SINGLE_COLUMN_IN, 1.9))
        axis.errorbar(
            means,
            positions,
            xerr=np.array([lower_errors, upper_errors]),
            fmt="none",
            ecolor="#333333",
            capsize=2.5,
            elinewidth=0.7,
            capthick=0.7,
            zorder=2,
        )
        axis.scatter(means, positions, c=colors, s=18, zorder=3)
        axis.set_yticks(positions)
        axis.set_yticklabels(labels)
        axis.invert_yaxis()
        direction = "$\\downarrow$" if metric in SPECKLE_METRICS_LOWER else "$\\uparrow$"
        axis.set_title(metric, pad=4)
        axis.text(0.97, 0.08, direction, transform=axis.transAxes, ha="right", va="bottom", fontsize=8)
        plots.ecti_axis(axis, grid=False)
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.45, zorder=0)
        figure.tight_layout(pad=0.35)
        output_path = os.path.join(figures_dir, f"mean_95ci_bca_{metric.lower()}.png")
        figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
        plt.close(figure)
        output_paths.append(output_path)
    print(f"  Saved {len(output_paths)} 95% BCa confidence-interval figures to: {figures_dir}")
    return output_paths


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
    plots.apply_ecti_plot_style()

    if not stat_results:
        print("  No statistical results to plot; skipping heatmap.")
        return

    competitors = sorted(set(d["competitor"] for d in stat_results))
    observed_metrics = {d["metric"] for d in stat_results}
    metrics = [metric for metric in ALL_SPECKLE_METRICS + STRUCTURAL_METRICS if metric in observed_metrics]
    metrics.extend(sorted(observed_metrics.difference(metrics)))

    # Preserve category grouping
    categories = {d["metric"]: d["category"] for d in stat_results}

    # Build matrix: 2 = WIN, 1 = TIE, 0 = LOSS
    heatmap_data = np.zeros((len(competitors), len(metrics)))
    for d in stat_results:
        row = competitors.index(d["competitor"])
        col = metrics.index(d["metric"])
        interp = d["interpretation"].lower()
        if "mhrqi significantly better" in interp:
            heatmap_data[row, col] = 2
        elif "comparable" in interp or "no significant" in interp or "n.s." in interp:
            heatmap_data[row, col] = 1
        # else 0 (competitor better)

    fig, ax = plt.subplots(figsize=(plots.ECTI_DOUBLE_COLUMN_IN, 3.1))

    cmap = mcolors.ListedColormap(["#C7C7C7", "#F2F2F2", "#4A4A4A"])
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
    ax.set_yticklabels([plots.ecti_method_style(c)[1] for c in competitors], fontsize=8)

    # Cell annotations
    for i in range(len(competitors)):
        for j in range(len(metrics)):
            val = heatmap_data[i, j]
            label = "MHRQI" if val == 2 else "n.s." if val == 1 else "Comparator"
            ax.text(
                j, i, label, ha="center", va="center",
                color="white" if val == 2 else "#222222", fontsize=7, fontweight="bold",
            )

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
                    arrowprops=dict(arrowstyle="-", color="#666666", lw=0.6))
        ax.text(mid, -0.8, cat, ha="center", va="bottom",
                fontsize=7, style="italic", color="#555555")

    ax.spines[:].set_visible(False)
    ax.set_title(
        "Paired comparisons after Holm adjustment",
        fontsize=10, pad=34
    )

    legend_elements = [
        Patch(facecolor="#4A4A4A", label="MHRQI significantly better"),
        Patch(facecolor="#F2F2F2", label="No significant difference"),
        Patch(facecolor="#C7C7C7", label="Comparator better"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=7)

    fig.tight_layout(pad=0.5)
    output_path = os.path.join(metrics_dir, "benchmark_summary_heatmap.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  Saved: benchmark_summary_heatmap.png")


if __name__ == "__main__":
    print("="*60)
    print("MHRQI Statistical Benchmark")
    print("(No synthetic clean reference - degraded reference only)")
    print("="*60)

    cfg = load_config()
    n_size = cfg.get("image_size", 16)

    # Run benchmark
    all_results, base_dir, metrics_dir = run_benchmark(
        n=n_size,
        strength=1.65,
        baseline_config=cfg.get("baselines"),
        config=cfg,
    )

    # Create results table and statistical tests
    summary, stat_results = create_results_table(
        all_results,
        metrics_dir,
        statistics_config=cfg.get("statistics"),
        bootstrap_seed=cfg.get("seeds", {}).get("bootstrap", 20260818),
    )

    # Retain two non-redundant paper figures: empirical distributions and 95% BCa estimates.
    plots.plot_metric_boxplots(all_results, run_dir=metrics_dir)
    write_summary_csv(summary, stat_results, metrics_dir)
    create_bootstrap_ci_plot(summary, all_results, metrics_dir)
    write_artifact_manifest(base_dir)

    print(f"\n{'='*60}")
    print(f"All results saved to: {base_dir}")
    print(f"Metrics and charts: {metrics_dir}")
    print("="*60)
