"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MHRQI - Multi-scale Hierarchical Representation of Quantum Images            ║
║  Main Pipeline: Encoding, Denoising, Benchmarking                           ║
║                                                                              ║
║  Author: Keno S. Jose                                                        ║
║  License: Apache 2.0                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import csv
import math
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mhrqi.benchmarks import BenchmarkSuite
from mhrqi.core.representation import MHRQI
from mhrqi.utils import general as utils
from mhrqi.utils import visualization as plots

CSV_PATH = Path("mhrqi_runs.csv")


def save_rows_to_csv(rows, csv_path=CSV_PATH):
    fieldnames = ["timestamp", "n", "bins", "shots", "shots_per_bin", "mse"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def main(
    shots=1000,
    n=4,
    d=2,
    denoise=False,
    use_shots=True,
    fast=False,
    verbose_plots=False,
    img_path=None,
    run_comparison=True,
):
    """
    Main MHRQI simulation pipeline.

    Args:
        shots: number of measurement shots (if use_shots=True)
        n: image dimension (will be resized to n x n)
        d: qudit dimension (2=qubit)
        denoise: whether to apply denoising circuit
        use_shots: if True, use shot-based simulation; if False, use statevector
        fast: if True, use lazy (statevector-based) upload for speed
        verbose_plots: if True, show additional debug plots
        img_path: path to input image (defaults to resources/drusen1.jpeg)
        run_comparison: if True, run comparison benchmarks against BM3D/NL-Means/SRAD

    Returns:
        tuple: (original_image, reconstructed_image, run_directory_path)
    """

    # Use default image if not specified
    if img_path is None:
        # Look for resources folder in the project root (3 levels up from mhrqi/cli/main.py)
        root_dir = Path(__file__).resolve().parent.parent.parent
        img_path = str(root_dir / "resources" / "drusen1.jpeg")

    myimg = cv2.imread(img_path)
    myimg = cv2.resize(myimg, (n, n))

    myimg = cv2.cvtColor(myimg, cv2.COLOR_RGB2GRAY)
    N = myimg.shape[1]
    angle_norm = utils.angle_map(myimg)
    normalized_img = np.clip(myimg.astype(np.float64) / 255.0, 0.0, 1.0)

    H, W = angle_norm.shape
    max_depth = utils.get_max_depth(N, d)
    hierarchical_coord_matrix = utils.generate_hierarchical_coord_matrix(N, d)

    # -------------------------
    # Circuit Construction & Upload
    # -------------------------
    model = MHRQI(max_depth)
    if fast:
        model.lazy_upload(hierarchical_coord_matrix, normalized_img)
    else:
        model.upload(hierarchical_coord_matrix, normalized_img)

    # -------------------------
    # Denoising (Extension)
    # -------------------------
    if denoise:
        model.apply_denoising()

    # Simulate
    start_time = time.perf_counter()
    result = model.simulate(shots=shots if use_shots else None, use_gpu=True)
    end_time = time.perf_counter()

    # -------------------------
    # Reconstruction
    # -------------------------
    newimg = result.reconstruct(use_denoising_bias=denoise)
    newimg = (np.clip(newimg, 0.0, 1.0) * 255).astype(np.uint8)

    # Save bias_stats for plotting if needed
    bias_stats = result.bias_stats
    # -------------------------
    # Verbose Plots (Bias Map)
    # -------------------------
    if verbose_plots and denoise:
        plots.plot_bias_map(bias_stats, normalized_img, N, d)

    # -------------------------
    # Create run directory
    # -------------------------
    run_dir = plots.get_run_dir()

    # Save settings
    settings = {
        "Image": os.path.basename(img_path) if img_path else "drusen1.jpeg",
        "Size": f"{n}x{n}",
        "Backend": "MHRQI (Qiskit)",
        "Fast Mode": fast,
        "Denoise": denoise,
        "Use Shots": use_shots,
        "Shots": shots if use_shots else "N/A (statevector)",
        "d (qudit dim)": d,
        "Simulation Time": f"{end_time - start_time:.2f}s",
    }
    plots.save_settings_plot(settings, run_dir)

    # Get a clean image name from path
    img_name = os.path.splitext(os.path.basename(img_path or "drusen1.jpeg"))[0]
    plots.show_image_comparison(myimg, newimg, run_dir=run_dir, img_name=img_name)

    # -------------------------
    # Run comparison benchmarks
    # -------------------------
    if run_comparison:
        evals_dir = os.path.join(run_dir, "evals")
        print(f"Running benchmarks... saving to {evals_dir}")

        suite = BenchmarkSuite(myimg, save_dir=evals_dir)
        suite.run(methods="all", proposed_image=newimg)
        suite.save_reports(prefix="comp")

    return myimg, newimg, run_dir


def main_cli():
    parser = argparse.ArgumentParser(
        description="MHRQI - Multi-scale Hierarchical Representation of Quantum Images"
    )
    parser.add_argument("--shots", type=int, default=1000, help="Number of measurement shots")
    parser.add_argument("-n", "--size", type=int, default=4, help="Image size (n x n)")
    parser.add_argument(
        "-d", "--dimension", type=int, default=2, help="Qudit dimension (default 2 for qubits)"
    )
    parser.add_argument("--denoise", action="store_true", help="Apply denoising circuit")
    parser.add_argument(
        "--statevector", action="store_true", help="Use statevector simulation instead of shots"
    )
    parser.add_argument("--fast", action="store_true", help="Use lazy upload for speed")
    parser.add_argument("--verbose", action="store_true", help="Show additional debug plots")
    parser.add_argument("--img", type=str, help="Path to input image")
    parser.add_argument(
        "--no-comparison",
        action="store_false",
        dest="comparison",
        help="Skip comparison benchmarks",
    )
    parser.set_defaults(comparison=True)

    args = parser.parse_args()

    main(
        shots=args.shots,
        n=args.size,
        d=args.dimension,
        denoise=args.denoise,
        use_shots=not args.statevector,
        fast=args.fast,
        verbose_plots=args.verbose,
        img_path=args.img,
        run_comparison=args.comparison,
    )


if __name__ == "__main__":
    main_cli()
