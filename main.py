"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MHRQI - Magnitude Hierarchical Representation of Quantum Images            ║
║  Main Pipeline: Encoding, Denoising, Benchmarking                           ║
║                                                                              ║
║  Author: Keno-00                                                             ║
║  License: Apache 2.0                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

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

import circuit  # Renamed from circuit_qiskit.py
import compare_to
import plots
import utils

CSV_PATH = Path("mhrqi_runs.csv")

def save_rows_to_csv(rows, csv_path=CSV_PATH):
    fieldnames = [
        "timestamp", "n", "bins", "shots", "shots_per_bin",
        "mse"
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)

def main(shots=1000, n=4, d=2, denoise=False, use_shots=True, fast=False, verbose_plots=False, img_path=None, run_comparison=True):
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
        img_path = os.path.join(os.path.dirname(__file__), "resources", "drusen1.jpeg")

    myimg = cv2.imread(img_path)
    myimg = cv2.resize(myimg, (n, n))


    myimg = cv2.cvtColor(myimg, cv2.COLOR_RGB2GRAY)
    N = myimg.shape[1]
    angle_norm = utils.angle_map(myimg)
    normalized_img = np.clip(myimg.astype(np.float64) / 255.0, 0.0, 1.0)


    H, W = angle_norm.shape
    L_max = utils.get_Lmax(N, d)
    sk = []
    for L in range(0, L_max):
        sk.append(N if L == 0 else utils.get_subdiv_size(L, N, d))
    hierarchy_matrix = []
    for r, c in np.ndindex(H, W):
        hcv = []
        for _, k in enumerate(sk):
            sub_hcv = utils.compute_register(r, c, d, k)
            hcv.extend(sub_hcv)
        hierarchy_matrix.append(hcv)

    # -------------------------
    # Circuit Construction
    # -------------------------
    qc, pos_regs, intensity_reg, bias = circuit.MHRQI_init(d, L_max)
    upload_fn = circuit.MHRQI_lazy_upload if fast else circuit.MHRQI_upload
    data_qc = upload_fn(qc, pos_regs, intensity_reg, d, hierarchy_matrix, normalized_img)

    # -------------------------
    # Denoising
    # -------------------------
    if denoise:
        data_qc = circuit.DENOISER(data_qc, pos_regs, intensity_reg, bias)

    # -------------------------
    # Simulation
    # -------------------------
    start_time = time.perf_counter()

    if use_shots:
        counts = circuit.simulate_counts(data_qc, shots, use_gpu=True)
        if denoise:
            bins, bias_stats = circuit.make_bins_counts(counts, hierarchy_matrix, denoise=True)
        else:
            bins = circuit.make_bins_counts(counts, hierarchy_matrix, denoise=False)
            bias_stats = None
    else:
        state_vector = circuit.simulate_statevector(data_qc)
        if denoise:
            bins, bias_stats = circuit.make_bins_sv(state_vector, hierarchy_matrix, denoise=True)
        else:
            bins = circuit.make_bins_sv(state_vector, hierarchy_matrix, denoise=False)
            bias_stats = None

    end_time = time.perf_counter()

    # -------------------------
    # Reconstruction
    # -------------------------
    newimg = utils.mhrqi_bins_to_image(bins, hierarchy_matrix, d, (N, N),
                                        bias_stats=bias_stats, original_img=None)
    newimg = (np.clip(newimg, 0.0, 1.0) * 255).astype(np.uint8)
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
        'Image': os.path.basename(img_path) if img_path else 'drusen1.jpeg',
        'Size': f'{n}x{n}',
        'Backend': 'MHRQI (Qiskit)',
        'Fast Mode': fast,
        'Denoise': denoise,
        'Use Shots': use_shots,
        'Shots': shots if use_shots else 'N/A (statevector)',
        'd (qudit dim)': d,
        'Simulation Time': f'{end_time - start_time:.2f}s'
    }
    plots.save_settings_plot(settings, run_dir)

    # Get a clean image name from path
    img_name = os.path.splitext(os.path.basename(img_path or 'drusen1.jpeg'))[0]
    plots.show_image_comparison(myimg, newimg, run_dir=run_dir, img_name=img_name)

    # -------------------------
    # Run comparison benchmarks
    # -------------------------
    if run_comparison:
        evals_dir = os.path.join(run_dir, "evals")
        print(f"Running benchmarks... saving to {evals_dir}")

        compare_to.compare_to(
            myimg,
            proposed_img=newimg,
            methods="all",
            plot=True,
            save=True,
            save_prefix="comp",
            save_dir=evals_dir,
            reference_image=None  # No synthetic reference - only no-ref metrics
        )

    return myimg, newimg, run_dir


if __name__ == "__main__":
    # Configuration
    n = 256  # Image size
    d = 2   # qudit dimension: 2=qubit

    # Single MHRQI backend (no choice needed)

    # Simulation settings
    use_shots = False       # False = statevector (exact), True = shot-based sampling
    shots_list = [10000000]
    fast = True             # Use lazy (statevector) upload for speed
    denoise = True           # Apply denoising circuit

    verbose_plots = True
    run_comparison = True


    # Testing mode
    do_tests = False
    if do_tests:
        bin_of_n = 2 * (n ** 2)
        for j in range(2, 10):
            shots_list.append(bin_of_n * j)

    # Collect trend data if doing multiple runs
    run_mse = []
    shots_used = []

    for shot_count in shots_list:
        # Reset run directory for new runs
        plots.reset_run_dir()

        gt_img, rec_img, run_dir = main(
            shots=shot_count,
            n=n,
            d=d,
            denoise=denoise,
            use_shots=use_shots,
            fast=fast,
            verbose_plots=verbose_plots,
            img_path="resources/non_medical/plane.png",
            run_comparison=run_comparison
        )

        # These are already saved in the run directory
        plots.plot_mse_map(gt_img, rec_img)

        # Collect trend data for multi-shot runs
        if len(shots_list) > 1:
            i_mse = plots.compute_mse(gt_img, rec_img)
            run_mse.append(i_mse)
            shots_used.append(shot_count)

        print(f"Run complete. Output saved to: {run_dir}")

    # Plot trends if multiple shot counts were tested
    if len(shots_list) > 1 and verbose_plots:
        plots.plot_shots_vs_mse(shots_used, run_mse)
