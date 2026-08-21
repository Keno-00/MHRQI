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
import json
import math
import os
import random
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
    simulation_device="CPU",
    seed_simulator=20260818,
    seed_transpiler=20260818,
    bit_depth=8,
    resize_interpolation="opencv_INTER_AREA",
    seed_python=20260818,
    seed_numpy=20260818,
    baseline_config=None,
):
    """
    Main MHRQI simulation pipeline.
    
    Args:
        shots: number of measurement shots (if use_shots=True)
        n: image dimension (will be resized to n x n)
        d: binary spatial subdivision factor (the implementation uses qubits)
        denoise: whether to apply denoising circuit
        use_shots: if True, use shot-based simulation; if False, use statevector
        fast: if True, use lazy (statevector-based) upload for speed
        verbose_plots: if True, show additional debug plots
        img_path: path to input image (defaults to resources/drusen1.jpeg)
        run_comparison: if True, run comparison benchmarks against BM3D/NL-Means/SRAD
        simulation_device: Qiskit Aer device, normally "CPU" or "GPU"
        seed_simulator: random seed used by the simulator
        seed_transpiler: random seed used by the transpiler
        bit_depth: bits used for intensity encoding
        resize_interpolation: OpenCV interpolation mode name or constant
        seed_python: seed for Python random module
        seed_numpy: seed for NumPy random module
        baseline_config: dict containing parameters for baseline algorithms
    
    Returns:
        tuple: (original_image, reconstructed_image, run_directory_path)
    """

    # Seed random generators
    if seed_python is not None:
        random.seed(seed_python)
    if seed_numpy is not None:
        np.random.seed(seed_numpy)

    # Use default image if not specified
    if img_path is None:
        img_path = os.path.join(os.path.dirname(__file__), "resources", "drusen1.jpeg")

    # Map interpolation method
    if isinstance(resize_interpolation, str):
        interp_name = resize_interpolation.replace("opencv_", "")
        interp_flag = getattr(cv2, interp_name, cv2.INTER_AREA)
    else:
        interp_flag = resize_interpolation

    myimg = cv2.imread(img_path)
    myimg = cv2.resize(myimg, (n, n), interpolation=interp_flag)


    myimg = cv2.cvtColor(myimg, cv2.COLOR_RGB2GRAY)
    N = myimg.shape[1]
    angle_norm = utils.angle_map(myimg)
    max_intensity = float((2 ** bit_depth) - 1)
    normalized_img = np.clip(myimg.astype(np.float64) / max_intensity, 0.0, 1.0)


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
    qc, pos_regs, intensity_reg, bias = circuit.MHRQI_init(d, L_max, bit_depth=bit_depth)
    upload_fn = circuit.MHRQI_lazy_upload if fast else circuit.MHRQI_upload
    data_qc = upload_fn(qc, pos_regs, intensity_reg, d, hierarchy_matrix, normalized_img)

    # -------------------------
    # Denoising
    # -------------------------
    if denoise:
        data_qc, _ = circuit.DENOISER(data_qc, pos_regs, intensity_reg, bias)

    # -------------------------
    # Simulation
    # -------------------------
    start_time = time.perf_counter()

    if use_shots:
        counts = circuit.simulate_counts(
            data_qc,
            shots,
            device=simulation_device,
            seed_simulator=seed_simulator,
            seed_transpiler=seed_transpiler,
        )
        if denoise:
            bins, bias_stats = circuit.make_bins_counts(counts, hierarchy_matrix, denoise=True)
        else:
            bins = circuit.make_bins_counts(counts, hierarchy_matrix, denoise=False)
            bias_stats = None
    else:
        state_vector = circuit.simulate_statevector(
            data_qc,
            device=simulation_device,
            seed_simulator=seed_simulator,
            seed_transpiler=seed_transpiler,
        )
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
    newimg = (np.clip(newimg, 0.0, 1.0) * max_intensity).astype(np.uint8)
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
        'Spatial subdivision factor d': d,
        'Bit Depth': bit_depth,
        'Resize Interpolation': str(resize_interpolation),
        'Python Seed': seed_python,
        'NumPy Seed': seed_numpy,
        'Simulation device': simulation_device,
        'Simulator seed': seed_simulator,
        'Transpiler seed': seed_transpiler,
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
            reference_image=None,  # No synthetic reference - only no-ref metrics
            baseline_config=baseline_config,
        )

    return myimg, newimg, run_dir


if __name__ == "__main__":
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "configs", "paper.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Configuration extracted from config file
    n = config.get("image_size", 256)
    d = config.get("subdivision_factor", 2)
    bit_depth = config.get("bit_depth", 8)
    resize_interpolation = config.get("resize_interpolation", "opencv_INTER_AREA")

    # Simulation settings
    benchmark_mode = config.get("benchmark_mode", "statevector")
    use_shots = (benchmark_mode != "statevector")
    shots_val = config.get("shots")
    shots_list = [shots_val] if shots_val is not None else [10000000]

    denoise = config.get("denoise", True)
    fast = config.get("fast", False)
    verbose_plots = config.get("verbose_plots", False)
    run_comparison = config.get("run_comparison", False)

    seeds = config.get("seeds", {})
    seed_python = seeds.get("python", 20260818)
    seed_numpy = seeds.get("numpy", 20260818)
    seed_simulator = seeds.get("simulator", 20260818)
    seed_transpiler = seeds.get("transpiler", 20260818)

    simulator_cfg = config.get("simulator", {})
    simulation_device = simulator_cfg.get("device", "CPU")

    baselines_cfg = config.get("baselines", {})

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
            run_comparison=run_comparison,
            simulation_device=simulation_device,
            seed_simulator=seed_simulator,
            seed_transpiler=seed_transpiler,
            bit_depth=bit_depth,
            resize_interpolation=resize_interpolation,
            seed_python=seed_python,
            seed_numpy=seed_numpy,
            baseline_config=baselines_cfg,
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
