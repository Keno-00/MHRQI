"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MHRQI - Multiscale-Hierarchical Representation of Quantum Images          ║
║  Main Pipeline: Encoding, Denoising, Benchmarking                           ║
║                                                                              ║
║  Author: Keno-00                                                             ║
║  License: Apache 2.0                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import configparser
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


def save_measurement_evidence(evidence_dir, counts, bins, bias_stats, use_shots, shots, n, bit_depth,
                              denoise, simulation_seconds, data_qc):
    """Persist the measurement record required to reproduce image reconstruction."""
    os.makedirs(evidence_dir, exist_ok=True)
    address_rows = []
    for address, values in sorted(bins.items()):
        count = float(values["count"])
        mean_intensity = values["intensity_sum"] / count if count else float("nan")
        variance = (
            max(0.0, values["intensity_squared_sum"] / count - mean_intensity ** 2)
            if count else float("nan")
        )
        outcome = bias_stats.get(address, {}) if bias_stats is not None else {}
        address_rows.append({
            "hierarchical_address": "".join(map(str, address)),
            "sample_count": count,
            "mean_intensity": mean_intensity,
            "intensity_variance": variance,
            "outcome_one_count": outcome.get("hit", ""),
            "outcome_zero_count": outcome.get("miss", ""),
        })
    with open(os.path.join(evidence_dir, "mhrqi_address_support.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "hierarchical_address", "sample_count", "mean_intensity", "intensity_variance",
                "outcome_one_count", "outcome_zero_count",
            ],
        )
        writer.writeheader()
        writer.writerows(address_rows)

    execution_record = {
        "record_schema": "mhrqi-measurement-v1",
        "mode": "shots" if use_shots else "statevector",
        "requested_shots": shots if use_shots else None,
        "observed_shots": int(sum(counts.values())) if counts is not None else None,
        "image_size": n,
        "spatial_addresses": n * n,
        "expected_samples_per_address": shots / float(n * n) if use_shots else None,
        "nonempty_addresses": len(address_rows),
        "bit_depth": bit_depth,
        "denoise": denoise,
        "simulation_seconds": simulation_seconds,
        "circuit_qubits": data_qc.num_qubits,
        "circuit_depth": data_qc.depth(),
        "circuit_operations": int(sum(data_qc.count_ops().values())),
    }
    with open(os.path.join(evidence_dir, "mhrqi_execution_record.json"), "w", encoding="utf-8") as handle:
        json.dump(execution_record, handle, indent=2)

    if counts is not None:
        with open(os.path.join(evidence_dir, "mhrqi_measurement_counts.json"), "w", encoding="utf-8") as handle:
            json.dump({str(bitstring): int(count) for bitstring, count in counts.items()}, handle)

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
    evidence_dir=None,
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
        evidence_dir: optional directory for raw measurement evidence and support logs
    
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
            bins, bias_stats = circuit.make_bins_counts(
                counts, hierarchy_matrix, bit_depth=bit_depth, denoise=True
            )
        else:
            bins = circuit.make_bins_counts(
                counts, hierarchy_matrix, bit_depth=bit_depth, denoise=False
            )
            bias_stats = None
    else:
        state_vector = circuit.simulate_statevector(
            data_qc,
            device=simulation_device,
            seed_simulator=seed_simulator,
            seed_transpiler=seed_transpiler,
        )
        if denoise:
            bins, bias_stats = circuit.make_bins_sv(
                state_vector, hierarchy_matrix, bit_depth=bit_depth, denoise=True
            )
        else:
            bins = circuit.make_bins_sv(
                state_vector, hierarchy_matrix, bit_depth=bit_depth, denoise=False
            )
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
    evidence_dir = evidence_dir or run_dir
    save_measurement_evidence(
        evidence_dir=evidence_dir,
        counts=counts if use_shots else None,
        bins=bins,
        bias_stats=bias_stats,
        use_shots=use_shots,
        shots=shots,
        n=N,
        bit_depth=bit_depth,
        denoise=denoise,
        simulation_seconds=end_time - start_time,
        data_qc=data_qc,
    )

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
    # Load configuration (supports paper.ini or paper.json)
    configs_dir = os.path.join(os.path.dirname(__file__), "configs")
    ini_path = os.path.join(configs_dir, "paper.ini")
    json_path = os.path.join(configs_dir, "paper.json")

    if os.path.exists(ini_path):
        cp = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
        cp.read(ini_path, encoding="utf-8")

        shots_raw = cp.get("execution", "shots", fallback="none").strip().lower()
        shots_val = int(shots_raw) if shots_raw.isdigit() else None

        config = {
            "schema_version": cp.getint("general", "schema_version", fallback=1),
            "experiment_name": cp.get("general", "experiment_name", fallback="ecti_revision_reference"),
            "image_size": cp.getint("image", "image_size", fallback=128),
            "resize_interpolation": cp.get("image", "resize_interpolation", fallback="opencv_INTER_AREA"),
            "subdivision_factor": cp.getint("image", "subdivision_factor", fallback=2),
            "bit_depth": cp.getint("image", "bit_depth", fallback=8),
            "benchmark_mode": cp.get("execution", "benchmark_mode", fallback="statevector"),
            "shots": shots_val,
            "denoise": cp.getboolean("execution", "denoise", fallback=True),
            "fast": cp.getboolean("execution", "fast", fallback=False),
            "verbose_plots": cp.getboolean("execution", "verbose_plots", fallback=False),
            "run_comparison": cp.getboolean("execution", "run_comparison", fallback=False),
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
                },
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
        }
    else:
        with open(json_path, "r", encoding="utf-8") as f:
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
