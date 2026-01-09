import cv2, math
import matplotlib.pyplot as plt
import utils
import circuit
import circuit_qiskit as circuit_2
import circuit_dd as circuit3
import numpy as np
import plots
from pathlib import Path
from datetime import datetime
import csv
import time
import compare_to
import matplotlib
import os


linuxmode = True


CSV_PATH = Path("mhrqi_runs.csv")

def save_rows_to_csv(rows, csv_path=CSV_PATH):
    fieldnames = [
        "timestamp", "n", "bins", "shots", "shots_per_bin",
        "mse", "psnr"
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)

def main(shots=1000, n=4, d=2, denoise=False, use_shots=True, backend='qiskit_mhrqi', fast=False, verbose_plots=False, img_path=None, run_comparison=True):
    """
    Main MHRQI/MHRQIB simulation pipeline.
    
    Args:
        shots: number of measurement shots (if use_shots=True)
        n: image dimension (will be resized to n x n)
        d: qudit dimension (2=qubit, 3=qutrit, etc.)
        denoise: whether to apply denoising circuit
        use_shots: if True, use shot-based simulation; if False, use statevector
        backend: one of 'qiskit_mhrqi', 'qiskit_mhrqib', 'mqt_mhrqi', 'mqt_dd_mhrqi'
        fast: if True, use lazy (statevector-based) upload for speed
        verbose_plots: if True, show additional debug plots
        img_path: path to input image (defaults to resources/drusen1.jpeg)
        run_comparison: if True, run comparison benchmarks against BM3D/NL-Means/SRAD
    
    Returns:
        tuple: (original_image, reconstructed_image, run_directory_path)
    """
    # Validate backend
    valid_backends = {'qiskit_mhrqi', 'qiskit_mhrqib', 'mqt_mhrqi', 'mqt_dd_mhrqi'}
    if backend not in valid_backends:
        raise ValueError(f"Invalid backend '{backend}'. Must be one of: {valid_backends}")
    
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
    # Dictionary Prep (for MQT backends)
    # -------------------------
    intensity_dict = None
    if 'mqt' in backend:
        intensity_dict = {}
        for i, hcv in enumerate(hierarchy_matrix):
            r, c = utils.compose_rc(hcv, d) 
            bitstring_key = ''.join(str(x) for x in hcv)
            intensity_dict[bitstring_key] = normalized_img[r, c]

    # -------------------------
    # Circuit Construction
    # -------------------------
    if backend == 'qiskit' or backend == 'qiskit_mhrqib':
        # Unified MHRQI with basis-encoded intensity and bias qubit
        qc, pos_regs, intensity_reg, bias = circuit_2.MHRQI_init_qiskit(d, L_max)
        upload_fn = circuit_2.MHRQIB_lazy_upload_intensity_qiskit if fast else circuit_2.MHRQIB_upload_intensity_qiskit
        data_qc = upload_fn(qc, pos_regs, intensity_reg, d, hierarchy_matrix, normalized_img)
        
    elif backend == 'mqt_dd_mhrqi':
        qc, reg = circuit3.MHRQI_init(d, L_max)
        bias = None
        pos_regs = None
        intensity_reg = None
        data_qc = circuit3.MHRQI_upload_intensity(qc, reg, intensity_dict, approx_threshold=0.1)
        
    elif backend == 'mqt_mhrqi':
        qc, reg = circuit.MHRQI_init(d, L_max)
        bias = None
        pos_regs = None
        intensity_reg = None
        upload_fn = circuit.MHRQI_lazy_upload_intensity if fast else circuit.MHRQI_upload_intensity
        if fast:
            data_qc = upload_fn(qc, reg, intensity_dict, approx_threshold=0.01)
        else:
            data_qc = upload_fn(qc, reg, d, hierarchy_matrix, angle_norm)

    # -------------------------
    # Denoising
    # -------------------------
    if denoise:
        if backend in ['qiskit', 'qiskit_mhrqib']:
            data_qc = circuit_2.DENOISER_qiskit(data_qc, pos_regs, intensity_reg, bias, strength=1.65)
        else:
            data_qc = circuit.DENOISER(data_qc, reg, d, L_max, time_step=0.5)

    # -------------------------
    # Simulation
    # -------------------------
    start_time = time.perf_counter()

    if backend in ['qiskit', 'qiskit_mhrqib']:
        if use_shots:
            counts = circuit_2.simulate_counts(data_qc, shots, use_gpu=True)
            print("finished simulation")
            # TODO: Add counts-based make_bins with denoise flag
            bins = circuit_2.make_bins_mhrqib_qiskit(counts, hierarchy_matrix)
            bias_stats = None
        else:
            state_vector = circuit_2.simulate_statevector(data_qc)
            print("finished simulation")
            if denoise:
                bins, bias_stats = circuit_2.make_bins_sv(state_vector, hierarchy_matrix, denoise=True)
            else:
                bins = circuit_2.make_bins_sv(state_vector, hierarchy_matrix, denoise=False)
                bias_stats = None
    else:
        # MQT
        bias_stats = None
        if use_shots:
            counts = circuit.MHRQI_simulate(data_qc, shots=shots)
            print("finished simulation")
            bins = utils.make_bins(counts, hierarchy_matrix)
        else:
            state_vector = circuit.MHRQI_simulate(data_qc)
            print("finished simulation")
            bins = utils.make_bins_sv(state_vector, hierarchy_matrix)
            
    end_time = time.perf_counter()
    print(f"Simulation time: {end_time - start_time:.4f} seconds")
            
    # -------------------------
    # Reconstruction
    # -------------------------
    if backend in ['qiskit', 'qiskit_mhrqib']:
        # Pass original normalized image for probability-based edge denoising
        newimg = utils.mhrqi_bins_to_image(bins, hierarchy_matrix, d, (N, N), 
                                            bias_stats=bias_stats,)
        newimg = (np.clip(newimg, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        newimg = plots.bins_to_image(bins, d, N, kind="p")
    # -------------------------
    # Verbose Plots (Homogeneity Map)
    # -------------------------
    if verbose_plots and denoise and backend in ['qiskit', 'qiskit_mhrqib']:
        # Extract edge_map for visualization
        total_prob = sum(bins[tuple(v)]['count'] for v in hierarchy_matrix if tuple(v) in bins)
        uniform_prob = total_prob / len(hierarchy_matrix) if len(hierarchy_matrix) > 0 else 1.0
        
        edge_map = {}
        for vec in hierarchy_matrix:
            key = tuple(vec)
            if key in bins:
                prob = bins[key]['count']
                r, c = utils.compose_rc(vec, d)
                edge_map[(r, c)] = min(prob / uniform_prob, 1.0) if uniform_prob > 0 else 0.5
        
        plots.plot_homogeneity_map(edge_map, normalized_img, N, d, L_max, threshold=0.4)

    # -------------------------
    # Create run directory
    # -------------------------
    run_dir = plots.get_run_dir()
    
    # Save settings
    settings = {
        'Image': os.path.basename(img_path) if img_path else 'drusen1.jpeg',
        'Size': f'{n}x{n}',
        'Backend': backend,
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
        
        # Prepare Reference (NL-Means) as "Ground Truth" for Full-Ref metrics
        nlmeans_ref = None
        print("Generating NL-Means reference...")
        try:
            input_float = compare_to.to_float01(myimg)
            nlmeans_ref = compare_to.denoise_nlmeans(input_float)
        except Exception as e:
            print(f"Warning: NL-Means generation failed: {e}")
        
        compare_to.compare_to(
            myimg,
            proposed_img=newimg,
            methods="all",
            plot=True,
            save=True,
            save_prefix="comp",
            save_dir=evals_dir,
            reference_image=nlmeans_ref
        )
    
    return myimg, newimg, run_dir


if __name__ == "__main__":
    # Configuration
    n = 128  # Image size (64x64 for fast testing)
    d = 2   # qudit dimension: 2=qubit
    
    # Backend options: 'qiskit_mhrqi', 'qiskit_mhrqib', 'mqt_mhrqi', 'mqt_dd_mhrqi'
    backend = 'qiskit_mhrqib'
    
    # Simulation settings
    use_shots = False       # False = statevector (exact), True = shot-based sampling
    shots_list = [10000000]
    fast = True             # Use lazy (statevector) upload for speed
    denoise = True          # Apply denoising circuit

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
    run_psnr = []
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
            backend=backend,
            fast=fast,
            verbose_plots=verbose_plots,
            run_comparison=run_comparison
        )
        
        # These are already saved in the run directory
        plots.plot_mse_map(gt_img, rec_img)
        plots.plot_psnr_map(gt_img, rec_img)
        
        # Collect trend data for multi-shot runs
        if len(shots_list) > 1:
            i_mse = plots.compute_mse(gt_img, rec_img)
            i_psnr = plots.compute_psnr(gt_img, rec_img)
            run_mse.append(i_mse)
            run_psnr.append(i_psnr)
            shots_used.append(shot_count)
        
        print(f"Run complete. Output saved to: {run_dir}")

    # Plot trends if multiple shot counts were tested
    if len(shots_list) > 1 and verbose_plots:
        plots.plot_shots_vs_mse(shots_used, run_mse)
        plots.plot_shots_vs_psnr(shots_used, run_psnr)