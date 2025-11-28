import cv2, math
import matplotlib.pyplot as plt
import utils
import circuit
import circuit_qiskit as circuit_2
import numpy as np
import plots
from pathlib import Path
from datetime import datetime
import csv

import matplotlib
import os

HEADLESS = matplotlib.get_backend().lower().endswith("agg")

if HEADLESS:
    linuxmode = False


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

def main(shots=1000, n=4, d=2, denoise=False, use_shots=True,compiler = False,qiskit_mode = False):
    myimg = cv2.imread("resources/cnv.jpeg")
    myimg = cv2.resize(myimg, (n,n))
    myimg = cv2.cvtColor(myimg, cv2.COLOR_RGB2GRAY)
    N = myimg.shape[1]
    
    angle_norm = utils.angle_map(myimg)
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
    

    if qiskit_mode:
        qc, pos_regs, intensity_reg, accumulator_reg, and_ancilla_reg, work_regs = circuit_2.MHRQI_init_qiskit(d, L_max)
        data_qc = circuit_2.MHRQI_upload_intensity_qiskit(qc, pos_regs, intensity_reg, d, hierarchy_matrix, angle_norm)
    else:
        qc, reg = circuit.MHRQI_init(d, L_max)
        data_qc = circuit.MHRQI_upload_intensity(qc, reg, d, hierarchy_matrix, angle_norm)
    
    if denoise:
        if qiskit_mode:
            data_qc = circuit_2.DENOISER_qiskit(qc, pos_regs, intensity_reg, accumulator_reg, and_ancilla_reg, work_regs, d, hierarchy_matrix, angle_norm)
        else:
            data_qc = circuit_2.DENOISER_qiskit(
                qc, pos_regs, intensity_reg,
                accumulator_reg, and_ancilla_reg, work_regs,
                d, hierarchy_matrix, angle_norm,
                beta=1.0,
                alpha=1.0,
                lambda_color=0.3,
                target_u=np.pi/8.0,
                target_v=np.pi/8.0,
                num_layers=3,          # <-- more Grover layers
            )
    
    # Simulate based on flag
    if use_shots:
        if qiskit_mode:
            counts = circuit_2.simulate_counts(qc, shots,linuxmode)
        else:
            if compiler:
                counts = circuit.MHRQI_compiled(data_qc, shots)
            else:
                counts = circuit.MHRQI_simulate(data_qc, shots)

        if qiskit_mode:
            if denoise:
                bins = circuit_2.make_bins_denoised_qiskit(counts,hierarchy_matrix)
            else:

                bins = circuit_2.make_bins_qiskit(counts,hierarchy_matrix)
        else:
            if denoise:
                bins = utils.make_bins_denoised(counts, hierarchy_matrix)
            else:
                bins = utils.make_bins(counts, hierarchy_matrix)
    else:

        if qiskit_mode:
            print("not implemented")
        else:
            state_vector = circuit.MHRQI_simulate_state_vector(data_qc)
            print(state_vector)


            if denoise:
                bins = utils.make_bins_sv_denoised(state_vector, hierarchy_matrix)
            else:
                bins = utils.make_bins_sv(state_vector, hierarchy_matrix)
    # plots.plot_hits_grid(bins,d,N,kind="hit")
    # plots.plot_hits_grid(bins,d,N,kind="miss")
    #print(bins)
    grid = plots.bins_to_grid(bins,d,N,kind="p")
    #grid,_,_,_=plots.plot_hits_grid(bins,d,N,kind="p")
    # plots.plot_hits_scatter(bins,d,N,kind="hit")
    # plots.plot_hits_scatter(bins,d,N,kind="miss")
    # plots.plot_hits_scatter(bins,d,N,kind="p")
    
    newimg = plots.grid_to_image_uint8(grid,0.0,1.0)
    plots.show_image_comparison(myimg,newimg)
    return myimg,newimg

#

    

if __name__ == "__main__":
    n = 729
    qudit_d =3 # qubit  =  2, qutrit =  3, ququart = 4
                #uses qiskit on qubits.
    if qudit_d ==2:
        qiskit_mode = True
    else:
        qiskit_mode = False

    bin_of_n = 2*(n**2)
    tests = 10
    do_tests = False
    shots = [900]
    run_psnr = []
    run_mse = []
    rows = []

    denoise = False
    use_shots = False
    compiler = False

    if do_tests:
        for j in range(2, tests):
            shots.append(bin_of_n * j)

    for i in shots:
        #print(f"Image size: {n}x{n}\nBins: {bin_of_n}\nCurrent Shots: {i}\nShots per Bin: {i/bin_of_n}")
        gt_img, rec_img = main(i,n,qudit_d,denoise,use_shots,compiler,qiskit_mode)
        #gt_img, rec_img = main_state_vector(n,qudit_d,denoise=True)
        #main(i, n)

#         plots.plot_mse_map(gt_img, rec_img)
#         plots.plot_psnr_map(gt_img, rec_img)

#         i_mse = plots.compute_mse(gt_img, rec_img)
#         i_psnr = plots.compute_psnr(gt_img, rec_img)

#         run_mse.append(i_mse)
#         run_psnr.append(i_psnr)

#         rows.append({
#             "timestamp": datetime.now().isoformat(timespec="seconds"),
#             "n": n,
#             "bins": bin_of_n,
#             "shots": i,
#             "shots_per_bin": i / bin_of_n,
#             "mse": float(i_mse),
#             "psnr": float(i_psnr),
#         })

#         print(f'===================\nCurrently running: {i} shots \nCurrent MSE: {i_mse}\nCurrent PSNR: {i_psnr}\n===================\n')
#     save_rows_to_csv(rows)

#     plots.plot_shots_vs_mse(shots, run_mse)
#     plots.plot_shots_vs_psnr(shots, run_psnr)
# #