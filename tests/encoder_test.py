"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MHRQI - Encoder Synthetic Test                                              ║
║  Controlled test: encode Lenna (128×128) through MHRQI encoder              ║
║  Outputs: MSE/PSNR comparison plot, MSE heatmap, complexity scaling JSON.   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python tests/encoder_test.py                              # n=128, lazy, shots
    python tests/encoder_test.py --shots 2000000              # explicit shot count
    python tests/encoder_test.py --sizes 2 4 8 128            # complexity scaling
    python tests/encoder_test.py --statevector                # exact statevector mode
"""

import argparse
import datetime
import json
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import circuit
import plots
import utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_dir(base="runs", subdir="synthetic_encoder"):
    """Create runs/<timestamp>/<subdir>/ following the project convention."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, ts, subdir)
    os.makedirs(path, exist_ok=True)
    return path


def _load_lenna_gray(n, lenna_path):
    """Load Lenna and resize to n×n grayscale uint8."""
    img = cv2.imread(lenna_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Lenna image not found at: {lenna_path}")
    img = cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)
    return img


def _run_encoder(img_gray_uint8, n, d=2, fast=True, use_shots=False, shots=2_000_000):
    """
    Run MHRQI encoder (no denoiser) on an n×n grayscale image.
    Returns (reconstructed_uint8, circuit_stats dict).
    """
    normalized = np.clip(img_gray_uint8.astype(np.float64) / 255.0, 0.0, 1.0)
    L_max = utils.get_Lmax(n, d)

    sk = [n if L == 0 else utils.get_subdiv_size(L, n, d) for L in range(L_max)]
    hierarchy_matrix = []
    for r, c in np.ndindex(n, n):
        hcv = []
        for k in sk:
            hcv.extend(utils.compute_register(r, c, d, k))
        hierarchy_matrix.append(hcv)

    qc, pos_regs, intensity_reg, bias = circuit.MHRQI_init(d, L_max)
    upload_fn = circuit.MHRQI_lazy_upload if fast else circuit.MHRQI_upload
    data_qc = upload_fn(qc, pos_regs, intensity_reg, d, hierarchy_matrix, normalized)

    # Exact circuit statistics from Qiskit
    circuit_stats = {
        "num_qubits":  data_qc.num_qubits,
        "gate_count":  data_qc.size(),
        "depth":       data_qc.depth(),
        "gate_counts": dict(data_qc.count_ops()),
    }

    if use_shots:
        counts = circuit.simulate_counts(data_qc, shots=shots, use_gpu=True)
        bins = circuit.make_bins_counts(counts, hierarchy_matrix, denoise=False)
    else:
        sv = circuit.simulate_statevector(data_qc)
        bins = circuit.make_bins_sv(sv, hierarchy_matrix, denoise=False)

    rec_float = utils.mhrqi_bins_to_image(bins, hierarchy_matrix, d, (n, n),
                                           bias_stats=None, original_img=None)
    return (np.clip(rec_float, 0.0, 1.0) * 255).astype(np.uint8), circuit_stats


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save_comparison_plot(orig, rec, mse, psnr, n, run_dir):
    """Save side-by-side comparison with MSE / PSNR annotated."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(orig, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Lenna Original ({n}×{n})")
    axes[0].axis("off")
    axes[1].imshow(rec, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("MHRQI Reconstructed")
    axes[1].axis("off")
    fig.suptitle(f"Encoder Test — MSE: {mse:.4f} | PSNR: {psnr:.2f} dB", fontsize=12)
    plt.tight_layout()
    out = os.path.join(run_dir, f"lenna_n{n}_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison: {out}")


def _save_mse_heatmap(orig, rec, n, run_dir):
    """Save per-pixel squared error heatmap."""
    se = (orig.astype(np.float32) - rec.astype(np.float32)) ** 2
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(se, cmap="RdYlGn_r", interpolation="nearest")
    ax.set_title(f"MSE Map — n={n}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="squared error")
    plt.tight_layout()
    out = os.path.join(run_dir, f"lenna_n{n}_mse_map.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved MSE map:    {out}")


def _save_scaling_plot(results, run_dir):
    """4-panel complexity scaling: MSE, PSNR, gate count, depth vs n."""
    sizes       = [r["n"] for r in results]
    mses        = [r["mse"] for r in results]
    psnrs       = [r["psnr_dB"] for r in results if r["psnr_dB"] != "inf"]
    sizes_psnr  = [r["n"]  for r in results if r["psnr_dB"] != "inf"]
    times       = [r["encode_time_s"] for r in results]
    gate_counts = [r["gate_count"] for r in results]
    depths      = [r["depth"] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("MHRQI Encoder — Complexity Scaling", fontsize=13)

    axes[0, 0].plot(sizes, mses, "o-", color="tomato")
    axes[0, 0].set_xlabel("n"); axes[0, 0].set_ylabel("MSE")
    axes[0, 0].set_title("MSE vs n"); axes[0, 0].grid(True)

    axes[0, 1].plot(sizes_psnr, psnrs, "o-", color="steelblue")
    axes[0, 1].set_xlabel("n"); axes[0, 1].set_ylabel("PSNR (dB)")
    axes[0, 1].set_title("PSNR vs n"); axes[0, 1].grid(True)

    axes[0, 2].plot(sizes, times, "o-", color="seagreen")
    axes[0, 2].set_xlabel("n"); axes[0, 2].set_ylabel("Encode time (s)")
    axes[0, 2].set_title("Encoding Time vs n"); axes[0, 2].grid(True)

    axes[1, 0].plot(sizes, gate_counts, "o-", color="orchid")
    axes[1, 0].set_xlabel("n"); axes[1, 0].set_ylabel("Gate count (exact)")
    axes[1, 0].set_title("Gate Count vs n"); axes[1, 0].grid(True)

    axes[1, 1].plot(sizes, depths, "o-", color="goldenrod")
    axes[1, 1].set_xlabel("n"); axes[1, 1].set_ylabel("Circuit depth (exact)")
    axes[1, 1].set_title("Circuit Depth vs n"); axes[1, 1].grid(True)

    axes[1, 2].axis("off")  # spare panel

    plt.tight_layout()
    out = os.path.join(run_dir, "encoder_scaling.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")

    plt.close(fig)
    print(f"  Saved scaling plot: {out}")


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run(sizes, fast, use_shots, shots, lenna_path):
    run_dir = _make_run_dir(subdir="synthetic_encoder")
    print(f"\n=== Encoder Synthetic Test ===")
    print(f"Output directory: {run_dir}")
    print(f"Lenna path:       {lenna_path}")
    print(f"Sizes to test:    {sizes}")
    print(f"Mode:             {'shots=' + str(shots) if use_shots else 'statevector'}")
    print(f"Fast (lazy) upload: {fast}\n")

    results = []

    for n in sizes:
        print(f"--- n={n} ---")
        orig = _load_lenna_gray(n, lenna_path)

        t0 = time.perf_counter()
        rec, cstats = _run_encoder(orig, n, fast=fast, use_shots=use_shots, shots=shots)
        elapsed = time.perf_counter() - t0

        mse  = plots.compute_mse(orig, rec)
        psnr = plots.compute_psnr(orig, rec)

        d_val = 2
        L_max  = utils.get_Lmax(n, d_val)

        entry = {
            "n":             n,
            "num_pixels":    n * n,
            "L_max":         L_max,
            "num_qubits":    cstats["num_qubits"],
            "gate_count":    cstats["gate_count"],
            "depth":         cstats["depth"],
            "gate_counts":   cstats["gate_counts"],
            "encode_time_s": round(elapsed, 4),
            "mse":           round(float(mse), 6),
            "psnr_dB":       round(float(psnr), 4) if psnr != float('inf') else "inf",
            "fast_upload":   fast,
            "mode":          "shots" if use_shots else "statevector",
            "shots":         shots if use_shots else None,
        }
        results.append(entry)

        print(f"  MSE:         {mse:.4f}")
        print(f"  PSNR:        {psnr:.2f} dB")
        print(f"  Encode time: {elapsed:.3f}s")
        print(f"  Qubits:      {cstats['num_qubits']}")
        print(f"  Gate count:  {cstats['gate_count']}")
        print(f"  Depth:       {cstats['depth']}")

        _save_comparison_plot(orig, rec, mse, psnr, n, run_dir)
        _save_mse_heatmap(orig, rec, n, run_dir)

    if len(results) > 1:
        _save_scaling_plot(results, run_dir)

    json_path = os.path.join(run_dir, "encoder_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved raw results: {json_path}")
    print(f"\n=== Done. All outputs in: {run_dir} ===\n")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MHRQI Encoder Synthetic Test (Lenna)")
    parser.add_argument("--n", type=int, default=128,
                        help="Image size n (power of 2). Default: 128")
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                        help="List of n values for complexity scaling, e.g. --sizes 2 4 8 128")
    parser.add_argument("--shots", type=int, default=2_000_000,
                        help="Number of shots. Default: 2000000")
    parser.add_argument("--statevector", action="store_true",
                        help="Use exact statevector simulation instead of shots")
    parser.add_argument("--no-fast", action="store_true",
                        help="Disable lazy upload (gate-by-gate, slow for large n)")
    parser.add_argument("--lenna", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "resources", "non_medical", "lenna.jpg"),
                        help="Path to Lenna image")
    args = parser.parse_args()

    sizes = args.sizes if args.sizes else [args.n]
    for s in sizes:
        if s < 2 or (s & (s - 1)) != 0:
            print(f"[ERROR] n={s} is not a power of 2.")
            sys.exit(1)

    run(sizes=sizes, fast=not args.no_fast,
        use_shots=not args.statevector, shots=args.shots,
        lenna_path=args.lenna)

