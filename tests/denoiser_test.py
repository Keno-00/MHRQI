"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MHRQI - Encoder + Denoiser Synthetic Test                                  ║
║  Controlled test: 128×128 synthetic image (gradient + speckle)              ║
║                                                                              ║
║  Image layout (128×128):                                                    ║
║    Left half  (cols 0–63):   vertical gradient (0→255, rows 0→127)         ║
║    Right half (cols 64–127): uniform gray (127) + multiplicative speckle    ║
║                                                                              ║
║  Runs at native n=128 using lazy (statevector) upload.                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python tests/denoiser_test.py               # n=128, statevector, lazy upload
    python tests/denoiser_test.py --n 32 --use-shots --shots 2000000
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

def _make_run_dir(base="runs", subdir="synthetic_denoiser"):
    """Create runs/<timestamp>/<subdir>/ following the project convention."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, ts, subdir)
    os.makedirs(path, exist_ok=True)
    return path


def _make_synthetic_image(size=128):
    """
    Build the controlled synthetic test image (size×size, uint8).

    Left half:  vertical gradient — intensity 0 (top) to 255 (bottom).
    Right half: uniform intensity=127, corrupted with multiplicative speckle.

    Returns:
        original  (uint8): clean reference image
        noisy     (uint8): speckle-corrupted version (left half unchanged)
    """
    img_clean = np.zeros((size, size), dtype=np.float64)
    half = size // 2

    # Left half — vertical gradient
    for r in range(size):
        img_clean[r, :half] = r * 255.0 / (size - 1)

    # Right half — uniform 127
    img_clean[:, half:] = 127.0

    # Multiplicative speckle on right half only
    rng = np.random.default_rng(seed=42)
    speckle = rng.rayleigh(scale=0.25, size=(size, size - half))
    speckle = speckle / (0.25 * np.sqrt(np.pi / 2.0))  # normalise mean ≈ 1

    img_noisy = img_clean.copy()
    img_noisy[:, half:] = img_clean[:, half:] * speckle
    img_noisy = np.clip(img_noisy, 0, 255)

    return img_clean.astype(np.uint8), img_noisy.astype(np.uint8)


def _run_encoder_denoiser(img_gray_uint8, n, d=2, fast=True, use_shots=False, shots=2_000_000):
    """
    Encode and denoise a grayscale image at circuit size n using MHRQI.

    The image is resized to n×n for the circuit only when n != image size.
    For n=128 (default) with a 128×128 input, no resize occurs.

    Returns:
        rec_uint8: reconstructed+denoised image at n×n (uint8)
    """
    h, w = img_gray_uint8.shape
    if (h, w) != (n, n):
        small = cv2.resize(img_gray_uint8, (n, n), interpolation=cv2.INTER_AREA)
    else:
        small = img_gray_uint8

    normalized = np.clip(small.astype(np.float64) / 255.0, 0.0, 1.0)
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
    data_qc, _ = circuit.DENOISER(data_qc, pos_regs, intensity_reg, bias)

    if use_shots:
        counts = circuit.simulate_counts(data_qc, shots=shots, use_gpu=True)
        bins, bias_stats = circuit.make_bins_counts(counts, hierarchy_matrix, denoise=True)
    else:
        sv = circuit.simulate_statevector(data_qc)
        bins, bias_stats = circuit.make_bins_sv(sv, hierarchy_matrix, denoise=True)

    rec_float = utils.mhrqi_bins_to_image(
        bins, hierarchy_matrix, d, (n, n),
        bias_stats=bias_stats, original_img=None
    )
    return (np.clip(rec_float, 0.0, 1.0) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save_3panel_plot(original, noisy, denoised,
                       mse_noisy, psnr_noisy, mse_denoised, psnr_denoised,
                       run_dir):
    """3-panel: Original | Noisy | Denoised with MSE/PSNR annotated."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Synthetic Original\n(Reference)")
    axes[0].axis("off")

    axes[1].imshow(noisy, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(
        f"Synthetic Noisy\n(Input)\nMSE={mse_noisy:.2f} | PSNR={psnr_noisy:.2f} dB"
    )
    axes[1].axis("off")

    axes[2].imshow(denoised, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(
        f"MHRQI Denoised\n(Output)\nMSE={mse_denoised:.2f} | PSNR={psnr_denoised:.2f} dB"
    )
    axes[2].axis("off")

    fig.suptitle("Encoder + Denoiser Synthetic Test", fontsize=13)
    plt.tight_layout()
    out = os.path.join(run_dir, "speckle_3panel.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 3-panel:     {out}")


def _save_mse_heatmap(ref, test, label, run_dir):
    """Per-pixel squared error heatmap."""
    se = (ref.astype(np.float32) - test.astype(np.float32)) ** 2
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(se, cmap="RdYlGn_r", interpolation="nearest")
    ax.set_title(f"MSE Map — {label}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="squared error")
    plt.tight_layout()
    fname = f"mse_map_{label.lower().replace(' ', '_')}.png"
    out = os.path.join(run_dir, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved MSE heatmap: {out}")


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run(n, fast, use_shots, shots, display_size=128):
    run_dir = _make_run_dir(subdir="synthetic_denoiser")
    print(f"\n=== Encoder + Denoiser Synthetic Test ===")
    print(f"Output directory:   {run_dir}")
    print(f"Synthetic img size: {display_size}×{display_size}")
    print(f"Circuit size n:     {n}×{n}")
    print(f"Mode:               {'shots=' + str(shots) if use_shots else 'statevector'}")
    print(f"Fast (lazy) upload: {fast}\n")

    original, noisy = _make_synthetic_image(size=display_size)
    cv2.imwrite(os.path.join(run_dir, "synthetic_original.png"), original)
    cv2.imwrite(os.path.join(run_dir, "synthetic_noisy.png"), noisy)
    print(f"  Saved synthetic images.")

    mse_noisy  = plots.compute_mse(original, noisy)
    psnr_noisy = plots.compute_psnr(original, noisy)
    print(f"  Noisy  — MSE: {mse_noisy:.4f} | PSNR: {psnr_noisy:.2f} dB")

    t0 = time.perf_counter()
    rec = _run_encoder_denoiser(noisy, n=n, fast=fast, use_shots=use_shots, shots=shots)
    elapsed = time.perf_counter() - t0
    print(f"  Simulation time: {elapsed:.2f}s")

    # If circuit ran at a different size, resize rec back for comparison
    if rec.shape != original.shape:
        rec_cmp = cv2.resize(rec, (original.shape[1], original.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
    else:
        rec_cmp = rec

    mse_denoised  = plots.compute_mse(original, rec_cmp)
    psnr_denoised = plots.compute_psnr(original, rec_cmp)
    print(f"  Denoised — MSE: {mse_denoised:.4f} | PSNR: {psnr_denoised:.2f} dB")

    _save_3panel_plot(original, noisy, rec_cmp,
                      mse_noisy, psnr_noisy, mse_denoised, psnr_denoised, run_dir)
    _save_mse_heatmap(original, noisy,    "noisy",    run_dir)
    _save_mse_heatmap(original, rec_cmp,  "denoised", run_dir)
    cv2.imwrite(os.path.join(run_dir, "denoised_output.png"), rec_cmp)

    d     = 2
    L_max = utils.get_Lmax(n, d)
    results = {
        "display_size":      display_size,
        "circuit_n":         n,
        "num_qubits":        2 * L_max + 8 + 1 + 2,
        "L_max":             L_max,
        "mode":              "shots" if use_shots else "statevector",
        "shots":             shots if use_shots else None,
        "fast_upload":       fast,
        "simulation_time_s": round(elapsed, 4),
        "metrics": {
            "noisy_vs_original": {
                "mse":     round(float(mse_noisy), 6),
                "psnr_dB": round(float(psnr_noisy), 4),
            },
            "denoised_vs_original": {
                "mse":     round(float(mse_denoised), 6),
                "psnr_dB": round(float(psnr_denoised), 4),
            },
        }
    }
    json_path = os.path.join(run_dir, "denoiser_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved raw results: {json_path}")
    print(f"\n=== Done. All outputs in: {run_dir} ===\n")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MHRQI Encoder+Denoiser Synthetic Test (128×128 gradient + speckle)"
    )
    parser.add_argument("--n", type=int, default=128,
                        help="Circuit size (power of 2). Default: 128 (native, lazy upload)")
    parser.add_argument("--shots", type=int, default=2_000_000,
                        help="Number of shots (only used with --use-shots). Default: 2000000")
    parser.add_argument("--use-shots", action="store_true",
                        help="Use shot-based simulation (recommended with --n 32)")
    parser.add_argument("--no-fast", action="store_true",
                        help="Disable lazy (statevector) upload")
    parser.add_argument("--display-size", type=int, default=128,
                        help="Synthetic image size. Default: 128")
    args = parser.parse_args()

    n = args.n
    if n < 2 or (n & (n - 1)) != 0:
        print(f"[ERROR] n={n} is not a power of 2.")
        sys.exit(1)

    run(n=n, fast=not args.no_fast, use_shots=args.use_shots,
        shots=args.shots, display_size=args.display_size)
