"""
Diagnostic: compare lazy vs gate-based MHRQI upload at n=4 and n=16.

For each pixel, prints:
  - expected reconstructed value (same as input for lossless encoding)
  - reconstructed value from lazy upload
  - reconstructed value from gate-based upload

Also dumps per-pixel table and saves a comparison image.
"""

import os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import circuit, utils, plots

LENNA = os.path.join(os.path.dirname(__file__), "..", "resources", "non_medical", "lenna.jpg")

def make_circuit(img_norm, n, d=2):
    """Build hierarchy_matrix and initialised qc for given image."""
    L_max = utils.get_Lmax(n, d)
    sk = [n if L == 0 else utils.get_subdiv_size(L, n, d) for L in range(L_max)]
    hm = []
    for r, c in np.ndindex(n, n):
        hcv = []
        for k in sk:
            hcv.extend(utils.compute_register(r, c, d, k))
        hm.append(hcv)
    qc, pos_regs, intensity_reg, bias = circuit.MHRQI_init(d, L_max)
    return qc, pos_regs, intensity_reg, bias, hm

def reconstruct(sv_or_counts, hm, n, denoise=False, use_sv=True):
    d = 2
    if use_sv:
        bins = circuit.make_bins_sv(sv_or_counts, hm, denoise=denoise)
    else:
        bins, _ = circuit.make_bins_counts(sv_or_counts, hm, denoise=denoise)
    rec = utils.mhrqi_bins_to_image(bins, hm, d, (n, n), bias_stats=None, original_img=None)
    return (np.clip(rec, 0, 1) * 255).astype(np.uint8)

def run(n=4):
    print(f"\n{'='*60}")
    print(f"  Debug upload — n={n}")
    print(f"{'='*60}")

    img_bgr = cv2.imread(LENNA, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img_bgr, (n, n), interpolation=cv2.INTER_AREA)
    img_norm = img.astype(np.float64) / 255.0

    d = 2
    qc, pos_regs, intensity_reg, bias, hm = make_circuit(img_norm, n, d)

    # ── LAZY UPLOAD ──────────────────────────────────────────────────────────
    import copy
    qc_lazy = copy.deepcopy(qc)
    qc_lazy = circuit.MHRQI_lazy_upload(qc_lazy, pos_regs, intensity_reg, d, hm, img_norm)
    sv_lazy = circuit.simulate_statevector(qc_lazy)
    rec_lazy = reconstruct(sv_lazy, hm, n)

    # ── GATE UPLOAD ───────────────────────────────────────────────────────────
    qc_gate, pos_regs2, intensity_reg2, bias2, hm2 = make_circuit(img_norm, n, d)
    qc_gate = circuit.MHRQI_upload(qc_gate, pos_regs2, intensity_reg2, d, hm2, img_norm)

    print(f"  Gate circuit: {qc_gate.size()} gates, depth {qc_gate.depth()}, {qc_gate.num_qubits} qubits")

    sv_gate = circuit.simulate_statevector(qc_gate)
    rec_gate = reconstruct(sv_gate, hm2, n)

    # ── STATEVECTOR COMPARISON ────────────────────────────────────────────────
    sv_lazy_arr = np.array(sv_lazy)
    sv_gate_arr = np.array(sv_gate)
    sv_diff = np.abs(sv_lazy_arr - sv_gate_arr)
    print(f"\n  Statevector comparison (n={n}):")
    print(f"    Max |sv_lazy - sv_gate| : {sv_diff.max():.6f}")
    print(f"    Non-zero lazy entries   : {np.count_nonzero(np.abs(sv_lazy_arr) > 1e-10)}")
    print(f"    Non-zero gate entries   : {np.count_nonzero(np.abs(sv_gate_arr) > 1e-10)}")

    # ── PER-PIXEL TABLE (n=4 only for readability) ────────────────────────────
    if n <= 8:
        print(f"\n  Per-pixel comparison (input | lazy | gate):")
        print(f"  {'(r,c)':<8}  {'Input':>6}  {'Lazy':>6}  {'Gate':>6}  {'ΔLazy':>7}  {'ΔGate':>7}")
        for r in range(n):
            for c in range(n):
                inp   = img[r, c]
                l_val = rec_lazy[r, c]
                g_val = rec_gate[r, c]
                print(f"  ({r},{c})    {inp:>6}  {l_val:>6}  {g_val:>6}  "
                      f"{inp-l_val:>+7}  {inp-g_val:>+7}")

    # ── METRICS ───────────────────────────────────────────────────────────────
    mse_lazy = plots.compute_mse(img, rec_lazy)
    mse_gate = plots.compute_mse(img, rec_gate)
    psnr_lazy = plots.compute_psnr(img, rec_lazy)
    psnr_gate = plots.compute_psnr(img, rec_gate)
    print(f"\n  Lazy  — MSE: {mse_lazy:.4f}  PSNR: {psnr_lazy:.2f} dB")
    print(f"  Gate  — MSE: {mse_gate:.4f}  PSNR: {psnr_gate:.2f} dB")

    # ── SAVE VISUAL ───────────────────────────────────────────────────────────
    import datetime
    import matplotlib.pyplot as plt
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("runs", ts, f"debug_upload_n{n}")
    os.makedirs(out_dir, exist_ok=True)

    # Upsample for visibility if very small
    disp = max(256 // n, 1) * n
    up = lambda x: cv2.resize(x, (disp, disp), interpolation=cv2.INTER_NEAREST)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(up(img),      cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Input ({n}×{n})")
    axes[1].imshow(up(rec_lazy), cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Lazy (MSE={mse_lazy:.2f})")
    axes[2].imshow(up(rec_gate), cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(f"Gate (MSE={mse_gate:.2f})")
    for ax in axes: ax.axis("off")
    fig.suptitle(f"Upload comparison n={n} — input | lazy | gate", fontsize=12)
    plt.tight_layout()
    out = os.path.join(out_dir, f"upload_comparison_n{n}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out}")

    # Error heatmaps
    for label, rec in [("lazy", rec_lazy), ("gate", rec_gate)]:
        se = (img.astype(np.float32) - rec.astype(np.float32)) ** 2
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(up(se.astype(np.uint8) if se.max() <= 255 else (se/se.max()*255).astype(np.uint8)),
                  cmap="RdYlGn_r", interpolation="nearest")
        # use float for real values
        se_disp = cv2.resize(se, (disp, disp), interpolation=cv2.INTER_NEAREST)
        fig2, ax2 = plt.subplots(figsize=(5,5))
        im2 = ax2.imshow(se_disp, cmap="RdYlGn_r")
        ax2.set_title(f"Error heatmap — {label} (n={n})")
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, label="sq error")
        plt.tight_layout()
        fig2.savefig(os.path.join(out_dir, f"error_{label}_n{n}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig2)
        plt.close(fig)

    print(f"  All outputs in: {out_dir}")
    return mse_lazy, mse_gate

if __name__ == "__main__":
    run(n=4)
    run(n=16)
