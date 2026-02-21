"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MHRQI - Magnitude Hierarchical Representation of Quantum Images             ║
║  Plotting and Metrics: Visualization, Quality Assessment, Benchmarking       ║
║                                                                              ║
║  Author: Keno-00                                                             ║
║  License: Apache 2.0                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import datetime
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import utils

if not hasattr(np, 'int'):
    np.int = int
import scipy.misc
from PIL import Image


def _imresize_patch(arr, size, interp='bilinear', mode=None):
    if mode == 'F':
        arr = arr.astype('float32')
        img = Image.fromarray(arr, mode='F')
    else:
        img = Image.fromarray(arr)

    if isinstance(size, float):
        new_size = (int(img.width * size), int(img.height * size))
    elif isinstance(size, int):
        new_size = (int(img.width * size / 100), int(img.height * size / 100))
    elif isinstance(size, tuple):
        new_size = (size[1], size[0]) # PIL (W, H)
    else:
        new_size = (img.width, img.height)

    resample = Image.BICUBIC
    if interp == 'nearest': resample = Image.NEAREST
    elif interp == 'bilinear': resample = Image.BILINEAR
    elif interp == 'lanczos': resample = Image.LANCZOS

    img = img.resize(new_size, resample=resample)
    return np.array(img)

if not hasattr(scipy.misc, 'imresize'):
    scipy.misc.imresize = _imresize_patch
# -------------------------------------------------------------------------------

HEADLESS = matplotlib.get_backend().lower().endswith("agg")

# -------------------------------------------------------------------------------
# Run directory management
# -------------------------------------------------------------------------------
_current_run_dir = None

def get_run_dir(run_dir=None):
    """
    Get or create the current run output directory.
    If run_dir is provided, use it. Otherwise, create a new timestamped directory.
    """
    global _current_run_dir
    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        _current_run_dir = run_dir
        return run_dir
    if _current_run_dir is not None:
        return _current_run_dir
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    _current_run_dir = os.path.join("runs", date_str)
    os.makedirs(_current_run_dir, exist_ok=True)
    return _current_run_dir

def reset_run_dir():
    """Reset the cached run directory (for new runs)."""
    global _current_run_dir
    _current_run_dir = None

def save_settings_plot(settings_dict, run_dir=None, filename="settings.png"):
    """
    Create a visual summary of the run settings and save it.
    
    Args:
        settings_dict: dict of setting names to values
        run_dir: output directory (uses get_run_dir() if None)
        filename: output filename
    """
    run_dir = get_run_dir(run_dir)

    fig, ax = plt.subplots(figsize=(6, max(2, len(settings_dict) * 0.4)))
    ax.axis('off')

    # Create table data
    table_data = [[k, str(v)] for k, v in settings_dict.items()]

    table = ax.table(
        cellText=table_data,
        colLabels=['Setting', 'Value'],
        loc='center',
        cellLoc='left',
        colWidths=[0.4, 0.6]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', weight='bold')

    plt.title("Run Settings", fontsize=12, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)



def _value_from_bin(v, kind="p", eps=0.0):
    if kind == "hit":  return float(v["hit"])
    if kind == "miss": return float(v["miss"])
    t = float(v["trials"])
    return (v["hit"] + eps) / (t + 2*eps) if t > 0 else np.nan  # phat

# --- grid helpers ---
def bins_to_grid(bins, d,N, kind="p", eps=0.0):
    size = N
    grid = np.full((size, size), np.nan)
    for hcv, v in bins.items():
        y, x = utils.compose_rc(hcv,d)
        grid[y,x] = _value_from_bin(v, kind, eps)
    return grid



def grid_to_image_uint8(grid, vmin=None, vmax=None, flip_vertical=False):
    """
    Convert an N x N grid into a uint8 image.
    NaN values are replaced with 0 before scaling.
    Optionally flips vertically to match image row convention (y down).
    """
    # Ensure float array
    work = np.array(grid, dtype=float)
    work[np.isnan(work)] = 0.0

    # Compute value range if not given
    finite = np.isfinite(grid)
    if not finite.any():
        vmin, vmax = 0.0, 1.0
    else:
        vmin = np.nanmin(grid) if vmin is None else vmin
        vmax = np.nanmax(grid) if vmax is None else vmax
        if vmax == vmin:
            vmax = vmin + 1.0

    # Scale to [0, 255]
    img = (np.clip(work, vmin, vmax) - vmin) / (vmax - vmin)
    img = (img * 255.0).round().astype(np.uint8)

    # Flip vertically if needed
    if flip_vertical:
        img = np.flipud(img)

    return img

def bins_to_image(bins, d, N, kind="p", eps=0.0, vmin=0.0, vmax=1.0):
    """
    Convert bins directly to uint8 image without intermediate grid.
    
    Args:
        bins: measurement bins dict
        d: qudit dimension
        N: image size
        kind: value type ("p" for p-hat probability, "hit", "miss")
        eps: smoothing epsilon for p-hat
        vmin, vmax: value range for scaling to [0, 255]
    
    Returns:
        uint8 image array (N x N)
    """
    img = np.zeros((N, N), dtype=np.uint8)

    for hcv, v in bins.items():
        y, x = utils.compose_rc(hcv, d)
        val = _value_from_bin(v, kind, eps)

        # Handle NaN as 0, then scale to uint8
        if np.isnan(val):
            val = 0.0
        scaled = (np.clip(val, vmin, vmax) - vmin) / (vmax - vmin)
        img[y, x] = int(round(scaled * 255.0))

    return img



def show_image_comparison(orig_img, recon_img, titles=("Original", "Reconstructed"), run_dir=None, img_name=None):
    """
    Plot two images side by side for visual comparison.
    Accepts 2D arrays (uint8 preferred). No resizing is done.
    
    Args:
        orig_img: original image
        recon_img: reconstructed image
        titles: tuple of titles for the two images
        run_dir: output directory (uses get_run_dir() if None)
        img_name: base name for saved images (uses 'reconstructed' if None)
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(orig_img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(titles[0])
    axes[0].set_xticks([]); axes[0].set_yticks([]) # Keep box, hide ticks

    axes[1].imshow(recon_img, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(titles[1])
    axes[1].set_xticks([]); axes[1].set_yticks([]) # Keep box, hide ticks

    plt.tight_layout()

    dir_path = get_run_dir(run_dir)
    img_base = img_name or "reconstructed"
    plt.savefig(os.path.join(dir_path, f"{img_base}_comparison.png"), dpi=150, bbox_inches="tight")
    recon_img_uint8 = recon_img.astype(np.uint8) if recon_img.dtype != np.uint8 else recon_img
    cv2.imwrite(os.path.join(dir_path, f"{img_base}.png"), recon_img_uint8)

# ---------------------------------------------
# utils
# ---------------------------------------------
def _to_float_array(img):
    """Return float32 array, shape (H,W[,C])."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        return arr.astype(np.float32)
    raise ValueError("img must be 2D or 3D array")

def _check_same_shape(a, b):
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")


# ---------------------------------------------
# metric helpers (+ per-pixel maps)
# ---------------------------------------------
import pypiqe
import skvideo.measure
from brisque import BRISQUE
from skimage.metrics import mean_squared_error, structural_similarity

# Lazy-loaded modules for FSIM (heavy dependencies)
_torch = None
_piq = None

def _ensure_torch_piq():
    """Lazy load torch and piq only when needed."""
    global _torch, _piq
    if _torch is None:
        import torch
        _torch = torch
    if _piq is None:
        import piq
        _piq = piq
    return _torch, _piq

def compute_fsim(img_ref, img_test):
    """
    Compute FSIM score using piq.
    Higher is better [0, 1].
    """
    torch, piq = _ensure_torch_piq()

    ref = _to_float_array(img_ref)
    test = _to_float_array(img_test)

    if ref.max() > 1.0: ref /= 255.0
    if test.max() > 1.0: test /= 255.0

    ref = np.clip(ref, 0.0, 1.0)
    test = np.clip(test, 0.0, 1.0)

    def to_tensor(arr):
        t = torch.from_numpy(arr).float()
        if t.ndim == 2:
            return t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 3:
            return t.permute(2, 0, 1).unsqueeze(0)
        return t

    ref_t = to_tensor(ref)
    test_t = to_tensor(test)

    try:
        score = piq.fsim(ref_t, test_t, data_range=1.0, reduction='none', chromatic=False)
        return float(score.item())
    except Exception:
        return float('nan')


def compute_niqe(img_input):
    """
    Compute NIQE score using scikit-video.
    Lower is better.
    """
    img = _to_float_array(img_input)
    if img.max() <= 1.0:
        img_u8 = (img * 255.0).astype(np.uint8)
    else:
        img_u8 = img.astype(np.uint8)

    if img_u8.ndim == 2:
        img_u8 = img_u8[np.newaxis, ...]
    elif img_u8.ndim == 3:
        if img_u8.shape[2] == 3:
            img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)[np.newaxis, ...]
        else:
            img_u8 = img_u8[np.newaxis, ...]

    try:
        return float(skvideo.measure.niqe(img_u8)[0])
    except Exception:
        return float('nan')

def compute_piqe(img_input):
    """
    Compute PIQE score using pypiqe.
    Lower is better.
    """
    img = _to_float_array(img_input)
    if img.max() <= 1.0:
        img_u8 = (img * 255.0).astype(np.uint8)
    else:
        img_u8 = img.astype(np.uint8)

    try:
        score, _, _, _ = pypiqe.piqe(img_u8)
        return float(score)
    except Exception:
        return float('nan')

def compute_brisque(img_input):
    """
    Compute BRISQUE score using 'brisque' library.
    Lower score is better.
    """
    img = _to_float_array(img_input)
    if img.max() <= 1.0:
        img_u8 = (img * 255.0).astype(np.uint8)
    else:
        img_u8 = img.astype(np.uint8)

    from PIL import Image, ImageOps

    if img.ndim == 2:
        img_rgb = np.stack((img_u8,)*3, axis=-1)
    else:
        img_rgb = img_u8

    pil_img = Image.fromarray(img_rgb)

    obj = BRISQUE(url=False)
    try:
        return obj.score(pil_img)
    except Exception:
        return float('nan')

def compute_mse(img_gt, img_test):
    """
    Return average MSE between two images using skimage.
    """
    gt = _to_float_array(img_gt)
    te = _to_float_array(img_test)
    _check_same_shape(gt, te)
    return mean_squared_error(gt, te)

def compute_psnr(img_gt, img_test, data_range=255.0):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in dB.
    Higher is better.
    """
    mse_val = compute_mse(img_gt, img_test)
    if mse_val == 0:
        return float('inf')
    return 10.0 * np.log10((data_range ** 2) / mse_val)

def plot_mse_map(img_gt, img_test, title="Per-pixel squared error", run_dir=None):
    """
    Show per-pixel squared error heatmap.
    Red = higher error, Green = lower error.
    """
    gt = _to_float_array(img_gt)
    te = _to_float_array(img_test)
    _check_same_shape(gt, te)

    if gt.ndim == 3:
        se = np.mean((gt - te)**2, axis=2)  # (H,W) average over channels
    else:
        se = (gt - te)**2

    plt.figure()
    im = plt.imshow(se, cmap="RdYlGn_r")
    plt.title(title)
    plt.axis("off")
    cbar = plt.colorbar(im)
    cbar.set_label("squared error")

    dir_path = get_run_dir(run_dir)
    plt.savefig(os.path.join(dir_path, "mse_map.png"), dpi=150, bbox_inches="tight")


def compute_ssim(img_gt, img_test, data_range=255.0):
    """
    Return Structural Similarity Index (SSIM) using skimage.
    """
    gt = _to_float_array(img_gt)
    te = _to_float_array(img_test)
    _check_same_shape(gt, te)
    # SSIM requires data_range to be specified if images are float but not -1..1 or 0..1
    # Our _to_float_array likely returns 0-255 range float if input was uint8.
    # But wait, main.py/compare_to.py passed 0-1 floats to CNR, let's verify usage.
    # Ideally we handle both.

    return structural_similarity(gt, te, data_range=data_range)

def compute_ssi(img_noisy, img_filtered, roi):
    """
    Compute Speckle Suppression Index (SSI).
    Formula: (std_filt / mean_filt) / (std_noisy / mean_noisy)
    Calculated on a homogeneous region.
    Lower is better.
    """
    img_n = _to_float_array(img_noisy)
    img_f = _to_float_array(img_filtered)

    if isinstance(roi, tuple) and len(roi) == 4:
        y, x, h, w = roi
        reg_n = img_n[y:y+h, x:x+w]
        reg_f = img_f[y:y+h, x:x+w]
    else:
        reg_n = img_n[roi]
        reg_f = img_f[roi]

    m_n = np.mean(reg_n)
    s_n = np.std(reg_n)

    m_f = np.mean(reg_f)
    s_f = np.std(reg_f)

    eps = 1e-10
    if m_n < eps or m_f < eps:
        return float('inf')

    cov_n = s_n / m_n
    cov_f = s_f / m_f

    if cov_n < eps:
        return float('inf')

    return float(cov_f / cov_n)

def compute_dr_iqa(img_dr, img_fd):
    """
    Compute Degraded Reference Image Quality Assessment (DR-IQA) score.
    Combines FSIM (similarity to degraded ref) and NIQE (naturalness of degraded ref).
    
    Score = FSIM(img_fd, img_dr) * (1 / (1 + NIQE(img_dr)))
    
    Inputs:
        img_dr: Degraded Reference image (numpy array)
        img_fd: Final Distorted/Restored image (numpy array)
    """
    # 1. Compute FSIM (Similarity between Restored and Noisy Ref)
    # Using existing compute_fsim which handles conversion to tensors
    fsim_val = compute_fsim(img_dr, img_fd)

    # 2. Compute NIQE of the Degraded Reference
    # Note: The original snippet computed NIQE on the DR (degraded reference).
    # This acts as a weighting factor based on how bad the reference is?
    # Or maybe it meant NIQE of the restored image?
    # User snippet: "NIQE_val = niqe_metric(img_dr)"
    # User snippet comment: # compute NIQE on DR
    niqe_dr = compute_niqe(img_dr)

    # 3. Combine
    if np.isnan(fsim_val) or np.isnan(niqe_dr):
        return float('nan')

    nr_q = 1.0 / (1.0 + niqe_dr)
    quality_score = fsim_val * nr_q

    return quality_score

def compute_smpi(img_original, img_filtered):
    """
    Calculates SMPI for SAR image evaluation.
    Lower values indicate better speckle suppression and mean preservation.
    """
    original = _to_float_array(img_original)
    filtered = _to_float_array(img_filtered)

    mean_o = np.mean(original)
    mean_f = np.mean(filtered)
    var_o = np.var(original)
    var_f = np.var(filtered)

    q = 1 + np.abs(mean_o - mean_f)

    if var_o == 0:
        return float('inf')

    smpi = q * (np.sqrt(var_f) / np.sqrt(var_o))
    return float(smpi)

def compute_omqdi(img_noisy, img_denoised):
    """
    Compute OMQDI (Objective Measure of Quality of Denoised Images).
    DOI: 10.1016/j.bspc.2021.102962
    
    Args:
        img_noisy: Noisy input image (single channel)
        img_denoised: Denoised output image (single channel)
    
    Returns:
        tuple: (OMQDI, EPF, NSF)
            - OMQDI: Combined metric Q1+Q2, ideal value is 2, range [1,2]
            - EPF: Edge-Preservation Factor (Q1), ideal value is 1, range [0,1]
            - NSF: Noise-Suppression Factor (Q2), ideal value is 1, range [0,1]
    """
    from IQA import OMQDI
    
    noisy = _to_float_array(img_noisy)
    denoised = _to_float_array(img_denoised)
    
    # Normalize to 0-1 if needed
    if noisy.max() > 1.0:
        noisy = noisy / 255.0
    if denoised.max() > 1.0:
        denoised = denoised / 255.0
    
    try:
        omqdi_val, epf, nsf = OMQDI(noisy, denoised)
        return (float(omqdi_val), float(epf), float(nsf))
    except Exception:
        return (float('nan'), float('nan'), float('nan'))

def compute_enl(img, roi=None):
    """
    Compute Equivalent Number of Looks (ENL).
    ENL = mean² / variance
    Higher values indicate better speckle suppression in homogeneous regions.
    
    Args:
        img: Input image
        roi: Optional tuple (y, x, h, w) for region of interest
    
    Returns:
        ENL value (higher is better)
    
    Citation: Ulaby et al., 1986
    """
    arr = _to_float_array(img)
    
    if roi is not None:
        y, x, h, w = roi
        region = arr[y:y+h, x:x+w]
    else:
        region = arr
    
    mean_val = np.mean(region)
    var_val = np.var(region)
    
    eps = 1e-10
    if var_val < eps:
        return 10000.0  # Cap at reasonable max
    
    enl = (mean_val ** 2) / var_val
    return float(min(enl, 10000.0))

def compute_epi(img_original, img_denoised):
    """
    Compute Edge Preservation Index (EPI).
    EPI = correlation of gradient magnitudes between original and denoised.
    Higher values indicate better edge preservation.
    
    Citation: Sattar et al., 1997
    """
    import cv2
    
    orig = _to_float_array(img_original)
    denoised = _to_float_array(img_denoised)
    
    # Ensure uint8 for Sobel
    orig_u8 = (orig * 255).astype(np.uint8) if orig.max() <= 1.0 else orig.astype(np.uint8)
    den_u8 = (denoised * 255).astype(np.uint8) if denoised.max() <= 1.0 else denoised.astype(np.uint8)
    
    # Compute Sobel gradients
    gx_o = cv2.Sobel(orig_u8, cv2.CV_64F, 1, 0, ksize=3)
    gy_o = cv2.Sobel(orig_u8, cv2.CV_64F, 0, 1, ksize=3)
    grad_orig = np.sqrt(gx_o**2 + gy_o**2)
    
    gx_d = cv2.Sobel(den_u8, cv2.CV_64F, 1, 0, ksize=3)
    gy_d = cv2.Sobel(den_u8, cv2.CV_64F, 0, 1, ksize=3)
    grad_den = np.sqrt(gx_d**2 + gy_d**2)
    
    # Correlation coefficient
    corr = np.corrcoef(grad_orig.flatten(), grad_den.flatten())[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0

def auto_detect_rois(img):
    """
    Auto-detect signal and background ROIs for CNR calculation.
    
    Signal ROI: High intensity region (top 10% of image)
    Background ROI: Low variance homogeneous region
    
    Returns:
        signal_roi: (y, x, h, w) tuple
        bg_roi: (y, x, h, w) tuple
    """
    arr = _to_float_array(img)
    h, w = arr.shape
    
    # Use 16x16 blocks
    block_size = min(16, h // 4, w // 4)
    if block_size < 4:
        block_size = 4
    
    # Find signal region (high intensity)
    threshold = np.percentile(arr, 90)
    signal_mask = arr > threshold
    
    # Find centroid of high-intensity region
    y_coords, x_coords = np.where(signal_mask)
    if len(y_coords) > 0:
        cy, cx = int(np.mean(y_coords)), int(np.mean(x_coords))
        # Clamp to valid range
        sy = max(0, min(cy - block_size // 2, h - block_size))
        sx = max(0, min(cx - block_size // 2, w - block_size))
        signal_roi = (sy, sx, block_size, block_size)
    else:
        signal_roi = (0, 0, block_size, block_size)
    
    # Find background region (lowest variance block)
    best_var = float('inf')
    bg_roi = (0, 0, block_size, block_size)
    
    for y in range(0, h - block_size, block_size // 2):
        for x in range(0, w - block_size, block_size // 2):
            block = arr[y:y+block_size, x:x+block_size]
            mu = np.mean(block)
            if mu > 0.95: # Skip saturated regions
                continue
                
            var = np.var(block)
            
            # Center bias
            dist_to_center = np.sqrt((y + block_size/2 - h/2)**2 + (x + block_size/2 - w/2)**2)
            dist_norm = dist_to_center / np.sqrt((h/2)**2 + (w/2)**2)
            
            # Cost = var + penalty for being away from center
            cost = var + 0.05 * dist_norm
            
            if cost < best_var and var > 0:
                best_var = cost
                bg_roi = (y, x, block_size, block_size)
    
    return signal_roi, bg_roi

def compute_cnr(img, signal_roi=None, bg_roi=None):
    """
    Compute Contrast-to-Noise Ratio (CNR).
    CNR = |mean_signal - mean_bg| / std_bg
    Higher values indicate better contrast.
    
    If ROIs not provided, auto-detects them.
    
    Returns:
        cnr_value: float
        signal_roi: (y, x, h, w)
        bg_roi: (y, x, h, w)
    """
    arr = _to_float_array(img)
    
    if signal_roi is None or bg_roi is None:
        signal_roi, bg_roi = auto_detect_rois(arr)
    
    sy, sx, sh, sw = signal_roi
    by, bx, bh, bw = bg_roi
    
    signal_region = arr[sy:sy+sh, sx:sx+sw]
    bg_region = arr[by:by+bh, bx:bx+bw]
    
    mean_signal = np.mean(signal_region)
    mean_bg = np.mean(bg_region)
    std_bg = np.std(bg_region)
    
    eps = 1e-10
    if std_bg < eps:
        return (10000.0, signal_roi, bg_roi)  # Cap at reasonable max
    
    cnr = abs(mean_signal - mean_bg) / std_bg
    return (float(min(cnr, 10000.0)), signal_roi, bg_roi)


# =============================================================================
# trend helpers (line graphs)
# =============================================================================

def plot_shots_vs_mse(shots, mse_values, title="Shots vs MSE", run_dir=None):
    """
    Line graph only for trend inspection.
    shots: list[int] or list[float]
    mse_values: list[float]
    """
    if len(shots) != len(mse_values):
        raise ValueError("shots and mse_values length mismatch")
    plt.figure()
    plt.plot(shots, mse_values)
    plt.xlabel("Shots")
    plt.ylabel("MSE")
    plt.title(title)
    plt.grid(True)

    dir_path = get_run_dir(run_dir)
    plt.savefig(os.path.join(dir_path, "shots_vs_mse.png"), dpi=150, bbox_inches="tight")



# =============================================================================
# ORGANIZED PLOTTING CLASSES
# =============================================================================

class MetricsPlotter:
    """Visualization for image quality metrics and comparison reports."""

    @staticmethod
    def print_summary_text(competitors, keys, title):
        """Prints metric table to console."""
        print("-" * 100)
        print(f" {title}")
        print("-" * 100)
        header = f"{'Method':<12}" + "".join([f"{k:<15}" for k in keys])
        print(header)
        print("-" * 100)
        for m in competitors:
            row = f"{m['name']:<12}"
            for k in keys:
                val = m['metrics'].get(k, float('nan'))
                row += f"{val:<15.4f}"
            print(row)
        print("-" * 100)
        print()

    @staticmethod
    def save_summary_report(ref_img, competitors, metric_keys, title, filename_suffix,
                            save_dir, include_original_in_table=False):
        """
        Generates a unified figure with images and metrics table.
        If ref_img is None, only plots competitors.
        
        Args:
            ref_img: Reference image (uint8) or None
            competitors: List of dicts with 'name', 'metrics', 'image' keys
            metric_keys: List of metric names to display
            title: Figure title
            filename_suffix: Output filename (without extension)
            save_dir: Directory to save the figure
            include_original_in_table: Whether to include 'Original' in the table
        """
        # Filter competitors for the table
        if include_original_in_table:
            table_methods = competitors
        else:
            table_methods = [c for c in competitors if c["name"] != "Original"]

        if not table_methods:
            return

        # Prepare Metrics Data & Ranking
        data_map = {m["name"]: m["metrics"] for m in table_methods}
        names = [m["name"] for m in table_methods]

        # Compute Ranks
        higher_better = {"OMQDI", "EPF", "ENL", "EPI", "CNR", "NSF"}  # Higher is better
        ranks = {k: {} for k in metric_keys}

        for k in metric_keys:
            is_higher = k in higher_better
            valid_items = [(name, data_map[name].get(k, float('nan'))) for name in names]
            valid_items = [x for x in valid_items if not np.isnan(x[1])]
            valid_items.sort(key=lambda x: x[1], reverse=is_higher)

            # Handle ties: items with same value get same rank
            current_rank = 1
            prev_val = None
            for i, (name, val) in enumerate(valid_items):
                if prev_val is not None and val == prev_val:
                    ranks[k][name] = current_rank  # Same rank for ties
                else:
                    current_rank = i + 1
                    ranks[k][name] = current_rank
                prev_val = val

        # Setup Figure
        has_ref_plot = (ref_img is not None)
        n_imgs = len(table_methods) + (1 if has_ref_plot else 0)

        fig_width = max(10, n_imgs * 2.5)
        fig_height = 6

        fig = plt.figure(figsize=(fig_width, fig_height))

        # GridSpec: Top row (Images), Bottom row (Table)
        gs = fig.add_gridspec(2, n_imgs, height_ratios=[1, 1], hspace=0.1)

        col_idx = 0

        # Plot Reference (Leftmost) if exists
        if has_ref_plot:
            ax_ref = fig.add_subplot(gs[0, col_idx])
            ax_ref.imshow(ref_img, cmap="gray", vmin=0, vmax=255)
            ax_ref.set_title("Reference", fontsize=10, fontweight='bold')
            ax_ref.set_xticks([]); ax_ref.set_yticks([])
            col_idx += 1

        # Plot Competitors
        for m in table_methods:
            ax = fig.add_subplot(gs[0, col_idx])
            ax.imshow(m["image"], cmap="gray", vmin=0, vmax=255)
            ax.set_title(m["name"], fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            col_idx += 1

        # Plot Table (Spanning all columns)
        ax_table = fig.add_subplot(gs[1, :])
        ax_table.axis('off')

        # Table Data
        cell_text = []
        for name in names:
            row = []
            for k in metric_keys:
                val = data_map[name].get(k, float('nan'))
                if np.isnan(val):
                    row.append("N/A")
                else:
                    r = ranks[k].get(name, "")
                    rank_str = f" (#{r})" if r else ""
                    row.append(f"{val:.4f}{rank_str}")
            cell_text.append(row)

        table = ax_table.table(
            cellText=cell_text,
            rowLabels=names,
            colLabels=metric_keys,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.5)

        # Style Header
        for j in range(len(metric_keys)):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(color='white', weight='bold')

        # Highlight #1 Ranks
        for i, name in enumerate(names):
            for j, k in enumerate(metric_keys):
                r = ranks[k].get(name, None)
                if r == 1:
                    table[(i+1, j)].set_facecolor('#C6EFCE')  # Green for Winner
                    table[(i+1, j)].set_text_props(weight='bold')

        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.95)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"{filename_suffix}.png")
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


class ImagePlotter:
    """Utilities for displaying and comparing images."""

    show_image_comparison = staticmethod(show_image_comparison)
    plot_mse_map = staticmethod(plot_mse_map)
    grid_to_image_uint8 = staticmethod(grid_to_image_uint8)
    bins_to_image = staticmethod(bins_to_image)


class TrendPlotter:
    """Line graphs for trend analysis."""

    plot_shots_vs_mse = staticmethod(plot_shots_vs_mse)


# =============================================================================
# HOMOGENEITY VISUALIZATION
# =============================================================================

def plot_bias_map(bias_stats, original_img, N, d, run_dir=None):
    """
    Visualize the confidence bias derived from the bias ancilla.
    
    Args:
        bias_stats: dict of {(qy1, qx1, ...): stats}
        original_img: original grayscale image
        N: image size
        d: qudit dimension
        run_dir: output directory
    """
    if bias_stats is None:
        print("No bias stats to plot.")
        return None

    # Create bias ratio map
    bias_map = np.zeros((N, N))
    for vec, stats in bias_stats.items():
        r, c = utils.compose_rc(vec, d)
        hit = stats.get('hit', 0)
        miss = stats.get('miss', 0)
        total = hit + miss
        ratio = hit / total if total > 0 else 0.5
        bias_map[r, c] = ratio

    # Save and display
    dir_path = get_run_dir(run_dir)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original
    if original_img is not None:
        axes[0].imshow(original_img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Bias Ratio Map
    im = axes[1].imshow(bias_map, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('Bias Confidence (Hit Ratio)')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, 'bias_map.png'), dpi=150, bbox_inches='tight')

    if not HEADLESS:
        plt.show()
    plt.close()

    return bias_map

