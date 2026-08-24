"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MHRQI - Multiscale-Hierarchical Representation of Quantum Images           ║
║  Plotting and Metrics: Visualization, Quality Assessment, Benchmarking       ║
║                                                                              ║
║  Author: Keno-00                                                             ║
║  License: Apache 2.0                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import datetime
import math
import os
import shutil

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
        new_size = (size[1], size[0])
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

HEADLESS = matplotlib.get_backend().lower().endswith("agg")

# ECTI-CIT figures are prepared for a 10-point, two-column manuscript.  The
# supplied ECTI class uses Computer Modern typography, italic captions, and a
# restrained black-rule layout; the plotting theme mirrors that print setting.
ECTI_SINGLE_COLUMN_IN = 3.33
ECTI_DOUBLE_COLUMN_IN = 6.85
ECTI_METHOD_STYLE = {
    "bm3d": ("#6B6B6B", "BM3D"),
    "nlmeans": ("#9A9A9A", "NLM"),
    "srad": ("#B8B8B8", "SRAD"),
    "siamesegan": ("#4A4A4A", "SiameseGAN"),
    "proposed": ("#111111", "MHRQI"),
}


def apply_ecti_plot_style():
    """Apply the journal-aligned visual defaults to all generated figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 10,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def ecti_axis(axis, grid=True):
    """Use the light rule and spacing treatment appropriate for print figures."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", pad=2)
    if grid:
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, zorder=0)


def ecti_panel_label(axis, panel_index):
    """Place a compact journal-style panel label in the upper-left corner."""
    label = chr(ord("a") + panel_index)
    axis.text(
        0.02,
        0.97,
        f"({label})",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color="black",
        zorder=5,
    )


def ecti_method_style(method):
    """Return the stable grayscale style and display name for a benchmark method."""
    return ECTI_METHOD_STYLE.get(method, ("#7A7A7A", method))


apply_ecti_plot_style()

# -------------------------------------------------------------------------------
# Run directory management
# -------------------------------------------------------------------------------
_current_run_dir = None

def copy_config_to_run_dir(run_dir=None, configs_dir=None):
    """
    Copy configuration file(s) into the specified run directory.

    Args:
        run_dir: Path to target run directory. If None, uses get_run_dir().
        configs_dir: Path to directory containing config files. Defaults to source/configs.
    """
    if run_dir is None:
        if _current_run_dir is None:
            return
        run_dir = _current_run_dir
    if configs_dir is None:
        configs_dir = os.path.join(os.path.dirname(__file__), "configs")

    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)

    if os.path.exists(configs_dir):
        for item in os.listdir(configs_dir):
            if item.endswith((".ini", ".json", ".cfg", ".yaml", ".yml")):
                src_path = os.path.join(configs_dir, item)
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, os.path.join(run_dir, item))

def get_run_dir(run_dir=None):
    """
    Get or create the current run output directory.

    Args:
        run_dir: If provided, use this path directly.

    Returns:
        Path to the run output directory.
    """
    global _current_run_dir
    if run_dir is not None:
        is_new = not os.path.exists(run_dir)
        os.makedirs(run_dir, exist_ok=True)
        _current_run_dir = run_dir
        if is_new:
            copy_config_to_run_dir(_current_run_dir)
        return run_dir
    if _current_run_dir is not None:
        return _current_run_dir
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    _current_run_dir = os.path.join("runs", date_str)
    is_new = not os.path.exists(_current_run_dir)
    os.makedirs(_current_run_dir, exist_ok=True)
    if is_new:
        copy_config_to_run_dir(_current_run_dir)
    return _current_run_dir

def reset_run_dir():
    """Reset the cached run directory."""
    global _current_run_dir
    _current_run_dir = None

def save_settings_plot(settings_dict, run_dir=None, filename="settings.png"):
    """
    Create a visual table of run settings and save it as a PNG.

    Args:
        settings_dict: Dict of setting names to values.
        run_dir: Output directory. Uses get_run_dir() if None.
        filename: Output filename.
    """
    run_dir = get_run_dir(run_dir)

    fig, ax = plt.subplots(figsize=(ECTI_DOUBLE_COLUMN_IN, max(1.6, len(settings_dict) * 0.24)))
    ax.axis('off')

    table_data = [[k, str(v)] for k, v in settings_dict.items()]

    table = ax.table(
        cellText=table_data,
        colLabels=['Setting', 'Value'],
        loc='center',
        cellLoc='left',
        colWidths=[0.4, 0.6]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.05)

    for i in range(2):
        table[(0, i)].set_facecolor('#F2F2F2')
        table[(0, i)].set_edgecolor('#333333')
        table[(0, i)].set_text_props(color='black', weight='bold')
    for cell in table.get_celld().values():
        cell.set_linewidth(0.4)
        cell.set_edgecolor('#808080')

    ax.set_title("Run settings", pad=6)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(run_dir, filename), dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def _value_from_bin(v, kind="p", eps=0.0):
    if kind == "hit":  return float(v["hit"])
    if kind == "miss": return float(v["miss"])
    t = float(v["trials"])
    return (v["hit"] + eps) / (t + 2*eps) if t > 0 else np.nan


def bins_to_grid(bins, d, N, kind="p", eps=0.0):
    size = N
    grid = np.full((size, size), np.nan)
    for hcv, v in bins.items():
        y, x = utils.compose_rc(hcv, d)
        grid[y, x] = _value_from_bin(v, kind, eps)
    return grid


def grid_to_image_uint8(grid, vmin=None, vmax=None, flip_vertical=False):
    """
    Convert an N x N grid into a uint8 image.

    NaN values are replaced with 0 before scaling.

    Args:
        grid: 2D numpy array.
        vmin: Minimum value for scaling. Inferred from data if None.
        vmax: Maximum value for scaling. Inferred from data if None.
        flip_vertical: If True, flip the image vertically.

    Returns:
        uint8 image array of shape (N, N).
    """
    work = np.array(grid, dtype=float)
    work[np.isnan(work)] = 0.0

    finite = np.isfinite(grid)
    if not finite.any():
        vmin, vmax = 0.0, 1.0
    else:
        vmin = np.nanmin(grid) if vmin is None else vmin
        vmax = np.nanmax(grid) if vmax is None else vmax
        if vmax == vmin:
            vmax = vmin + 1.0

    img = (np.clip(work, vmin, vmax) - vmin) / (vmax - vmin)
    img = (img * 255.0).round().astype(np.uint8)

    if flip_vertical:
        img = np.flipud(img)

    return img

def bins_to_image(bins, d, N, kind="p", eps=0.0, vmin=0.0, vmax=1.0):
    """
    Convert bins directly to a uint8 image.

    Args:
        bins: Measurement bins dict.
        d: Qudit dimension.
        N: Image size.
        kind: Value type ("p" for p-hat probability, "hit", "miss").
        eps: Smoothing epsilon for p-hat.
        vmin: Minimum value for scaling to [0, 255].
        vmax: Maximum value for scaling to [0, 255].

    Returns:
        uint8 image array of shape (N, N).
    """
    img = np.zeros((N, N), dtype=np.uint8)

    for hcv, v in bins.items():
        y, x = utils.compose_rc(hcv, d)
        val = _value_from_bin(v, kind, eps)
        if np.isnan(val):
            val = 0.0
        scaled = (np.clip(val, vmin, vmax) - vmin) / (vmax - vmin)
        img[y, x] = int(round(scaled * 255.0))

    return img


def show_image_comparison(orig_img, recon_img, titles=("Original", "Reconstructed"), run_dir=None, img_name=None):
    """
    Plot two images side by side and save the comparison.

    Args:
        orig_img: Original image (2D uint8 preferred).
        recon_img: Reconstructed image (2D uint8 preferred).
        titles: Tuple of display titles for the two images.
        run_dir: Output directory. Uses get_run_dir() if None.
        img_name: Base name for saved files.
    """
    fig, axes = plt.subplots(1, 2, figsize=(ECTI_SINGLE_COLUMN_IN, 1.75))
    axes[0].imshow(orig_img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(titles[0])
    axes[0].set_xticks([]); axes[0].set_yticks([])
    ecti_panel_label(axes[0], 0)

    axes[1].imshow(recon_img, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(titles[1])
    axes[1].set_xticks([]); axes[1].set_yticks([])
    ecti_panel_label(axes[1], 1)

    fig.tight_layout(pad=0.2, w_pad=0.25)

    dir_path = get_run_dir(run_dir)
    img_base = img_name or "reconstructed"
    fig.savefig(
        os.path.join(dir_path, f"{img_base}_comparison.png"), dpi=300,
        bbox_inches="tight", pad_inches=0.02,
    )
    recon_img_uint8 = recon_img.astype(np.uint8) if recon_img.dtype != np.uint8 else recon_img
    cv2.imwrite(os.path.join(dir_path, f"{img_base}.png"), recon_img_uint8)
    plt.close(fig)


# -------------------------------------------------------------------------------
# Internal utilities
# -------------------------------------------------------------------------------

def _to_float_array(img):
    """Return float32 array of shape (H, W) or (H, W, C)."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        return arr.astype(np.float32)
    raise ValueError("img must be 2D or 3D array")

def _check_same_shape(a, b):
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")


# -------------------------------------------------------------------------------
# Metric functions
# -------------------------------------------------------------------------------
import pypiqe
import skvideo.measure
from brisque import BRISQUE
from skimage.metrics import mean_squared_error, structural_similarity

_torch = None
_piq = None

def _ensure_torch_piq():
    """Lazy-load torch and piq."""
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
    Compute FSIM (Feature Similarity Index) using piq.

    Args:
        img_ref: Reference image.
        img_test: Test image.

    Returns:
        FSIM score in [0, 1]. Higher is better.
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
    Compute NIQE (Natural Image Quality Evaluator) using scikit-video.

    Args:
        img_input: Input image.

    Returns:
        NIQE score. Lower is better.
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


def compute_mse(img_gt, img_test):
    """
    Compute mean squared error between two images.

    Args:
        img_gt: Ground truth image.
        img_test: Test image.

    Returns:
        MSE value.
    """
    gt = _to_float_array(img_gt)
    te = _to_float_array(img_test)
    _check_same_shape(gt, te)
    return mean_squared_error(gt, te)

def compute_psnr(img_gt, img_test, data_range=255.0):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in dB.

    Args:
        img_gt: Ground truth image.
        img_test: Test image.
        data_range: Dynamic range of the images.

    Returns:
        PSNR in dB. Higher is better.
    """
    mse_val = compute_mse(img_gt, img_test)
    if mse_val == 0:
        return float('inf')
    return 10.0 * np.log10((data_range ** 2) / mse_val)

def plot_mse_map(img_gt, img_test, title="Per-pixel squared error", run_dir=None):
    """
    Save a per-pixel squared error heatmap.

    Args:
        img_gt: Ground truth image.
        img_test: Test image.
        title: Plot title.
        run_dir: Output directory.
    """
    gt = _to_float_array(img_gt)
    te = _to_float_array(img_test)
    _check_same_shape(gt, te)

    if gt.ndim == 3:
        se = np.mean((gt - te)**2, axis=2)
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
    Compute Structural Similarity Index (SSIM).

    Args:
        img_gt: Ground truth image.
        img_test: Test image.
        data_range: Dynamic range of the images.

    Returns:
        SSIM score. Higher is better.
    """
    gt = _to_float_array(img_gt)
    te = _to_float_array(img_test)
    _check_same_shape(gt, te)
    return structural_similarity(gt, te, data_range=data_range)

def compute_ssi(img_noisy, img_filtered, roi):
    """
    Compute Speckle Suppression Index (SSI).

    SSI = (std_filtered / mean_filtered) / (std_noisy / mean_noisy),
    evaluated on a homogeneous ROI. Lower is better.

    Args:
        img_noisy: Noisy input image.
        img_filtered: Filtered output image.
        roi: Region of interest as (y, x, h, w) or array index.

    Returns:
        SSI value. Lower is better.
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

def compute_smpi(img_original, img_filtered):
    """
    Compute Speckle Mean Preservation Index (SMPI).

    Lower values indicate better speckle suppression with mean preservation.

    Args:
        img_original: Original noisy image.
        img_filtered: Filtered image.

    Returns:
        SMPI value. Lower is better.
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
        img_noisy: Noisy input image (single channel).
        img_denoised: Denoised output image (single channel).

    Returns:
        Tuple (OMQDI, EPF, NSF):
            - OMQDI: Combined metric Q1 + Q2, ideal value 2, range [1, 2].
            - EPF: Edge-Preservation Factor (Q1), ideal value 1, range [0, 1].
            - NSF: Noise-Suppression Factor (Q2), ideal value 1, range [0, 1].
    """
    from compare_to import OMQDI

    noisy = _to_float_array(img_noisy)
    denoised = _to_float_array(img_denoised)

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

    ENL = mean² / variance. Evaluated on a homogeneous ROI if provided.

    Args:
        img: Input image.
        roi: Optional (y, x, h, w) region of interest.

    Returns:
        ENL value. Higher is better. Capped at 10000.

    Reference:
        Ulaby et al., 1986.
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
        return 10000.0

    enl = (mean_val ** 2) / var_val
    return float(min(enl, 10000.0))

def compute_epi(img_original, img_denoised):
    """
    Compute Edge Preservation Index (EPI).

    EPI is the Pearson correlation of Sobel gradient magnitudes between
    the original and denoised images. Higher values indicate better edge
    preservation.

    Args:
        img_original: Original image.
        img_denoised: Denoised image.

    Returns:
        EPI in [-1, 1]. Higher is better.

    Reference:
        Sattar et al., 1997.
    """
    import cv2

    orig = _to_float_array(img_original)
    denoised = _to_float_array(img_denoised)

    orig_u8 = (orig * 255).astype(np.uint8) if orig.max() <= 1.0 else orig.astype(np.uint8)
    den_u8 = (denoised * 255).astype(np.uint8) if denoised.max() <= 1.0 else denoised.astype(np.uint8)

    gx_o = cv2.Sobel(orig_u8, cv2.CV_64F, 1, 0, ksize=3)
    gy_o = cv2.Sobel(orig_u8, cv2.CV_64F, 0, 1, ksize=3)
    grad_orig = np.sqrt(gx_o**2 + gy_o**2)

    gx_d = cv2.Sobel(den_u8, cv2.CV_64F, 1, 0, ksize=3)
    gy_d = cv2.Sobel(den_u8, cv2.CV_64F, 0, 1, ksize=3)
    grad_den = np.sqrt(gx_d**2 + gy_d**2)

    corr = np.corrcoef(grad_orig.flatten(), grad_den.flatten())[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0

def auto_detect_rois(img):
    """
    Auto-detect signal and background ROIs for CNR calculation.

    Signal ROI: centroid of the top-10% intensity region.
    Background ROI: lowest-variance block with center bias.

    Args:
        img: Input image.

    Returns:
        Tuple (signal_roi, bg_roi), each as (y, x, h, w).
    """
    arr = _to_float_array(img)
    h, w = arr.shape

    block_size = min(16, h // 4, w // 4)
    if block_size < 4:
        block_size = 4

    threshold = np.percentile(arr, 90)
    signal_mask = arr > threshold

    y_coords, x_coords = np.where(signal_mask)
    if len(y_coords) > 0:
        cy, cx = int(np.mean(y_coords)), int(np.mean(x_coords))
        sy = max(0, min(cy - block_size // 2, h - block_size))
        sx = max(0, min(cx - block_size // 2, w - block_size))
        signal_roi = (sy, sx, block_size, block_size)
    else:
        signal_roi = (0, 0, block_size, block_size)

    best_var = float('inf')
    bg_roi = (0, 0, block_size, block_size)

    for y in range(0, h - block_size, block_size // 2):
        for x in range(0, w - block_size, block_size // 2):
            block = arr[y:y+block_size, x:x+block_size]
            mu = np.mean(block)
            if mu > 0.95:
                continue
            var = np.var(block)
            dist_to_center = np.sqrt((y + block_size/2 - h/2)**2 + (x + block_size/2 - w/2)**2)
            dist_norm = dist_to_center / np.sqrt((h/2)**2 + (w/2)**2)
            cost = var + 0.05 * dist_norm
            if cost < best_var and var > 0:
                best_var = cost
                bg_roi = (y, x, block_size, block_size)

    return signal_roi, bg_roi

def compute_cnr(img, signal_roi=None, bg_roi=None):
    """
    Compute Contrast-to-Noise Ratio (CNR).

    CNR = |mean_signal - mean_bg| / std_bg.
    ROIs are auto-detected if not provided.

    Args:
        img: Input image.
        signal_roi: Optional (y, x, h, w) signal region.
        bg_roi: Optional (y, x, h, w) background region.

    Returns:
        Tuple (cnr_value, signal_roi, bg_roi). Higher CNR is better.
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
        return (10000.0, signal_roi, bg_roi)

    cnr = abs(mean_signal - mean_bg) / std_bg
    return (float(min(cnr, 10000.0)), signal_roi, bg_roi)


# -------------------------------------------------------------------------------
# Trend helpers
# -------------------------------------------------------------------------------

def plot_shots_vs_mse(shots, mse_values, title="Shots vs MSE", run_dir=None):
    """
    Plot and save a shots vs MSE trend graph.

    Args:
        shots: List of shot counts.
        mse_values: List of MSE values.
        title: Plot title.
        run_dir: Output directory.
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


# ===============================================================================
# Organized plotting classes
# ===============================================================================

class MetricsPlotter:
    """Visualization for image quality metrics and comparison reports."""

    @staticmethod
    def print_summary_text(competitors, keys, title):
        """Print a formatted metric table to stdout."""
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
        Generate and save a unified figure with images and a ranked metrics table.

        Args:
            ref_img: Reference image (uint8) or None.
            competitors: List of dicts with 'name', 'metrics', 'image' keys.
            metric_keys: List of metric names to display.
            title: Figure title.
            filename_suffix: Output filename without extension.
            save_dir: Directory to save the figure.
            include_original_in_table: Whether to include 'Original' in the table.
        """
        if include_original_in_table:
            table_methods = competitors
        else:
            table_methods = [c for c in competitors if c["name"] != "Original"]

        if not table_methods:
            return

        data_map = {m["name"]: m["metrics"] for m in table_methods}
        names = [m["name"] for m in table_methods]

        higher_better = {"OMQDI", "EPF", "ENL", "EPI", "CNR", "NSF"}
        ranks = {k: {} for k in metric_keys}

        for k in metric_keys:
            is_higher = k in higher_better
            valid_items = [(name, data_map[name].get(k, float('nan'))) for name in names]
            valid_items = [x for x in valid_items if not np.isnan(x[1])]
            valid_items.sort(key=lambda x: x[1], reverse=is_higher)

            current_rank = 1
            prev_val = None
            for i, (name, val) in enumerate(valid_items):
                if prev_val is not None and val == prev_val:
                    ranks[k][name] = current_rank
                else:
                    current_rank = i + 1
                    ranks[k][name] = current_rank
                prev_val = val

        has_ref_plot = (ref_img is not None)
        n_imgs = len(table_methods) + (1 if has_ref_plot else 0)

        fig_width = ECTI_DOUBLE_COLUMN_IN
        fig_height = 3.0

        fig = plt.figure(figsize=(fig_width, fig_height))

        gs = fig.add_gridspec(2, n_imgs, height_ratios=[1, 1], hspace=0.1)

        col_idx = 0

        if has_ref_plot:
            ax_ref = fig.add_subplot(gs[0, col_idx])
            ax_ref.imshow(ref_img, cmap="gray", vmin=0, vmax=255)
            ax_ref.set_title("Reference", fontsize=7, pad=3)
            ax_ref.set_xticks([]); ax_ref.set_yticks([])
            col_idx += 1

        for m in table_methods:
            ax = fig.add_subplot(gs[0, col_idx])
            ax.imshow(m["image"], cmap="gray", vmin=0, vmax=255)
            ax.set_title(m["name"], fontsize=7, pad=3)
            ax.set_xticks([]); ax.set_yticks([])
            col_idx += 1

        ax_table = fig.add_subplot(gs[1, :])
        ax_table.axis('off')

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
        table.set_fontsize(6.5)
        table.scale(1.0, 1.05)

        for j in range(len(metric_keys)):
            table[(0, j)].set_facecolor('#F2F2F2')
            table[(0, j)].set_text_props(color='black', weight='bold')

        for i, name in enumerate(names):
            for j, k in enumerate(metric_keys):
                r = ranks[k].get(name, None)
                if r == 1:
                    table[(i+1, j)].set_text_props(weight='bold')
        for cell in table.get_celld().values():
            cell.set_linewidth(0.35)
            cell.set_edgecolor('#808080')

        fig.suptitle(title, fontsize=9.5, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.94], pad=0.25, h_pad=0.4)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"{filename_suffix}.png")
            fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
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


# ===============================================================================
# Denoiser confidence visualization
# ===============================================================================

def plot_bias_map(bias_stats, original_img, N, d, run_dir=None):
    """
    Visualize the denoiser confidence map derived from the outcome ancilla.

    Args:
        bias_stats: Dict mapping position vectors to hit/miss stats.
        original_img: Original grayscale image in [0, 1].
        N: Image size.
        d: Qudit dimension.
        run_dir: Output directory.

    Returns:
        Confidence ratio map as a 2D numpy array, or None if no stats.
    """
    if bias_stats is None:
        print("No bias stats to plot.")
        return None

    bias_map = np.zeros((N, N))
    for vec, stats in bias_stats.items():
        r, c = utils.compose_rc(vec, d)
        hit = stats.get('hit', 0)
        miss = stats.get('miss', 0)
        total = hit + miss
        ratio = hit / total if total > 0 else 0.5
        bias_map[r, c] = ratio

    dir_path = get_run_dir(run_dir)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    if original_img is not None:
        axes[0].imshow(original_img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

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


# ===============================================================================
# Benchmark Distribution & Error Bar Plots
# ===============================================================================

def plot_metric_boxplots(all_results, metrics=None, methods=None, run_dir=None, filename="metrics_boxplots.png"):
    """
    Generate box plots illustrating the distribution of benchmark metrics across methods.

    Args:
        all_results: Dict mapping img_name -> method_name -> metric_name -> value
        metrics: List of metric keys to plot
        methods: List of method names
        run_dir: Output directory
        filename: Output image filename
    """
    if methods is None:
        observed = {method for image_results in all_results.values() for method in image_results}
        methods = [method for method in ['bm3d', 'nlmeans', 'srad', 'siamesegan', 'proposed'] if method in observed]
    
    if metrics is None:
        metrics = ['SSI', 'SMPI', 'ENL', 'CNR', 'NSF', 'EPF', 'EPI', 'OMQDI']

    available_metrics = []
    for m in metrics:
        for img_res in all_results.values():
            for meth in methods:
                if meth in img_res and m in img_res[meth]:
                    if m not in available_metrics:
                        available_metrics.append(m)

    if not available_metrics:
        print("No valid metric data found for boxplots.")
        return None

    method_labels = ["Siamese\nGAN" if m == 'siamesegan' else ecti_method_style(m)[1] for m in methods]
    colors = [ecti_method_style(m)[0] for m in methods]
    dir_path = get_run_dir(run_dir)
    figures_dir = os.path.join(dir_path, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    output_paths = []

    for metric in available_metrics:
        data = []
        for meth in methods:
            vals = []
            for img_res in all_results.values():
                if meth in img_res and metric in img_res[meth]:
                    v = img_res[meth][metric]
                    if not np.isnan(v):
                        vals.append(v)
            data.append(vals)

        figure, axis = plt.subplots(figsize=(ECTI_SINGLE_COLUMN_IN, 1.9))
        bp = axis.boxplot(
            data,
            tick_labels=method_labels,
            patch_artist=True,
            vert=False,
            widths=0.58,
            boxprops=dict(linewidth=0.65),
            medianprops=dict(color='black', linewidth=1.0),
            whiskerprops=dict(linewidth=0.65),
            capprops=dict(linewidth=0.65),
            flierprops=dict(marker='o', markersize=2.5, markeredgewidth=0.4, alpha=0.6),
        )

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        direction = "$\\downarrow$" if metric in {"SSI", "SMPI"} else "$\\uparrow$"
        axis.set_title(f"{metric} {direction}", pad=5)
        axis.invert_yaxis()
        ecti_axis(axis, grid=False)
        axis.grid(axis='x', color='#D9D9D9', linewidth=0.45, zorder=0)
        figure.tight_layout(pad=0.35)
        out_path = os.path.join(figures_dir, f"boxplot_{metric.lower()}.png")
        figure.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.03)
        plt.close(figure)
        output_paths.append(out_path)

    print(f"  Saved {len(output_paths)} metric distribution figures to: {figures_dir}")
    return output_paths


def plot_metric_errorbars(all_results, metrics=None, methods=None, run_dir=None, filename="metrics_errorbars.png"):
    """
    Generate journal-scale, two-panel point-estimate figures across methods.

    Each output contains at most two metric panels, labelled (a) and (b). This
    keeps individual plots readable in the ECTI-CIT two-column layout and lets
    the manuscript caption define the statistical quantity once per figure.

    Args:
        all_results: Dict mapping img_name -> method_name -> metric_name -> value
        metrics: List of metric keys to plot
        methods: List of method names
        run_dir: Output directory
        filename: Output filename stem. Numbered panel figures are written to
            the run directory.
    """
    if methods is None:
        observed = {method for image_results in all_results.values() for method in image_results}
        methods = [method for method in ['bm3d', 'nlmeans', 'srad', 'siamesegan', 'proposed'] if method in observed]
    
    if metrics is None:
        metrics = ['SSI', 'SMPI', 'ENL', 'CNR', 'NSF', 'EPF', 'EPI', 'OMQDI']

    available_metrics = []
    for m in metrics:
        for img_res in all_results.values():
            for meth in methods:
                if meth in img_res and m in img_res[meth]:
                    if m not in available_metrics:
                        available_metrics.append(m)

    if not available_metrics:
        print("No valid metric data found for errorbar plots.")
        return None

    method_labels = ["Siamese\nGAN" if m == 'siamesegan' else ecti_method_style(m)[1] for m in methods]
    colors = [ecti_method_style(m)[0] for m in methods]
    dir_path = get_run_dir(run_dir)
    filename_stem, extension = os.path.splitext(filename)
    extension = extension or ".png"
    output_paths = []

    for figure_index, metric_pair_start in enumerate(range(0, len(available_metrics), 2), start=1):
        panel_metrics = available_metrics[metric_pair_start:metric_pair_start + 2]
        figure, axes = plt.subplots(
            1,
            len(panel_metrics),
            figsize=(ECTI_DOUBLE_COLUMN_IN, 2.15),
            squeeze=False,
        )

        for panel_index, (axis, metric) in enumerate(zip(axes[0], panel_metrics)):
            means = []
            stds = []
            for meth in methods:
                vals = []
                for img_res in all_results.values():
                    if meth in img_res and metric in img_res[meth]:
                        value = img_res[meth][metric]
                        if not np.isnan(value):
                            vals.append(value)
                means.append(float(np.mean(vals)) if vals else 0.0)
                stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

            x_pos = np.arange(len(methods))
            axis.errorbar(
                x_pos, means, yerr=stds, fmt='none', capsize=2.5,
                capthick=0.7, elinewidth=0.7, color='#333333', zorder=2,
            )
            for x_pos_value, mean_value, color in zip(x_pos, means, colors):
                axis.plot(x_pos_value, mean_value, 'o', color=color, markersize=4.2, zorder=3)

            direction = "$\\downarrow$" if metric in {"SSI", "SMPI"} else "$\\uparrow$"
            axis.set_title(f"{metric} {direction}", pad=4)
            axis.set_ylabel("Value")
            axis.set_xticks(x_pos)
            axis.set_xticklabels(method_labels, rotation=0)
            ecti_panel_label(axis, panel_index)
            ecti_axis(axis)

        figure.tight_layout(pad=0.5, w_pad=0.9)
        suffix = f"_{figure_index:02d}" if len(available_metrics) > 2 else ""
        out_path = os.path.join(dir_path, f"{filename_stem}{suffix}{extension}")
        figure.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.03)
        plt.close(figure)
        output_paths.append(out_path)

    print(f"  Saved {len(output_paths)} two-panel metric estimate figures to: {dir_path}")
    return output_paths


class BenchmarkPlotter:
    """Utilities for benchmark distribution and error bar plots."""

    plot_metric_boxplots = staticmethod(plot_metric_boxplots)
    plot_metric_errorbars = staticmethod(plot_metric_errorbars)

