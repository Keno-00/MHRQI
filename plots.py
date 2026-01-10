"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MHRQI - Magnitude Hierarchical Representation of Quantum Images            ║
║  Plotting and Metrics: Visualization, Quality Assessment, Benchmarking      ║
║                                                                              ║
║  Author: Keno-00                                                             ║
║  License: Apache 2.0                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib.pyplot as plt
import utils

import matplotlib
import os
import datetime

import cv2
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
    axes[0].set_title(titles[0]); 
    axes[0].set_xticks([]); axes[0].set_yticks([]) # Keep box, hide ticks

    axes[1].imshow(recon_img, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(titles[1]); 
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
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from brisque import BRISQUE
import pypiqe
import skvideo.measure
import torch
import piq

def compute_fsim(img_ref, img_test):
    """
    Compute FSIM score using piq.
    Higher is better [0, 1].
    """
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

def compute_psnr(img_gt, img_test, max_pixel=255.0):
    """
    Return average PSNR (dB) using skimage.
    """
    gt = _to_float_array(img_gt)
    te = _to_float_array(img_test)
    _check_same_shape(gt, te)
    return peak_signal_noise_ratio(gt, te, data_range=max_pixel)

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
    
    if m_n == 0 or m_f == 0:
        return float('inf')
        
    cov_n = s_n / m_n
    cov_f = s_f / m_f
    
    if cov_n == 0:
        return float('inf')
        
    return cov_f / cov_n

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



def plot_psnr_map(img_gt, img_test, max_pixel=255.0, clip_db=60.0, title="Per-pixel PSNR proxy (dB)", run_dir=None):
    """
    Visualize a per-pixel PSNR-like map derived from per-pixel squared error:
        psnr_i = 10*log10(max_pixel^2 / (se_i + eps))
    For visualization only; clip at clip_db to keep scale readable.
    Green = higher PSNR (better), Red = lower PSNR (worse).
    """
    eps = 1e-12
    gt = _to_float_array(img_gt)
    te = _to_float_array(img_test)
    _check_same_shape(gt, te)

    if gt.ndim == 3:
        se = np.mean((gt - te)**2, axis=2)  # (H,W)
    else:
        se = (gt - te)**2

    psnr_map = 10.0 * np.log10((max_pixel ** 2) / (se + eps))
    psnr_map = np.clip(psnr_map, 0.0, clip_db)

    plt.figure()
    im = plt.imshow(psnr_map, cmap="RdYlGn")
    plt.title(title)
    plt.axis("off")
    cbar = plt.colorbar(im)
    cbar.set_label("PSNR (dB, clipped)")
    
    dir_path = get_run_dir(run_dir)
    plt.savefig(os.path.join(dir_path, "psnr_map.png"), dpi=150, bbox_inches="tight")

# ---------------------------------------------
# trend helpers (line graphs)
# ---------------------------------------------
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

def plot_shots_vs_psnr(shots, psnr_values, title="Shots vs PSNR", run_dir=None):
    """
    Line graph only for trend inspection.
    shots: list[int] or list[float]
    psnr_values: list[float]
    """
    if len(shots) != len(psnr_values):
        raise ValueError("shots and psnr_values length mismatch")
    plt.figure()
    plt.plot(shots, psnr_values)
    plt.xlabel("Shots")
    plt.ylabel("PSNR (dB)")
    plt.title(title)
    plt.grid(True)
    
    dir_path = get_run_dir(run_dir)
    plt.savefig(os.path.join(dir_path, "shots_vs_psnr.png"), dpi=150, bbox_inches="tight")


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
        higher_better = {"FSIM", "SSIM", "PSNR", "DR-IQA"}
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
    plot_psnr_map = staticmethod(plot_psnr_map)
    grid_to_image_uint8 = staticmethod(grid_to_image_uint8)
    bins_to_image = staticmethod(bins_to_image)


class TrendPlotter:
    """Line graphs for trend analysis."""
    
    plot_shots_vs_mse = staticmethod(plot_shots_vs_mse)
    plot_shots_vs_psnr = staticmethod(plot_shots_vs_psnr)


# =============================================================================
# HOMOGENEITY VISUALIZATION
# =============================================================================

def plot_homogeneity_map(edge_map, original_img, N, d, L_max, threshold=0.3, run_dir=None):
    """
    Visualize homogeneous (green) and heterogeneous (red) blocks.
    
    Args:
        edge_map: dict of {(r,c): edge_weight} from measurement probability
        original_img: original grayscale image (normalized [0,1])
        N: image size
        d: qudit dimension
        L_max: max hierarchy level
        threshold: homogeneity threshold for edge weight variance
        run_dir: output directory
    
    Creates RGB image:
        - Green channel: homogeneous blocks (can flatten)
        - Red channel: heterogeneous blocks (preserve structure)
        - Brightness: edge weight value
    """
    # Create RGB output image
    rgb_img = np.zeros((N, N, 3), dtype=np.float32)
    
    # Helper functions
    def get_block_bounds(r, c, k, N, d):
        block_size = N // (d ** k)
        if block_size < 1:
            return None
        block_r = (r // block_size) * block_size
        block_c = (c // block_size) * block_size
        return block_r, block_c, block_size
    
    def is_block_homogeneous(r, c, k, N, d, edge_map, threshold):
        bounds = get_block_bounds(r, c, k, N, d)
        if bounds is None:
            return False
        block_r, block_c, block_size = bounds
        
        weights = []
        for dr in range(block_size):
            for dc in range(block_size):
                sr, sc = block_r + dr, block_c + dc
                if 0 <= sr < N and 0 <= sc < N:
                    weights.append(edge_map.get((sr, sc), 0.5))
        
        if len(weights) < 2:
            return False
        return (max(weights) - min(weights)) < threshold
    
    def get_homogeneity_depth(r, c, N, d, edge_map, L_max, threshold):
        depth = 0
        finest_level = max(L_max - 1, 1)
        for k in range(finest_level, 0, -1):
            if is_block_homogeneous(r, c, k, N, d, edge_map, threshold):
                depth += 1
            else:
                break
        return depth
    
    # Process each pixel
    edge_threshold = 0.85  # Match the threshold in utils.py
    
    for r in range(N):
        for c in range(N):
            edge_w = edge_map.get((r, c), 0.5)
            
            # Base intensity from original image
            base_intensity = original_img[r, c] if original_img is not None else 0.5
            
            if edge_w < edge_threshold:
                # PIXEL-PRECISE PRESERVE: Red channel (edge detected)
                # Stronger red for stronger edges (lower weight)
                red_intensity = 0.5 + 0.5 * (1.0 - edge_w / edge_threshold)
                rgb_img[r, c, 0] = red_intensity  # Red
                rgb_img[r, c, 1] = base_intensity * 0.2  # Dim green
                rgb_img[r, c, 2] = base_intensity * 0.2  # Dim blue
            else:
                # Flatten region - check if block has any preserve
                # If block has preserve, show pixel-precise flatten (lighter green)
                # If block is purely flat, show block-based flatten (bright green)
                
                # Check finest level block for preserve neighbors
                block_size = N // (d ** (L_max - 1))
                if block_size < 1:
                    block_size = 1
                block_r = (r // block_size) * block_size
                block_c = (c // block_size) * block_size
                
                has_preserve_in_block = False
                for dr in range(block_size):
                    for dc in range(block_size):
                        sr, sc = block_r + dr, block_c + dc
                        if 0 <= sr < N and 0 <= sc < N:
                            if edge_map.get((sr, sc), 0.5) < edge_threshold:
                                has_preserve_in_block = True
                                break
                    if has_preserve_in_block:
                        break
                
                if has_preserve_in_block:
                    # Block has preserve - use PIXEL-PRECISE flatten (lighter green)
                    # Intensity based on own edge weight
                    green_intensity = 0.3 + 0.4 * (edge_w - edge_threshold) / (1.0 - edge_threshold)
                    rgb_img[r, c, 1] = green_intensity  # Lighter green
                    rgb_img[r, c, 0] = base_intensity * 0.2
                    rgb_img[r, c, 2] = base_intensity * 0.2
                else:
                    # Purely flat block - BLOCK-BASED flatten (bright green)
                    depth = get_homogeneity_depth(r, c, N, d, edge_map, L_max, threshold)
                    if depth > 0:
                        green_intensity = 0.5 + 0.5 * min(depth / (L_max - 1), 1.0)
                    else:
                        green_intensity = 0.5
                    rgb_img[r, c, 1] = green_intensity  # Bright green
                    rgb_img[r, c, 0] = base_intensity * 0.15
                    rgb_img[r, c, 2] = base_intensity * 0.15
    
    # Save and display
    dir_path = get_run_dir(run_dir)
    
    # Figure 1: Homogeneity map
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    if original_img is not None:
        axes[0].imshow(original_img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Edge weight map
    edge_img = np.zeros((N, N))
    for r in range(N):
        for c in range(N):
            edge_img[r, c] = edge_map.get((r, c), 0.5)
    axes[1].imshow(edge_img, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[1].set_title('Edge Weights (Red=High=Noisy, Green=Low=Edge)')
    axes[1].axis('off')
    
    # Homogeneity colored map
    rgb_clipped = np.clip(rgb_img, 0, 1)
    axes[2].imshow(rgb_clipped)
    axes[2].set_title('Homogeneity Map (Green=Flatten, Red=Preserve)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, 'homogeneity_map.png'), dpi=150, bbox_inches='tight')
    
    if not HEADLESS:
        plt.show()
    plt.close()
    
    return rgb_img

