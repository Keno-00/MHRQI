import numpy as np
import matplotlib.pyplot as plt
import utils


def _value_from_bin(v, kind="p", eps=0.0):
    if kind == "hit":  return float(v["hit"])
    if kind == "miss": return float(v["miss"])
    t = float(v["trials"])
    return (v["hit"] + eps) / (t + 2*eps) if t > 0 else np.nan  # phat

def plot_hits_scatter(bins, d,N, kind="p", eps=0.0, s_min=10, s_max=200, title=None):
    xs, ys, vals = [], [], []
    for hcv, v in bins.items():
        y, x = utils.compose_rc(hcv,d)
        xs.append(x)
        ys.append(y)
        vals.append(_value_from_bin(v, kind, eps))
    xs, ys, vals = np.array(xs), np.array(ys), np.array(vals)

    finite = np.isfinite(vals)
    if finite.any():
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if vmax > vmin:
            sizes = s_min + (s_max - s_min) * (vals - vmin) / (vmax - vmin)
        else:
            sizes = np.full_like(vals, (s_min + s_max) / 2.0, dtype=float)
    else:
        sizes = np.full_like(vals, (s_min + s_max) / 2.0, dtype=float)

    fig, ax = plt.subplots()
    ax.scatter(xs[finite], ys[finite], s=sizes[finite])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, N-1)
    ax.set_ylim(0, N-1)
    ax.grid(True, linewidth=0.5)
    ax.set_title(title or f"Hits ({'p̂' if kind=='p' else kind})")
    plt.show()

    return xs, ys, vals, sizes, fig, ax

# --- heatmap grid ---
def bins_to_grid(bins, d,N, kind="p", eps=0.0):
    size = N
    grid = np.full((size, size), np.nan)
    for hcv, v in bins.items():
        y, x = utils.compose_rc(hcv,d)
        grid[y,x] = _value_from_bin(v, kind, eps)
    return grid

# heatmap → return grid, fig, ax, im
def plot_hits_grid(bins, d, N,kind="p", eps=0.0, title=None):
    grid = bins_to_grid(bins, d, N,kind, eps)
    fig, ax = plt.subplots()
    im = ax.imshow(grid, origin="lower", interpolation="nearest")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or f"Hits grid ({'p̂' if kind=='p' else kind})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.show()
    return grid, fig, ax, im



def grid_to_image_uint8(grid, vmin=None, vmax=None, flip_vertical=False):
    """
    Convert an N×N grid into a uint8 image.
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

def bins_to_image_uint8(bins, d, N,kind="p", eps=0.0, vmin=None, vmax=None, flip_vertical=True):
    grid = bins_to_grid(bins, d, N,kind=kind, eps=eps)
    return grid_to_image_uint8(grid, vmin=vmin, vmax=vmax, flip_vertical=flip_vertical)


def show_image_comparison(orig_img, recon_img, titles=("Original", "Reconstructed")):
    """
    Plot two images side by side for visual comparison.
    Accepts 2D arrays (uint8 preferred). No resizing is done.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(orig_img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(titles[0]); axes[0].axis("off")

    axes[1].imshow(recon_img, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(titles[1]); axes[1].axis("off")

    plt.tight_layout()
    plt.show()

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

def _per_pixel_diff(gt, test):
    """
    Per-pixel absolute difference (H,W) regardless of channels.
    If color, average absolute diff across channels.
    """
    if gt.ndim == 2:
        return np.abs(gt - test)
    # color: average per-pixel abs diff over channels
    return np.mean(np.abs(gt - test), axis=2)

# ---------------------------------------------
# metric helpers (+ per-pixel maps)
# ---------------------------------------------
def compute_mse(img_gt, img_test):
    """
    Return average MSE between two images.
    Works for grayscale or color (averages over channels).
    """
    gt = _to_float_array(img_gt)
    te = _to_float_array(img_test)
    _check_same_shape(gt, te)

    diff = gt - te
    if diff.ndim == 2:
        mse = np.mean(diff**2)
    else:
        mse = np.mean(np.mean(diff**2, axis=2))  # average over channels, then pixels
    return float(mse)

def plot_mse_map(img_gt, img_test, title="Per-pixel squared error"):
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
    # RdYlGn_r maps low->green, high->red
    im = plt.imshow(se, cmap="RdYlGn_r")
    plt.title(title)
    plt.axis("off")
    cbar = plt.colorbar(im)
    cbar.set_label("squared error")
    plt.show()

def compute_psnr(img_gt, img_test, max_pixel=255.0):
    """
    Return average PSNR (dB) computed from the global MSE.
    PSNR = 10*log10(max_pixel^2 / MSE). Infinite if identical.
    """
    mse = compute_mse(img_gt, img_test)
    if mse == 0:
        return float("inf")
    psnr = 10.0 * np.log10((max_pixel ** 2) / mse)
    return float(psnr)

def plot_psnr_map(img_gt, img_test, max_pixel=255.0, clip_db=60.0, title="Per-pixel PSNR proxy (dB)"):
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
    # RdYlGn maps low->red, high->green
    im = plt.imshow(psnr_map, cmap="RdYlGn")
    plt.title(title)
    plt.axis("off")
    cbar = plt.colorbar(im)
    cbar.set_label("PSNR (dB, clipped)")
    plt.show()

# ---------------------------------------------
# trend helpers (line graphs)
# ---------------------------------------------
def plot_shots_vs_mse(shots, mse_values, title="Shots vs MSE"):
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
    plt.show()

def plot_shots_vs_psnr(shots, psnr_values, title="Shots vs PSNR"):
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
    plt.show()