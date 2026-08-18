"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MHRQI - Magnitude Hierarchical Representation of Quantum Images            ║
║  Utility Functions: Encoding, Reconstruction, Sibling Smoothing             ║
║                                                                              ║
║  Author: Keno-00                                                             ║
║  License: Apache 2.0                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import math

import numpy as np


def angle_map(img, bit_depth=8):
    """
    Map pixel intensities to quantum rotation angles via arcsin encoding.

    Args:
        img: Grayscale image as integer array.
        bit_depth: Bit depth of the image (default 8).

    Returns:
        Array of angles in [0, π].
    """
    max_val = (1 << bit_depth) - 1
    u = np.clip(img.astype(np.float64) / max_val, 0.0, 1.0)
    theta = 2.0 * np.arcsin(np.sqrt(u))
    return theta

def get_Lmax(N, d):
    """
    Compute the maximum hierarchy level for image size N and qudit dimension d.

    Args:
        N: Image side length.
        d: Qudit dimension.

    Returns:
        L_max = floor(log_d(N)).
    """
    L_max = math.floor(math.log(N, d))
    return L_max

def get_subdiv_size(k, N, d):
    """
    Compute the subdivision size at hierarchy level k.

    Args:
        k: Hierarchy level.
        N: Image side length.
        d: Qudit dimension.

    Returns:
        Side length of subregions at level k.
    """
    s = N / (d**k)
    return s


def compute_register(r, c, d, sk_prev):
    """
    Compute the qudit register values (qy, qx) for pixel (r, c) at a given scale.

    Args:
        r: Row index.
        c: Column index.
        d: Qudit dimension.
        sk_prev: Subdivision size at the previous level.

    Returns:
        Tuple (qy, qx).
    """
    qx = min(math.floor((c % sk_prev) * (d / sk_prev)), d - 1)
    qy = min(math.floor((r % sk_prev) * (d / sk_prev)), d - 1)
    return qy, qx

def compose_rc(hcv, d=2):
    """
    Convert a hierarchical coordinate vector to (row, col) pixel coordinates.

    Args:
        hcv: Sequence of qudit values (qy0, qx0, qy1, qx1, ...). Length must be even.
        d: Qudit dimension.

    Returns:
        Tuple (r, c).

    Raises:
        ValueError: If hcv length is odd or any digit is out of range.
    """
    if len(hcv) % 2 != 0:
        raise ValueError("hcv length must be even (pairs of qy,qx).")

    qy_digits = hcv[0::2]
    qx_digits = hcv[1::2]

    r = 0
    c = 0
    for digit in qy_digits:
        if not (0 <= digit < d):
            raise ValueError("qy digit out of range for given d.")
        r = r * d + int(digit)

    for digit in qx_digits:
        if not (0 <= digit < d):
            raise ValueError("qx digit out of range for given d.")
        c = c * d + int(digit)

    return r, c


def mhrqi_bins_to_image(bins, hierarchy_matrix, d, image_shape, bias_stats=None, original_img=None):
    """
    Reconstruct an image from measurement bins with optional confidence-weighted smoothing.

    When bias_stats is provided, each pixel is blended with its 8-neighborhood
    weighted by its denoiser confidence. Neighbors with confidence below
    CONFIDENCE_THRESHOLD are used as context; high-confidence pixels are
    trusted as-is.

    Args:
        bins: Measurement bins dict mapping position tuples to intensity stats.
        hierarchy_matrix: List of hierarchical coordinate vectors.
        d: Qudit dimension.
        image_shape: Output image shape as (H, W).
        bias_stats: Optional dict mapping position tuples to hit/miss counts.
        original_img: Optional pre-computed baseline image to use as source.

    Returns:
        Reconstructed image as a float array of shape image_shape.
    """
    img = np.zeros(image_shape)
    N = image_shape[0]

    reconstructed_baseline = np.zeros(image_shape)
    for vec in hierarchy_matrix:
        key = tuple(vec)
        if key in bins and bins[key].get('count', 0) > 0:
            avg_intensity = bins[key]['intensity_sum'] / bins[key]['count']
            r, c = compose_rc(vec, d)
            reconstructed_baseline[r, c] = avg_intensity

    source_img = original_img if original_img is not None else reconstructed_baseline

    if bias_stats is None:
        return source_img

    CONFIDENCE_THRESHOLD = 0.7

    confidence_map = np.ones(image_shape) * 0.5
    for vec in hierarchy_matrix:
        key = tuple(vec)
        r, c = compose_rc(vec, d)
        if key in bias_stats:
            hit = bias_stats[key].get('hit', 0)
            miss = bias_stats[key].get('miss', 0)
            total = hit + miss
            confidence_map[r, c] = hit / total if total > 0 else 0.5

    for vec in hierarchy_matrix:
        key = tuple(vec)
        r, c = compose_rc(vec, d)
        confidence = confidence_map[r, c]

        trusted_neighbor_vals = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N:
                    if confidence_map[nr, nc] <= CONFIDENCE_THRESHOLD:
                        trusted_neighbor_vals.append(source_img[nr, nc])

        if len(trusted_neighbor_vals) == 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < N and 0 <= nc < N:
                        trusted_neighbor_vals.append(source_img[nr, nc])

        context_avg = np.median(trusted_neighbor_vals) if trusted_neighbor_vals else source_img[r, c]

        img[r, c] = (confidence * source_img[r, c]) + ((1 - confidence) * context_avg)

    return img
