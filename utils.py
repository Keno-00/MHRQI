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
    max_val = (1 << bit_depth) - 1  # 255
    u = np.clip(img.astype(np.float64) / max_val, 0.0, 1.0)
    theta = 2.0*np.arcsin(np.sqrt(u))       #RY(theta) to  psi = theta/2
    return theta

def get_Lmax(N,d):
    L_max = math.floor(math.log(N,d))
    return L_max

def get_subdiv_size(k,N,d):
    s = N/(d**k)
    return s


def compute_register(r, c, d, sk_prev):
    qx = min(math.floor((c%sk_prev)*(d/sk_prev)),d-1)
    qy = min(math.floor((r%sk_prev)*(d/sk_prev)),d-1)
    return qy,qx

def compose_rc(hcv, d=2):

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
    Confidence-weighted 8-neighbor reconstruction.
    Only uses HIGH-CONFIDENCE neighbors for context.
    """
    img = np.zeros(image_shape)
    N = image_shape[0]

    # Baseline
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

    # Build confidence map first
    confidence_map = np.ones(image_shape) * 0.5
    for vec in hierarchy_matrix:
        key = tuple(vec)
        r, c = compose_rc(vec, d)
        if key in bias_stats:
            hit = bias_stats[key].get('hit', 0)
            miss = bias_stats[key].get('miss', 0)
            total = hit + miss
            confidence_map[r, c] = hit / total if total > 0 else 0.5

    # CONFIDENCE_THRESHOLD: only trust these neighbors
    CONFIDENCE_THRESHOLD = 0.7

    # Reconstruct
    for vec in hierarchy_matrix:
        key = tuple(vec)
        r, c = compose_rc(vec, d)
        confidence = confidence_map[r, c]

        # Collect ONLY high-confidence neighbors
        trusted_neighbor_vals = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N:
                    # CHANGE: Check neighbor confidence before using
                    if confidence_map[nr, nc] <= CONFIDENCE_THRESHOLD:
                        trusted_neighbor_vals.append(source_img[nr, nc])

        # Fallback: if no trusted neighbors, use all neighbors
        if len(trusted_neighbor_vals) == 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < N and 0 <= nc < N:
                        trusted_neighbor_vals.append(source_img[nr, nc])

        context_avg = np.median(trusted_neighbor_vals) if trusted_neighbor_vals else source_img[r, c]

        # Proportional blend
        img[r, c] = (confidence * source_img[r, c]) + ((1 - confidence) * context_avg)

    return img
