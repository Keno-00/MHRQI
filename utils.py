"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MHRQI - Magnitude Hierarchical Representation of Quantum Images            ║
║  Utility Functions: Encoding, Reconstruction, Sibling Smoothing             ║
║                                                                              ║
║  Author: Keno-00                                                             ║
║  License: Apache 2.0                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from collections import defaultdict
from typing import Any, Mapping, Optional, List,Iterable
import math


############################
# $1$ Classical Image Precompute functions
###########################

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

##############
# $2$ Decompute and regen image functions
#############

def bits_to_int(bits: Iterable[int], msb_first: bool = True) -> int:
    value = 0
    if msb_first:
        for b in bits:
            value = (value << 1) | (1 if b else 0)
    else:
        # LSB-first: bit 0 contributes 2^0, bit 1 contributes 2^1, etc.
        # We iterate with power-of-two accumulation.
        pow2 = 1
        for b in bits:
            if b:
                value += pow2
            pow2 <<= 1
    return value


def sort_by_binary(big_array: List[List[int]], msb_first: bool = True) -> List[List[int]]:
    return sorted(big_array, key=lambda arr: bits_to_int(arr, msb_first=msb_first))
def digits_to_int(digits: Iterable[int], base: int, msb_first: bool = True) -> int:
    value = 0
    if msb_first:
        for d in digits:
            value = value * base + d
    else:
        powb = 1
        for d in digits:
            value += d * powb
            powb *= base
    return value


def empty_bin():
    return {"miss": 0.0, "hit": 0.0, "trials": 0.0}


def sort_by_dary(big_array: List[List[int]], d: int, msb_first: bool = True) -> List[List[int]]:
    return sorted(big_array, key=lambda arr: digits_to_int(arr, d, msb_first=msb_first))



def expand_and_sort(big_array: List[List[int]], msb_first: bool = True) -> List[List[int]]:
    sorted_initial = sort_by_binary(big_array, msb_first=msb_first)
    expanded = []
    for arr in sorted_initial:
        expanded.append(arr + [0])
        expanded.append(arr + [1])
    # Detect if binary
    is_binary = all(val in (0, 1) for arr in expanded for val in arr)
    if is_binary:
        sorted_final = sort_by_binary(expanded, msb_first=msb_first)
    else:
        max_val = max(val for arr in expanded for val in arr)
        d = max_val + 1
        sorted_final = sort_by_dary(expanded, d, msb_first=msb_first)
    return sorted_final

def make_bins(counts, hierarchy_matrix):
    bins = defaultdict(empty_bin)
    sorted = expand_and_sort(hierarchy_matrix)

    for i in counts:
        curr = sorted[i].copy()
        h = curr.pop()
        key = tuple(curr)
        if h == 1:
            bins[key]["hit"] += 1
        else:
            bins[key]["miss"] += 1
        bins[key]["trials"] += 1

    return bins

def make_bins_sv(state_vector, hierarchy_matrix, d=2):
    bins = defaultdict(empty_bin)
    sorted = expand_and_sort(hierarchy_matrix)
    sv = np.array(state_vector)  # ensure its a NumPy array
    sv_flat = sv.flatten()      # flatten in row-major order

    for index,i in enumerate(sv_flat): #for each i na makuha natin, hanapin natin yung respective register
        prb =np.abs(i)**2
        curr = sorted[index].copy()
        #print(curr)
        h = curr.pop()
        key = tuple(curr)
        if h == 1:
            bins[key]["hit"] = prb
        else:
            bins[key]["miss"] = prb
        bins[key]["trials"] += prb

    return bins


# NOTE: The *_denoised binning functions were removed.
# If MQT qudits denoising is implemented in circuit.py, you will need either:
# 1. Selective measurement support (like Qiskit) to exclude ancilla qubits, OR
# 2. Post-measurement binning functions that filter out ancilla bits
# See circuit.py for related notes.



def mhrqi_bins_to_image(bins, hierarchy_matrix, d, image_shape, bias_stats=None, original_img=None):
    """
    Convert unified MHRQI bins to image with hierarchical seam-aware denoising.
    
    Args:
        bins: bins from make_bins_sv (contains probability 'count' per position)
        hierarchy_matrix: position state matrix
        d: dimension
        image_shape: target image shape (rows, cols)
        bias_stats: optional dict (not used in statevector mode)
        original_img: REQUIRED - original normalized image [0,1] for denoising
    
    Returns:
        img: reconstructed image with edge-preserving denoising
    
    Hierarchical Seam-Aware Denoising:
        - Extract edge weights from measurement probability distribution
        - Smooth ONLY within sibling blocks at each hierarchy level
        - Stronger smoothing for non-edges, preserve edges
    """
    img = np.zeros(image_shape)
    N = image_shape[0]
    L_max = int(np.log2(N))
    
    # Extract edge map from measurement probability
    # INTERPRETATION: 
    #   HIGH probability = frequently measured = NOISY/UNIFORM region = should FLATTEN
    #   LOW probability = rarely measured = EDGE/STRUCTURE = should PRESERVE
    total_prob = sum(bins[tuple(v)]['count'] for v in hierarchy_matrix if tuple(v) in bins)
    n_pixels = len(hierarchy_matrix)
    uniform_prob = total_prob / n_pixels if n_pixels > 0 else 1.0
    
    edge_map = {}
    for vec in hierarchy_matrix:
        key = tuple(vec)
        if key in bins and bins[key]['count'] > 0:
            prob = bins[key]['count']
            r, c = compose_rc(vec, d)
            # edge_weight: HIGH prob = noisy (flatten), LOW prob = edge (preserve)
            edge_map[(r, c)] = min(prob / uniform_prob, 1.0) if uniform_prob > 0 else 0.5
    
    # If no original image, just return intensity from bins
    if original_img is None:
        for vec in hierarchy_matrix:
            key = tuple(vec)
            if key in bins and bins[key]['count'] > 0:
                avg_intensity = bins[key]['intensity_sum'] / bins[key]['count']
                r, c = compose_rc(vec, d)
                img[r, c] = avg_intensity
        return img
    
    # Helper: get siblings at level k (same parent block)
    def get_siblings(r, c, k, N, d):
        """Get sibling pixel coordinates at hierarchy level k"""
        block_size = N // (d ** k)
        if block_size < 1:
            return []
        block_r = (r // block_size) * block_size
        block_c = (c // block_size) * block_size
        siblings = []
        for dr in range(block_size):
            for dc in range(block_size):
                sr, sc = block_r + dr, block_c + dc
                if (sr, sc) != (r, c) and 0 <= sr < N and 0 <= sc < N:
                    siblings.append((sr, sc))
        return siblings
    
    # Helper: check if block at level k is homogeneous
    def is_block_homogeneous(r, c, k, N, d, edge_map, threshold=0.25):
        """Check if block is homogeneous (uniform, no edges)"""
        block_size = N // (d ** k)
        if block_size < 2:
            return False
        block_r = (r // block_size) * block_size
        block_c = (c // block_size) * block_size
        
        weights = []
        for dr in range(block_size):
            for dc in range(block_size):
                sr, sc = block_r + dr, block_c + dc
                if 0 <= sr < N and 0 <= sc < N:
                    weights.append(edge_map.get((sr, sc), 0.5))
        
        if len(weights) < 2:
            return False
        
        # Two criteria for homogeneity:
        # 1. Low variance (all similar weights)
        # 2. High mean (no edges in block - edges have low weight)
        variance = max(weights) - min(weights)
        mean_weight = sum(weights) / len(weights)
        
        # Block is homogeneous only if uniform AND no edges
        return variance < threshold and mean_weight > 0.6
    
    # Helper: check if block contains any edges (low weight pixels)
    def block_has_edges(r, c, k, N, d, edge_map, edge_threshold=0.91):  # Higher = more aggressive preserve
        """Check if block contains pixels with low edge weight (edges)"""
        block_size = N // (d ** k)
        if block_size < 1:
            return False
        block_r = (r // block_size) * block_size
        block_c = (c // block_size) * block_size
        
        for dr in range(block_size):
            for dc in range(block_size):
                sr, sc = block_r + dr, block_c + dc
                if 0 <= sr < N and 0 <= sc < N:
                    if edge_map.get((sr, sc), 0.5) < edge_threshold:
                        return True
        return False
    
    # =========================================
    # PIXEL-PRECISE EDGE-WEIGHT BASED DENOISING
    # =========================================
    # Directly use edge_weight at each pixel:
    #   Low weight (< threshold) = edge = preserve
    #   High weight (> threshold) = flat = can smooth
    # Smoothing strength depends on:
    #   1. How flat this pixel is (edge_weight)
    #   2. How flat its neighbors are (neighbor preserve ratio)
    
    edge_threshold = 0.85  # Higher = more preserve (was 0.81, now 0.85)
    
    for r in range(N):
        for c in range(N):
            orig_intensity = original_img[r, c]
            edge_w = edge_map.get((r, c), 0.5)
            
            if edge_w < edge_threshold:
                # Edge pixel - preserve original
                final_intensity = orig_intensity
            else:
                # Flat region - check neighbors for precise flatten
                smooth_level = L_max - 1
                siblings = get_siblings(r, c, smooth_level, N, d)
                
                if siblings:
                    # Count flat vs preserve neighbors
                    flat_siblings = [(sr, sc) for sr, sc in siblings 
                                    if edge_map.get((sr, sc), 0.5) >= edge_threshold]
                    preserve_siblings = [(sr, sc) for sr, sc in siblings 
                                        if edge_map.get((sr, sc), 0.5) < edge_threshold]
                    
                    # Neighbor ratio: 1.0 = all flat, 0.0 = all preserve
                    flat_ratio = len(flat_siblings) / len(siblings)
                    
                    if flat_siblings and flat_ratio > 0.67:
                        # NON-LINEAR STRENGTH: exponential curve
                        # flat_ratio^3 → only VERY flat regions get strong smoothing
                        # 1.0^3 = 1.0 (full strength)
                        # 0.8^3 = 0.51 (half strength)
                        # 0.7^3 = 0.34 (weak)
                        
                        base_strength = (edge_w - edge_threshold) / (1.0 - edge_threshold)
                        
                        # Non-linear: cube of flat_ratio
                        flatness_power = flat_ratio ** 3
                        
                        # Final smooth strength (100% for absolutely flat)
                        smooth_strength = base_strength * flatness_power * 1.0
                        
                        sibling_avg = sum(original_img[sr, sc] for sr, sc in flat_siblings) / len(flat_siblings)
                        final_intensity = (1 - smooth_strength) * orig_intensity + smooth_strength * sibling_avg
                    else:
                        # Too many preserve neighbors - don't smooth
                        final_intensity = orig_intensity
                else:
                    final_intensity = orig_intensity
            
            img[r, c] = final_intensity
    
    return img





