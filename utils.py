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

def expand_and_sort_denoised(big_array: List[List[int]], msb_first: bool = True) -> List[List[int]]:
    sorted_initial = sort_by_binary(big_array, msb_first=msb_first)
    expanded = []
    for arr in sorted_initial:
        # Add both mean (0/1) and color (0/1) = 4 combinations
        for mean_bit in [0, 1]:
            for color_bit in [0, 1]:
                expanded.append(arr + [mean_bit, color_bit])
    
    # Detect if binary
    is_binary = all(val in (0, 1) for arr in expanded for val in arr)
    if is_binary:
        sorted_final = sort_by_binary(expanded, msb_first=msb_first)
    else:
        max_val = max(val for arr in expanded for val in arr)
        d = max_val + 1
        sorted_final = sort_by_dary(expanded, d, msb_first=msb_first)
    return sorted_final


def empty_bin():
    return {"miss": 0.0, "hit": 0.0, "trials": 0.0}

def make_bins(counts, hierarchy_matrix):
    bins = defaultdict(empty_bin)
    sorted = expand_and_sort(hierarchy_matrix)
    print(sorted)
    print(counts)

    for i in counts: #for each i na makuha natin, hanapin natin yung respective register
        print(i) # i is a state count
        curr = sorted[i].copy()
        print(curr)
        h = curr.pop()
        key = tuple(curr)
        if h == 1:
            bins[key]["hit"] += 1
        else:
            bins[key]["miss"] += 1
        bins[key]["trials"] += 1

    return bins

def make_bins_denoised(counts, hierarchy_matrix):
    bins = defaultdict(empty_bin)
    sorted_expanded = expand_and_sort_denoised(hierarchy_matrix)
    
    for i in counts:  # i is an index
        curr = sorted_expanded[i].copy()
        
        mean_bit = curr.pop()
        color_bit = curr.pop()
        key = tuple(curr)
        
        if mean_bit == 0:
            if color_bit == 1:
                bins[key]["hit"] += 1
            else:
                bins[key]["miss"] += 1
            bins[key]["trials"] += 1
    
    return bins

def make_bins_sv(state_vector, hierarchy_matrix, d=2):
    bins = defaultdict(empty_bin)
    sorted = expand_and_sort(hierarchy_matrix)
    print(sorted)
    print(state_vector)
    sv = np.array(state_vector)  # ensure its a NumPy array
    sv_flat = sv.flatten()      # flatten in row-major order

    for index,i in enumerate(sv_flat): #for each i na makuha natin, hanapin natin yung respective register
        prb =np.abs(i)**2
        curr = sorted[index].copy()
        print(curr)
        h = curr.pop()
        key = tuple(curr)
        if h == 1:
            bins[key]["hit"] = prb
        else:
            bins[key]["miss"] = prb
        bins[key]["trials"] += prb

    return bins





def make_bins_sv_denoised(state_vector, hierarchy_matrix, d=2):
    bins = defaultdict(empty_bin)
    sorted = expand_and_sort_denoised(hierarchy_matrix)
    
    sv = np.array(state_vector)
    sv_flat = sv.flatten()

    for index, i in enumerate(sv_flat):
        prb = np.abs(i)**2
        curr = sorted[index].copy()
        
        mean_bit = curr.pop()
        color_bit = curr.pop()
        key = tuple(curr)
        
        # Only count if mean_bit == 0 (uncompute succeeded)
        if mean_bit == 0:
            if color_bit == 1:
                bins[key]["hit"] += prb
            else:
                bins[key]["miss"] += prb
            bins[key]["trials"] += prb

    return bins


def p_hat(bins, hcv, eps=0.0):
    v = bins[hcv]
    t = v["trials"]
    return (v["hit"] + eps) / (t + 2*eps) if t else float("nan") # hit over hit+miss


####################
# $3$ denoiser functions
####################

def v_eff(theta_a, interaction, alpha):
    sin_a_half = math.sin(theta_a / 2)
    j = alpha * abs(sin_a_half**2)
    
    
    return j+ interaction


def g_matrix(thetas, beta):
    """
    Compute g for all pairwise interactions between values in thetas list.
    Returns a matrix where entry (i,j) is g(thetas[i], thetas[j], beta).
    """
    thetas = np.array(thetas)
    theta_a_grid, theta_b_grid = np.meshgrid(thetas, thetas, indexing='ij')
    sin_a_half = np.sin(theta_a_grid / 2)
    sin_b_half = np.sin(theta_b_grid / 2)
    return beta * np.abs(sin_a_half**2 - sin_b_half**2)




def interactions(matrix):
    """
    Compute the sum of each row in the matrix, returning a column vector.
    """
    return np.sum(matrix, axis=1, keepdims=True)





#res = g_matrix([1,1.2,0.9,2],1)
#print(interactions(res))
