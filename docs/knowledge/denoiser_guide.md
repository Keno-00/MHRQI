# MHRQI Denoiser Guide

**What the quantum denoiser actually does.**

---

## Overview

The MHRQI denoiser applies a **partial Grover diffusion operator** to position qubits at each level of the hierarchy. This creates controlled mixing between neighboring pixels based on their hierarchical relationship.

**Simple explanation:** Pixels are gently blended with their "siblings" (neighbors in the hierarchy) from coarse to fine levels.

---

## Current Implementation

Function: `DENOISER_qiskit()` in `circuit_qiskit.py`

### Step-by-Step

1. **For each hierarchy level k** (from L-1 down to 0):
   - Get position qubits for that level: `qy[k]`, `qx[k]`
   - Calculate mixing strength based on level (finer = stronger)
   
2. **Apply partial Grover diffusion**:
   ```
   H(qy), H(qx)          # Create superposition
   X(qy), X(qx)          # Flip bits
   CP(angle, qy, qx)     # Controlled phase
   X(qy), X(qx)          # Unflip  
   H(qy), H(qx)          # Return from superposition
   ```

3. **Bias qubit marking** (optional):
   - Mark high-intensity pixels with rotation on bias qubit
   - Used during reconstruction to weight contributions

---

## What This Actually Achieves

### Grover Diffusion Operator

The sequence `H-X-CP-X-H` is a **partial Grover reflection**:
- Reflects quantum amplitudes toward their mean
- `angle = 0` → no mixing
- `angle = π` → full Grover (complete reflection)

We use partial mixing (angle < π) to gently smooth without over-diffusing.

### Level-Dependent Strength

```python
level_weight = (k + 1) / num_levels  # 0.17 to 1.0 for 6 levels
angle = diffusion_strength * level_weight
```

- **Fine levels** (k=L-1, k=L-2): Strong mixing → noise reduction
- **Coarse levels** (k=0, k=1): Gentle mixing → preserve structure

### What It Does to Pixels

At each level k, the Hadamard creates a superposition that mixes:
- Current pixel
- Its 3 siblings at that level (4-way split in quad-tree)

The CP gate applies phase rotation proportional to the mixing strength, which after measurement has the effect of statistically pulling pixel values toward their local mean.

---

## Edge Preservation (Classical Post-Processing)

**Location:** `mhrqi_bins_to_image()` in `utils.py`

After quantum measurement, we reconstruct the image with edge-aware classical smoothing:

1. **Extract edge map** from measurement probability:
   - High probability → flat region → smooth more
   - Low probability → edge → preserve

2. **Sibling-based smoothing**:
   - For flat pixels: average with siblings
   - For edge pixels: keep original value

3. **Adaptive strength**:
   - Depends on local flatness
   - Non-linear (only very flat regions get strong smoothing)

---

## Parameters

| Parameter | Type | Effect |
|-----------|------|--------|
| `strength` | float (0-2) | Overall diffusion intensity |
| `bias` | qubit | Optional bias qubit for weighting |
| `method` | `'bias'` or `'uniform'` | Bias marking vs uniform mixing |

**Typical value:** `strength = 1.65` for good balance

---

## What This Is NOT

❌ **NOT** Discrete-Time Quantum Walks (DTQW) - that code was removed  
❌ **NOT** edge-detecting within the quantum circuit - edges detected classically  
❌ **NOT** gradient-based anisotropic diffusion - no gradient computation in circuit  
❌ **NOT** block-matching or NL-means adaptation  

---

## Limitations

1. **No quantum edge detection**: Edge preservation happens classically after measurement
2. **Shot noise**: Shot-based mode introduces sampling variance
3. **Statevector only**: Full denoising requires statevector simulation (expensive)
4. **Fixed hierarchy**: Cannot adapt structure to image content

---

## Code Reference

**Main denoiser:**
- `circuit_qiskit.py:481-585` - `DENOISER_qiskit()`

**Classical reconstruction:**
- `utils.py:157-329` - `mhrqi_bins_to_image()`

**Fallback:**
- `circuit_qiskit.py:661-679` - `_uniform_averaging()` (when bias unavailable)

---

## Summary

The denoiser is a **hierarchical partial Grover diffusion** that mixes pixel values with their siblings at multiple scales. Edge preservation is achieved through classical post-processing using measurement probability as an edge indicator.

It's simple, effective, and based on well-understood quantum operators.
