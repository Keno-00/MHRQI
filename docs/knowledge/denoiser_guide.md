# MHRQI Denoiser Guide

**What the denoiser actually does - honest assessment.**

---

## Current Implementation Status

The MHRQI denoiser is **hybrid quantum-classical**:
- **Quantum part:** Hierarchical position diffusion creates probability distribution
- **Classical part:** Sibling-based smoothing uses this distribution for edge detection

**This is not pure quantum denoising** - it's quantum-assisted edge-preserving image reconstruction.

---

## Quantum Circuit Component

Function: `DENOISER_qiskit()` in `circuit_qiskit.py`

### What It Actually Does

1. **For each hierarchy level k** (from L-1 down to 0):
   - Apply **partial Grover diffusion** to position qubits
   - This modifies the measurement probability distribution
   - Does NOT directly change intensity values (they're in computational basis)

2. **Grover diffusion operator:**
   ```
   H(qy), H(qx)          # Create superposition over 4 siblings
   X(qy), X(qx)          # Flip bits
   CP(angle, qy, qx)     # Controlled phase (partial reflection)
   X(qy), X(qx)          # Unflip  
   H(qy), H(qx)          # Return from superposition
   ```

3. **Level-dependent strength:**
   - Fine levels: stronger phase rotation (more mixing)
   - Coarse levels: weaker phase rotation (preserve structure)

### What This Achieves (Quantum)

- ✓ Creates hierarchical entanglement in position qubits
- ✓ Modifies measurement probability distribution
- ✓ Pixels in flat regions get higher measurement probability
- ✓ Pixels on edges get lower measurement probability

### What This Does NOT Do (Not Quantum)

- ✗ Intensity values are NOT modified quantum-mechanically
- ✗ No quantum averaging of intensity values
- ✗ No quantum arithmetic or value propagation
- ✗ Basis-encoded data (|intensity⟩) unchanged by position operations

**Reality:** Grover diffusion on position creates an edge indicator (measurement probability), not actual denoising.

---

## Classical Reconstruction Component

**Location:** `mhrqi_bins_to_image()` in `utils.py`

This is where **actual smoothing happens** - classically, after measurement.

### Step-by-Step

1. **Measure quantum circuit** → get probability distribution over pixels

2. **Extract edge map** from measurement probability:
   ```python
   edge_weight = min(measured_prob / uniform_prob, 1.0)
   
   if edge_weight < 0.85:
       # Low probability = edge → preserve
   else:
       # High probability = flat → smooth
   ```

3. **Sibling-based smoothing:**
   ```python
   if pixel is flat and siblings are flat:
       smooth_strength = edge_weight * flatness^3
       new_value = (1-smooth_strength)*original + smooth_strength*sibling_avg
   else:
       new_value = original  # preserve edges
   ```

### What This Achieves (Classical)

- ✓ Actual intensity averaging (classical computation)
- ✓ Edge preservation (thresholding on probability)
- ✓ Adaptive smoothing strength
- ✓ Hierarchical sibling structure (classical grouping)

---

## Honest Assessment: Where's The Quantum?

### Quantum Contributions

1. **Hierarchical probability distribution** - comes from quantum circuit structure
2. **Natural multi-scale encoding** - position qubits organized by level
3. **Edge indicator via interference** - measurement probability reflects local structure

### Classical Contributions

1. **Actual smoothing** - averaging intensity values
2. **Edge detection** - thresholding on probability
3. **Sibling identification** - grouping pixels classically
4. **Adaptive strength** - non-linear weighting formula

### The Truth

**This is a classical hierarchical filter guided by a quantum probability distribution.**

The quantum circuit doesn't denoise - it creates an edge map. The classical post-processing does the denoising using that edge map.

---

## Why This Approach?

**Fundamental limitation:** Intensity values are basis-encoded (|I⟩ = |10110101⟩).

Standard quantum operations on position qubits **cannot modify** computational basis states in other registers.

**To truly denoise quantum-mechanically would require:**
- Quantum arithmetic (adders, comparators) → expensive
- Quantum SWAP operations with neighbors → complex addressing
- Quantum averaging circuits → many ancilla qubits

**We chose:** Hybrid approach that's feasible on current hardware.

---

## Parameters

| Parameter | Type | Effect |
|-----------|------|--------|
| `strength` | float (0-2) | Grover diffusion intensity |
| `bias` | qubit | Optional bias qubit (currently unused effectively) |
| `method` | `'bias'` or `'uniform'` | Bias marking vs uniform (both end up classical) |

**Typical value:** `strength = 1.65`

---

## Limitations

1. **Not pure quantum** - smoothing happens classically
2. **Basis encoding constraint** - can't quantum-average computational basis values
3. **Measurement required** - must collapse state to get probabilities
4. **Edge detection classical** - probability threshold is classical logic
5. **No quantum speedup** - classical post-processing is bottleneck

---

## Future Directions

To make this truly quantum would require:
- Quantum arithmetic circuits for averaging
- Different encoding (amplitude-based instead of basis)
- Or accept hybrid nature and focus on quality/application

---

## Code References

**Quantum circuit:**
- `circuit_qiskit.py:335-585` - `DENOISER_qiskit()`
- `circuit_qiskit.py:421-579` - Grover diffusion loop

**Classical reconstruction:**
- `utils.py:157-329` - `mhrqi_bins_to_image()`
- `utils.py:209-222` - `get_siblings()`

---

## Summary

The MHRQI denoiser is an **honest hybrid approach**:
- Uses quantum circuit to create hierarchical edge-aware probability distribution
- Uses classical post-processing to perform actual smoothing
- Makes no claims of pure quantum denoising
- Focuses on practical image quality using available quantum resources

**It works, but it's not magic - it's quantum-assisted classical filtering.**

