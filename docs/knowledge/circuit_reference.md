# Circuit Module Reference

Documentation for `circuit.py` - the core quantum circuit implementation.

## Overview

This module contains all quantum circuit construction, simulation, and binning functions for MHRQI encoding and denoising.

---

## Circuit Construction

### MHRQI_init(d, L_max, bit_depth=8)

Initialize an MHRQI quantum circuit with proper register layout.

**Parameters:**
- `d`: Dimension (2 for standard qubit-based encoding)
- `L_max`: Hierarchy depth (log₂ of image side length)
- `bit_depth`: Bits for intensity encoding (default: 8 for 0-255 grayscale)

**Returns:**
- `qc`: QuantumCircuit
- `pos_regs`: Position qubit registers
- `intensity_reg`: Intensity qubit register
- `bias_qubit`: Bias register for denoising
- `hierarchy_matrix`: Position state matrix

**Qubit Layout:**
| Register | Count | Purpose |
|----------|-------|---------|
| Position | 2×L_max | Hierarchical coordinates |
| Intensity | bit_depth | Basis-encoded grayscale |
| Bias | 1 | Denoising confidence flag |
| Work | 2 | Ancillas for gates |

---

### MHRQI_upload(qc, pos_regs, intensity_reg, d, hierarchy_matrix, img)

Upload image intensities using multi-controlled X gates.

**Use when:** You need explicit gate decomposition (e.g., for circuit analysis).

---

### MHRQI_lazy_upload(qc, pos_regs, intensity_reg, d, hierarchy_matrix, img)

Upload image intensities using direct statevector initialization.

**Use when:** Running simulations (much faster than gate-based upload).

---

## Denoising

### DENOISER(qc, pos_regs, intensity_reg, bias=None, brightness_shift=10)

Apply parent-child consistency denoising circuit.

**Algorithm:**
1. Create sibling superposition (H gates on finest position qubits)
2. Encode parent block average into ancilla via CRY rotations
3. Compare pixel MSB to parent average (XNOR logic)
4. Mark inconsistent pixels in bias qubit
5. Uncompute all operations except bias

**Parameters:**
- `brightness_shift`: Offset applied before/after denoising (default: 10)

---

### ACCIDENT_DISCOVERY(qc, pos_regs, intensity_reg, bias=None)

Experimental 4-bit variant of denoiser.

**Status:** Under investigation as potential quantizer/transcoder.

---

## Simulation

### simulate_statevector(qc, use_gpu=True)

Run statevector simulation (exact, no sampling noise).

---

### simulate_counts(qc, shots=1024, use_gpu=True)

Run shot-based measurement simulation.

---

## Binning

### make_bins_counts(counts, hierarchy_matrix, bit_depth=8, denoise=False)

Convert measurement counts to pixel intensity bins.

**Returns:**
- `bins`: Dictionary mapping position → intensity statistics
- `bias_stats`: (if denoise=True) hit/miss counts for confidence

---

### make_bins_sv(state_vector, hierarchy_matrix, bit_depth=8, denoise=False)

Convert statevector amplitudes to pixel intensity bins (sparse processing).
