# MHRQI Encoder Guide

**How images are encoded into quantum circuits.**

---

## Overview

MHRQI encodes grayscale images into quantum states using:
1. **Hierarchical position qubits** - Quad-tree coordinate encoding
2. **Basis-encoded intensity** - Binary representation of pixel values

For an N×N image where N = 2^L, we use **2L + 8 + 3 qubits**.

---

## Hierarchical Coordinate Vector (HCV)

### Position Encoding

Each pixel (r, c) gets a hierarchical coordinate vector:

```
HCV = (qy₀, qx₀, qy₁, qx₁, ..., qy_{L-1}, qx_{L-1})
```

Where each pair (qy_k, qx_k) represents the kth level of the hierarchy.

### Example: 4×4 Image (L=2)

| Pixel | HCV | Binary |
|-------|-----|--------|
| (0,0) | (0,0,0,0) | \|0000⟩ |
| (0,3) | (0,1,1,1) | \|0111⟩ |
| (3,0) | (1,0,1,0) | \|1010⟩ |
| (3,3) | (1,1,1,1) | \|1111⟩ |

**Recovery formula:**
```
r = qy₀ * 2^(L-1) + qy₁ * 2^(L-2) + ... + qy_{L-1} * 2^0
c = qx₀ * 2^(L-1) + qx₁ * 2^(L-2) + ... + qx_{L-1} * 2^0
```

---

## Intensity Encoding

Pixel values (0-255) are stored in **8 qubits** using standard binary:

```
I = 127 → |01111111⟩
I = 0   → |00000000⟩
I = 255 → |11111111⟩
```

### Normalization

If input image is normalized to [0,1]:
```python
intensity_int = int(normalized_value * 255)
intensity_bits = format(intensity_int, '08b')
```

---

## Quantum State

The full MHRQI quantum state is:

```
|MHRQI⟩ = (1/√N²) Σ_{r,c} |I(r,c)⟩_{intensity} ⊗ |HCV(r,c)⟩_{position}
```

All pixels in uniform superposition over positions, each carrying its intensity value.

---

## Circuit Construction

### Register Allocation

| Register | Qubits | Purpose |
|----------|--------|---------|
| Position | 2L | Hierarchical coordinates |
| Intensity | 8 | Binary pixel value |
| Bias | 1 | Denoising weight (optional) |
| AND ancilla | 1 | Multi-controlled gates |
| Work | 2 | Denoiser operations |

**Total:** 2L + 12 qubits

### Two Upload Methods

#### 1. Gate-Based Upload (Slow, Educational)

```python
for each pixel (r, c):
    1. Compute HCV
    2. Set up multi-controlled X on position qubits
    3. Apply CX to intensity qubits where bit = 1
    4. Uncompute
```

**Circuit depth:** O(N² × L × 8)  
**Use when:** Learning, small images

#### 2. Statevector Initialization (Fast)

```python
# Directly construct amplitude vector
statevector = zeros(2^(2L+8+3))
for each pixel (r, c):
    pos_index = HCV_to_index(r, c)
    int_index = intensity_value(r, c)
    combined_index = (pos_index << 8) | int_index
    statevector[combined_index] = 1/√N²
```

**Circuit depth:** O(1) - single gate  
**Use when:** Simulation, large images

---

## Code Location

**HCV computation:**
- `utils.py:25-50` - `compute_register()`, `compose_rc()`

**Gate-based upload:**
- `circuit_qiskit.py:130-187` - `MHRQIB_upload_intensity_qiskit()`

**Fast upload:**
- `circuit_qiskit.py:235-332` - `MHRQIB_lazy_upload_intensity_qiskit()`

**Initialization:**
- `circuit_qiskit.py:57-105` - `MHRQI_init_qiskit()`

---

## Measurement and Reconstruction

### Measurement

Measure all qubits in computational basis:
- Position bits → pixel coordinates (r, c)
- Intensity bits → pixel value I

### Reconstruction

**Statevector mode:**
```python
for each basis state |pos⟩|int⟩:
    probability = |amplitude|²
    mapped_pixel[pos] = int * probability
```

**Shot-based mode:**
```python
for each shot:
    pos, int = measure()
    bins[pos]["intensity_sum"] += int
    bins[pos]["count"] += 1

average_intensity[pos] = bins[pos]["intensity_sum"] / bins[pos]["count"]
```

---

## Comparison with Other Encodings

| Property | MHRQI | NEQR | FRQI |
|----------|-------|------|------|
| Position | Hierarchical | Flat binary | Flat binary |
| Intensity | 8-bit basis | 8-bit basis | Amplitude (angle) |
| Qubits (64×64) | 2·6 + 8 + 3 = 23 | 2·6 + 8 = 20 | 2·6 + 1 = 13 |
| Multi-scale | ✓ Natural | ✗ Requires extra work | ✗ No |
| Exact intensity | ✓ Always | ✓ Always | ✗ Statistical |

---

## Key Equations

| Concept | Formula |
|---------|---------|
| HCV definition | `(qy₀, qx₀, ..., qy_{L-1}, qx_{L-1})` |
| Row recovery | `r = Σ qy_k × 2^(L-1-k)` |
| Intensity encoding | `I = Σ i_j × 2^j` for j=0..7 |
| Total qubits | `2L + 8 + 3` |

---

## Summary

MHRQI encoding uses hierarchical position representation (HCV) with basis-encoded intensity. The hierarchy enables natural multi-scale processing, while basis encoding preserves exact pixel values.

Two upload methods exist: gate-based (slow, exact circuit) and statevector (fast, simulation-only).
