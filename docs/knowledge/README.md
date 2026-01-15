# MHRQI Knowledge Base

**Documentation for the MHRQI quantum image processing framework.**

---

## Quick Navigation

### Getting Started
- [What is MHRQI?](#what-is-mhrqi) - Core concepts

### Implementation Guides
- [Encoder Guide](encoder_guide.md) - How images are encoded
- [Benchmarking](benchmarking.md) - Metrics and comparison methodology

---

## What is MHRQI?

**MHRQI** = **Magnitude Hierarchical Representation of Quantum Images**

A method for encoding and processing grayscale images using quantum circuits.

### Core Ideas

1. **Hierarchical Position Encoding**
   - Quad-tree decomposition of pixel coordinates
   - Example: (3,3) in 4×4 → level0:(bottom-right) + level1:(bottom-right)

2. **Basis-Encoded Intensity**
   - Grayscale value (0-255) stored in 8 qubits
   - |01111111⟩ = 127 = mid-gray

3. **Quantum Denoising**
   - Compare pixel intensity to parent block average
   - Mark consistent/inconsistent pixels via bias qubit
   - Classical reconstruction uses confidence weighting

---

## Current Status

### ✅ Implemented
- Hierarchical qubit encoding (d=2)
- Statevector and gate-based circuits
- Basis-to-parent consistency denoiser
- Benchmarking against BM3D, NL-Means, SRAD
- Metrics: FSIM, SSIM, NIQE, PIQE, BRISQUE, SMPI, SSI

### 🚧 Experimental
- ACCIDENT_DISCOVERY circuit (quantizer behavior)
- GPU acceleration (cuStateVec)

---

## File Structure

```
docs/
├── knowledge/
│   ├── README.md           # This file
│   ├── encoder_guide.md    # How encoding works
│   └── benchmarking.md     # Metrics and comparison
└── site/                   # Website HTML/CSS/JS
```

---

## Terminology

| Term | Meaning |
|------|---------|
| HCV | Hierarchical Coordinate Vector - position encoding |
| Siblings | Pixels sharing same parent block |
| Bias qubit | Marks preserve (0) vs flatten (1) |
| L_max | Hierarchy depth = log₂(image_size) |

---

**Last updated:** 2026-01-15
