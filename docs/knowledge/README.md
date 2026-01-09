# MHRQI Knowledge Base

**Factual, plain-language documentation of the MHRQI quantum image processing framework.**

---

## Quick Navigation

### Getting Started
- [Main README](../../README.md) - Project overview and quick start
- [What is MHRQI?](#what-is-mhrqi) - Core concepts explained simply

### Implementation Guides
- [Encoder Guide](encoder_guide.md) - How images are encoded
- [Denoiser Guide](denoiser_guide.md) - How quantum denoising works
- [Sibling Smoothing](sibling_smoothing.md) - Hierarchical structure and smoothing

### Reference
- [Benchmarking](benchmarking.md) - Metrics and comparison methodology

---

## What is MHRQI?

**MHRQI** = **Magnitude Hierarchical Representation of Quantum Images**

It's a method for encoding and processing grayscale images using quantum circuits.

### Three Core Ideas

1. **Hierarchical Position Encoding**
   - Instead of encoding pixel (r,c) as flat binary, use levels
   - Like a quad-tree: coarse position → medium → fine → pixel
   - Example: (3,3) in 4×4 image = level0:(bottom-right) + level1:(bottom-right)

2. **Basis-Encoded Intensity**
   - Grayscale value (0-255) stored directly in 8 qubits
   - Computational basis: |01111111⟩ = 127 = mid-gray

3. **Level-by-Level Denoising**
   - Apply smoothing at each hierarchy level
   - Coarse levels: blend large blocks gently
   - Fine levels: smooth noise more aggressively

---

## Current Status

### ✅ Implemented
- Hierarchical qubit encoding (qubits only, d=2)
- Statevector and gate-based circuit construction
- Grover-like diffusion denoiser
- Classical sibling-based smoothing in reconstruction
- Benchmarking against BM3D, NL-Means, SRAD
- Comprehensive metrics (FSIM, SSIM, NIQE, PIQE, SMPI, etc.)

### 🚧 Experimental
- MQT qudits backend (d>2)
- Decision diagram backend
- GPU acceleration (cuStateVec)

### ❌ Not Implemented
- DTQW (Discrete-Time Quantum Walks) - removed
- H-BM3D, H-NLM adaptations - not built
- Angle-based MHRQI (rotation encoding) - being phased out

---

## Documentation Philosophy

We prioritize:
- ✅ **Factual** - Only document what exists in code
- ✅ **Plain language** - No unnecessary jargon or math flourish
- ✅ **Well-organized** - Clear structure, minimal redundancy
- ❌ **No hallucinations** - Don't claim unproven results

---

## File Structure

```
docs/
├── knowledge/                  # You are here
│   ├── README.md              # This file
│   ├── encoder_guide.md       # How encoding works
│   ├── denoiser_guide.md      # How denoising works
│   ├── sibling_smoothing.md   # Hierarchical structure explained
│   └── benchmarking.md        # Metrics and comparison
└── site/                      # Website HTML/CSS/JS
    ├── index.html
    └── ...
```

---

## Terminology Clarifications

### MHRQI vs MHRQIB
**As of 2026:** MHRQIB (basis-encoded) **IS** the definitive MHRQI. 

The older angle-based encoding (rotation) is being deprecated. Throughout the code you may see `backend='qiskit_mhrqib'` - this is what we now simply call MHRQI.

### Seams and Siblings
- **Hierarchical block**: Group of pixels sharing parent in quad-tree
- **Siblings**: Pixels in same parent block but different child blocks
- **Seam**: Potential discontinuity at block boundaries  
- **Sibling smoothing**: Averaging within parent blocks to avoid seams

No complex mathematical formalism needed - it's just grouped averaging.

---

## Further Help

- **Code structure**: See [main README](../../README.md)
- **Running experiments**: Check `main.py` configuration options
- **Benchmarking**: See [benchmarking.md](benchmarking.md)

---

**Last updated:** 2026-01-09
