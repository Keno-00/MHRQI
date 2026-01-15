# Encoder Guide: MHRQI-B

This guide explains how classical images are translated into the **Magnitude-Hierarchical Representation of Quantum Images (Basis Encoding)**.

## Core Encoding: MHRQI-B

MHRQI-B (Basis-Encoded MHRQI) stores pixel intensities directly as binary integers in a quantum register. This is more robust for near-term hardware than older angle-based (rotation) methods.

### 1. Hierarchical Position
The image is decomposed using a quad-tree structure (for $d=2$). Each level of the hierarchy represents a spatial scale:
- **Level 0:** Global position (which quadrant)
- **Level 1:** Sub-quadrant position
- **Level k:** Fine-grained pixel position

The **Hierarchical Coordinate Vector (HCV)** maps $(x,y)$ to a sequence of indices $Q_k$.

### 2. Basis-Encoded Intensity
Instead of rotating a qubit by $\theta$ (intensity), we use a register of $n$ qubits (typically $n=8$ for 256 grayscale levels).
- A pixel with intensity $127$ (mid-gray) is encoded as the state $|01111111\rangle$.
- This allows for exact representation of digital values.

### 3. Phase Gradient Sensing
In the `MHRQI_lazy_upload` implementation (`circuit.py`), we classically compute the image gradient (using Sobel operators) and embed this information into the **phase** of the quantum state.

- **Flat regions (low gradient):** Phase $\approx 0$
- **Edges (high gradient):** Phase $\approx \pi$

This "Phase Marking" allows the quantum denoiser to distinguish between noise and structural edges through interference.

## Circuit Implementation

The encoding is performed in `MHRQI_upload` or the optimized `MHRQI_lazy_upload`.

```python
# MHRQI_lazy_upload (Simplified logic)
def MHRQI_lazy_upload(qc, ...):
    # 1. Compute Sobel gradient classically
    gradient_map = compute_sobel(img)
    
    # 2. Map gradient to phase
    # Flat = low phase, Edge = high phase
    state = apply_phase(state, gradient_map)
    
    # 3. Initialize statevector
    qc.initialize(state)
```

## Why this works
By combining basis-encoded values with phase-encoded structure, we create a "composite" state where the intensity is readable via measurement, but the spatial relationship (edges) affects how those measurements are weighted during denoising.
