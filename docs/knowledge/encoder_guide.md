# Encoder

This guide explains how classical images are encoded in the **Magnitude-Hierarchical Representation of Quantum Images**.

---

## Components of MHRQI

MHRQI stores pixel coordinates and intensities in two quantum registers. 

### 1. Hierarchical Position

The image is decomposed using a quad-tree structure (for d=2). Each level of the hierarchy represents a spatial scale:
- **Level 0:** Global position (which quadrant)
- **Level 1:** Sub-quadrant position
- **Level k:** Fine-grained pixel position

The **Hierarchical Coordinate Vector (HCV)** maps (r,c) to a sequence of indices.

The position register is initialized to the state |0⟩ for each qubit. To each qubit, hadamard gates are applied. This creates a superposition of all possible positions.



### 2. Basis-Encoded Intensity

Instead of rotating a qubit by θ (intensity), we use a register of n qubits (typically n=8 for 256 grayscale levels).
- A pixel with intensity 127 (mid-gray) is encoded as the state |01111111⟩.
- This allows for exact representation of digital values.

---

## Circuit Implementation

The encoding is performed in `MHRQI_upload` or the optimized `MHRQI_lazy_upload`.

### MHRQI_upload (Gate-based)
Uses multi-controlled X gates to flip intensity bits based on position:
```
For each pixel position:
  1. Prepare control qubits (X gates where ctrl_state=0)
  2. Apply MCX to set and_ancilla
  3. Apply CX from ancilla to intensity qubits for each '1' bit
  4. Uncompute MCX
  5. Restore control qubits
```

### MHRQI_lazy_upload (Statevector-based)
Directly initializes the statevector for faster simulation:
```python
# Build statevector directly (no gates)
state = np.zeros(2**num_qubits, dtype=complex)
for each pixel:
    idx = position_bits + intensity_bits
    state[idx] = 1.0
state = normalize(state)
qc.append(SetStatevector(state), qc.qubits)
```

This bypasses gate synthesis entirely, ideal for statevector simulation.

---

## Qubit Layout

| Register | Qubits | Purpose |
|----------|--------|---------|
| Position (q_y_k, q_x_k) | 2 × L_max | Hierarchical pixel position |
| Intensity | 8 | Basis-encoded grayscale (0-255) |
| Bias | 1 | Denoising confidence flag |
| Work | 2 | Ancilla for multi-controlled gates |

For a 256×256 image: L_max = 8, total = 16 + 8 + 1 + 2 = **27 qubits**

---

## See Also

- [Benchmarking](benchmarking.md) - How denoising results are evaluated
- `circuit.py` - Implementation source
