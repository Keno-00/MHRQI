# MHRQI Sibling Smoothing

**How hierarchical structure enables seam-free image processing.**

---

## The Hierarchical Structure

MHRQI encodes pixel positions using a quad-tree (for qubits, d=2).

### Example: 4×4 Image (L=2 levels)

```
Level 0: Entire image
├─ Level 1: Top-left quadrant (4 pixels)
│  ├─ Pixel (0,0)
│  ├─ Pixel (0,1)
│  ├─ Pixel (1,0)
│  └─ Pixel (1,1)
├─ Level 1: Top-right quadrant
├─ Level 1: Bottom-left quadrant
└─ Level 1: Bottom-right quadrant
```

Each pixel's position is encoded as a path through this tree.

---

## Siblings

**Siblings** are pixels that share the same parent block but belong to different child blocks.

### At Different Levels

**Level 1 siblings** (same half, different quadrants):
- (0,0) and (0,2) - same top half, different left/right
- (0,1) and (0,3) - same top half, different left/right

**Level 2 siblings** (same quadrant):
- (0,0) and (0,1) - both in top-left quadrant, differ in finest level

---

## The Seam Problem

When processing blocks independently, discontinuities appear at block boundaries.

**Example:**
```
Block A: pixels smoothed within → avg = 100
Block B: pixels smoothed within → avg = 200
Boundary between A and B: sharp jump from 100 to 200 = SEAM
```

---

## Sibling Smoothing (Classical)

**Location:** `get_siblings()` and smoothing logic in `utils.py:157-329`

### How It Works

1. **Identify siblings** at each level:
   ```python
   def get_siblings(r, c, k, N, d):
       # Find all pixels in same level-k parent block
       # excluding the pixel itself
   ```

2. **Check if homogeneous**:
   - Use edge map from measurement probability
   - If most siblings are "flat" (high probability) → smooth
   - If many siblings are "edges" (low probability) → preserve

3. **Adaptive averaging**:
   ```python
   if pixel is flat and siblings are flat:
       smooth_strength = edge_weight * flatness^3
       new_value = (1 - smooth_strength) * original + smooth_strength * sibling_avg
   else:
       new_value = original  # preserve
   ```

### Key Insight

By smoothing **within** parent blocks (among siblings), we ensure continuity across block boundaries. No sharp jumps.

---

## Quantum Circuit Contribution

**Location:** `DENOISER_qiskit()` in `circuit_qiskit.py`

The quantum denoiser creates mixing at each hierarchy level:

```python
for k in range(num_levels):
    # Hadamard on position qubits at level k
    qc.h(qy[k])
    qc.h(qx[k])
    
    # Partial Grover diffusion
    # (mixes with 3 siblings at this level)
    
    qc.h(qy[k])
    qc.h(qx[k])
```

After measurement, pixels that are siblings at level k will have statistically mixed values.

---

## Edge Preservation

Not all siblings should be averaged equally.

### Measurement Probability as Edge Indicator

During quantum evolution, the measurement probability distribution reflects image structure:
- **Flat regions**: High, uniform probability → many shots
- **Edges**: Low probability → few shots

We use this as an edge map:

```python
edge_weight = min(prob / uniform_prob, 1.0)

if edge_weight < 0.85:
    # This is an edge - preserve
else:
    # Flat region - can smooth
```

---

## No Complicated Math Needed

The core idea is simple:
1. Group pixels hierarchically
2. Smooth within groups (siblings)
3. Use probability as edge indicator
4. Adaptive strength based on local context

No theorems, no proofs - just practical grouped averaging.

---

## Code References

**Sibling identification:**
- `utils.py:209-222` - `get_siblings()`

**Smoothing logic:**
- `utils.py:269-329` - Edge-weight based adaptive smoothing

**Quantum mixing:**
- `circuit_qiskit.py:556-579` - Grover diffusion loop

---

## Summary

**Sibling smoothing** prevents seam artifacts by respecting hierarchical relationships. Instead of processing blocks independently, we mix pixels within parent blocks at multiple scales. This creates smooth transitions across boundaries while preserving edges through adaptive weighting.

It's hierarchical averaging - nothing more, nothing less.
