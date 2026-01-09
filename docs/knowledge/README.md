# MHRQI Knowledge Base

**Purpose:** Factual, plain-language documentation of the MHRQI (Multi-scale Hierarchical Representation of Quantum Images) project.

**Guiding Principles:**
- ✅ **Factual above all** - Only document what exists in code
- ✅ **Plain language** - Avoid flourish, obtuse narratives, made-up jargon
- ✅ **Well-organized** - Clear structure, no redundancies
- ❌ **No hallucinations** - Don't make up terms or overclaim capabilities

---

## Current Status

### ✅ Implemented
- MHRQI encoding (angle-based, Qiskit)
- MHRQIB encoding (basis-encoded, Qiskit)
- Hierarchical position encoding using qubit levels
- Lazy upload (statevector initialization)
- Classical sibling-based smoothing in `utils.py`
- Partial Grover diffusion denoiser in `circuit_qiskit.py`
- Benchmarking against BM3D, NL-Means, SRAD
- Multiple image quality metrics (SSIM, PSNR, NIQE, PIQE, SMPI, DR-IQA, FSIM, BRISQUE)

### 🚧 Partially Implemented / Needs Verification
- MQT qudit backend (exists but less mature than Qiskit)
- Decision Diagram backend (`circuit_dd.py`)
- GPU acceleration via cuStateVec

### ❌ Not Implemented / Made Up
- DTQW (Discrete Time Quantum Walks) - removed as dead code
- Block matching filter - removed as dead code
- Many theoretical claims in `seam_theory.md` need validation

---

## Documentation To-Do List

### 1. Core Concepts (Plain Explanations)
- [ ] What is MHRQI? (hierarchical image encoding)
- [ ] Why use hierarchy? (multiscale processing)
- [ ] Position encoding (how qubit levels map to pixels)
- [ ] Intensity encoding differences:
  - [ ] MHRQI (angle-based, single qubit)
  - [ ] MHRQIB (basis-encoded, multiple qubits)

### 2. Implementation Details
- [ ] **Encoding Process**
  - [ ] How `utils.compute_register()` creates hierarchy
  - [ ] How `angle_map()` / basis encoding works
  - [ ] Lazy vs gate-based upload
  
- [ ] **Denoising**
  - [ ] What the current denoiser actually does (`DENOISER_qiskit`)
  - [ ] Partial Grover diffusion (what it is, not what it could be)
  - [ ] Classical sibling smoothing (`mhrqi_bins_to_image`)
  - [ ] Edge preservation via walker probability
  
- [ ] **Simulation Backends**
  - [ ] Qiskit (main, well-tested)
  - [ ] MQT qudits (experimental)
  - [ ] Decision diagrams (experimental)

### 3. Mathematical Foundations
- [ ] **Verify and rewrite `seam_theory.md`**
  - [ ] Audit claims against actual code
  - [ ] Remove made-up terms
  - [ ] Keep only what's implemented
  - [ ] Plain language rewrites
  
- [ ] **Denoiser math** (`mhrqib_denoiser_math.md`)
  - [ ] Verify formulas match code
  - [ ] Document actual Grover diffusion operator used
  - [ ] No overclaiming
  
- [ ] **Encoder math** (`mhrqib_encoder_math.md`)
  - [ ] Verify encoding formulas
  - [ ] Document statevector structure

### 4. Benchmarking
- [ ] What metrics are calculated
- [ ] How to interpret them
- [ ] Comparison methodology
- [ ] Known limitations

### 5. Usage Guide
- [ ] How to run simulations
- [ ] Command-line parameters
- [ ] Backend selection
- [ ] Output interpretation

---

## Files to Audit

### High Priority - Likely Contains Hallucinations
- [ ] `seam_theory.md` - Mathematical formalism needs code validation
  - Claims about "Theorem 1", "Proposition 2", etc. need verification
  - Check if RX gates actually used for sibling mixing (found: yes, lines 458-459 in circuit_qiskit.py)
  - Check if cross-boundary coupling actually implemented
  
- [ ] `hierarchical_denoising_math.md` - Check against actual denoiser implementation
  
- [ ] `mhrqib_denoiser_math.md` - Verify math matches code

### Medium Priority
- [ ] `mhrqib_encoder_math.md` - Verify encoding formulas
- [ ] `benchmarking.md` - Check consistency with `compare_to.py`

---

## Audit Findings So Far

### seam_theory.md
**Status:** Contains valid concepts BUT overformalized
- ✅ **Real:** Sibling concept exists in `utils.py` (`get_siblings()`)
- ✅ **Real:** Seam-aware smoothing implemented in `mhrqi_bins_to_image()`
- ⚠️ **Overclaimed:** Mathematical theorems not proven in code
- ⚠️ **Overclaimed:** "Hierarchically Consistent Smoothing" formula not implemented
- ❌ **Wrong:** Hadamard mixing for siblings - code uses RX gates instead
- ❌ **Missing:** Controlled cross-boundary mixing operator not in code

**Action:** Rewrite as plain explanation of actual sibling smoothing implementation

### circuit_qiskit.py DENOISER
**Actual implementation:**
```python
# Partial Grover diffusion at each level
for k in range(num_levels):
    qy = pos_regs[2 * k][0]
    qx = pos_regs[2 * k + 1][0]
    level_weight = (k + 1) / num_levels
    angle = diffusion_strength * level_weight
    
    # Partial diffusion using CP (controlled phase)
    qc.h(qy)
    qc.h(qx)
    qc.x(qy)
    qc.x(qx)
    qc.cp(angle, qy, qx)  # Key operation
    qc.x(qy)
    qc.x(qx)
    qc.h(qy)
    qc.h(qx)
```

This is NOT the same as the claimed "S_k sibling coupling" in `seam_theory.md`.

---

## Next Steps

1. Create plain-language rewrites of each doc
2. Remove overclaimed mathematics
3. Document what actually exists
4. Update website narrative to be tame and factual
