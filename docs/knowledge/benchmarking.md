# MHRQI Benchmarking

**Comparison methodology and metrics.**

---

## Purpose

The benchmarking pipeline compares MHRQI against established classical denoisers to:
- **Validate functionality** - Ensure MHRQI produces reasonable outputs
- **Understand trade-offs** - Where quantum approach differs from classical
- **Demonstrate methodology** - Show how quantum denoising can be evaluated

**This is NOT for claiming superiority.** We calculate metrics for analysis, not competition.

---

## Compared Methods

### Classical Denoisers

1. **BM3D** (Block-Matching and 3D Filtering)
   - State-of-the-art classical denoiser
   - Uses collaborative filtering in transform domain
   - Typically strongest performer on natural images

2. **NL-Means** (Non-Local Means)
   - Weighted average of similar patches
   - Good for textured images
   - Often used as "clean reference" when no ground truth available

3. **SRAD** (Speckle Reducing Anisotropic Diffusion)
   - PDE-based method for speckle noise
   - Designed for ultrasound/radar images
   - Edge-preserving diffusion

### MHRQI (Proposed)

- Quantum hierarchical diffusion + classical reconstruction
- See [denoiser_guide.md](denoiser_guide.md) for details

---

## Metrics Calculated

### Full Reference
*Requires clean ground truth image*

- **FSIM** (Feature Similarity Index): Phase congruency + gradient similarity  
  Higher is better (0-1)

- **SSIM** (Structural Similarity Index): Luminance, contrast, structure  
  Higher is better (0-1)

### Speckle & Consistency
*Compares output to NOISY input for structure preservation*

- **SSI** (Speckle Suppression Index): Coefficient of variation ratio  
  Lower is better (measures noise reduction in flat regions)

- **SMPI** (Speckle Mean Preservation Index): Mean preservation + variance reduction  
  Lower is better

- **DR-IQA** (Degraded Reference IQA): FSIM × (1/(1+NIQE))  
  Higher is better (balances fidelity + naturalness)

### No Reference
*Perceptual quality without any reference*

- **NIQE** (Naturalness Image Quality Evaluator): Statistical naturalness  
  Lower is better

- **PIQE** (Perception-based IQE): Block distortion estimate  
  Lower is better

- **BRISQUE** (Blind/Referenceless IQE): SVM-based quality score  
  Lower is better

---

## Running Benchmarks

```bash
python compare_to.py
```

This will:
1. Load test image (default: `resources/drusen1.jpeg`)
2. Apply all denoisers
3. Calculate all metrics
4. Generate comparison plots and tables in `runs/<timestamp>/evals/`

---

## Output Structure

```
runs/<timestamp>/evals/
├── full_reference_summary.png    # FSIM, SSIM table + images
├── speckle_summary.png           # SSI, SMPI, DR-IQA table + images
├── no_reference_summary.png      # NIQE, PIQE, BRISQUE table + images
└── denoised_*.png               # Individual results
```

Each summary includes:
- Reference image (if applicable)
- All competitor images side-by-side
- Metrics table with rankings (1=best, N=worst)

---

## ROI Selection

For speckle metrics (SSI, SMPI), a homogeneous region is auto-selected:

```python
def auto_homogeneous_roi(img, win=20, stride=10):
    # Slide 20×20 window
    # Find region with lowest variance (most uniform)
    # Return coordinates for metric calculation
```

This ensures fair comparison in noise-free areas.

---

## Interpretation

### What Metrics Mean

**FSIM, SSIM** → How similar to reference?  
**SSI, SMPI** → How much noise reduced?  
**DR-IQA** → Balance of fidelity + naturalness  
**NIQE, PIQE, BRISQUE** → Does it look natural?

### Caveats

- **No single "best" metric** - Different metrics favor different aspects
- **Reference matters** - Full-ref metrics depend on chosen reference
- **Context-dependent** - Speckle metrics only relevant for speckle noise
- **No ground truth** - Real OCT images have no "clean" version

---

## Code Location

**Benchmark script:**
- `compare_to.py` - Main comparison logic

**Metric implementations:**
- `plots.py:246-486` - All metric functions

**Denoiser wrappers:**
- `compare_to.py:40-54` - BM3D, NL-Means, SRAD

---

## Methodology Notes

### Reference Image Selection

When no ground truth exists (real images):
- **Use NL-Means output** as "degraded reference"
- Calculate DR-IQA using this reference
- Full-ref metrics compare ALL methods to this reference

### Ranking

Rankings are:
- **Per-metric** (not overall score)
- **Within denoisers only** (exclude "Original" from ranking)
- **1 = best, N = worst** for that specific metric

---

## Summary

Benchmarking demonstrates MHRQI's functionality and characteristics relative to established methods. Metrics are calculated for analysis and understanding, not for performance claims.

Different metrics capture different aspects of denoising quality - no single number tells the whole story.
