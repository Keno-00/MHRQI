# Benchmarking & Metrics

MHRQI is evaluated against state-of-the-art classical image processing methods.

## Comparison Baselines

- **BM3D (Block-Matching and 3D filtering):** The "golden standard" for classical denoising.
- **NL-Means (Non-Local Means):** A patch-based filter.
- **SRAD (Speckle Reducing Anisotropic Diffusion):** Designed for ultrasound/OCT speckle noise.

## Core Metrics

We use three categories of metrics:

### 1. Full-Reference (FR)
Requires a "noise-free" reference image.
- **SSIM (Structural Similarity Index):** Measures perceptual degradation.
- **FSIM (Feature Similarity Index):** Weights edges and phase congruency (primary benchmark).

### 2. No-Reference (NR)
Assesses quality without an original image.
- **NIQE (Natural Image Quality Evaluator):** Measures "naturalness".
- **PIQE (Perception based Image Quality Evaluator):** Measures local artifacts.
- **BRISQUE:** Measures distortion using natural scene statistics.

### 3. Speckle-Specific
- **SMPI (Speckle Suppression and Mean Preservation Index):** Evaluates speckle removal without losing structure.
- **SSI (Speckle Suppression Index):** Standard ratio of variance.
- **DR-IQA:** Degraded Reference IQA combining FSIM and NIQE.

## Benchmark Pipeline

The `compare_to.py` script automates evaluation:
1. Load a medical image (noisy input)
2. Run all denoisers on the same input
3. Calculate metrics across all categories
4. Generate visual reports

## How to Read Results

| Metric | Better Direction |
|--------|-----------------|
| FSIM, SSIM | Closer to **1.0** |
| NIQE, PIQE, BRISQUE | **Lower** values |
| SSI, SMPI | **Lower** values (closer to 0) |
