# Benchmarking & Metrics

MHRQI is evaluated against state-of-the-art classical and quantum image processing methods. This guide details the metrics and methodologies used in our comparison framework.

## Comparison Baselines

We compare MHRQI against:
- **BM3D (Block-Matching and 3D filtering):** The "golden standard" for classical denoising.
- **NL-Means (Non-Local Means):** A patch-based filter.
- **SRAD (Speckle Reducing Anisotropic Diffusion):** Specifically designed for ultrasound/OCT speckle noise.
- **FRQI/NEQR:** Traditional quantum image representations (for encoding efficiency comparisons).

## Core Metrics

We use three categories of metrics to provide a holistic view of performance:

### 1. Full-Reference (FR)
Requires a "noise-free" original image.
- **SSIM (Structural Similarity Index):** Measures perceptual degradation.
- **FSIM (Feature Similarity Index):** Heavily weights edges and phase congruency (our primary benchmark).
- **PSNR (Peak Signal-to-Noise Ratio):** Standard logarithmic error measure.

### 2. No-Reference (NR)
Assesses quality without an original image (crucial for real-world medical data).
- **NIQE (Natural Image Quality Evaluator):** Measures "naturalness".
- **PIQE (Perception based Image Quality Evaluator):** Measures local artifacts.

### 3. Speckle-Specific
- **SMPI (Speckle Suppression and Metallicity Preservation Index):** Evaluates how well speckle is removed without losing bright "highlights" (metallicities) in medical scans.
- **SSI (Speckle Suppression Index):** Standard ratio of variance.

## The Benchmark Pipeline

The `compare_to.py` script automates the evaluation:
1. Load a base medical image.
2. Add synthetic noise (Gaussian, Poisson, or Speckle).
3. Run all denoisers on the same noisy input.
4. Calculate metrics across all categories.
5. Generate a side-by-side visual report (`plots.py`).

## How to Read Results

- **FSIM/SSIM:** Closer to **1.0** is better.
- **NIQE/PIQE:** Smaller values indicate higher perceptual quality.
- **SMPI:** Balanced values indicate successful speckle suppression without edge blurring.
