# Benchmarking & Metrics

MHRQI is evaluated against state-of-the-art classical image processing methods for OCT speckle denoising.

## Target Scenario

- **No clean reference image** available (real medical imaging scenario)
- **Multiplicative speckle noise** in OCT B-scans
- **No training datasets** required for any metric

## Comparison Baselines

- **BM3D** (Block-Matching and 3D filtering) — Golden standard for Gaussian denoising
- **NL-Means** (Non-Local Means) — Patch-based filter
- **SRAD** (Speckle Reducing Anisotropic Diffusion) — Designed for ultrasound/OCT speckle

---

## Metric Categories

### 1. Speckle Reduction

| Metric | Direction | Description | Citation |
|--------|-----------|-------------|----------|
| **SSI** | Lower ↓ | Speckle Suppression Index (CoV ratio) | Yu & Acton, 2002 |
| **SMPI** | Lower ↓ | Speckle Suppression & Mean Preservation Index | Sattar et al., 2012 |
| **NSF** | Lower ↓ | Noise-Suppression Factor (from OMQDI) | Jagalingam & Hegde, 2021 |
| **ENL** | Higher ↑ | Equivalent Number of Looks (mean²/variance) | Ulaby et al., 1986 |
| **CNR** | Higher ↑ | Contrast-to-Noise Ratio (auto ROI) | Standard |

> [!NOTE]
> SSI can be biased toward aggressive smoothing. ENL and CNR provide balance.

### 2. Structural Similarity

| Metric | Direction | Description | Citation |
|--------|-----------|-------------|----------|
| **EPF** | Higher ↑ | Edge-Preservation Factor (wavelet-based) | Jagalingam & Hegde, 2021 |
| **EPI** | Higher ↑ | Edge Preservation Index (Sobel gradient correlation) | Sattar et al., 1997 |
| **OMQDI** | Higher ↑ | Objective Measure of Quality of Denoised Images | Jagalingam & Hegde, 2021 |

---

## Benchmark Pipeline

The `compare_to.py` script automates evaluation:

1. Load a noisy OCT image
2. Run all denoisers on the same input
3. Calculate metrics across all categories
4. Generate visual reports (excluding noisy original from rankings)

## Statistical Testing

The `statistical_benchmark.py` runs Wilcoxon signed-rank tests across 9 medical images to determine statistically significant differences between methods.
