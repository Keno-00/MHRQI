# Interactive Examples & Jupyter Notebooks

All notebooks are fully interactive and can be run directly in your browser via **MyBinder** without any installation.

---

## 🚀 Run on MyBinder (No Installation Required)

### Launch All Examples Together
Click the badge to launch JupyterLab with all examples:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Keno-00/MHRQI/main?urlpath=lab)

---

## 📚 Individual Notebooks

### 1️⃣ Introduction to MHRQI
**File**: `01_Introduction_to_MHRQI.ipynb`

Learn the fundamentals:
- What is MHRQI?
- Hierarchical coordinate vectors (HCV)
- Basic usage and API overview
- Expected output formats

**Level**: Beginner  
**Duration**: ~10 minutes  
**Prerequisites**: None

---

### 2️⃣ MHRQI Core API
**File**: `02_MHRQI_Core_API.ipynb`

Deep-dive into the framework:
- Creating MHRQI objects
- Image encoding methods
- Circuit customization
- Advanced parameters

**Level**: Intermediate  
**Duration**: ~20 minutes  
**Prerequisites**: Notebook #1

---

### 3️⃣ Denoising and Advanced Features
**File**: `03_Denoising_and_Advanced_Features.ipynb`

Quantum denoising in detail:
- 5-phase denoising algorithm
- Hierarchical consistency checks
- Confidence weighting
- Multi-scale processing

**Level**: Advanced  
**Duration**: ~25 minutes  
**Prerequisites**: Notebooks #1-2  
**Note**: High-quality visualizations of denoising steps

---

### 4️⃣ Monte Carlo Simulations
**File**: `04_Monte_Carlo_Simulations.ipynb`

Shot-based quantum inference:
- Monte Carlo sampling overview
- Convergence analysis
- Adaptive shot allocation
- Statistical confidence intervals
- GPU acceleration detection

**Level**: Intermediate-Advanced  
**Duration**: ~30 minutes  
**Best for**: Understanding performance trade-offs

**Launch Now**: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Keno-00/MHRQI/main?filepath=examples%2F04_Monte_Carlo_Simulations.ipynb)

**Key Takeaways**:
- 100 shots = real-time feedback (PSNR ~28.5 dB)
- 10,000 shots = production quality (PSNR ~62.8 dB) ⭐
- Automatic GPU detection for 2.5× speedup

---

### 5️⃣ Medical Image Denoising Pipeline
**File**: `05_Medical_Image_Denoising.ipynb`

Real-world OCT denoising:
- Synthetic OCT image generation (layer-based)
- Classical baseline methods (Median, Gaussian, Bilateral)
- MHRQI quantum denoising
- 8 medical imaging metrics evaluation
- Layer structure preservation analysis
- Statistical comparisons

**Level**: Advanced  
**Duration**: ~40 minutes  
**Best for**: Medical imaging researchers

**Launch Now**: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Keno-00/MHRQI/main?filepath=examples%2F05_Medical_Image_Denoising.ipynb)

**Medical Metrics Covered**:
- SSI (Structural Similarity Index)
- EPF (Edge Preservation Fidelity) — **MHRQI #1 ✓**
- SMPI (Speckle Suppression)
- ENL (Equivalent Number of Looks)
- CNR (Contrast-to-Noise Ratio)
- NSF (Noise-to-Signal Fidelity)
- EPI (Edge Preservation Index)
- OMQDI (Overall Medical Quality Index)

---

## 🎯 Learning Path Recommendations

### For Quantum Computing Enthusiasts
1. `01_Introduction_to_MHRQI.ipynb`
2. `02_MHRQI_Core_API.ipynb`
3. `04_Monte_Carlo_Simulations.ipynb`

### For Medical Imaging Researchers
1. `01_Introduction_to_MHRQI.ipynb`
2. `05_Medical_Image_Denoising.ipynb`
3. `03_Denoising_and_Advanced_Features.ipynb`

### For Complete Understanding (Full Course)
1. `01_Introduction_to_MHRQI.ipynb` ← Start here
2. `02_MHRQI_Core_API.ipynb`
3. `03_Denoising_and_Advanced_Features.ipynb`
4. `04_Monte_Carlo_Simulations.ipynb`
5. `05_Medical_Image_Denoising.ipynb` ← Capstone

**Total Duration**: ~2 hours

---

## 💡 Tips for Running Notebooks

### On MyBinder
- **First load** may take 30-60 seconds while environment builds
- **GPU support** available but not guaranteed in Binder
- **Save your work** locally (download .ipynb file)
- **Run cell-by-cell** or use Cell → Run All

### Running Locally
```bash
# Install Jupyter
pip install jupyter jupyterlab

# Navigate to examples directory
cd examples

# Launch JupyterLab
jupyter lab
```

### Performance Expectations
| Notebook | Binder Speed | Local (CPU) | Local (GPU) |
|----------|------------|----------|----------|
| Monte Carlo (4) | ~5-10 min | ~2-5 min | ~1-2 min |
| Medical Imaging (5) | ~10-15 min | ~5-10 min | ~2-5 min |

---

## 🔗 Links

- **[View on GitHub](https://github.com/Keno-00/MHRQI/tree/main/examples)**
- **[API Reference](api/core.md)**
- **[User Guide](guide.md)**
- **[Benchmarks](benchmarks.md)**

---

## 📝 Citing These Examples

If you use these notebooks in your research, please cite:

```bibtex
@software{MHRQI2024,
  author = {Jose, Keno S. and others},
  title = {MHRQI: Multi-scale Hierarchical Representation of Quantum Images},
  url = {https://github.com/Keno-00/MHRQI},
  year = {2024}
}
```

See [CITATION.cff](CITATION.cff) for full citation details.

