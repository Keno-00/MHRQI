# MHRQI - Magnitude Hierarchical Representation of Quantum Images

**Author**: Keno S. Jose  
**License**: Apache 2.0

A quantum image processing framework that encodes images using hierarchical position qubits and applies quantum denoising.

---

## What is MHRQI?

MHRQI encodes grayscale images into quantum circuits using:

1. **Hierarchical position encoding** - Pixels organized in a quad-tree structure using qubit levels
2. **Basis-encoded intensities** - 8-bit pixel values stored directly in computational basis states
3. **Quantum denoising** - Grover-like diffusion operators applied at each hierarchy level

This approach enables multi-scale image processing where coarse and fine details are naturally separated by the hierarchy.

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd MHRQI

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- Qiskit >= 1.0
- Qiskit Aer (GPU support recommended)
- OpenCV, NumPy, Matplotlib
- For benchmarking: BM3D, SRAD, scikit-video

---

## Quick Start

```python
from main import main

# Run MHRQI with quantum denoising
original, reconstructed, run_dir = main(
    n=64,                 # Image size (64x64)
    d=2,                  # Qubit-based hierarchy
    denoise=True,         # Apply quantum denoiser
    use_shots=False,      # Statevector (exact simulation)
    backend='qiskit_mhrqib',
    fast=True,            # Use statevector initialization
    verbose_plots=True,
    run_comparison=True   # Compare with BM3D, NL-Means, SRAD
)
```

Results saved to `runs/<timestamp>/`

---

## How It Works

### 1. Hierarchical Encoding

For an N×N image where N = 2^L:
- **Position**: 2L qubits encode pixel coordinates hierarchically
- **Intensity**: 8 qubits store grayscale value (0-255) in binary

**Example** (4×4 image, L=2):
- Pixel (0,0) → qubits: `|0000⟩` (position) + `|intensity⟩`
- Pixel (3,3) → qubits: `|1111⟩` (position) + `|intensity⟩`

### 2. Quantum-Assisted Denoising

**Hybrid approach** combining quantum and classical processing:

**Quantum component:**
- Applies partial Grover diffusion at each hierarchy level
- Creates probability distribution indicating flat vs edge regions
- Hierarchical structure naturally encoded in circuit

**Classical component:**
- Uses probability distribution for edge detection
- Performs sibling-based smoothing in flat regions
- Preserves edges via adaptive weighting

**Note:** This is quantum-assisted classical filtering, not pure quantum denoising. Intensity values (basis-encoded) are smoothed classically using quantum-derived edge information.

### 3. Measurement and Reconstruction

- **Statevector mode**: Direct amplitude extraction (exact)
- **Shot-based mode**: Statistical reconstruction from measurements

---

## Project Structure

```
MHRQI/
├── main.py              # Pipeline entry point
├── circuit_qiskit.py    # Qiskit implementation (primary)
├── utils.py             # Hierarchy encoding, image reconstruction
├── plots.py             # Visualization and metrics
├── compare_to.py        # Classical benchmark comparison
├── docs/
│   ├── knowledge/       # Technical documentation
│   └── site/            # Website files
└── runs/                # Output directory
```

---

## Benchmarking

Compare MHRQI against classical denoisers:

```bash
python compare_to.py
```

### Compared Methods
- **BM3D** - Block-matching 3D filtering
- **NL-Means** - Non-local means
- **SRAD** - Speckle reducing anisotropic diffusion

### Metrics Calculated

| Category | Metrics | Purpose |
|----------|---------|---------|
| Full Reference | FSIM, SSIM | Similarity to reference |
| No Reference | NIQE, PIQE, BRISQUE | Perceptual quality |
| Speckle Consistency | SSI, SMPI | Noise suppression |

**Note:** Benchmarks are for methodology demonstration, not performance claims.

---

## Documentation

- **[Knowledge Base](/docs/knowledge/README.md)** - Technical guides and implementation details
- **[Website](/docs/site/)** - Interactive visualization and demos.
can be found locally at `docs/site/index.html` or by going here: [kenojose.site](https://kenojose.site)

---

## Configuration

Key parameters in `main.py`:

| Parameter | Values | Description |
|-----------|--------|-------------|
| `backend` | `'qiskit_mhrqib'` (primary) | Simulation backend |
| `n` | 4, 8, 16, 32, 64, 128 | Image dimension (n×n) |
| `d` | 2 (qubits) | Hierarchy base |
| `denoise` | `True`, `False` | Enable quantum denoising |
| `fast` | `True` | Use statevector init (vs gates) |
| `use_shots` | `True`, `False` | Shot-based or statevector measurement |

---

## License

This project is licensed under the **Apache License 2.0**.

See [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@software{jose2026mhrqi,
  title={MHRQI: Magnitude Hierarchical Representation of Quantum Images},
  author={Jose, Keno S.},
  year={2026},
  license={Apache-2.0}
}
```

---

## Acknowledgments

This project demonstrates quantum image processing using hierarchical quantum structures for multi-scale analysis and denoising.
