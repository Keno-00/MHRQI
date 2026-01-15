# MHRQI

**Magnitude-Hierarchical Representation of Quantum Images**

An undergraduate thesis project exploring quantum image encoding and denoising for medical imaging applications.

---

## What is MHRQI?

MHRQI is a quantum image representation that combines:
- **Hierarchical position encoding** using a quad-tree structure
- **Basis-encoded intensity** (8-qubit grayscale values)
- **Quantum denoising** via parent-child consistency checks

This approach localizes pixel information in a hierarchical tree rather than flat binary encoding, potentially reducing sensitivity to local perturbations.

## Current Implementation

The codebase includes:
- Quantum circuit construction for encoding grayscale images
- A denoising circuit that marks pixels as consistent/inconsistent with parent blocks
- Classical reconstruction with confidence-weighted smoothing
- Benchmarking against classical methods (BM3D, NL-Means, SRAD)

### Simulation Only

⚠️ This implementation runs on **classical simulation** (Qiskit Aer). It has not been tested on real quantum hardware.

## Quick Start

```bash
# Clone
git clone https://github.com/Keno-00/MHRQI.git
cd MHRQI

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run on a test image
python main.py
```

## Project Structure

```
MHRQI/
├── circuit.py              # Quantum circuit construction
├── main.py                 # Main pipeline
├── utils.py                # Encoding/reconstruction utilities
├── plots.py                # Visualization and metrics
├── compare_to.py           # Classical denoiser comparison
├── statistical_benchmark.py # Batch benchmarking
└── docs/
    ├── knowledge/          # Technical documentation
    └── site/               # Interactive demo website
```

## Documentation

- [Knowledge Base](docs/knowledge/README.md) - Technical documentation
- [Demo Site](docs/site/index.html) - Interactive visualizations

## Requirements

- Python 3.9+
- Qiskit
- NumPy, OpenCV, Matplotlib
- scikit-image, scikit-video
- bm3d, brisque, pypiqe (for benchmarking)

## Author

**Keno S. Jose**  
Undergraduate Thesis Project

## License

Apache 2.0
