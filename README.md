# MHRQI: Multiscale-Hierarchical Representation of Quantum Images

MHRQI is an undergraduate thesis project developed at the Polytechnic University of the Philippines Manila. The project explores quantum image representation and quantum denoising for medical images, specifically retinal OCT B-scan images affected by speckle noise.

***

## The Core Idea

In medical imaging, speckle noise damages subtle tissue boundaries. When we store images in standard quantum representations, pixels are often treated as flat coordinate registers without spatial hierarchy. 

MHRQI builds a tree structure for the image using a quadtree decomposition. 

1. **Hierarchical Position Encoding**: The image is split into quadrants recursively. Each quadrant level is tracked by position qubits.
2. **Basis-Encoded Intensity**: Grayscale pixel values use an 8-qubit register to store exact discrete brightness levels.
3. **Quantum Denoising**: Quantum circuits evaluate parent-child consistency between spatial blocks. When a child block strongly diverges from its parent summary, the circuit marks it as inconsistent.
4. **Reconstruction**: Classical post-processing applies confidence-weighted smoothing to reconstruct the final denoised image from quantum measurement outcomes.

***

## Current Implementation

The repository contains the complete pipeline for image preparation, circuit generation, simulation, and statistical comparison.

- **Quantum Circuit Construction**: Builds Qiskit quantum circuits for multiscale hierarchical state preparation and consistency checks.
- **Simulation**: Executes circuits using Qiskit Aer statevector simulation or shot-based measurement simulation.
- **Baseline Comparators**: Evaluates denoising quality against classical filters including BM3D, Non-Local Means, and SRAD, as well as deep learning models like SiameseGAN.
- **Statistical Rigor**: Computes non-parametric Wilcoxon signed-rank tests, BCa bootstrap confidence intervals, and Holm p-value adjustments across full image benchmarks.

> [!NOTE]
> **Simulation Only**: All experiments currently run on classical simulation using Qiskit Aer. This codebase has not been run on physical quantum hardware.

***

## Quick Start

### 1. Requirements

- Python 3.12
- Qiskit 2.2+ and Qiskit Aer
- OpenCV, NumPy, SciPy, Matplotlib, scikit-image

### 2. Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/Keno-00/MHRQI.git
cd MHRQI

python -m venv .venv
# On Windows PowerShell
.venv\Scripts\Activate.ps1
# On Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Running the Pipeline

To run the default single-image pipeline:

```bash
python main.py
```

To run the complete statistical benchmark suite:

```bash
python statistical_benchmark.py
```

Configuration parameters such as image dimensions, simulation method, and baseline settings can be adjusted in `configs/paper.ini`.

***

## Repository Structure

```text
MHRQI/
├── circuit.py                # Quantum circuit construction and oracle design
├── main.py                   # Single-run pipeline for encoding and reconstruction
├── utils.py                  # Image processing and quadtree utilities
├── plots.py                  # Visualization and metric generation
├── compare_to.py             # Classical and deep learning baseline filters
├── statistical_benchmark.py   # Batch benchmarking and hypothesis testing
├── configs/
│   └── paper.ini             # Master experiment configuration
├── resources/                # Test datasets and sample images
└── docs/                     # Technical documentation and notes
```

***

## License

This project is licensed under the Apache License 2.0.
