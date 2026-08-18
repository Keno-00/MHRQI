# Fixed CPU Reference Environment

This environment is the first reproducible reference profile for the ECTI revision.

It is based on the successful disposable Python 3.12 rehearsal. A clean resolver required SciPy 1.13.1 because BM3D 4.0.3 requires BM4D 4.2.5, which requires SciPy 1.13 or newer. Tifffile is pinned to 2024.8.30 because later 2026 releases require NumPy 2.1 or newer, while this profile preserves NumPy 1.26.4. It is not claimed to be the exact historical HPC environment because the historical SRAD repository revision has not been recovered.

Create and verify it on Windows with:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r environment\requirements-lock.txt
.\.venv\Scripts\python.exe scripts\capture_environment.py --output revision_outputs\environment.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The GPU profile will be defined separately for Linux x86_64 and the COARE A100 environment. CPU and GPU results must be compared before the GPU profile is accepted.
