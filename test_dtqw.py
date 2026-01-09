"""
Simple test script for DTQW experimental denoiser.

Usage: python test_dtqw.py

This will run a tiny 4x4 image through both:
1. Current hybrid approach (qiskit_mhrqib)
2. Experimental DTQW approach (qiskit_mhrqib_dtqw)

Then compare results.
"""

import cv2
import numpy as np
import main

# Create simple test image (4x4 with noise)
print("Creating 4x4 test image with noise...")
clean = np.array([
    [100, 100, 100, 100],
    [100, 200, 200, 100],
    [100, 200, 200, 100],
    [100, 100, 100, 100]
], dtype=np.uint8)

# Add salt-and-pepper noise
noisy = clean.copy()
noisy[0, 1] = 50   # noise pixel
noisy[2, 2] = 250  # noise pixel

# Save test image
cv2.imwrite('test_4x4.png', noisy)

print("\n" + "="*60)
print("TEST 1: Current Hybrid Approach")
print("="*60)

orig1, recon1, dir1 = main.main(
    shots=1024,
    n=4,
    d=2,
    denoise=True,
    use_shots=False,  # Use statevector for deterministic results
    backend='qiskit_mhrqib',
    fast=True,
    verbose_plots=False,
    img_path='test_4x4.png',
    run_comparison=False
)

print("\n" + "="*60)
print("TEST 2: Experimental DTQW Approach")
print("="*60)

orig2, recon2, dir2 = main.main(
    shots=1024,
    n=4,
    d=2,
    denoise=True,
    use_shots=False,  # Use statevector
    backend='qiskit_mhrqib_dtqw',
    fast=True,
    verbose_plots=False,
    img_path='test_4x4.png',
    run_comparison=False
)

# Compare results
print("\n" + "="*60)
print("COMPARISON")
print("="*60)

print("\nOriginal (noisy):")
print(orig1)

print("\nHybrid approach result:")
print(recon1)

print("\nDTQW experimental result:")
print(recon2)

# Compute simple metrics
def simple_mse(a, b):
    return np.mean((a.astype(float) - b.astype(float)) ** 2)

mse_hybrid = simple_mse(clean, recon1)
mse_dtqw = simple_mse(clean, recon2)

print(f"\nMSE vs clean image:")
print(f"  Hybrid:  {mse_hybrid:.2f}")
print(f"  DTQW:    {mse_dtqw:.2f}")
print(f"  Winner:  {'DTQW' if mse_dtqw < mse_hybrid else 'Hybrid'} (lower is better)")

print("\n" + "="*60)
print("Experiment complete!")
print("="*60)
