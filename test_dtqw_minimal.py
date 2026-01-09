"""
Minimal test for DTQW experimental denoiser - no external dependencies.
Tests if the circuit can be built without errors.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
import sys
sys.path.insert(0, '/home/keno/Documents/source_repos/MHRQI')

import dtqw_experimental

print("="*60)
print("DTQW Experimental Denoiser - Circuit Build Test")
print("="*60)

# Create minimal circuit for 4x4 image (N=4, d=2, L=2 levels)
print("\nCreating quantum circuit...")
N = 4
d = 2
L_max = 2  # log2(4) = 2 levels

# Registers
pos_regs = []
for k in range(L_max):
    pos_regs.append(QuantumRegister(1, f'qy{k}'))
    pos_regs.append(QuantumRegister(1, f'qx{k}'))

intensity_reg = QuantumRegister(8, 'intensity')

qc = QuantumCircuit()
for reg in pos_regs:
    qc.add_register(reg)
qc.add_register(intensity_reg)

print(f"Initial circuit: {qc.num_qubits} qubits")
print(f"Position qubits: {L_max * 2}")
print(f"Intensity qubits: 8")

# Test DTQW denoiser
print("\nApplying DTQW denoiser...")
try:
    qc_dtqw = dtqw_experimental.DTQW_denoiser_experimental(
        qc, pos_regs, intensity_reg, base_steps=2, strength=1.0
    )
    print(f"✓ DTQW circuit built successfully!")
    print(f"Final circuit: {qc_dtqw.num_qubits} qubits (added {qc_dtqw.num_qubits - qc.num_qubits} ancillas)")
    print(f"Circuit depth: {qc_dtqw.depth()}")
    print(f"Gate count: {qc_dtqw.size()}")
    
    # Check ancilla count
    ancilla_count = qc_dtqw.num_qubits - qc.num_qubits
    if ancilla_count <= 2:
        print(f"✓ Ancilla overhead acceptable: {ancilla_count} qubits")
    else:
        print(f"⚠ Warning: {ancilla_count} ancillas (expected ≤2)")
    
    success = True
except Exception as e:
    print(f"✗ DTQW circuit build FAILED:")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
    success = False

# Test bit-serial alternative
print("\n" + "="*60)
print("Testing bit-serial alternative...")
print("="*60)

qc2 = QuantumCircuit()
for reg in pos_regs:
    qc2.add_register(reg)
qc2.add_register(intensity_reg)

try:
    qc_serial = dtqw_experimental.DTQW_denoiser_bitserial(
        qc2, pos_regs, intensity_reg, strength=1.0
    )
    print(f"✓ Bit-serial DTQW circuit built successfully!")
    print(f"Final circuit: {qc_serial.num_qubits} qubits (added {qc_serial.num_qubits - qc2.num_qubits} ancillas)")
    print(f"Circuit depth: {qc_serial.depth()}")
    print(f"Gate count: {qc_serial.size()}")
    
    serial_success = True
except Exception as e:
    print(f"✗ Bit-serial circuit build FAILED:")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
    serial_success = False

print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"Controlled rotation DTQW: {'PASS ✓' if success else 'FAIL ✗'}")
print(f"Bit-serial DTQW:          {'PASS ✓' if serial_success else 'FAIL ✗'}")

if success or serial_success:
    print("\n✓ At least one DTQW approach builds successfully!")
    print("  Next: Test on actual images to see if it improves denoising")
else:
    print("\n✗ Both DTQW approaches failed - needs debugging")

print("="*60)
