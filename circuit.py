"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MHRQI - Magnitude Hierarchical Representation of Quantum Images             ║
║  Qiskit Implementation                                                       ║
║                                                                              ║
║  Author: Keno-00                                                             ║
║  License: Apache 2.0                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import itertools  # Added for dynamic parent state generation
import random  # TODO: Remove if confirmed unused after testing
from collections import defaultdict

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import Aer, AerSimulator
from qiskit_aer.library import SetStatevector

import utils


# -------------------------
# Helper primitives
# -------------------------
def _prepare_controls_on_states(qc: QuantumCircuit, controls, ctrl_states):
    """Flip controls where ctrl_state == 0 so that all required controls are 1."""
    for q, s in zip(controls, ctrl_states):
        if int(s) == 0:
            qc.x(q)


def _restore_controls(qc: QuantumCircuit, controls, ctrl_states):
    for q, s in zip(controls, ctrl_states):
        if int(s) == 0:
            qc.x(q)


def apply_multi_controlled_ry(qc, controls, ctrl_states, target, ancilla_for_and, work_ancillas, angle):
    _prepare_controls_on_states(qc, controls, ctrl_states)

    try:
        if len(controls) == 0:
            qc.ry(angle, target)
            return

        if len(controls) == 1:
            # Single control: use controlled RY directly
            qc.cry(angle, controls[0], target)
            return

        qc.reset(ancilla_for_and)

        qc.mcx(controls, ancilla_for_and)
        qc.cry(angle, ancilla_for_and, target)
        qc.mcx(controls, ancilla_for_and)  # Uncompute

        qc.reset(ancilla_for_and)

    finally:
        _restore_controls(qc, controls, ctrl_states)

# -------------------------
# Circuit construction
# -------------------------

def MHRQI_init(d, L_max, bit_depth=8):
    """
    Unified MHRQI circuit initialization.
    
    Creates a quantum circuit with:
    - Position qubits (2*L_max): Hierarchical position encoding
    - Intensity qubits (bit_depth): Basis-encoded grayscale values
    - Bias qubit (1): Smoothness indicator for denoising
    - Work qubits (2): Gradient computation
    
    Args:
        d: dimension (2 for images)
        L_max: hierarchy depth (log2 of image side)
        bit_depth: bits for intensity encoding (default 8 for 0-255)
    
    Returns:
        qc: QuantumCircuit
        pos_regs: list of position qubit registers
        intensity_reg: register containing intensity qubits
        bias_qubit: single bias qubit for denoising
    """
    pos_qubits = []

    # Position qubits (2 per level: y and x)
    for k in range(L_max):
        pos_qubits.append(QuantumRegister(1, f"q_y_{k}"))
        pos_qubits.append(QuantumRegister(1, f"q_x_{k}"))

    # Intensity qubits (basis-encoded grayscale)
    intensity = QuantumRegister(bit_depth, 'intensity')

    # Bias qubit for denoising (hit/miss weighting)
    bias = QuantumRegister(1, 'bias')

    # Work qubits for gradient computation and multi-controlled operations
    # work[0] is used as and_ancilla during upload
    work = QuantumRegister(2, 'work')

    qc = QuantumCircuit(*pos_qubits, intensity, bias, work)

    # Initialize position qubits in uniform superposition
    for reg in pos_qubits:
        qc.h(reg[0])

    # Bias qubit starts in |0⟩ (marked during denoising)

    return qc, pos_qubits, intensity, bias


def MHRQI_upload(qc: QuantumCircuit, pos_regs, intensity_reg, d, hierarchy_matrix, img):
    """
    Upload intensity values using MHRQI(basis) encoding.
    Each pixel's intensity is encoded in binary across intensity_reg qubits.
    
    Args:
        qc: QuantumCircuit
        pos_regs: position qubit registers
        intensity_reg: intensity qubit register (multiple qubits)
        d: dimension
        hierarchy_matrix: matrix of position states
        img: normalized image (0-1 range will be scaled to 0-255)
    """
    controls = [reg[0] for reg in pos_regs]
    intensity_qubits = list(intensity_reg)
    bit_depth = len(intensity_qubits)

    # Get work register for ancilla
    and_ancilla = None
    for r in qc.qregs:
        if r.name == 'work':
            and_ancilla = r[0]
            break

    # For each pixel position
    for vec in hierarchy_matrix:
        ctrl_states = list(vec)
        r, c = utils.compose_rc(vec, d)

        # Get pixel intensity and convert to integer (0-255)
        pixel_value = float(img[r, c])
        # Assuming img is normalized to [0, 1], scale to [0, 2^bit_depth - 1]
        intensity_int = int(pixel_value * (2**bit_depth - 1))

        # Convert intensity to binary representation
        intensity_bits = format(intensity_int, f'0{bit_depth}b')

        # Apply X gates to set intensity qubits based on binary representation
        _prepare_controls_on_states(qc, controls, ctrl_states)

        if len(controls) > 0:
            # Multi-controlled operations for each intensity bit
            qc.mcx(controls, and_ancilla)

            for bit_idx, bit_val in enumerate(intensity_bits):
                if bit_val == '1':
                    qc.cx(and_ancilla, intensity_qubits[bit_idx])

            qc.mcx(controls, and_ancilla)
        else:
            # No position control needed
            for bit_idx, bit_val in enumerate(intensity_bits):
                if bit_val == '1':
                    qc.x(intensity_qubits[bit_idx])

        _restore_controls(qc, controls, ctrl_states)

    return qc

# -------------------------
# Lazy Upload (Faster)
# -------------------------


def MHRQI_lazy_upload(qc: QuantumCircuit, pos_regs, intensity_reg, d, hierarchy_matrix, img):
    """
    Fast MHRQI(basis) upload using direct statevector initialization.
    """
    # finalized! dont change!
    qubit_to_idx = {q: i for i, q in enumerate(qc.qubits)}
    pos_indices = [qubit_to_idx[reg[0]] for reg in pos_regs]
    intensity_indices = [qubit_to_idx[q] for q in intensity_reg]
    num_qubits = qc.num_qubits
    dim = 2 ** num_qubits
    bit_depth = len(intensity_indices)
    state = np.zeros(dim, dtype=complex)
    num_pos = 2 ** len(pos_indices)
    all_indices = pos_indices + intensity_indices
    is_sequential = all(all_indices[i] == i for i in range(len(all_indices)))
    if is_sequential:
        for vec in hierarchy_matrix:
            p = 0
            for i, val in enumerate(vec):
                if val:
                    p |= (1 << i)
            r, c = utils.compose_rc(vec, d)
            pixel_value = float(img[r, c])
            intensity_int = int(pixel_value * (2**bit_depth - 1))
            base_idx = p
            intensity_offset = 0
            for bit_idx in range(bit_depth):
                if (intensity_int >> bit_idx) & 1:
                    intensity_offset |= (1 << (len(pos_indices) + bit_idx))
            state[base_idx + intensity_offset] = 1.0
    else:
        print("Warn: Qubits not sequential, falling back to gate-based upload.")
        return MHRQI_upload(qc, pos_regs, intensity_reg, d, hierarchy_matrix, img)
    state_norm = np.linalg.norm(state)
    if state_norm > 0:
        state = state / state_norm
    qc.append(SetStatevector(state), qc.qubits)
    return qc


def _adjust_brightness(qc, intensity_reg, k):
    """
    Add constant k to intensity register (reversible).
    To subtract, pass negative k (uses two's complement internally).
    
    This is a simplified constant adder using XOR cascades.
    For brightness shift: subtract before denoise, add after.
    
    Args:
        qc: QuantumCircuit to apply gates to
        intensity_reg: intensity qubit register
        k: constant to add (negative to subtract)
    """
    n = len(list(intensity_reg))
    intensity_qubits = list(intensity_reg)

    # Handle negative k via two's complement
    if k < 0:
        k = ((~(-k)) + 1) & ((1 << n) - 1)
    else:
        k = k & ((1 << n) - 1)

    if k == 0:
        return

    # Simplified constant addition using carry propagation
    # For each bit position where k has a 1, we need to flip and propagate carry
    # Using ripple-carry style: X on bit i, then MCX for carry propagation

    for i in range(n):
        if (k >> i) & 1:
            # Add 1 at position i
            if i == 0:
                qc.x(intensity_qubits[0])
            else:
                # Flip bit i, propagate carry from lower bits
                # Carry in = all lower bits that were 1 before flip will generate carry
                # Simplified: just flip current bit, MCX from all lower to current
                controls = intensity_qubits[:i]
                qc.mcx(controls, intensity_qubits[i])
                # Then flip bit i unconditionally
                qc.x(intensity_qubits[i])


def DENOISER(qc: QuantumCircuit, pos_regs, intensity_reg, bias=None, brightness_shift=10):
    denoise_qc = QuantumCircuit(*qc.qregs)
    num_levels = len(pos_regs) // 2

    #========================================
    # Ancilla allocation
    #========================================

    work_qubits = []
    for r in denoise_qc.qregs:
        if r.name == 'work':
            work_qubits = list(r)
            break

    if len(work_qubits) < 2 or bias is None:
        print("WARNING: Insufficient ancillas")
        return qc

    parent_avg_ancilla = work_qubits[0]
    consistency_ancilla = work_qubits[1]
    bias_qubit = bias[0]
    intensity_qubits = list(intensity_reg)

    # ==========================================
    # CHECK FINEST LEVEL VS PARENT
    # ==========================================

    finest_level = num_levels - 1

    if finest_level == 0:
        denoise_qc.x(bias_qubit)
        qc.compose(denoise_qc, inplace=True)
        return qc

    qy_fine = pos_regs[2 * finest_level][0]
    qx_fine = pos_regs[2 * finest_level + 1][0]

    # === Brightness shift (subtract before denoise) ===
    if brightness_shift != 0:
        _adjust_brightness(denoise_qc, intensity_reg, -brightness_shift)

    # === Sibling superposition ===
    # DO NOT CHANGE!
    denoise_qc.h(qy_fine) # superposition (Fine.1.1)
    denoise_qc.h(qx_fine) # superposition (Fine.1.2)

    # === Parent average encoding ===
    # DO NOT CHANGE!

    intensity_msb = intensity_qubits[-1]
    intensity_msb_1 = intensity_qubits[-2] if len(intensity_qubits) > 1 else intensity_msb
    intensity_msb_2 = intensity_qubits[-3] if len(intensity_qubits) > 2 else intensity_msb_1
    intensity_msb_3 = intensity_qubits[-4] if len(intensity_qubits) > 3 else intensity_msb_2
    intensity_msb_4 = intensity_qubits[-5] if len(intensity_qubits) > 4 else intensity_msb_3
    intensity_msb_5 = intensity_qubits[-6] if len(intensity_qubits) > 5 else intensity_msb_4

    # Rotate ancilla based on intensity
    # DO NOT CHANGE!
    denoise_qc.x(intensity_msb) # not (MSB0.1)
    denoise_qc.x(intensity_msb_1) # not (MSB1.1)
    denoise_qc.x(intensity_msb_2) # not (MSB2.1)
    denoise_qc.x(intensity_msb_3) # not (MSB3.1)
    denoise_qc.x(intensity_msb_4) # not (MSB4.1)
    denoise_qc.x(intensity_msb_5) # not (MSB5.1)


    denoise_qc.cry(np.pi / 8, intensity_msb, parent_avg_ancilla) # rotate (MSB0.1)
    denoise_qc.cry(np.pi / 4, intensity_msb_1, parent_avg_ancilla) # rotate (MSB1.1)
    denoise_qc.cry(np.pi / 2, intensity_msb_2, parent_avg_ancilla) # rotate (MSB2.1)
    denoise_qc.cry(np.pi, intensity_msb_3, parent_avg_ancilla) # rotate (MSB3.1)
    denoise_qc.cry(np.pi * 2, intensity_msb_4, parent_avg_ancilla) # rotate (MSB4.1)
    denoise_qc.cry(np.pi * 4, intensity_msb_5, parent_avg_ancilla) # rotate (MSB5.1)

    denoise_qc.x(intensity_msb) # not_dag (MSB0.1)
    denoise_qc.x(intensity_msb_1) # not_dag (MSB1.1)
    denoise_qc.x(intensity_msb_2) # not_dag (MSB2.1)
    denoise_qc.x(intensity_msb_3) # not_dag (MSB3.1)
    denoise_qc.x(intensity_msb_4) # not_dag (MSB4.1)
    denoise_qc.x(intensity_msb_5) # not_dag (MSB5.1)

    # === Uncompute sibling superposition ===
    # DO NOT CHANGE!
    denoise_qc.h(qx_fine) # superposition_dag (Fine.2.2)
    denoise_qc.h(qy_fine) # superposition_dag (Fine.2.1)

    # Now parent_avg_ancilla holds information about parent block
    # And we're back to single pixel state

    # === Compare pixel to parent average ===
    # DO NOT CHANGE!
    # If pixel intensity matches parent_avg_ancilla → consistent
    # If different → inconsistent

    # Check if MSB matches parent average state
    # If parent_avg is in |1⟩ and pixel MSB is 1 → match
    # If parent_avg is in |0⟩ and pixel MSB is 0 → match

    # CNOT from MSB to consistency_ancilla, controlled on parent_avg
    # This marks matches
    # DO NOT CHANGE!

    denoise_qc.x(parent_avg_ancilla) # not
    denoise_qc.ccx(intensity_msb, parent_avg_ancilla, consistency_ancilla) #ccnot (PARENT.1)
    denoise_qc.x(parent_avg_ancilla) # not_dag


    denoise_qc.x(intensity_msb) # not
    denoise_qc.ccx(intensity_msb, parent_avg_ancilla, consistency_ancilla) #ccnot (PARENT.2)
    denoise_qc.x(parent_avg_ancilla) # not_dag
    denoise_qc.x(intensity_msb) # not_dag

    # === Set bias ===
    # If consistency_ancilla = 1 → consistent → preserve
    denoise_qc.x(consistency_ancilla) # not
    denoise_qc.cx(consistency_ancilla, bias_qubit) #cnot

    # === Uncompute ===
    # Uncompute consistency check (XNOR)
    denoise_qc.x(intensity_msb) # not (MSB0.2)
    denoise_qc.x(parent_avg_ancilla) # not (PARENT.2)
    denoise_qc.ccx(intensity_msb, parent_avg_ancilla, consistency_ancilla) #ccnot_dag (PARENT.2)
    denoise_qc.x(parent_avg_ancilla) # not_dag (PARENT.2)
    denoise_qc.x(intensity_msb) # not_dag (MSB0.2)

    denoise_qc.ccx(intensity_msb, parent_avg_ancilla, consistency_ancilla) #ccnot_dag (PARENT.1)

    # Uncompute parent average encoding
    denoise_qc.h(qy_fine) # superposition_dag (Fine.2.1)
    denoise_qc.h(qx_fine) # superposition_dag (Fine.2.2)

    denoise_qc.x(intensity_msb) # not (MSB0.2)
    denoise_qc.x(intensity_msb_1) # not (MSB1.2)
    denoise_qc.x(intensity_msb_2) # not (MSB2.2)
    denoise_qc.x(intensity_msb_3) # not (MSB3.2)
    denoise_qc.x(intensity_msb_4) # not (MSB4.2)
    denoise_qc.x(intensity_msb_5) # not (MSB5.2)

    denoise_qc.cry(-np.pi / 8, intensity_msb, parent_avg_ancilla) # rotate_dag (MSB0.1)
    denoise_qc.cry(-np.pi / 4, intensity_msb_1, parent_avg_ancilla) # rotate_dag (MSB1.1)
    denoise_qc.cry(-np.pi / 2, intensity_msb_2, parent_avg_ancilla) # rotate_dag (MSB2.1)
    denoise_qc.cry(-np.pi, intensity_msb_3, parent_avg_ancilla) # rotate_dag (MSB3.1)
    denoise_qc.cry(-np.pi * 2, intensity_msb_4, parent_avg_ancilla) # rotate_dag (MSB4.1)
    denoise_qc.cry(-np.pi * 4, intensity_msb_5, parent_avg_ancilla) # rotate_dag (MSB5.1)

    denoise_qc.x(intensity_msb) # not_dag (MSB0.2)
    denoise_qc.x(intensity_msb_1) # not_dag (MSB1.2)
    denoise_qc.x(intensity_msb_2) # not_dag (MSB2.2)
    denoise_qc.x(intensity_msb_3) # not_dag (MSB3.2)
    denoise_qc.x(intensity_msb_4) # not_dag (MSB4.2)
    denoise_qc.x(intensity_msb_5) # not_dag (MSB5.2)

    denoise_qc.h(qx_fine) # superposition_dag (Fine.1.2)
    denoise_qc.h(qy_fine) # superposition_dag (Fine.1.1)



    #denoise_qc.ry(np.pi/8, bias_qubit) # rotate_dag (Bias)

    # === Brightness shift (add back after denoise) ===
    if brightness_shift != 0:
        _adjust_brightness(denoise_qc, intensity_reg, -brightness_shift)

    qc.compose(denoise_qc, inplace=True)

    return qc


def ACCIDENT_DISCOVERY(qc: QuantumCircuit, pos_regs, intensity_reg, bias=None, ):
    denoise_qc = QuantumCircuit(*qc.qregs)
    num_levels = len(pos_regs) // 2

    #========================================
    # Ancilla allocation
    #========================================

    work_qubits = []
    for r in denoise_qc.qregs:
        if r.name == 'work':
            work_qubits = list(r)
            break

    if len(work_qubits) < 2 or bias is None:
        print("WARNING: Insufficient ancillas")
        return qc

    parent_avg_ancilla = work_qubits[0]
    consistency_ancilla = work_qubits[1]
    bias_qubit = bias[0]
    intensity_qubits = list(intensity_reg)

    # ==========================================
    # CHECK FINEST LEVEL VS PARENT
    # ==========================================

    finest_level = num_levels - 1

    if finest_level == 0:
        denoise_qc.x(bias_qubit)
        print(denoise_qc.draw(output='text'))
        qc.compose(denoise_qc, inplace=True)
        return qc

    qy_fine = pos_regs[2 * finest_level][0]
    qx_fine = pos_regs[2 * finest_level + 1][0]

    # === Sibling superposition ===
    # DO NOT CHANGE!
    denoise_qc.h(qy_fine) # superposition (Fine.1.1)
    denoise_qc.h(qx_fine) # superposition (Fine.1.2)

    # === Parent average encoding ===
    # DO NOT CHANGE!

    intensity_msb = intensity_qubits[-1]
    intensity_msb_1 = intensity_qubits[-2] if len(intensity_qubits) > 1 else intensity_msb
    intensity_msb_2 = intensity_qubits[-3] if len(intensity_qubits) > 2 else intensity_msb_1
    intensity_msb_3 = intensity_qubits[-4] if len(intensity_qubits) > 3 else intensity_msb_2

    # Rotate ancilla based on intensity
    # DO NOT CHANGE!
    # denoise_qc.x(intensity_msb) # not (MSB0.1)
    # denoise_qc.x(intensity_msb_1) # not (MSB1.1)
    # denoise_qc.x(intensity_msb_2) # not (MSB2.1)
    # denoise_qc.x(intensity_msb_3) # not (MSB3.1)


    denoise_qc.cry(np.pi / 8, intensity_msb, parent_avg_ancilla) # rotate (MSB0.1)
    denoise_qc.cry(np.pi / 4, intensity_msb_1, parent_avg_ancilla) # rotate (MSB1.1)
    denoise_qc.cry(np.pi / 2, intensity_msb_2, parent_avg_ancilla) # rotate (MSB2.1)
    denoise_qc.cry(np.pi, intensity_msb_3, parent_avg_ancilla) # rotate (MSB3.1)

    # denoise_qc.x(intensity_msb) # not_dag (MSB0.1)
    # denoise_qc.x(intensity_msb_1) # not_dag (MSB1.1)
    # denoise_qc.x(intensity_msb_2) # not_dag (MSB2.1)
    # denoise_qc.x(intensity_msb_3) # not_dag (MSB3.1)

    # === Uncompute sibling superposition ===
    # DO NOT CHANGE!
    denoise_qc.h(qx_fine) # superposition_dag (Fine.2.2)
    denoise_qc.h(qy_fine) # superposition_dag (Fine.2.1)

    # Now parent_avg_ancilla holds information about parent block
    # And we're back to single pixel state

    # === Compare pixel to parent average ===
    # DO NOT CHANGE!
    # If pixel intensity matches parent_avg_ancilla → consistent
    # If different → inconsistent

    # Check if MSB matches parent average state
    # If parent_avg is in |1⟩ and pixel MSB is 1 → match
    # If parent_avg is in |0⟩ and pixel MSB is 0 → match

    # CNOT from MSB to consistency_ancilla, controlled on parent_avg
    # This marks matches
    # DO NOT CHANGE!

    denoise_qc.x(parent_avg_ancilla) # not
    denoise_qc.ccx(intensity_msb, parent_avg_ancilla, consistency_ancilla) #ccnot (PARENT.1)
    denoise_qc.x(parent_avg_ancilla) # not_dag


    denoise_qc.x(intensity_msb) # not
    denoise_qc.ccx(intensity_msb, parent_avg_ancilla, consistency_ancilla) #ccnot (PARENT.2)
    denoise_qc.x(parent_avg_ancilla) # not_dag
    denoise_qc.x(intensity_msb) # not_dag

    # === Set bias ===
    # If consistency_ancilla = 1 → consistent → preserve
    denoise_qc.cx(consistency_ancilla, bias_qubit) #cnot

    # === Uncompute ===
    # Uncompute consistency check (XNOR)
    denoise_qc.x(intensity_msb) # not (MSB0.2)
    denoise_qc.x(parent_avg_ancilla) # not (PARENT.2)
    denoise_qc.ccx(intensity_msb, parent_avg_ancilla, consistency_ancilla) #ccnot_dag (PARENT.2)
    denoise_qc.x(parent_avg_ancilla) # not_dag (PARENT.2)
    denoise_qc.x(intensity_msb) # not_dag (MSB0.2)

    denoise_qc.ccx(intensity_msb, parent_avg_ancilla, consistency_ancilla) #ccnot_dag (PARENT.1)

    # Uncompute parent average encoding
    denoise_qc.h(qy_fine) # superposition_dag (Fine.2.1)
    denoise_qc.h(qx_fine) # superposition_dag (Fine.2.2)

    # denoise_qc.x(intensity_msb) # not (MSB0.2)
    # denoise_qc.x(intensity_msb_1) # not (MSB1.2)
    # denoise_qc.x(intensity_msb_2) # not (MSB2.2)
    # denoise_qc.x(intensity_msb_3) # not (MSB3.2)


    denoise_qc.cry(-np.pi / 8, intensity_msb, parent_avg_ancilla) # rotate_dag (MSB0.1)
    denoise_qc.cry(-np.pi / 4, intensity_msb_1, parent_avg_ancilla) # rotate_dag (MSB1.1)
    denoise_qc.cry(-np.pi / 2, intensity_msb_2, parent_avg_ancilla) # rotate_dag (MSB2.1)
    denoise_qc.cry(-np.pi, intensity_msb_3, parent_avg_ancilla) # rotate_dag (MSB3.1)

    # denoise_qc.x(intensity_msb) # not_dag (MSB0.2)
    # denoise_qc.x(intensity_msb_1) # not_dag (MSB1.2)
    # denoise_qc.x(intensity_msb_2) # not_dag (MSB2.2)
    # denoise_qc.x(intensity_msb_3) # not_dag (MSB3.2)

    denoise_qc.h(qx_fine) # superposition_dag (Fine.1.2)
    denoise_qc.h(qy_fine) # superposition_dag (Fine.1.1)

    # NOTE: This circuit behaves as a Computational Basis State 8bit to 4bit quantizer
    #       and potentially as a Basis-to-Angle Encoding converter/transcoder.

    qc.compose(denoise_qc, inplace=True)

    return qc


# -------------------------
# Simulation helpers
# -------------------------

def simulate_statevector(qc: QuantumCircuit, use_gpu=True):
    backend = Aer.get_backend('statevector_simulator',device='GPU' if use_gpu else 'CPU')
    transpiled = transpile(qc, backend)
    job = backend.run(transpiled)
    result = job.result()
    return result.get_statevector()


def simulate_counts(qc: QuantumCircuit, shots=1024, use_gpu=True):
    pos_qubits = []
    for reg in qc.qregs:
        if reg.name.startswith('q_y_') or reg.name.startswith('q_x_'):
            pos_qubits.extend(reg)

    qubits_to_measure = []
    qubits_to_measure.extend(pos_qubits)

    # Measure all intensity qubits
    intensity_reg = None
    for reg in qc.qregs:
        if reg.name == 'intensity':
            intensity_reg = reg
            break
    if intensity_reg is not None:
        qubits_to_measure.extend(list(intensity_reg))

    # Measure bias qubit if present
    bias_reg = None
    for reg in qc.qregs:
        if reg.name == 'bias':
            bias_reg = reg
            break
    if bias_reg is not None:
        qubits_to_measure.extend(list(bias_reg))

    creg = ClassicalRegister(len(qubits_to_measure), 'c')
    qc_measure = qc.copy()
    qc_measure.add_register(creg)
    qc_measure.measure(qubits_to_measure, creg)

    # Choose backend: try GPU AerSimulator if requested, else CPU qasm_simulator.
    if use_gpu:
        try:
            # Enable cuStateVec for accelerated statevector simulation
            backend = AerSimulator(method='statevector', device='GPU', cuStateVec_enable=True)
        except Exception as e:
            # Fallback if GPU backend is not available in this environment
            print(f"GPU backend error: {e}. Falling back to CPU.")
            backend = Aer.get_backend('qasm_simulator')
    else:
        backend = Aer.get_backend('qasm_simulator')

    transpiled = transpile(qc_measure, backend)
    result = backend.run(transpiled, shots=shots).result()
    return result.get_counts()


# -------------------------
# Binning helpers
# -------------------------


def make_bins_counts(counts, hierarchy_matrix, bit_depth=8, denoise=False):
    """
    Create bins from measurement counts for MHRQI(basis) encoding.
    Reconstructs intensity values from multiple qubit measurements.
    
    Args:
        counts: measurement results from Qiskit
        hierarchy_matrix: position state matrix
        bit_depth: number of intensity qubits
        denoise: if True, also extract bias statistics
    
    Returns:
        bins: dict mapping position to intensity statistics
        bias_stats: (optional) dict mapping position to bias statistics
    """
    bins = defaultdict(lambda: {"intensity_sum": 0, "count": 0, "intensity_squared_sum": 0})
    bias_stats = defaultdict(lambda: {
        "hit": 0, "miss": 0,
        "intensity_hit": 0, "intensity_miss": 0
    }) if denoise else None

    pos_len = len(hierarchy_matrix[0])

    for bitstring, count in counts.items():
        b = bitstring[::-1]  # Reverse for little-endian

        # Qubit layout: [position] [intensity] [bias if denoise]
        expected_len = pos_len + bit_depth + (1 if denoise else 0)

        if len(b) < expected_len:
            continue

        # Extract position bits
        pos_bits = tuple(int(b[i]) for i in range(pos_len))

        # Extract intensity bits and reconstruct integer value
        intensity_bits = [int(b[pos_len + i]) for i in range(bit_depth)]
        intensity_value = sum(bit * (2 ** idx) for idx, bit in enumerate(intensity_bits))

        # Normalize to [0, 1]
        intensity_normalized = intensity_value / (2**bit_depth - 1)

        # Accumulate statistics
        bins[pos_bits]["intensity_sum"] += intensity_normalized * count
        bins[pos_bits]["intensity_squared_sum"] += (intensity_normalized ** 2) * count
        bins[pos_bits]["count"] += count

        # Extract bias bit if present
        if denoise:
            bias_bit = int(b[pos_len + bit_depth])
            if bias_bit == 1:
                bias_stats[pos_bits]["hit"] += count
                bias_stats[pos_bits]["intensity_hit"] += intensity_normalized * count
            else:
                bias_stats[pos_bits]["miss"] += count
                bias_stats[pos_bits]["intensity_miss"] += intensity_normalized * count

    if denoise:
        return bins, bias_stats
    return bins

def make_bins_sv(state_vector, hierarchy_matrix, bit_depth=8, denoise=False):
    """
    Create bins from statevector for unified MHRQI encoding.
    OPTIMIZED: Only processes non-zero amplitudes (sparse).
    """
    bins = defaultdict(lambda: {"intensity_sum": 0, "count": 0, "intensity_squared_sum": 0})
    bias_stats = defaultdict(lambda: {
        "hit": 0, "miss": 0,
        "intensity_hit": 0, "intensity_miss": 0
    }) if denoise else None

    pos_len = len(hierarchy_matrix[0])
    sv = np.array(state_vector).flatten()
    total_qubits = int(np.log2(len(sv)))

    # Qubit layout: [position] [intensity] [bias] [work]
    bias_idx = pos_len + bit_depth if denoise else None

    # ========== SPARSE EXTRACTION (optimized) ==========
    # Only process non-zero amplitudes
    probs = np.abs(sv) ** 2
    nonzero_mask = probs > 1e-10
    nonzero_indices = np.where(nonzero_mask)[0]
    nonzero_probs = probs[nonzero_mask]

    for idx, prb in zip(nonzero_indices, nonzero_probs):
        # Extract position bits using bitwise operations
        # Assuming little-endian bit layout (Qiskit convention)
        pos_bits_list = []
        for i in range(pos_len):
            pos_bits_list.append((idx >> i) & 1)
        pos_bits = tuple(pos_bits_list)

        # Extract intensity bits
        intensity_value = 0
        for i in range(bit_depth):
            if (idx >> (pos_len + i)) & 1:
                intensity_value |= (1 << i)

        intensity_normalized = intensity_value / (2**bit_depth - 1)

        # Accumulate
        bins[pos_bits]["intensity_sum"] += intensity_normalized * prb
        bins[pos_bits]["intensity_squared_sum"] += (intensity_normalized ** 2) * prb
        bins[pos_bits]["count"] += prb

        # Bias extraction
        if denoise and bias_idx is not None:
            bias_bit = (idx >> bias_idx) & 1
            if bias_bit == 1:
                bias_stats[pos_bits]["hit"] += prb
                bias_stats[pos_bits]["intensity_hit"] += intensity_normalized * prb
            else:
                bias_stats[pos_bits]["miss"] += prb
                bias_stats[pos_bits]["intensity_miss"] += intensity_normalized * prb

    if denoise:
        return bins, bias_stats
    return bins
