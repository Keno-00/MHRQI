"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MHRQI - Magnitude Hierarchical Representation of Quantum Images             ║
║  Qiskit Implementation                                                       ║
║                                                                              ║
║  Author: Keno-00                                                             ║
║  License: Apache 2.0                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import itertools
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
    - outcome qubit (1): Smoothness indicator for denoising
    - Work qubits (2): Gradient computation
    
    Args:
        d: dimension (2 for images)
        L_max: hierarchy depth (log2 of image side)
        bit_depth: bits for intensity encoding (default 8 for 0-255)
    
    Returns:
        qc: QuantumCircuit
        pos_regs: list of position qubit registers
        intensity_reg: register containing intensity qubits
        outcome_qubit: single outcome qubit for denoising
    """
    pos_qubits = []

    # Position qubits (2 per level: y and x)
    for k in range(L_max):
        pos_qubits.append(QuantumRegister(1, f"q_y_{k}"))
        pos_qubits.append(QuantumRegister(1, f"q_x_{k}"))

    # Intensity qubits (basis-encoded grayscale)
    intensity = QuantumRegister(bit_depth, 'intensity')

    # outcome qubit for denoising (hit/miss weighting)
    outcome = QuantumRegister(1, 'outcome')

    # Work qubits for gradient computation and multi-controlled operations
    # work[0] is used as and_ancilla during upload
    work = QuantumRegister(2, 'work')

    qc = QuantumCircuit(*pos_qubits, intensity, outcome, work)

    # Place position qubits in uniform superposition
    for reg in pos_qubits:
        qc.h(reg[0])

    # outcome qubit starts in |0⟩ (marked during denoising)

    return qc, pos_qubits, intensity, outcome


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
    and_ancilla = None
    for r in qc.qregs:
        if r.name == 'work':
            and_ancilla = r[0]
            break
    for vec in hierarchy_matrix:
        ctrl_states = list(vec)
        r, c = utils.compose_rc(vec, d)
        pixel_value = float(img[r, c])
        intensity_int = int(pixel_value * (2**bit_depth - 1))
        intensity_bits = format(intensity_int, f'0{bit_depth}b')
        _prepare_controls_on_states(qc, controls, ctrl_states)
        if len(controls) > 0:
            qc.mcx(controls, and_ancilla)
            for bit_idx, bit_val in enumerate(intensity_bits):
                if bit_val == '1':
                    qc.cx(and_ancilla, intensity_qubits[bit_idx])
            qc.mcx(controls, and_ancilla)
        else:
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
    qubit_to_idx = {q: i for i, q in enumerate(qc.qubits)}
    pos_indices = [qubit_to_idx[reg[0]] for reg in pos_regs]
    intensity_indices = [qubit_to_idx[q] for q in intensity_reg]
    num_qubits = qc.num_qubits
    dim = 2 ** num_qubits
    bit_depth = len(intensity_indices)
    state = np.zeros(dim, dtype=complex)
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


def DENOISER(qc: QuantumCircuit, pos_regs, intensity_reg, outcome=None):
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

    if len(work_qubits) < 2 or outcome is None:
        print("WARNING: Insufficient ancillas")
        return qc

    parent_avg_ancilla = work_qubits[0]
    consistency_ancilla = work_qubits[1]
    outcome_qubit = outcome[0]
    intensity_qubits = list(intensity_reg)

    # ==========================================
    # CHECK FINEST LEVEL VS PARENT
    # ==========================================

    finest_level = num_levels - 1

    if finest_level == 0:
        denoise_qc.x(outcome_qubit)
        qc.compose(denoise_qc, inplace=True)
        return qc

    qy_fine = pos_regs[2 * finest_level][0]
    qx_fine = pos_regs[2 * finest_level + 1][0]

    # === Sibling superposition ===
    denoise_qc.h(qy_fine)
    denoise_qc.h(qx_fine)

    # === Parent average encoding ===

    intensity_msb = intensity_qubits[-1]
    intensity_msb_1 = intensity_qubits[-2] if len(intensity_qubits) > 1 else intensity_msb
    intensity_msb_2 = intensity_qubits[-3] if len(intensity_qubits) > 2 else intensity_msb_1
    intensity_msb_3 = intensity_qubits[-4] if len(intensity_qubits) > 3 else intensity_msb_2

    # Flip MSB bits before rotation
    denoise_qc.x(intensity_msb)
    denoise_qc.x(intensity_msb_1)
    denoise_qc.x(intensity_msb_2)
    denoise_qc.x(intensity_msb_3)

    # Rotate ancilla proportionally to intensity bits
    denoise_qc.cry(np.pi / 16, intensity_msb, parent_avg_ancilla)
    denoise_qc.cry(np.pi / 8, intensity_msb_1, parent_avg_ancilla)
    denoise_qc.cry(np.pi / 4, intensity_msb_2, parent_avg_ancilla)
    denoise_qc.cry(np.pi / 2, intensity_msb_3, parent_avg_ancilla)

    # Unflip MSB bits
    denoise_qc.x(intensity_msb)
    denoise_qc.x(intensity_msb_1)
    denoise_qc.x(intensity_msb_2)
    denoise_qc.x(intensity_msb_3)

    # === Uncompute sibling superposition ===
    denoise_qc.h(qx_fine)
    denoise_qc.h(qy_fine)

    # === Compare pixel to parent average ===
    # XNOR logic: consistent if MSB matches parent average
    denoise_qc.x(parent_avg_ancilla)
    denoise_qc.ccx(intensity_msb, parent_avg_ancilla, consistency_ancilla)
    denoise_qc.x(parent_avg_ancilla)

    denoise_qc.x(intensity_msb)
    denoise_qc.ccx(intensity_msb, parent_avg_ancilla, consistency_ancilla)
    denoise_qc.x(parent_avg_ancilla)
    denoise_qc.x(intensity_msb)

    # === Set outcome ===
    denoise_qc.x(consistency_ancilla)
    denoise_qc.cx(consistency_ancilla, outcome_qubit)

    # === Uncompute ===
    denoise_qc.x(intensity_msb)
    denoise_qc.x(parent_avg_ancilla)
    denoise_qc.ccx(intensity_msb, parent_avg_ancilla, consistency_ancilla)
    denoise_qc.x(parent_avg_ancilla)
    denoise_qc.x(intensity_msb)

    denoise_qc.ccx(intensity_msb, parent_avg_ancilla, consistency_ancilla)

    # Uncompute parent average encoding
    denoise_qc.h(qy_fine)
    denoise_qc.h(qx_fine)

    denoise_qc.x(intensity_msb)
    denoise_qc.x(intensity_msb_1)
    denoise_qc.x(intensity_msb_2)
    denoise_qc.x(intensity_msb_3)

    denoise_qc.cry(-np.pi / 16, intensity_msb, parent_avg_ancilla)
    denoise_qc.cry(-np.pi / 8, intensity_msb_1, parent_avg_ancilla)
    denoise_qc.cry(-np.pi / 4, intensity_msb_2, parent_avg_ancilla)
    denoise_qc.cry(-np.pi / 2, intensity_msb_3, parent_avg_ancilla)

    denoise_qc.x(intensity_msb)
    denoise_qc.x(intensity_msb_1)
    denoise_qc.x(intensity_msb_2)
    denoise_qc.x(intensity_msb_3)

    denoise_qc.h(qx_fine)
    denoise_qc.h(qy_fine)


    qc.compose(denoise_qc, inplace=True)

    return qc, denoise_qc




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
    intensity_reg = None
    for reg in qc.qregs:
        if reg.name == 'intensity':
            intensity_reg = reg
            break
    if intensity_reg is not None:
        qubits_to_measure.extend(list(intensity_reg))
    outcome_reg = None
    for reg in qc.qregs:
        if reg.name == 'outcome':
            outcome_reg = reg
            break
    if outcome_reg is not None:
        qubits_to_measure.extend(list(outcome_reg))
    creg = ClassicalRegister(len(qubits_to_measure), 'c')
    qc_measure = qc.copy()
    qc_measure.add_register(creg)
    qc_measure.measure(qubits_to_measure, creg)
    if use_gpu:
        try:
            backend = AerSimulator(method='statevector', device='GPU', cuStateVec_enable=True)
        except Exception as e:
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
    bins = defaultdict(lambda: {"intensity_sum": 0, "count": 0, "intensity_squared_sum": 0})
    outcome_stats = defaultdict(lambda: {
        "hit": 0, "miss": 0,
        "intensity_hit": 0, "intensity_miss": 0
    }) if denoise else None
    pos_len = len(hierarchy_matrix[0])
    for bitstring, count in counts.items():
        b = bitstring[::-1]  # Reverse for little-endian
        expected_len = pos_len + bit_depth + (1 if denoise else 0)
        if len(b) < expected_len:
            continue
        pos_bits = tuple(int(b[i]) for i in range(pos_len))
        intensity_bits = [int(b[pos_len + i]) for i in range(bit_depth)]
        intensity_value = sum(bit * (2 ** idx) for idx, bit in enumerate(intensity_bits))
        intensity_normalized = intensity_value / (2**bit_depth - 1)
        bins[pos_bits]["intensity_sum"] += intensity_normalized * count
        bins[pos_bits]["intensity_squared_sum"] += (intensity_normalized ** 2) * count
        bins[pos_bits]["count"] += count
        if denoise:
            outcome_bit = int(b[pos_len + bit_depth])
            if outcome_bit == 1:
                outcome_stats[pos_bits]["hit"] += count
                outcome_stats[pos_bits]["intensity_hit"] += intensity_normalized * count
            else:
                outcome_stats[pos_bits]["miss"] += count
                outcome_stats[pos_bits]["intensity_miss"] += intensity_normalized * count
    if denoise:
        return bins, outcome_stats
    return bins

def make_bins_sv(state_vector, hierarchy_matrix, bit_depth=8, denoise=False):
    bins = defaultdict(lambda: {"intensity_sum": 0, "count": 0, "intensity_squared_sum": 0})
    outcome_stats = defaultdict(lambda: {
        "hit": 0, "miss": 0,
        "intensity_hit": 0, "intensity_miss": 0
    }) if denoise else None
    pos_len = len(hierarchy_matrix[0])
    sv = np.array(state_vector).flatten()
    outcome_idx = pos_len + bit_depth if denoise else None
    probs = np.abs(sv) ** 2
    nonzero_mask = probs > 1e-10
    nonzero_indices = np.where(nonzero_mask)[0]
    nonzero_probs = probs[nonzero_mask]
    for idx, prb in zip(nonzero_indices, nonzero_probs):
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
        if denoise and outcome_idx is not None:
            outcome_bit = (idx >> outcome_idx) & 1
            if outcome_bit == 1:
                outcome_stats[pos_bits]["hit"] += prb
                outcome_stats[pos_bits]["intensity_hit"] += intensity_normalized * prb
            else:
                outcome_stats[pos_bits]["miss"] += prb
                outcome_stats[pos_bits]["intensity_miss"] += intensity_normalized * prb
    if denoise:
        return bins, outcome_stats
    return bins


if __name__ == "__main__":

    qc, pos_qubits, intensity, outcome= MHRQI_init(2,3)
    DENOISER(qc, pos_qubits, intensity, outcome)