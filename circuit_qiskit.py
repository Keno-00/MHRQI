from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer, AerSimulator

from qiskit.circuit.library import MCXGate
import numpy as np
import utils
from collections import defaultdict


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


def apply_multi_controlled_ry(qc: QuantumCircuit, controls, ctrl_states, target, ancilla_for_and, work_ancillas, angle):
    """
    Compute logical AND of controls into ancilla_for_and, apply CRY from ancilla to target, then uncompute.

    ctrl_states is assumed to be in the same order as `controls`:
        controls[i] is in state ctrl_states[i].
    """
    # Bring all controls to 1 where required
    _prepare_controls_on_states(qc, controls, ctrl_states)

    try:
        # If no controls, just rotate the target
        if len(controls) == 0:
            qc.ry(angle, target)
            return

        if len(controls) == 1:
            # Single control: use controlled RY directly
            qc.cry(angle, controls[0], target)
            return

        # Compute AND into ancilla_for_and using a multi-controlled X
        qc.mcx(controls, ancilla_for_and)

        # Apply controlled RY from ancilla to target
        qc.cry(angle, ancilla_for_and, target)

        # Uncompute the AND
        qc.mcx(controls, ancilla_for_and)

    finally:
        # Restore flipped controls regardless of success/failure
        _restore_controls(qc, controls, ctrl_states)


# -------------------------
# Circuit construction
# -------------------------

def MHRQI_init_qiskit(d, L_max):
    """Create circuit and registers in Qiskit. For d==2 this maps directly to qubits."""
    # Position registers: q_y(k), q_x(k) each occupying 1 qubit
    pos_qubits = []
    for k in range(L_max):
        pos_qubits.append(QuantumRegister(1, f"q_y_{k}"))
        pos_qubits.append(QuantumRegister(1, f"q_x_{k}"))

    intensity = QuantumRegister(1, 'intensity')

    # Work ancillas for multi-control decomposition: currently 0 to avoid entanglement issues
    work_count = 0
    work = QuantumRegister(work_count, 'work')

    # Ancilla to hold accumulator and AND-target
    accumulator = QuantumRegister(1, 'accumulator')
    and_ancilla = QuantumRegister(1, 'and_ancilla')

    # Build a flat QuantumCircuit including all registers
    qc = QuantumCircuit(*pos_qubits, intensity, accumulator, and_ancilla, work)

    # Apply H on each position qubit
    for reg in pos_qubits:
        qc.h(reg[0])

    return qc, pos_qubits, intensity


def MHRQI_upload_intensity_qiskit(qc: QuantumCircuit, pos_regs, intensity_reg, d, hierarchy_matrix, img):
    """
    Upload intensity values by rotating the intensity ancilla conditioned on the position control states.
    This uses RY rotations to encode intensity into the intensity qubit.
    """
    # Control qubits in the same order as vec
    controls = [reg[0] for reg in pos_regs]
    intensity_qubit = intensity_reg[0]

    # Work ancillas are currently unused in apply_multi_controlled_ry
    work_qubits = []

    # Locate AND ancilla
    and_ancilla = None
    for r in qc.qregs:
        if r.name == 'and_ancilla':
            and_ancilla = r[0]
            break

    for vec in hierarchy_matrix:
        # ctrl_states exactly matches vec
        ctrl_states = list(vec)

        # Coordinates based on the same vec ordering as in original code
        r, c = utils.compose_rc(vec, d)
        theta = float(img[r, c])

        apply_multi_controlled_ry(qc, controls, ctrl_states, intensity_qubit, and_ancilla, work_qubits, theta)

    return qc


def DENOISER_qiskit(qc, pos_regs, d, time_step=0.4, steps=1, use_v_cycle=True):
    """
    Multiscale Hamiltonian Denoiser with Trotterization and V-Cycle.
    
    Improvements:
    1. Trotterization: Splits time_step into smaller chunks to simulate 
       continuous flow rather than discrete jumps.
    2. V-Cycle: Applies Fine -> Coarse -> Fine diffusion to remove 
       block artifacts introduced by the coarse smoothing.
    
    Args:
        qc: The QuantumCircuit.
        pos_regs: List of QuantumRegisters.
        time_step: Total diffusion strength.
        steps: Number of Trotter steps (repetitions). Higher = smoother quality.
        use_v_cycle: If True, performs Fine -> Coarse -> Fine.
    """
    
    # Calculate dt per step
    dt = time_step / steps
    
    print(f"--- Applying MHRQI V-Cycle Denoiser (Total t={time_step}, Steps={steps}) ---")

    # Helper: The Mixing Kernel (QFT -> Rz -> IQFT)
    def apply_mixing_kernel(target_qubit, t_val):
        # 1. Mix (Hadamard)
        qc.h(target_qubit)
        # 2. Filter (Kinetic Phase) - angle is -2.0 * t
        qc.rz(-2.0 * t_val, target_qubit)
        # 3. Unmix (Inverse Hadamard)
        qc.h(target_qubit)

    # We need at least the leaves to do anything
    if len(pos_regs) < 2:
        return qc

    # Identification of layers
    # Leaves (Fine grain): The last two indices [-2, -1]
    leaf_y = pos_regs[-2][0]
    leaf_x = pos_regs[-1][0]
    
    # Parents (Coarse grain): The pair before leaves [-4, -3], if they exist
    has_parents = len(pos_regs) >= 4
    parent_y = pos_regs[-4][0] if has_parents else None
    parent_x = pos_regs[-3][0] if has_parents else None

    # --- TROTTER LOOP ---
    for s in range(steps):
        # 1. DOWN SWEEP (Fine -> Coarse)
        
        # Apply Fine Diffusion (Leaves)
        apply_mixing_kernel(leaf_y, dt)
        apply_mixing_kernel(leaf_x, dt)
        
        # Apply Coarse Diffusion (Parents) if available
        if has_parents:
            # Coarse diffusion usually needs to be weaker or equal to fine
            # We use dt * 0.7 to avoid over-blurring structure
            dt_coarse = dt * 0.7
            apply_mixing_kernel(parent_y, dt_coarse)
            apply_mixing_kernel(parent_x, dt_coarse)

        # 2. UP SWEEP (Back to Fine) - The "V" in V-Cycle
        # This removes artifacts created by moving the parent boundaries
        if use_v_cycle:
            # We apply a very light "polish" on the leaves again
            apply_mixing_kernel(leaf_y, dt)
            apply_mixing_kernel(leaf_x, dt)
            
    return qc

def DENOISER_qiskit(qc, pos_regs, d, strength=0.2):
    """
    Symmetrized MHRQI Denoiser.
    
    Improvement:
    Implements Strang Splitting (Symmetrized Trotterization) to reduce 
    grid artifacts (anisotropy) inherent in sequential axis rotation.
    
    Sequence: Y(t/2) -> X(t) -> Y(t/2)
    
    This approximates Isotropic Diffusion (circular spreading) more accurately 
    than simple sequential diffusion, resulting in a more natural texture.
    """
    print(f"--- Applying Symmetrized Denoiser (Strength={strength}) ---")

    def apply_op(target, t_val):
        if t_val <= 1e-4: return
        qc.h(target)
        qc.rz(-2.0 * t_val, target)
        qc.h(target)

    if len(pos_regs) < 2: return qc

    # Indices
    leaf_y = pos_regs[-2][0]
    leaf_x = pos_regs[-1][0]
    
    has_parents = len(pos_regs) >= 4
    if has_parents:
        parent_y = pos_regs[-4][0]
        parent_x = pos_regs[-3][0]

    # --- PHASE 1: HALF-STEP Y (Leaves & Parents) ---
    # We apply half the strength to the Y-axis first.
    
    # Leaves (Fine)
    apply_op(leaf_y, strength / 2.0)
    
    # Parents (Coarse) - Scaled down to 50% relative strength
    if has_parents:
        coarse_strength = strength * 0.5
        apply_op(parent_y, coarse_strength / 2.0)

    # --- PHASE 2: FULL-STEP X (Leaves & Parents) ---
    # We apply the full strength to the X-axis in the middle.
    
    # Leaves
    apply_op(leaf_x, strength)
    
    # Parents
    if has_parents:
        apply_op(parent_x, coarse_strength)

    # --- PHASE 3: HALF-STEP Y (Leaves & Parents) ---
    # We complete the symmetry by applying the second half of Y.
    
    # Leaves
    apply_op(leaf_y, strength / 2.0)
    
    # Parents
    if has_parents:
        apply_op(parent_y, coarse_strength / 2.0)

    return qc

# -------------------------
# Simulation helpers
# -------------------------

def simulate_statevector(qc: QuantumCircuit):
    backend = Aer.get_backend('statevector_simulator')
    transpiled = transpile(qc, backend)
    job = backend.run(transpiled)
    result = job.result()
    return result.get_statevector()
  

def simulate_counts(qc: QuantumCircuit, shots=1024, use_gpu=True):
    """
    Measure qubits in the fixed order:
        [q_y_0, q_x_0, q_y_1, q_x_1, ..., intensity]
    Ancillas (accumulator, and_ancilla, work) are ignored in measurement.

    If use_gpu is True and a GPU-enabled AerSimulator is available
    (e.g. in WSL with qiskit-aer-gpu), run on GPU; otherwise fall back
    to the CPU qasm_simulator.
    """
    pos_qubits = []
    intensity_qubit = None

    for reg in qc.qregs:
        if reg.name.startswith('q_y_') or reg.name.startswith('q_x_'):
            pos_qubits.extend(reg)
        elif reg.name == 'intensity':
            intensity_qubit = reg[0]

    qubits_to_measure = []
    qubits_to_measure.extend(pos_qubits)
    if intensity_qubit is not None:
        qubits_to_measure.append(intensity_qubit)

    creg = ClassicalRegister(len(qubits_to_measure), 'c')
    qc_measure = qc.copy()
    qc_measure.add_register(creg)
    qc_measure.measure(qubits_to_measure, creg)

    # Choose backend: try GPU AerSimulator if requested, else CPU qasm_simulator.
    if use_gpu:
        try:
            backend = AerSimulator(device='GPU')
        except Exception:
            # Fallback if GPU backend is not available in this environment
            backend = Aer.get_backend('qasm_simulator')
    else:
        backend = Aer.get_backend('qasm_simulator')

    transpiled = transpile(qc_measure, backend)
    result = backend.run(transpiled, shots=shots).result()
    return result.get_counts()


# -------------------------
# Binning helpers
# -------------------------

def make_bins_qiskit(counts, hierarchy_matrix):
    """
    Measurement order: [pos_bits..., intensity]
    Qiskit counts strings are big-endian w.r.t classical bits,
    so we reverse the bitstring so index 0 corresponds to the first measured qubit.

    After reversing:
        b[0..pos_len-1] = position bits
        b[pos_len]      = intensity
    """
    bins = defaultdict(utils.empty_bin)
    pos_len = len(hierarchy_matrix[0])   # number of position bits

    for bitstring, count in counts.items():
        b = bitstring[::-1]

        if len(b) < pos_len + 1:
            # Safety guard if something is inconsistent
            continue

        pos_bits = [int(b[i]) for i in range(pos_len)]
        intensity_bit = int(b[pos_len])

        key = tuple(pos_bits)
        if intensity_bit == 1:
            bins[key]["hit"] += count
        else:
            bins[key]["miss"] += count
        bins[key]["trials"] += count

    return bins


def make_bins_denoised_qiskit(counts, hierarchy_matrix):
    """
    Denoised binning in Qiskit:
    - Ancillas (including accumulator) are not measured.
    - We treat the denoised color encoding exactly like the raw one:
      positions + intensity only.
    """
    return make_bins_qiskit(counts, hierarchy_matrix)


# -------------------------
# Test pattern helpers
# -------------------------

def make_gradient_tests(N, max_angle=np.pi):
    """
    Return 4 angle matrices (N x N) for order tests:

    - left_to_right:  black -> white from left to right
    - right_to_left:  black -> white from right to left
    - top_to_bottom:  black -> white from top to bottom
    - bottom_to_top:  black -> white from bottom to top

    "Black" = 0 rotation, "white" = max_angle rotation.
    """
    angles = {}

    # Left to right: column index increases intensity
    angle_lr = np.zeros((N, N))
    for r in range(N):
        for c in range(N):
            angle_lr[r, c] = (c / (N - 1)) * max_angle
    angles["left_to_right"] = angle_lr

    # Right to left: column index decreases intensity
    angle_rl = np.zeros((N, N))
    for r in range(N):
        for c in range(N):
            angle_rl[r, c] = ((N - 1 - c) / (N - 1)) * max_angle
    angles["right_to_left"] = angle_rl

    # Top to bottom: row index increases intensity
    angle_tb = np.zeros((N, N))
    for r in range(N):
        for c in range(N):
            angle_tb[r, c] = (r / (N - 1)) * max_angle
    angles["top_to_bottom"] = angle_tb

    # Bottom to top: row index decreases intensity
    angle_bt = np.zeros((N, N))
    for r in range(N):
        for c in range(N):
            angle_bt[r, c] = ((N - 1 - r) / (N - 1)) * max_angle
    angles["bottom_to_top"] = angle_bt

    return angles


# -------------------------
# Example usage (multi-pattern test)
# -------------------------
if __name__ == '__main__':
    import plots
    import matplotlib.pyplot as plt

    denoiser = True  # set True if you also want to test denoiser behaviour
    linuxmode = True

    # Choose image size via L_max
    d = 2
    L_max = 4          # 4 -> 16x16, 3 -> 8x8, 2 -> 4x4, etc.
    N = 2 ** L_max

    # Build hierarchy consistent with L_max
    hier = []
    for y0 in (0, 1):
        for x0 in (0, 1):
            if L_max == 1:
                hier.append([y0, x0])
                continue
            for y1 in (0, 1):
                for x1 in (0, 1):
                    if L_max == 2:
                        hier.append([y0, x0, y1, x1])
                        continue
                    for y2 in (0, 1):
                        for x2 in (0, 1):
                            if L_max == 3:
                                hier.append([y0, x0, y1, x1, y2, x2])
                                continue
                            for y3 in (0, 1):
                                for x3 in (0, 1):
                                    hier.append([y0, x0, y1, x1, y2, x2, y3, x3])

    print("Hierarchy size:", len(hier))
    print("L_max:", L_max, "N:", N)

    # Build test angle patterns
    tests = make_gradient_tests(N, max_angle=np.pi)

    for name, angle in tests.items():
        print("\n==============================")
        print(f"Pattern: {name}")
        print("==============================")
        print("Input angle sample (top-left 4x4):")
        print(angle[:4, :4])

        # Initialise circuit for this pattern
        qc, pos_regs, intensity_reg, accumulator_reg, and_ancilla_reg, work_regs = MHRQI_init_qiskit(d, L_max)

        # Upload intensities
        qc = MHRQI_upload_intensity_qiskit(qc, pos_regs, intensity_reg, d, hier, angle)

        # Optional denoiser
        if denoiser:

            # Use t=0.4 for a good balance of smoothing vs detail
            qc = DENOISER_qiskit(qc, pos_regs, d,)

        # Simulate and bin
        counts = simulate_counts(qc, 2000000, linuxmode)
        if denoiser:
            bins = make_bins_denoised_qiskit(counts, hier)
        else:
            bins = make_bins_qiskit(counts, hier)

        # Pick one key to inspect (e.g. first hierarchy vector)
        test_vec = hier[0]
        print("Bins for", test_vec, ":", bins[tuple(test_vec)])

        # Reconstruct image
        img = plots.bins_to_image_uint8(bins, d, N)

        # Simple numeric check: min/max and first/last row means
        print("Reconstructed img min/max:", img.min(), img.max())
        print("Row 0 mean:", img[0].mean(), "Row N-1 mean:", img[-1].mean())
        print("Col 0 mean:", img[:, 0].mean(), "Col N-1 mean:", img[:, -1].mean())

        # Plot
        plots.show_image(img, f"QISKIT MHRQI - {name}")
