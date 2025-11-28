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

    return qc, pos_qubits, intensity, accumulator, and_ancilla, work


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


def DENOISER_qiskit(qc: QuantumCircuit, pos_regs, intensity_reg, accumulator_reg,
                    and_ancilla_reg, work_regs, d, hierarchy_matrix, img,
                    beta=1.0, alpha=1.0,
                    lambda_color=0.3,
                    target_u=np.pi/8.0,
                    target_v=np.pi/8.0,
                    num_layers=3):
    """
    Denoiser with multiple Grover-like layers.

    Pipeline:
    - Compute u_p (parent means) and v_eff (effective potentials) from the image.
    - Normalize u_p and v_eff to small angles *for the accumulator/Grover block only*.
    - Load scaled u_p and v_eff into the accumulator once.
    - For each Grover layer:
        1) Apply Grover-like diffuser on positions + accumulator.
        2) Update color qubit toward RAW u_p with lambda_color, conditioned on accumulator == 1.
        3) Apply inverse diffuser.
    - Uncompute energies from accumulator using the SAME scaled values.
    """
    controls = [reg[0] for reg in pos_regs]
    color_qubit = intensity_reg[0]
    accumulator_qubit = accumulator_reg[0]
    and_ancilla = and_ancilla_reg[0]
    work_qubits = [q for q in work_regs]

    # -------------------------
    # Precompute v_eff and u_p for parent vectors
    # -------------------------
    children = [[y, x] for y in range(d) for x in range(d)]
    v_eff_dict_raw = {}
    u_p_dict_raw = {}

    for vec in hierarchy_matrix:
        v_eff_dict_raw[str(vec)] = 0.0
        u_p_dict_raw[str(vec)] = 0.0

    for vec in hierarchy_matrix:
        parents = vec[:-2]
        pc_vectors = [parents + child for child in children]

        img_vals = []
        for v in pc_vectors:
            rr, cc = utils.compose_rc(v, d)
            img_vals.append(float(img[rr, cc]))

        u_p = float(np.mean(img_vals))
        g_res = utils.g_matrix(img_vals, beta)
        interactions = utils.interactions(g_res)

        for idx, v in enumerate(pc_vectors):
            rr, cc = utils.compose_rc(v, d)
            theta = float(img[rr, cc])
            v_eff = utils.v_eff(theta, interactions[idx].item(), alpha)
            v_eff_dict_raw[str(v)] = float(v_eff)
            u_p_dict_raw[str(v)] = u_p

    # -------------------------
    # Diagnostics: ranges before scaling
    # -------------------------
    theta_min, theta_max = float(img.min()), float(img.max())
    u_vals_raw = np.array(list(u_p_dict_raw.values()))
    v_vals_raw = np.array(list(v_eff_dict_raw.values()))

    if u_vals_raw.size == 0:
        u_vals_raw = np.array([0.0])
    if v_vals_raw.size == 0:
        v_vals_raw = np.array([0.0])

    print("theta range (input):", theta_min, theta_max)
    print("u_p range (raw):", float(u_vals_raw.min()), float(u_vals_raw.max()))
    print("v_eff range (raw):", float(v_vals_raw.min()), float(v_vals_raw.max()))

    # -------------------------
    # Normalization: scale u_p and v_eff for accumulator/Grover block
    # -------------------------
    max_u_raw = float(np.max(np.abs(u_vals_raw))) or 1.0
    max_v_raw = float(np.max(np.abs(v_vals_raw))) or 1.0

    scale_u = target_u / max_u_raw
    scale_v = target_v / max_v_raw

    print("scale_u (acc):", scale_u, "scale_v (acc):", scale_v)
    print("lambda_color:", lambda_color, "num_layers:", num_layers)

    scaled_u_p_acc = {k: scale_u * v for k, v in u_p_dict_raw.items()}
    scaled_v_eff_acc = {k: scale_v * v for k, v in v_eff_dict_raw.items()}

    # For diagnostics: raw color update range
    color_updates_raw = []
    for vec in hierarchy_matrix:
        rr, cc = utils.compose_rc(vec, d)
        theta = float(img[rr, cc])
        u_p_now_raw = u_p_dict_raw[str(vec)]
        color_updates_raw.append(u_p_now_raw - theta)
    color_updates_raw = np.array(color_updates_raw)
    print("raw (u_p - theta) range:", float(color_updates_raw.min()), float(color_updates_raw.max()))

    # -------------------------
    # Helpers: Grover diffuser and its inverse
    # -------------------------
    def grover_diffuse():
        # Forward diffuser: H+Z, phase flips, H back
        for reg in controls:
            qc.h(reg)
            qc.z(reg)
        qc.h(accumulator_qubit)
        qc.z(accumulator_qubit)

        for vec in hierarchy_matrix:
            ctrl_states = list(vec)
            _prepare_controls_on_states(qc, controls, ctrl_states)
            if len(controls) > 1:
                mcx = MCXGate(len(controls))
                qc.append(mcx, [*controls, and_ancilla])
                qc.z(and_ancilla)
                mcx = MCXGate(len(controls))
                qc.append(mcx, [*controls, and_ancilla])
            else:
                qc.z(controls[0])
            _restore_controls(qc, controls, ctrl_states)

        qc.h(accumulator_qubit)
        for reg in controls:
            qc.h(reg)

    def grover_diffuse_dagger():
        # Inverse diffuser: reverse sequence of H/Z and phase flips
        qc.h(accumulator_qubit)
        for reg in controls:
            qc.h(reg)
            qc.z(reg)
        qc.z(accumulator_qubit)

        for vec in hierarchy_matrix:
            ctrl_states = list(vec)
            _prepare_controls_on_states(qc, controls, ctrl_states)
            if len(controls) > 1:
                mcx = MCXGate(len(controls))
                qc.append(mcx, [*controls, and_ancilla])
                qc.z(and_ancilla)
                mcx = MCXGate(len(controls))
                qc.append(mcx, [*controls, and_ancilla])
            else:
                qc.z(controls[0])
            _restore_controls(qc, controls, ctrl_states)

        for reg in controls:
            qc.h(reg)

        qc.h(accumulator_qubit)

    # -------------------------
    # 1) Load energies into accumulator once
    # -------------------------
    for vec in hierarchy_matrix:
        ctrl_states = list(vec)
        u_p_now_acc = scaled_u_p_acc[str(vec)]
        v_pc_acc = scaled_v_eff_acc[str(vec)]
        apply_multi_controlled_ry(
            qc, controls, ctrl_states,
            accumulator_qubit, and_ancilla, work_qubits, u_p_now_acc
        )
        apply_multi_controlled_ry(
            qc, controls, ctrl_states,
            accumulator_qubit, and_ancilla, work_qubits, v_pc_acc
        )

    # -------------------------
    # 2) Grover layers: diffuser -> color update -> inverse diffuser
    # -------------------------
    combined_controls = controls + [accumulator_qubit]

    for layer in range(num_layers):
        print(f"Applying Grover layer {layer+1}/{num_layers}")
        grover_diffuse()

        # Color update toward RAW u_p, conditioned on accumulator == 1
        for vec in hierarchy_matrix:
            ctrl_states = list(vec) + [1]  # accumulator == 1
            rr, cc = utils.compose_rc(vec, d)
            theta = float(img[rr, cc])
            u_p_now_raw = u_p_dict_raw[str(vec)]
            rot_angle = lambda_color * (u_p_now_raw - theta)
            apply_multi_controlled_ry(
                qc, combined_controls, ctrl_states,
                color_qubit, and_ancilla, work_qubits, rot_angle
            )

        grover_diffuse_dagger()

    # -------------------------
    # 3) Uncompute energies from accumulator (scaled values)
    # -------------------------
    for vec in hierarchy_matrix:
        ctrl_states = list(vec)
        v_pc_acc = scaled_v_eff_acc[str(vec)]
        u_p_now_acc = scaled_u_p_acc[str(vec)]
        apply_multi_controlled_ry(
            qc, controls, ctrl_states,
            accumulator_qubit, and_ancilla, work_qubits, -v_pc_acc
        )
        apply_multi_controlled_ry(
            qc, controls, ctrl_states,
            accumulator_qubit, and_ancilla, work_qubits, -u_p_now_acc
        )

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


def simulate_counts(qc: QuantumCircuit, shots=1024, use_gpu=False):
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

    denoiser = False  # set True if you also want to test denoiser behaviour
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
            qc = DENOISER_qiskit(
                qc, pos_regs, intensity_reg,
                accumulator_reg, and_ancilla_reg, work_regs,
                d, hier, angle,
                beta=1.0,
                alpha=1.0,
                lambda_color=0.3,
                target_u=np.pi/8.0,
                target_v=np.pi/8.0,
                num_layers=3,          # <-- more Grover layers
            )


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
