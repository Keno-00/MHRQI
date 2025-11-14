# MHRQI port to Qiskit
# Converted from mqt.qudits implementation to a qubit-based circuit in Qiskit.
# Assumptions and notes:
# - The original used qudits of dimension d; this port assumes d==2 (qubits).
# - Control conditions on state 0 are handled by X-flips before/after the multi-control.
# - Multi-control rotations are implemented by computing the AND of controls onto a single ancilla
#   using the Qiskit mct (multi-controlled Toffoli), then applying a controlled RY (cry) from that
#   ancilla to the target, and finally uncomputing the ancilla.
# - The helper functions rely on a user-provided `utils` module for image-related math (same as
#   in the original). Keep the original utils in PYTHONPATH when running this code.
# - The port follows the original structure (init, upload_intensity, denoiser, simulation).

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer

from qiskit.circuit.library import RYGate
from qiskit.circuit.library import MCXGate
import numpy as np
import math
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


def apply_multi_controlled_ry(qc: QuantumCircuit, controls, ctrl_states, target, ancilla_for_and, work_ancillas, angle):
    """
    Compute logical AND of controls into ancilla_for_and, apply CRY from ancilla to target, then uncompute.

    This implementation attempts to use the QuantumCircuit.mcx primitive when available (it
    selects a decomposition suited to the local Qiskit). If `mcx` is not present on the
    QuantumCircuit object, it falls back to appending the `MCXGate` directly. The function
    always restores any control flips applied for ctrl_states.
    """
    # Bring all controls to 1 where required
    _prepare_controls_on_states(qc, controls, ctrl_states)

    try:
        # If no controls, just rotate the target
        if len(controls) == 0:
            qc.ry(angle, target)
            _restore_controls(qc, controls, ctrl_states)
            return

        if len(controls) == 1:
            # Single control: use controlled RY directly
            qc.cry(angle, controls[0], target)
            _restore_controls(qc, controls, ctrl_states)
            return

        # Compute AND into ancilla_for_and using a multi-controlled X
        try:
            # prefer the built-in mcx which chooses a decomposition appropriate to the backend
            qc.mcx(controls, ancilla_for_and, ancilla_scratch=work_ancillas if len(work_ancillas) > 0 else None)
        except TypeError:
            # older/newer qc.mcx signatures may differ; try without named arg
            try:
                qc.mcx(controls, ancilla_for_and, work_ancillas)
            except Exception:
                # fall back to appending MCXGate
                mcx = MCXGate(len(controls))
                qc.append(mcx, [*controls, ancilla_for_and])
        except AttributeError:
            mcx = MCXGate(len(controls))
            qc.append(mcx, [*controls, ancilla_for_and])

        # apply controlled RY from ancilla to target
        qc.cry(angle, ancilla_for_and, target)

        # uncompute the AND
        try:
            qc.mcx(controls, ancilla_for_and, ancilla_scratch=work_ancillas if len(work_ancillas) > 0 else None)
        except TypeError:
            try:
                qc.mcx(controls, ancilla_for_and, work_ancillas)
            except Exception:
                mcx = MCXGate(len(controls))
                qc.append(mcx, [*controls, ancilla_for_and])
        except AttributeError:
            mcx = MCXGate(len(controls))
            qc.append(mcx, [*controls, ancilla_for_and])

    finally:
        # restore flipped controls regardless of success/failure
        _restore_controls(qc, controls, ctrl_states)

# -------------------------
# Circuit construction
# -------------------------

def MHRQI_init_qiskit(d, L_max):
    """Create circuit and registers in Qiskit. For d==2 this maps directly to qubits."""
    # position registers: q_x(k), q_y(k) each occupying 1 qubit here
    pos_qubits = []
    for k in range(L_max):
        pos_qubits.append(QuantumRegister(1, f"q_y_{k}"))
        pos_qubits.append(QuantumRegister(1, f"q_x_{k}"))

    intensity = QuantumRegister(1, 'intensity')

    # Work ancillas for multi-control decomposition: allocate a small pool
    # Reserve max(1, num_position_qubits - 2) ancillas to support mct decomposition
    num_position_qubits = len(pos_qubits)
    work_count = max(1, num_position_qubits - 2) if num_position_qubits > 1 else 1
    work = QuantumRegister(work_count, 'work')

    # Ancilla to hold accumulator and also an AND-target ancilla
    accumulator = QuantumRegister(1, 'accumulator')
    and_ancilla = QuantumRegister(1, 'and_ancilla')

    # Build a flat QuantumCircuit including all registers
    qc = QuantumCircuit(*pos_qubits, intensity, work, accumulator, and_ancilla)

    # Put Hadamards on position qubits (original used circuit.h(i) across registers)
    # Flatten index mapping: position registers are the first 2*L_max registers
    flat_pos_indices = []
    for reg in pos_qubits:
        flat_pos_indices.append(qc.qubits[qc.qregs.index(reg)])

    # Apply H on each position qubit
    for reg in pos_qubits:
        qc.h(reg[0])

    return qc, pos_qubits, intensity, accumulator, and_ancilla, work


def MHRQI_upload_intensity_qiskit(qc: QuantumCircuit, pos_regs, intensity_reg, d, hierarchy_matrix, img):
    """
    Upload intensity values by rotating the intensity ancilla conditioned on the position control states.
    This uses RY rotations to encode intensity into the intensity qubit.

    Note: The original used a custom 'r' gate with parameters [0,1,theta,0]. Here we use RY(theta).
    """
    # collect control qubit indices and control states
    # position registers are in pos_regs as QuantumRegister objects
    controls = [reg[0] for reg in pos_regs]

    # ancilla indices
    intensity_qubit = intensity_reg[0]

    # work ancillas and and_ancilla are expected at fixed register names in qc
    # Locate them
    work_qubits = [q for r in qc.qregs for q in r][:0]  # placeholder; we'll find properly
    # Instead, gather by names
    work_qubits = []
    for r in qc.qregs:
        if r.name.startswith('work'):
            work_qubits = [r[i] for i in range(len(r))]
            break
    and_ancilla = None
    for r in qc.qregs:
        if r.name == 'and_ancilla':
            and_ancilla = r[0]
            break

    # For each hierarchy vector, apply a controlled rotation encoding the pixel intensity
    for vec in hierarchy_matrix:
        ctrl_states = list(vec)
        # angle derived from image value at composed (r,c)
        r, c = utils.compose_rc(vec, d)
        theta = float(img[r, c])
        # apply multi-controlled RY
        apply_multi_controlled_ry(qc, controls, ctrl_states, intensity_qubit, and_ancilla, work_qubits, theta)

    return qc


def DENOISER_qiskit(qc: QuantumCircuit, pos_regs, intensity_reg, accumulator_reg, and_ancilla_reg, work_regs, d, hierarchy_matrix, img, beta=1.0, alpha=1.0):
    """
    Port of the denoiser algorithm. This reproduces the control structure and math but
    implements oracles and Hamiltonians with controlled RY rotations on ancilla qubits.
    """
    # Flatten lists
    controls = [reg[0] for reg in pos_regs]
    color_qubit = intensity_reg[0]
    accumulator_qubit = accumulator_reg[0]
    and_ancilla = and_ancilla_reg[0]
    work_qubits = [q for q in work_regs]

    # Precompute v_eff and u_p for parent vectors
    children = [[x, y] for x in range(d) for y in range(d)]
    v_eff_dict = {}
    u_p_dict = {}

    for vec in hierarchy_matrix:
        v_eff_dict[str(vec)] = 0.0

    for vec in hierarchy_matrix:
        parents = vec[:-2]
        pc_vectors = [parents + child for child in children]
        img_vals = []
        for v in pc_vectors:
            rr, cc = utils.compose_rc(v, d)
            img_vals.append(float(img[rr, cc]))
        u_p = np.mean(img_vals)
        g_res = utils.g_matrix(img_vals, beta)
        interactions = utils.interactions(g_res)

        for idx, v in enumerate(pc_vectors):
            rr, cc = utils.compose_rc(v, d)
            theta = float(img[rr, cc])
            v_eff = utils.v_eff(theta, interactions[idx].item(), alpha)
            v_eff_dict[str(v)] = v_eff
            u_p_dict[str(v)] = u_p

    # For each parent vector, implement parent-mean and oracle rotations on the accumulator
    for vec in hierarchy_matrix:
        ctrl_states = list(vec)
        u_p_now = u_p_dict[str(vec)]
        v_pc = v_eff_dict[str(vec)]
        # parent mean rotation on accumulator
        apply_multi_controlled_ry(qc, controls, ctrl_states, accumulator_qubit, and_ancilla, work_qubits, u_p_now)
        # oracle (v_pc) rotation
        apply_multi_controlled_ry(qc, controls, ctrl_states, accumulator_qubit, and_ancilla, work_qubits, v_pc)

    # Diffuser-like sequence on position controls + accumulator
    # Apply H then Z on position and accumulator, perform phase flips for each ctrl_state
    for reg in controls:
        qc.h(reg)
        qc.z(reg)
    qc.h(accumulator_qubit)
    qc.z(accumulator_qubit)

    # Phase flip using multi-control Z (implemented here via controlled RZ by mapping)
    for vec in hierarchy_matrix:
        ctrl_states = list(vec)
        # bring controls to required states
        _prepare_controls_on_states(qc, controls, ctrl_states)
        # compute AND into and_ancilla
        if len(controls) > 1:
            mcx = MCXGate(len(controls))
            qc.append(mcx, [*controls, and_ancilla])
            # apply Z on and_ancilla (phase flip)
            qc.z(and_ancilla)
            mcx = MCXGate(len(controls))
            qc.append(mcx, [*controls, and_ancilla])
        else:
            qc.z(controls[0])
        _restore_controls(qc, controls, ctrl_states)

    qc.h(accumulator_qubit)
    for reg in controls:
        qc.h(reg)

    # Final adjustment: rotate color qubit conditioned on positions and accumulator==1
    # Build controls + accumulator
    combined_controls = controls + [accumulator_qubit]

    for vec in hierarchy_matrix:
        ctrl_state = list(vec) + [1]
        # compute required rotation: u_p_now - theta
        vector = list(vec)
        rr, cc = utils.compose_rc(vector, d)
        theta = float(img[rr, cc])
        u_p_now = u_p_dict[str(vector)]
        rot_angle = u_p_now - theta
        apply_multi_controlled_ry(qc, combined_controls, ctrl_state, color_qubit, and_ancilla, work_qubits, rot_angle)

    # Uncompute diffuser sequence (inverse of earlier)
    # Repeat earlier phase-flip uncompute
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

    # Uncompute parent mean and oracle on accumulator (apply negatives)
    for vec in hierarchy_matrix:
        ctrl_states = list(vec)
        v_pc = v_eff_dict[str(vec)]
        u_p_now = u_p_dict[str(vec)]
        apply_multi_controlled_ry(qc, controls, ctrl_states, accumulator_qubit, and_ancilla, work_qubits, -v_pc)
        apply_multi_controlled_ry(qc, controls, ctrl_states, accumulator_qubit, and_ancilla, work_qubits, -u_p_now)

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


def simulate_counts(qc: QuantumCircuit, shots=1024):
    # append measurement on all qubits
    creg = ClassicalRegister(len(qc.qubits), 'c')
    qc_measure = qc.copy()
    qc_measure.add_register(creg)
    qc_measure.measure(range(len(qc.qubits)), creg)

    backend = Aer.get_backend('qasm_simulator')
    transpiled = transpile(qc_measure, backend)
    # assemble removed; Aer backends accept circuits directly
    result = backend.run(transpiled, shots=shots).result()  # modern API
    return result.get_counts()


# -------------------------
# Example usage (mirrors original main)
# -------------------------
if __name__ == '__main__':
    hier = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0], [1, 1, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1]]
    angle = np.array([
        [1.12912503, 1.48835001, 1.27226413, 2.29527494],
        [1.15499668, 1.51979383, 1.98659597, 1.74819593],
        [1.28863582, 1.77215425, 1.97803837, 1.5825613],
        [1.55118723, 1.25580953, 1.81235433, 2.13797533]
    ])

    d = 2
    qc, pos_regs, intensity_reg, accumulator_reg, and_ancilla_reg, work_regs = MHRQI_init_qiskit(d, 2)
    qc = MHRQI_upload_intensity_qiskit(qc, pos_regs, intensity_reg, d, hier, angle)
    qc = DENOISER_qiskit(qc, pos_regs, intensity_reg, accumulator_reg, and_ancilla_reg, work_regs, d, hier, angle)
    sv = simulate_counts(qc)
    print(sv)
