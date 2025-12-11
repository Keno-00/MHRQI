import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mqt.qudits.quantum_circuit import QuantumCircuit, QuantumRegister
from mqt.qudits.quantum_circuit.gate import ControlData
from mqt.qudits.compiler import QuditCompiler
from mqt.qudits.simulation import MQTQuditProvider
from mqt.qudits.visualisation import plot_counts
import utils 
import plots 
import math

# ---------------------------------------------------------
# 1. HELPERS
# ---------------------------------------------------------
def append_qft_d(circuit, qudit_idx, d):
    circuit.h(qudit_idx)

# ---------------------------------------------------------
# 2. INITIALIZATION (UPDATED FOR PAIRS)
# ---------------------------------------------------------
def MHRQI_init(d, L_max):
    circuit = QuantumCircuit()
    q_x = []
    q_y = []
    
    # Create registers
    for k in range(L_max):
        q_y.append(QuantumRegister(f"q_y({k})", 1, [d]))
        q_x.append(QuantumRegister(f"q_x({k})", 1, [d]))
    
    intensity_reg = QuantumRegister("intensity", 1, [2]) 
    
    MHRQI_registers = []
    for k in range(L_max):
        MHRQI_registers.append(q_y[k])
        MHRQI_registers.append(q_x[k])
        
    MHRQI_registers.append(intensity_reg)
    
    # Add to circuit
    for reg in MHRQI_registers:
        circuit.append(reg)
    
    # Initialize superposition (All positions)
    for i in range(len(MHRQI_registers)-1):
        append_qft_d(circuit, i, d) 
        
    return circuit, MHRQI_registers

def MHRQI_upload_intensity(circuit, reg, d, hierarchy_matrix, img):
    num_regs = len(reg)
    color_idx = num_regs - 1
    control_indices = list(range(num_regs - 1))
    
    for hier_path in hierarchy_matrix:
        ctrl_states = list(hier_path)
        ctrl = ControlData(control_indices, ctrl_states)
        r, c = utils.compose_rc(hier_path, d)
        pixel_val = float(img[r, c])
        
        # [Level_0, Level_1, Theta, Phi]
        circuit.r(color_idx, [0, 1, pixel_val, 0.0], ctrl)  
    print(f"Number of operations: {len(circuit.instructions)}")
    print(f"Number of qudits in the circuit: {circuit.num_qudits}")  
        
    return circuit

# ---------------------------------------------------------
# 3. THE HIERARCHICAL DENOISER (TARGETING RIGHTMOST)
# ---------------------------------------------------------
# ---------------------------------------------------------
# REVISED DENOISER (FROM SCRATCH)
# ---------------------------------------------------------

def APPLY_MIXING_KERNEL(circuit, qudit_index, d, time_step):
    """
    Implements the Hamiltonian Evolution H_mix on a single leaf qudit.
    Sequence: QFT (Mix) -> Phase (Filter) -> IQFT (Unmix)
    """
    # 1. QFT (The "Mix")
    # In MQT, .h() is the Generalized Chrestenson (QFT) gate
    circuit.h(qudit_index)

    # 2. KINETIC PHASE (The "Filter")
    # We apply a phase penalty e^{-i * t} to the high-frequency state |1>.
    # State |0> (DC/Average) gets phase 0.
    # State |1> (AC/Noise) gets phase -t.
    
    if d == 2:
        # Rz rotates between |0> and |1>.
        # We assume the standard definition Rz(theta) ~ diag(e^{-i theta/2}, e^{i theta/2})
        # To get a relative phase of e^{-i t}, we set theta = t * 2 (roughly).
        # We explicitly target levels [0, 1].
        
        angle = -1.0 * time_step
        circuit.rz(qudit_index, [0, 1, angle])
        
    else:
        # For d > 2, we would penalize higher frequencies more (k^2).
        # For this prototype, we just penalize |1>.
        angle = -1.0 * time_step
        circuit.rz(qudit_index, [0, 1, angle])

    # 3. IQFT (The "Unmix")
    # Return to spatial basis
    circuit.h(qudit_index)


def HIERARCHICAL_DENOISER(circuit, reg, d, L_max, time_step=0.5):
    print(f"--- Applying Hierarchical Hamiltonian Denoising (t={time_step}) ---")
    
    # -----------------------------------------------------
    # 1. MAP THE REGISTERS
    # -----------------------------------------------------
    # Current Layout: [y0, x0, y1, x1, ..., yL, xL, Intensity]
    # We need to find the indices of the "Leaves" (The last pair).
    
    total_qudits = len(reg)
    intensity_idx = total_qudits - 1
    
    # The Leaves are the last position pair before intensity.
    leaf_x_idx = total_qudits - 2
    leaf_y_idx = total_qudits - 3
    
    # The Parents are everything else.
    parent_indices = list(range(0, total_qudits - 3))
    
    print(f"DEBUG: Mapping...")
    print(f"  - Parents (Fixed Branches): {parent_indices}")
    print(f"  - Leaves (Successors to Mix): Y={leaf_y_idx}, X={leaf_x_idx}")
    print(f"  - Intensity Payload: {intensity_idx}")

    # -----------------------------------------------------
    # 2. APPLY HAMILTONIAN (The "Mix")
    # -----------------------------------------------------
    # By operating on the leaves while the parents are entangled, 
    # we mathematically perform the sum over all fixed parent branches.
    
    # Apply to Vertical Leaf (yL)
    APPLY_MIXING_KERNEL(circuit, leaf_y_idx, d, time_step)
    
    # Apply to Horizontal Leaf (xL)
    APPLY_MIXING_KERNEL(circuit, leaf_x_idx, d, time_step)

    return circuit

# ---------------------------------------------------------
# 4. SIMULATION TOOLS
# ---------------------------------------------------------
def MHRQI_simulate(circuit,shots=None):
    provider = MQTQuditProvider()
    backend = provider.get_backend("sparse_statevec")

    if shots is None:
        type = "sv"
    else:
        type = "counts"

    if type == "sv":
        print("\n[Simulation] Calculating State Vector...")
        job = backend.run(circuit,)
        result = job.result()
        return result.get_state_vector()
    else:
        print(f"\n[Simulation] Running with {shots} shots...")
        from mqt.qudits.simulation.noise_tools.noise import Noise, NoiseModel
        noise_model = NoiseModel()
        job = backend.run(circuit, shots=shots, noise_model=noise_model)
        result = job.result()
        return result.get_counts()

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    d = 2
    L_max = 2
    
    # Test Hierarchy
    hier = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 1], 
            [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 1, 0], [0, 1, 1, 1], 
            [1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0], [1, 1, 0, 1], 
            [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1]]
            
    angles_noisy = np.random.rand(4, 4) * np.pi 

    qc, reg = MHRQI_init(d, L_max)
    qc = MHRQI_upload_intensity(qc, reg, d, hier, angles_noisy)
    qc = HIERARCHICAL_DENOISER(qc, reg, d, L_max, time_step=1.5)
    
    sv = MHRQI_simulate(qc)
    counts, compiled_qc = MHRQI_simulate(qc,shots=4096)
    
    # Reconstruct Image
    bins = utils.make_bins_sv(sv, hier)
    grid = plots.bins_to_grid(bins, d, 4, kind="p") 
    img = plots.grid_to_image_uint8(grid)
    
    # Save
    print("Saving results...")
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title("Denoised Result")
    plt.savefig("runs/denoised_result.png")
    plt.close()
    print("Done.")