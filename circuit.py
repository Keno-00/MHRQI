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
from itertools import product

# ---------------------------------------------------------
# 1. HELPERS
# ---------------------------------------------------------
def append_qft_d(circuit, qudit_idx, d):
    circuit.h(qudit_idx)

def build_image_statevector(pixel_dict, dimensions):
    """
    Build a quantum statevector encoding image data.
    Measurement probabilities will equal normalized pixel intensities.
    
    Args:
        pixel_dict: Dictionary mapping position bitstrings to normalized intensities
                   e.g., {'0132': 0.6, '0000': 0.3, ...}
        dimensions: List of qudit dimensions, e.g., [4, 4, 4, 4, 2]
                   Last dimension must be 2 (intensity ancilla)
    
    Returns:
        Normalized complex statevector
    """
    assert dimensions[-1] == 2, "Last dimension must be 2 (ancilla qubit)"
    
    total_dim = np.prod(dimensions)
    state_vector = np.zeros(total_dim, dtype=complex)
    
    num_position_qudits = len(dimensions) - 1
    position_dims = dimensions[:-1]
    position_ranges = [range(d) for d in position_dims]
    
    for position_state in product(*position_ranges):
        position_key = ''.join(map(str, position_state))
        intensity = pixel_dict.get(position_key, 0.0)
        
        # Calculate base index
        base_idx = 0
        multiplier = 1
        for i in range(num_position_qudits - 1, -1, -1):
            base_idx += position_state[i] * multiplier
            multiplier *= position_dims[i]
        
        idx_ancilla_0 = base_idx * 2 + 0
        idx_ancilla_1 = base_idx * 2 + 1
        
        state_vector[idx_ancilla_0] = np.sqrt(1.0 - intensity) 
        state_vector[idx_ancilla_1] = np.sqrt(intensity)         
    
    # Renormalize
    norm = np.linalg.norm(state_vector)
    if norm > 0:
        state_vector = state_vector / norm
    
    return state_vector


# ---------------------------------------------------------
# 2. INITIALIZATION (UPDATED FOR PAIRS)
# ---------------------------------------------------------
def MHRQI_init(d, L_max):
    circuit = QuantumCircuit()
    q_x = []
    q_y = []
    dims = []
    # Create registers
    for k in range(L_max):
        q_y.append(QuantumRegister(f"q_y({k})", 1, [d]))
        q_x.append(QuantumRegister(f"q_x({k})", 1, [d]))
        dims.append(d)
        dims.append(d)
    intensity_reg = QuantumRegister("intensity", 1, [2]) 
    dims.append(2)
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
    return circuit, dims

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


def MHRQI_lazy_upload_intensity(circuit, dims, intensity_dict, approx_threshold=0.01):
    filtered_dict = {
        k: v for k, v in intensity_dict.items()
        if v > approx_threshold
    }
    statevector = build_image_statevector(filtered_dict, dims)
    statevector = np.asarray(statevector, dtype=complex)
    norm = np.linalg.norm(statevector)
    if norm == 0:
        raise ValueError("Zero-norm statevector.")
    statevector /= norm
    circuit.set_initial_state(statevector)

    print(f"Number of operations: {len(circuit.instructions)}")
    print(f"Number of qudits in the circuit: {circuit.num_qudits}")

    return circuit

        

# ---------------------------------------------------------
# 3. THE HIERARCHICAL DENOISER 
# ---------------------------------------------------------
# NOTE: MQT qudits does not support selective measurement like Qiskit.
# If denoising uses ancilla qubits, reconstruction requires either:
# 1. Implementing selective measurement in MQT, OR
# 2. Post-measurement binning functions that filter out ancilla bits
#    (Previously in utils.py as *_denoised functions, now removed)
# ---------------------------------------------------------


def APPLY_MIXING_KERNEL(circuit, qudit_index, d, time_step):
    """
    Applies the diffusion kernel (Mix -> Phase -> Unmix) to a single dimension.
    """
    # 1. Transform to Frequency Domain
    circuit.h(qudit_index)

    # 2. Filter High Frequencies
    # This acts as the "Kinetic Energy" penalty in the Hamiltonian
    angle = -1.0 * time_step
    
    # Apply phase penalty to the "difference" states
    if d == 2:
        circuit.rz(qudit_index, [0, 1, angle])
    else:
        # For d > 2, we ideally check the library for proper phase syntax
        # penalizing all non-zero states.
        circuit.rz(qudit_index, [0, 1, angle])

    # 3. Transform back to Spatial Domain
    # CRITICAL NOTE: If d > 2, you technically need the inverse QFT here.
    # For d=2, H is its own inverse. 
    if d == 2:
        circuit.h(qudit_index)
    else:
        # Attempt to access the adjoint if available in your version of MQT
        try:
            circuit.h(qudit_index).dag()
        except:
            # Fallback if .dag() isn't exposed on the gate directly
            circuit.h(qudit_index)


def DENOISER(circuit, reg, d, L_max, time_step=0.5):
    print(f"--- Applying Multi-Scale Hamiltonian Denoising (t={time_step}) ---")
    
    # -----------------------------------------------------
    # STRATEGY: Global Diffusion
    # -----------------------------------------------------
    # To remove the "Squares", we cannot just mix the leaves.
    # We must mix ALL spatial registers. This allows information
    # to flow across the quadrant boundaries.
    
    # The register structure is [y0, x0, y1, x1, ... yL, xL, Intensity]
    # We want to iterate over all x and y registers.
    
    # Excluding the last register (Intensity)
    spatial_indices = range(len(reg) - 1)
    
    # We apply the kernel to every spatial dimension.
    # This creates a "Hypercube" diffusion across the whole image grid.
    for idx in spatial_indices:
        # We can scale the time_step if we want different diffusion 
        # at different scales (e.g., diffuse parents less than leaves).
        # For now, we keep it uniform to ensure smooth blending.
        APPLY_MIXING_KERNEL(circuit, idx, d, time_step)

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
    qc = DENOISER(qc, reg, d, L_max, time_step=1.5)
    
    sv = MHRQI_simulate(qc)
    counts = MHRQI_simulate(qc,shots=4096)
    
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