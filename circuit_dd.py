import numpy as np
from itertools import product
from mqt.qudits.quantum_circuit import QuantumCircuit, QuantumRegister
from mqt.qudits.quantum_circuit.gate import ControlData
from mqt.qudits.compiler.state_compilation.state_preparation import StatePrep
from mqt.qudits.quantum_circuit import QuantumCircuit

# ---------------------------------------------------------
# 1. HELPERS
# ---------------------------------------------------------

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



def MHRQI_init(d, L_max):
    circuit = QuantumCircuit()
    q_regs = []
    dims = []
    # Create registers
    for k in range(L_max):
        q_y = QuantumRegister(f"q_y({k})", 1, [d])
        q_x = QuantumRegister(f"q_x({k})", 1, [d])
        q_regs.append(q_y)
        q_regs.append(q_x)
        dims.append(d)
        dims.append(d)
    intensity_reg = QuantumRegister("intensity", 1, [2])
    q_regs.append(intensity_reg)
    dims.append(2)  # Intensity ancilla

    # Add to circuit
    for reg in q_regs:
        circuit.append(reg)

    return circuit, dims

def MHRQI_upload_intensity(circuit, reg, intensity_dict, approx_threshold=0.01):
    # Filter out zero intensities to reduce statevector size
    filtered_dict = {k: v for k, v in intensity_dict.items() if v > 1e-6}
    statevector = build_image_statevector(filtered_dict, reg)
    state_prep = StatePrep(circuit, statevector, approx=True)
    # Temporarily modify the threshold
    import mqt.qudits.core.micro_dd as mdd
    original_cut = mdd.cut_branches
    def custom_cut(contributions, threshold):
        return original_cut(contributions, approx_threshold)
    mdd.cut_branches = custom_cut
    circuit = state_prep.compile_state()
    mdd.cut_branches = original_cut  # Restore

    print(f"Number of operations: {len(circuit.instructions)}")
    print(f"Number of qudits in the circuit: {circuit.num_qudits}")
    
    return circuit