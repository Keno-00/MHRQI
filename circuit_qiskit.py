"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MHRQI - Magnitude Hierarchical Representation of Quantum Images            ║
║  Qiskit Implementation                                                       ║
║                                                                              ║
║  Author: Keno-00                                                             ║
║  License: Apache 2.0                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer, AerSimulator

from qiskit.circuit.library import MCXGate
from qiskit_aer.library import SetStatevector
import numpy as np
import itertools  # Added for dynamic parent state generation
import utils
from collections import defaultdict
import random



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

def MHRQI_init_qiskit(d, L_max, bit_depth=8):
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
    
    # Ancilla for multi-controlled operations
    and_ancilla = QuantumRegister(1, 'and_ancilla')
    
    # Work qubits for gradient computation
    work = QuantumRegister(2, 'work')
    
    qc = QuantumCircuit(*pos_qubits, intensity, bias, and_ancilla, work)
    
    # Initialize position qubits in uniform superposition
    for reg in pos_qubits:
        qc.h(reg[0])
    
    # Bias qubit starts in |0⟩ (marked during denoising)
    
    return qc, pos_qubits, intensity, bias

def MHRQI_upload_intensity_qiskit(qc: QuantumCircuit, pos_regs, intensity_reg, d, hierarchy_matrix, img):
    """
    Upload intensity values by rotating the intensity ancilla conditioned on the position control states.
    """
    controls = [reg[0] for reg in pos_regs]
    intensity_qubit = intensity_reg[0]
    work_qubits = []

    and_ancilla = None
    for r in qc.qregs:
        if r.name == 'and_ancilla':
            and_ancilla = r[0]
            break

    for vec in hierarchy_matrix:
        ctrl_states = list(vec)
        r, c = utils.compose_rc(vec, d)
        theta = float(img[r, c])

        apply_multi_controlled_ry(qc, controls, ctrl_states, intensity_qubit, and_ancilla, work_qubits, theta)

    return qc

def MHRQIB_upload_intensity_qiskit(qc: QuantumCircuit, pos_regs, intensity_reg, d, hierarchy_matrix, img):
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
    
    # Get ancilla
    and_ancilla = None
    for r in qc.qregs:
        if r.name == 'and_ancilla':
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

def MHRQI_lazy_upload_intensity_qiskit(qc: QuantumCircuit, pos_regs, intensity_reg, d, hierarchy_matrix, img):
    """
    Upload intensity values by directly defining the statevector. Possibly applied with only one custom gate
    """
    qubit_to_idx = {q: i for i, q in enumerate(qc.qubits)}
    pos_indices = [qubit_to_idx[reg[0]] for reg in pos_regs]
    intensity_idx = qubit_to_idx[intensity_reg[0]]
    
    num_qubits = qc.num_qubits
    dim = 2 ** num_qubits
    
    state = np.zeros(dim, dtype=complex)
    
    num_pos = 2 ** len(pos_indices)
    norm = 1.0 / np.sqrt(num_pos)
    
    is_sequential = all(pos_indices[i] == i for i in range(len(pos_indices)))
    
    if is_sequential:
        state[0:num_pos] = norm
        
        intensity_stride = 2**intensity_idx
        
        for vec in hierarchy_matrix:
            p = 0
            for i, val in enumerate(vec):
                if val:
                    p |= (1 << i)
            
            r, c = utils.compose_rc(vec, d)
            theta = float(img[r, c])
            
            state[p] = norm * np.cos(theta / 2.0)
            state[p + intensity_stride] = norm * np.sin(theta / 2.0)
            
    else:
        print("Warning: Qubits not sequential, lazy upload might be incorrect or currently skipped.")
        return MHRQI_upload_intensity_qiskit(qc, pos_regs, intensity_reg, d, hierarchy_matrix, img)

    qc.append(SetStatevector(state), qc.qubits)
    return qc

def MHRQIB_lazy_upload_intensity_qiskit(qc: QuantumCircuit, pos_regs, intensity_reg, d, hierarchy_matrix, img):
    """
    Fast MHRQI(basis) upload using direct statevector initialization.
    ADDITIONALLY: Encodes local gradient as phase for edge detection.
    
    Gradient is computed classically (Sobel) and embedded in phase:
    - High gradient (edge) → phase ≈ π
    - Low gradient (flat) → phase ≈ 0
    
    This allows the denoiser to detect edges through phase interference.
    """
    from scipy import ndimage
    
    qubit_to_idx = {q: i for i, q in enumerate(qc.qubits)}
    pos_indices = [qubit_to_idx[reg[0]] for reg in pos_regs]
    intensity_indices = [qubit_to_idx[q] for q in intensity_reg]
    
    num_qubits = qc.num_qubits
    dim = 2 ** num_qubits
    bit_depth = len(intensity_indices)
    
    state = np.zeros(dim, dtype=complex)
    
    num_pos = 2 ** len(pos_indices)
    norm = 1.0 / np.sqrt(num_pos)
    
    # Compute gradient map for phase encoding
    N = int(np.sqrt(len(hierarchy_matrix)))
    img_2d = np.zeros((N, N))
    for vec in hierarchy_matrix:
        r, c = utils.compose_rc(vec, d)
        if 0 <= r < N and 0 <= c < N:
            img_2d[r, c] = float(img[r, c])
    
    # MULTI-SCALE GRADIENT DETECTION
    # Blur first to ignore high-frequency noise, then detect structure gradients
    # sigma controls scale: larger = catches broader gradients, ignores fine noise
    sigma = 1.5  # Gaussian blur radius
    blurred = ndimage.gaussian_filter(img_2d, sigma=sigma)
    
    # Sobel on blurred image - detects structure gradients, not noise
    grad_x = ndimage.sobel(blurred, axis=1)
    grad_y = ndimage.sobel(blurred, axis=0)
    gradient_map = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize gradient to [0, 1]
    grad_max = gradient_map.max() if gradient_map.max() > 0 else 1.0
    gradient_map = gradient_map / grad_max
    
    # Check if position and intensity qubits are sequential
    all_indices = pos_indices + intensity_indices
    is_sequential = all(all_indices[i] == i for i in range(len(all_indices)))
    
    if is_sequential:
        # Fast path: sequential qubit layout
        for vec in hierarchy_matrix:
            # Calculate position index
            p = 0
            for i, val in enumerate(vec):
                if val:
                    p |= (1 << i)
            
            r, c = utils.compose_rc(vec, d)
            pixel_value = float(img[r, c])
            intensity_int = int(pixel_value * (2**bit_depth - 1))
            
            # Get gradient for amplitude weighting
            # INTERPRETATION (matching utils.py):
            #   HIGH probability = noisy/uniform = flatten
            #   LOW probability = edge = preserve
            gradient = gradient_map[r, c] if 0 <= r < N and 0 <= c < N else 0.0
            
            # STRONGER contrast: edges get much lower amplitude
            # Range [0.1, 1.0] - edges nearly 10x lower than flat regions
            edge_weight = 1.0 - 0.9 * gradient  # Flat=1.0, Strong edge=0.1
            
            # Base index for this position
            base_idx = p
            
            # Set intensity bits
            intensity_offset = 0
            for bit_idx in range(bit_depth):
                if (intensity_int >> bit_idx) & 1:
                    intensity_offset |= (1 << (len(pos_indices) + bit_idx))
            
            # Apply amplitude weighted by gradient (not just phase)
            state[base_idx + intensity_offset] = edge_weight
    else:
        print("Warning: Qubits not sequential, falling back to gate-based upload.")
        return MHRQIB_upload_intensity_qiskit(qc, pos_regs, intensity_reg, d, hierarchy_matrix, img)
    
    # Normalize state (required for valid quantum state)
    state_norm = np.linalg.norm(state)
    if state_norm > 0:
        state = state / state_norm
    
    qc.append(SetStatevector(state), qc.qubits)
    return qc


def DENOISER_qiskit(qc: QuantumCircuit, pos_regs, intensity_reg, bias=None, strength=0.5, method='bias'):
    """
    Unified MHRQI denoiser using bias qubit for hit/miss weighting.
    
    Based on: docs/knowledge/denoiser_guide.md, docs/knowledge/sibling_smoothing.md
    
    Bias Qubit Algorithm:
    1. Initialize bias qubit in superposition (H)
    2. For each hierarchy level k:
       a. Create neighbor superposition (H on position qubits)
       b. Compute gradient parity via XOR of intensity bits
       c. Phase mark bias qubit if at edge (CZ)
       d. Uncompute gradient and position superposition
    3. Measure: position + intensity + bias
    
    Reconstruction uses: hits / (hits + misses) where hit = bias measured |1⟩
    
    Args:
        qc: QuantumCircuit with MHRQI-encoded image
        pos_regs: Position qubit registers  
        intensity_reg: Intensity qubit register
        bias: Bias qubit for hit/miss weighting
        strength: Marking intensity in [0, 1]
        method: 'bias' (bias qubit marking) or 'uniform' (simple averaging)
    
    Returns:
        qc: Circuit with denoising operations applied
    
    Seam Theory: Phase marking + diffusion at each level couples sibling blocks,
    ensuring cross-boundary information flow via quantum interference.
    """
    num_levels = len(pos_regs) // 2
    
    # Fallback if no bias qubit provided
    if bias is None or method == 'uniform':
        return _uniform_averaging(qc, pos_regs, intensity_reg, strength, num_levels)
    
    # Get work qubits for gradient computation
    work_qubits = []
    for r in qc.qregs:
        if r.name == 'work':
            work_qubits = list(r)
            break
    
    if len(work_qubits) < 1:
        return _uniform_averaging(qc, pos_regs, intensity_reg, strength, num_levels)
    
    # Work qubits for oracle computation
    shift_qubit = work_qubits[0]    # Position shift qubit
    grad_qubit = work_qubits[1]      # Gradient accumulator
    bias_qubit = bias[0]
    
    # =========================================
    # HYBRID DENOISER: Phase Encoding + Unitary Diffusion
    # =========================================
    # Mathematical foundation (from analysis):
    # 
    # LIMITATION: Cannot compute I_p - I_p' coherently with single intensity
    # register because intensity is entangled with position:
    #   |Ψ⟩ = (1/√N) Σ_p |p⟩ ⊗ |I_p⟩
    #
    # SOLUTION: Classical gradient is encoded as PHASE in lazy_upload:
    #   amplitude_p = norm * exp(i * phase_p)
    #   where phase_p = gradient(p) * π
    #
    # This circuit applies UNITARY DIFFUSION (no resets!) to:
    # - Create interference between positions (H gates)
    # - Amplify positions with similar phases (flat regions)
    # - Dampen positions with different phases (edges)
    #
    # The Grover diffusion G = 2|ψ₀⟩⟨ψ₀| - I reflects about mean,
    # pushing amplitudes toward consensus within each hierarchical block.
    
    diffusion_strength = strength * np.pi / 4  # Scale to reasonable range
    
    # Apply Grover-like diffusion at each hierarchical level
    # Coarse levels: larger blocks, weaker smoothing
    # Fine levels: smaller blocks, stronger smoothing
    for k in range(num_levels):
        qy = pos_regs[2 * k][0]
        qx = pos_regs[2 * k + 1][0]
        
        # Level-dependent diffusion weight (finer = stronger)
        level_weight = (k + 1) / num_levels  # 0.17 to 1.0 for 6 levels
        angle = diffusion_strength * level_weight
        
        # ========== PARTIAL GROVER DIFFUSION ==========
        # H⊗H → X⊗X → CP(angle) → X⊗X → H⊗H
        # Full Grover uses CZ (angle=π), we use partial for tunable smoothing
        
        qc.h(qy)
        qc.h(qx)
        qc.x(qy)
        qc.x(qx)
        qc.cp(angle, qy, qx)  # Partial phase flip
        qc.x(qy)
        qc.x(qx)
        qc.h(qy)
        qc.h(qx)
    
    # Bias qubit: mark based on intensity MSB for reconstruction weighting
    intensity_msb = intensity_reg[-1]
    qc.cry(np.pi / 4, intensity_msb, bias_qubit)
    
    return qc



def _uniform_averaging(qc, pos_regs, intensity_reg, strength, num_levels):
    """
    Fallback: Simple uniform averaging without edge preservation.
    Used when work qubits are not available.
    """
    for k in range(num_levels):
        qy = pos_regs[2 * k][0]
        qx = pos_regs[2 * k + 1][0]
        
        weight = ((k + 1) / num_levels) ** 2
        angle = strength * weight
        
        if angle < 1e-6:
            continue
        
        qc.rx(angle, qy)
        qc.rx(angle, qx)
    
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

def make_bins_qiskit(counts, hierarchy_matrix):
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




def make_bins_mhrqib_qiskit(counts, hierarchy_matrix, bit_depth=8):
    """
    Create bins from measurement counts for MHRQI(basis) encoding.
    Reconstructs intensity values from multiple qubit measurements.
    
    Args:
        counts: measurement results from Qiskit
        hierarchy_matrix: position state matrix
        bit_depth: number of intensity qubits
    
    Returns:
        bins: dict mapping position to intensity statistics
    """
    bins = defaultdict(lambda: {"intensity_sum": 0, "count": 0, "intensity_squared_sum": 0})
    pos_len = len(hierarchy_matrix[0])
    
    for bitstring, count in counts.items():
        b = bitstring[::-1]  # Reverse for little-endian
        
        if len(b) < pos_len + bit_depth:
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
    
    # Qubit layout: [position] [intensity] [bias] [and_ancilla] [work]
    bias_idx = pos_len + bit_depth if denoise else None
    
    # ========== SPARSE EXTRACTION (optimized) ==========
    # Only process non-zero amplitudes
    probs = np.abs(sv) ** 2
    nonzero_mask = probs > 1e-10
    nonzero_indices = np.where(nonzero_mask)[0]
    nonzero_probs = probs[nonzero_mask]
    
    for idx, prb in zip(nonzero_indices, nonzero_probs):
        # Convert index to binary (little-endian in Qiskit convention)
        bitstring = format(idx, f'0{total_qubits}b')[::-1]
        
        # Extract position bits
        pos_bits = tuple(int(bitstring[i]) for i in range(pos_len))
        
        # Extract intensity bits
        intensity_bits = [int(bitstring[pos_len + i]) for i in range(bit_depth)]
        intensity_value = sum(bit * (2 ** i) for i, bit in enumerate(intensity_bits))
        intensity_normalized = intensity_value / (2**bit_depth - 1)
        
        # Accumulate
        bins[pos_bits]["intensity_sum"] += intensity_normalized * prb
        bins[pos_bits]["intensity_squared_sum"] += (intensity_normalized ** 2) * prb
        bins[pos_bits]["count"] += prb
        
        # Bias extraction
        if denoise and bias_idx is not None and bias_idx < len(bitstring):
            bias_bit = int(bitstring[bias_idx])
            if bias_bit == 1:
                bias_stats[pos_bits]["hit"] += prb
                bias_stats[pos_bits]["intensity_hit"] += intensity_normalized * prb
            else:
                bias_stats[pos_bits]["miss"] += prb
                bias_stats[pos_bits]["intensity_miss"] += intensity_normalized * prb
    
    if denoise:
        return bins, bias_stats
    return bins

