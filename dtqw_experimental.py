"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  EXPERIMENTAL: DTQW Denoiser with Controlled Rotations                      ║
║  WARNING: This is experimental code. May not improve over hybrid approach.   ║
║                                                                              ║
║  Approach: Use controlled rotations on intensity bits instead of quantum    ║
║  arithmetic. Minimal ancilla overhead (2 qubits).                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Theory:
-------
1. Create superposition over siblings at each hierarchy level (H gates on position)
2. Apply controlled rotations to intensity qubits (mixer based on coin flip)
3. Interference when collapsing superposition -> quantum mixing
4. No explicit neighbor addressing or quantum arithmetic needed

Key difference from hybrid approach:
- Hybrid: Grover diffusion creates probability distribution (measured classically)
- DTQW: Controlled rotations attempt to mix intensity values quantum-mechanically

Limitations:
- Still uncertain if this actually propagates intensity information
- May just create noise without classical post-processing
- Needs empirical testing to validate
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


def DTQW_denoiser_experimental(qc: QuantumCircuit, pos_regs, intensity_reg, 
                               base_steps=3, strength=1.0):
    """
    EXPERIMENTAL: Discrete-Time Quantum Walk denoiser using controlled rotations.
    
    WARNING: This is experimental. Not guaranteed to work better than hybrid approach.
    
    Args:
        qc: QuantumCircuit with MHRQI-encoded image
        pos_regs: Position qubit registers (hierarchical)
        intensity_reg: Intensity qubit register (8 qubits, basis-encoded)
        base_steps: Number of walk steps per level
        strength: Overall mixing strength (scales rotation angles)
    
    Returns:
        qc: Circuit with DTQW operations applied
    
    Ancilla usage: +2 qubits (coin + control flag)
    """
    num_levels = len(pos_regs) // 2
    
    # Allocate minimal ancillas
    coin = QuantumRegister(1, 'dtqw_coin')
    ctrl = QuantumRegister(1, 'dtqw_ctrl')
    qc.add_register(coin)
    qc.add_register(ctrl)
    
    # Process each hierarchy level
    for k in range(num_levels):
        qy = pos_regs[2*k][0]
        qx = pos_regs[2*k+1][0]
        
        # Level-dependent parameters
        level_weight = (k + 1) / num_levels  # Finer levels = stronger
        num_steps = max(1, int(base_steps * level_weight))
        base_angle = np.pi * strength * level_weight * 0.25  # Conservative mixing
        
        # Perform multiple walk steps at this level
        for step in range(num_steps):
            # ============================================
            # DTQW Step: Coin + Shift + Mix
            # ============================================
            
            # 1. Create superposition over 4 siblings at this level
            qc.h(qy)
            qc.h(qx)
            
            # 2. Coin flip for walk direction/mixing
            qc.h(coin[0])
            
            # 3. Controlled rotation on each intensity bit
            # This is where mixing happens (hopefully!)
            # Higher bits get weaker rotation to preserve structure
            for i, int_bit in enumerate(intensity_reg):
                # Bit-dependent angle (MSB preserved more)
                bit_weight = 1.0 / (2 ** ((7 - i) / 4))  # Decreases for higher bits
                angle = base_angle * bit_weight
                
                # Controlled rotation: mix based on coin state
                qc.cry(angle, coin[0], int_bit)
            
            # 4. Optional: controlled phase for interference
            # Mark ctrl qubit if coin flipped and position changed
            qc.ccx(coin[0], qy, ctrl[0])  # Set ctrl if coin=1 and qy=1
            qc.cz(ctrl[0], qx)            # Phase flip
            qc.ccx(coin[0], qy, ctrl[0])  # Uncompute
            
            # 5. Reverse superposition (interference step)
            qc.h(coin[0])
            qc.h(qy)
            qc.h(qx)
    
    return qc


def DTQW_denoiser_bitserial(qc: QuantumCircuit, pos_regs, intensity_reg, 
                            strength=1.0):
    """
    EXPERIMENTAL ALTERNATIVE: Bit-serial DTQW (process one bit at a time).
    
    More conservative approach - only 1 ancilla qubit, but deeper circuit.
    
    Args:
        qc: QuantumCircuit with MHRQI-encoded image
        pos_regs: Position qubit registers
        intensity_reg: Intensity qubit register
        strength: Mixing strength
    
    Returns:
        qc: Circuit with bit-serial DTQW
    
    Ancilla usage: +1 qubit
    """
    num_levels = len(pos_regs) // 2
    
    # Only 1 ancilla needed (reused for each bit)
    phase_anc = QuantumRegister(1, 'dtqw_phase')
    qc.add_register(phase_anc)
    
    for k in range(num_levels):
        qy = pos_regs[2*k][0]
        qx = pos_regs[2*k+1][0]
        
        level_weight = (k + 1) / num_levels
        angle = np.pi * strength * level_weight * 0.25
        
        # Process each intensity bit independently
        for i, int_bit in enumerate(intensity_reg):
            # Bit-dependent weight
            bit_weight = 1.0 / (2 ** ((7 - i) / 4))
            bit_angle = angle * bit_weight
            
            # Create sibling superposition
            qc.h(qy)
            qc.h(qx)
            
            # Phase marking based on bit value
            # This creates conditional mixing via interference
            qc.cx(int_bit, phase_anc[0])
            qc.ry(bit_angle, phase_anc[0])
            qc.cx(int_bit, phase_anc[0])  # Uncompute
            
            # Interference
            qc.h(qy)
            qc.h(qx)
    
    return qc
