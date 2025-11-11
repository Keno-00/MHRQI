import numpy as np
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit import QuantumRegister
from mqt.qudits.quantum_circuit.gate import ControlData
import utils
from mqt.qudits.simulation import MQTQuditProvider
from mqt.qudits.visualisation import plot_counts, plot_state
import math


SWAP = np.array([
    [1,0,0,0],
    [0,0,1,0],
    [0,1,0,0],
    [0,0,0,1]
], dtype=np.complex128)
#CUSTOM TOFFOLI, SWAP GATE

def MHRQI_init(d,L_max):
    circuit = QuantumCircuit()
    q_x = []
    q_y = []
    for k in range(L_max):
        print(f"Qy @ {k}")
        q_y.append(QuantumRegister(f"q_y({k})", 1, [d]))
        print(f"Qx @ {k}")
        q_x.append(QuantumRegister(f"q_x({k})", 1, [d]))
    intensity_reg = QuantumRegister("intensity", 1, [2]) 
    
    MHRQI_registers = [*q_x , *q_y, intensity_reg]
    #print(MHRQI_registers)
    for reg in (MHRQI_registers):
        #print(f"circuit append @ {reg}")
        circuit.append(reg)
    for i in range(0,int(len(MHRQI_registers)-1)):
        #print(f"h @ {i}")
        circuit.h(i)
    #print(circuit)
    return(circuit,MHRQI_registers)

def MHRQI_upload_intensity(circuit,reg,d,hierarchy_matrix,img):
    #print(reg)
    MHRQI_register = []
    controls = []
    for i in reg:
        print(i[0])
        MHRQI_register.append(i[0])
    
    #print(MHRQI_register)
    #print(len(reg))

    for i in range (0,int(len(reg)-1)):
        controls.append(i)
    #print(controls)
        

    for i in hierarchy_matrix:
        ctrl_states = list(i)
        ctrl = ControlData(controls,ctrl_states)
        r,c = utils.compose_rc(i,d)
        #print(f"rc: ({r,c}), pix val {img[r,c]}")
        circuit.r(int(len(reg)-1), [0, 1,float(img[r,c]),0.0] ,ctrl)    
    return circuit


def DENOISER_(circuit,reg,d,hierarchy_matrix,img,beta = 1, alpha = 1):
    controls = []
    hier_levels = int((len(reg)-1)/2)
    v_eff_dict = {}

    # index of color ancilla is equal to len(reg)-1
    # index of v_eff accumulator ancilla is equal to len(reg) 
    v_eff_ancilla = int(len(reg))
    # index of position register is from 0 to len(reg)-2



    accumulator = QuantumRegister("accumulator", 1, [2])

    circuit.append(accumulator)

    
    #print(hier_levels)
    children = [[x, y] for x in range(d) for y in range(d)]
    print(children)
    print(hierarchy_matrix)

    #for testing one parent
    #hierarchy_matrix = [[0,0,0,0],]

    for i in range (0,int(len(reg)-1)): # will not include color ancilla. we dont want to include it so this is okay.
        controls.append(i)

    
    for i in hierarchy_matrix:
        v_eff_dict[str(i)] = 0.0

    for i in hierarchy_matrix:
        #print(i)
        parents = i[:-2]
        #print(parents)
        pc_vectors = [parents + child for child in children]

        img_val = []
        for vector in pc_vectors:
            print(vector)
            r,c = utils.compose_rc(vector,d)
            img_val.append(float(img[r,c]))
        print(f"child intensities of parent : {vector[:-2]}")
        print(img_val)

        print("g of intensities")
        g_res = utils.g_matrix(img_val,beta)
        print(g_res)
        interactions = utils.interactions(g_res)

        print("interactions")
        print(interactions)
        #print(parents)
        for idx, vector in enumerate(pc_vectors):
            print(interactions)
            #print(vector)
            r,c = utils.compose_rc(vector,d)
            theta = float(img[r,c])

            v_eff = utils.v_eff(theta,interactions[idx],alpha )

            print(f"veff of {vector}:  {v_eff}")
            v_eff_dict[str(vector)] = v_eff

    print(v_eff_dict)

    for i in hierarchy_matrix:

        print(i)
        ctrl_states = list(i)
        ctrl = ControlData(controls,ctrl_states)
        v_pc = v_eff_dict[str(i)]
        print(v_pc)
        #print(f"rc: ({r,c}), pix val {img[r,c]}")
        circuit.r(v_eff_ancilla, [0, 1,float(v_pc),0.0] ,ctrl)      ### HAMILTONIAN encoding potential energy of interaction of energies. 

    del ctrl_states
    del ctrl
    del controls
    controls = []

    for i in range (0,int(len(reg)-1)): # position control reg
        controls.append(i)
    controls.append(v_eff_ancilla) # adding v_eff as a control perfectly conditioned on intra parent energy interactions to determine coherences

    for i in hierarchy_matrix:
        print(i)
        ctrl_states = list(i)
        for ctrl_state in ctrl_states:
            ctrl_state.append(1)

        ctrl = ControlData(controls,ctrl_states)
        









               

            
    return circuit
        


    #append muna ancilla
    



### SIMULATE

def MHRQI_simulate(circuit,shots):
    provider = MQTQuditProvider()
    provider.backends("fake")
    backend = provider.get_backend("faketraps2trits",shots=shots) 
    from mqt.qudits.simulation.noise_tools.noise import NoiseModel
    nm = NoiseModel()
    job = backend.run(circuit, shots=shots, noise_model=nm)
    result = job.result()

    state_vector = result.get_state_vector()
    counts = result.get_counts()
    plot_state(state_vector, circuit)
    plot_counts(counts, circuit)
    return counts,state_vector        

def MHRQI_simulate_state_vector(circuit):
    provider = MQTQuditProvider()
    provider.backends("sim")
    backend = provider.get_backend("tnsim") 

    job = backend.run(circuit)
    result = job.result()
    
    state_vector = result.get_state_vector()
    plot_state(state_vector,circuit)
    return state_vector

def p_acc_one(sv, acc_idx, endian='little'):
    n = int(np.log2(len(sv)))
    # binary mask bit position depending on endianness
    bitpos = acc_idx if endian=='little' else (n-1-acc_idx)
    inds = np.arange(len(sv))
    mask = ((inds >> bitpos) & 1).astype(bool)
    return float(np.sum(np.abs(sv[mask])**2))

if __name__ == "__main__":
    hier = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0], [1, 1, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1]]
    angle = np.array([
    [1.12912503, 1.48835001, 1.27226413, 2.29527494],
    [1.15499668, 1.51979383, 1.98659597, 1.74819593],
    [1.28863582, 1.77215425, 1.97803837, 1.5825613],
    [1.55118723, 1.25580953, 1.81235433, 2.13797533]
])
    angles_coherent = np.array([
    [1.5, 1.5, 1.5, 1.5],
    [1.5, 1.5, 1.5, 1.5],
    [1.5, 1.5, 1.5, 1.5],
    [1.5, 1.5, 1.5, 1.5]
])
    angles_incoherent = np.array([
    [0.5, 2.0, 0.5, 2.0],
    [2.0, 0.5, 2.0, 0.5],
    [0.5, 2.0, 0.5, 2.0],
    [2.0, 0.5, 2.0, 0.5]
])
    angles_all_zero = np.array([
    [1.0, 1.0, 2.0, 2.0],
    [1.0, 1.0, 2.0, 2.0],
    [2.0, 2.0, 2.0, 2.0],
    [2.0, 2.0, 2.0, 2.0]
])

    d = 2
    qc,reg = MHRQI_init(d,2)
    data_qc = MHRQI_upload_intensity(qc,reg,d,hier,angle )
    denoise_qc = DENOISER_(data_qc,reg,d, hier,angle)
    #sv = MHRQI_simulate_state_vector(denoise_qc)
    #print(sv)
    #bins = utils.make_bins_sv(sv,hier)

    

