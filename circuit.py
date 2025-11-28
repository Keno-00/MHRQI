import numpy as np
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit import QuantumRegister
from mqt.qudits.quantum_circuit.gate import ControlData
from mqt.qudits.compiler import QuditCompiler
import utils
from mqt.qudits.simulation import MQTQuditProvider
from mqt.qudits.visualisation import plot_counts, plot_state
import math



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
        # print(i[0])
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
    v_eff_dict = {}
    u_p_dict = {}

    # index of color ancilla is equal to len(reg)-1
    color_idx = len(reg)-1
    # index of v_eff accumulator ancilla is equal to len(reg) 
    accumulator_idx = int(len(reg))
    # index of position register is from 0 to len(reg)-2
    accumulator = QuantumRegister("accumulator", 1, [2])
    circuit.append(accumulator)
    # print(hierarchy_matrix)
    children = [[y, x] for x in range(d) for y in range(d)]
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
            #print(vector)
            r,c = utils.compose_rc(vector,d)
            img_val.append(float(img[r,c]))
        u_p = np.mean(img_val)

        g_res = utils.g_matrix(img_val,beta)
        interactions = utils.interactions(g_res)
        for idx, vector in enumerate(pc_vectors):
            r,c = utils.compose_rc(vector,d)
            theta = float(img[r,c])
            v_eff = utils.v_eff(theta, interactions[idx].item(), alpha)
            v_eff_dict[str(vector)] = v_eff
            u_p_dict[str(vector)] = u_p
 ##################################################################################################    
 #-------------------------------------------------------------------------------------------------
    for i in hierarchy_matrix:
        ctrl_states = list(i)
        ctrl = ControlData(controls,ctrl_states)
        v_pc = v_eff_dict[str(i)]
        u_p_now = u_p_dict[str(i)]
        circuit.r(accumulator_idx, [0, 1,float(u_p_now),0.0] ,ctrl)      ### HAMILTONIAN parent mean
        circuit.r(accumulator_idx, [0, 1,0.0,float(v_pc)] ,ctrl)      ### ORACLE
    del ctrl_states
    del ctrl
    del controls
    controls = []
    ctrl_states = []
    for i in range (0,int(len(reg)-1)): # position control reg
       controls.append(i)

    for i in hierarchy_matrix:
       ctrl_states.append(list(i))                                             # diffuser
    # print(ctrl_states)
    for i in controls:
        circuit.h(i)
        circuit.z(i)
    circuit.h(accumulator_idx)
    circuit.z(accumulator_idx)
    for ctrl_state in ctrl_states:
       ctrl_state = list(ctrl_state)
    #    print(f"diffuser on: {ctrl_state}")                                 #CZ
       ctrl = ControlData(controls,ctrl_state)
       circuit.z(accumulator_idx,ctrl)
    #circuit.x(accumulator_idx)
    circuit.h(accumulator_idx)
    for i in controls:
        circuit.h(i)
        #circuit.z(i)
    #final hamiltonian "ADJUSTED"
    del ctrl_states
    del ctrl
    del controls
    controls = []
    ctrl_states = []
    for i in range (0,int(len(reg)-1)): # position control reg
        controls.append(i)
    controls.append(accumulator_idx)
    for i in hierarchy_matrix:
        # print(i)
        state =  list(i)
        state.append(1)
        # print(state)
        ctrl_states.append(state)
    # print(ctrl_states)
    for ctrl_state in ctrl_states:
        ctrl_state = list(ctrl_state)
        # print(f"adjust on: {ctrl_state}")                                 #CZ
        vector = ctrl_state[:-1]
        r,c = utils.compose_rc(vector,d)
        theta = float(img[r,c])
        u_p_now = u_p_dict[str(vector)]
        rot_angle = u_p_now - theta
        # print(f"adjusting {vector} to mean by {rot_angle}")
        ctrl = ControlData(controls,ctrl_state)
        circuit.r(color_idx, [0, 1,float(rot_angle),0.0] ,ctrl) 
#-----------------------------------------------------------------------------
#########################################################################3####

    ######
    # uncompute after adjust
    ######
    del ctrl_states
    del ctrl
    del controls
    controls = []
    ctrl_states = []
    for i in range (0,int(len(reg)-1)): # position control reg
        controls.append(i)

    for i in hierarchy_matrix:
        ctrl_states.append(list(i))                                             # UNCOMPUTING diffuser
    # print(ctrl_states)
    circuit.h(accumulator_idx)
    for i in controls:
        circuit.h(i)
        circuit.z(i)
    circuit.z(accumulator_idx)
    for ctrl_state in ctrl_states:
        ctrl_state = list(ctrl_state)
        # print(f"uncompute diffuser on: {ctrl_state}")                                 
        ctrl = ControlData(controls,ctrl_state)
        circuit.z(accumulator_idx,ctrl)

    for i in controls:
        circuit.h(i)
        #circuit.z(i)
    #circuit.z(accumulator_idx)
    circuit.h(accumulator_idx)
    

    for i in hierarchy_matrix:                                     # UNCOMPUTING 

        # print(i)
        ctrl_states = list(i)
        ctrl = ControlData(controls,ctrl_states)
        v_pc = v_eff_dict[str(i)]
        u_p_now = u_p_dict[str(i)]
        # print(v_pc)
        #print(f"rc: ({r,c}), pix val {img[r,c]}")
        circuit.r(accumulator_idx, [0, 1,0.0,float(-v_pc)] ,ctrl)      ### ORACLE
        circuit.r(accumulator_idx, [0, 1,float(-u_p_now),0.0] ,ctrl)      ### HAMILTONIAN parent mean
        

            
    return circuit


        


    #append muna ancilla
    



### SIMULATE

def MHRQI_simulate(circuit,shots):
    provider = MQTQuditProvider()
    backend = provider.get_backend("sparse_statevec",shots=shots)

    # qudit_compiler = QuditCompiler()
    # passes = ["LogLocQRPass", "ZPropagationOptPass", "ZRemovalOptPass"]
    # passes = ["ZPropagationOptPass", "ZRemovalOptPass"]
    # circuit = qudit_compiler.compile(backend, circuit, passes)

    from mqt.qudits.simulation.noise_tools.noise import NoiseModel
    nm = NoiseModel()
    job = backend.run(circuit, shots=shots, noise_model=nm,use_gpu=True)
    result = job.result()

    counts = result.get_counts()
    plot_counts(counts, circuit)
    print(f"Number of operations: {len(circuit.instructions)}")
    print(f"Number of qudits in the circuit: {circuit.num_qudits}")
    return counts

def MHRQI_simulate_state_vector(circuit):
    provider = MQTQuditProvider()
    provider.backends("sim")
    backend = provider.get_backend("sparse_statevec")

    # Simulators can run the circuit directly without compilation
    job = backend.run(circuit,use_gpu=True)
    result = job.result()

    state_vector = result.get_state_vector()

    print(f"Number of operations: {len(circuit.instructions)}")
    print(f"Number of qudits in the circuit: {circuit.num_qudits}")
    plot_state(state_vector, circuit)
    return state_vector


def MHRQI_compiled(circuit,shots):
    import matplotlib.pyplot as plt
    import networkx as nx
    provider = MQTQuditProvider()
    provider.backends("fake")
    backend_ion = provider.get_backend("faketraps2trits", shots=shots)

    mapping = backend_ion.energy_level_graphs

    pos = nx.circular_layout(mapping[0])
    nx.draw(mapping[0], pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=12, font_weight="bold")
    plt.show()
    qudit_compiler = QuditCompiler()
    passes = ["PhyLocQRPass", "ZPropagationOptPass", "ZRemovalOptPass"]  # Physical passes for hardware
    compiled_circuit_qr = qudit_compiler.compile(backend_ion, circuit, passes)

    print(f"Number of operations: {len(compiled_circuit_qr.instructions)}")
    print(f"Number of qudits in the circuit: {compiled_circuit_qr.num_qudits}")
    job = backend_ion.run(compiled_circuit_qr)

    result = job.result()
    counts = result.get_counts()

    plot_counts(counts, compiled_circuit_qr)



if __name__ == "__main__":
    import plots
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
    data_qc = MHRQI_upload_intensity(qc,reg,d,hier,angles_coherent )
    #data_qc = DENOISER_(data_qc,reg,d, hier,angles_coherent)
    sv = MHRQI_simulate_state_vector(data_qc)
    print(sv)
    bins = utils.make_bins_sv(sv,hier)
    grid = plots.bins_to_grid(bins,d,4,kind="p")
    img = plots.grid_to_image_uint8(grid)
    plots.show_image(img,"test")
    

