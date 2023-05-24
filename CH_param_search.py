from CircuitClass import Distribution_func
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

c6_n_qubits = 3
n_loss_param = 2
training_lengths = 1


dev = qml.device('default.qubit', wires=14)

@qml.qnode(device=dev)
def H(x):
    for i in range(c6_n_qubits*n_loss_param):
        qml.Hadamard(wires=i)
    return qml.state()



def Fidelity_with_SuperPos(parameters, iterations, c6_n_qubits, n_loss_param, training_lengths, rr, seed, Hadamard_index):
    inst1 = Distribution_func(counting_qubits=c6_n_qubits, qf_qubits=n_loss_param,
                              training_length=training_lengths, r=rr, seed=seed,
                              n_loss_param=n_loss_param, Hadamard_index=Hadamard_index
                              , superpos_hyperparameter=0.2)
    list_cost=[]
    list_param=[]
    lr=0.01
    opt = qml.AdamOptimizer(lr)

    @qml.qnode(device=dev)
    def C6(parame):
        mapped_C6 = qml.map_wires(inst1.Circuit6,
                                  {i:i for i in range(c6_n_qubits*n_loss_param)})
        mapped_C6(parame, c6_n_qubits*n_loss_param)
        return qml.state()
    def Fid(param):
        Fid = qml.qinfo.fidelity(C6, H,
                                 wires0=[i for i in range(c6_n_qubits * n_loss_param)],
                                 wires1=[i for i in range(c6_n_qubits * n_loss_param)])(
            param, (3))
        return 1-Fid

    for it in range(iterations):

        parameters, cost = opt.step_and_cost(Fid, parameters)
        print('cost', cost)
        print(it)
        # print('Fidelity', Fid)
        list_cost.append(cost)
        list_param.append(parameters)

    return list_cost, list_param
for n in range(1,14):
    if n==5:
        break
    n_ = n
    n_count_params = (n_ ** 2 + 3 * n_)
    param_shape = (training_lengths, n_count_params)
    list_cost, list_param = Fidelity_with_SuperPos(parameters=np.random.uniform(size=param_shape,
                                               requires_grad=True), iterations=300, c6_n_qubits=1, n_loss_param=n, training_lengths=training_lengths, rr=3, seed=22, Hadamard_index=False)
    print('last cost of', n, 'with cost:', list_cost[-1])
