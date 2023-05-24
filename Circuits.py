import pennylane as qml
from pennylane import numpy as np
import itertools

def Circuit6(parameters, n_qubits, Length):
    n_params = (n_qubits**2 + 3*n_qubits) * Length
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    parameters = np.reshape(parameters, (Length, int(n_params/Length)))
    for l in range(Length):
        for i, params in enumerate(parameters[l][:n_qubits]):
            qml.RX(params, wires=i)
        for i, params in enumerate(parameters[l][n_qubits:2 * n_qubits]):
            qml.RZ(params, wires=i)

        step = 0
        for i, j in itertools.product(range(n_qubits), range(n_qubits)):
            if i == j:
                continue
            else:
                qml.CRZ(parameters[l][2 * n_qubits + step], wires=[i, j])
                step += 1

        for i, params in enumerate(
                parameters[l][2 * n_qubits + step:3 * n_qubits + step]):
            qml.RX(params, wires=i)
        for i, params in enumerate(
                parameters[l][3 * n_qubits + step:4 * n_qubits + step]):
            qml.RZ(params, wires=i)

def quantum_function(angles, r, qf_weights):
    for r_ in range(r):
        for i, angle in enumerate(angles):
            qml.Rot(qf_weights[r_][i][0], qf_weights[r_][i][1], qf_weights[r_][i][2],
                    0)
            qml.RX(angle, 0)
dev_search_sep = qml.device('default.qubit', wires=n_parameters * counting_qubits + 2)

dev_cc_eval_sep = qml.device('default.qubit', wires=counting_qubits * n_parameters,
                             shots=2 ** counting_qubits_sing)


def multi_d_quantum_function(angles,r, qf_weights):
    dev_qf_eval = qml.device('default.qubit', wires=1)
    @qml.qnode(dev_qf_eval)
    def inside_multi_d_quantum_function(angles,r, qf_weights):
        quantum_function(angles, r, qf_weights)
        return qml.expval(qml.PauliZ(0))
    return inside_multi_d_quantum_function(angles,r, qf_weights)


def Circuit6_probs_sing(parameters):
    dev_cc_eval_sing = qml.device('default.qubit', wires=counting_qubits_sing,
                                  shots=2 ** counting_qubits_sing)
    @qml.qnode(dev_cc_eval_sing)
    def inside_Circuit6_probs_sing(parameters):
        Circuit6(parameters, counting_qubits_sing, counting_length)
        return qml.counts(all_outcomes=True)
    return inside_Circuit6_probs_sing(parameters)


@qml.qnode(dev_cc_eval_sep)
def Circuit6_probs_sep(parameters):
    wire_maps = [{i: i + counting_qubits * j for i in range(counting_qubits)} for j
                 in range(n_parameters)]
    for par in range(int(len(wire_maps))):
        mapped_quantum_function = qml.map_wires(Circuit6, wire_maps[par])
        mapped_quantum_function(parameters[par], counting_qubits, counting_length)
    return qml.counts(all_outcomes=True)