import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import StronglyEntanglingLayers
import pandas as pd

n_count_qubits = 3
r = 11
trainable_block_layers = 4
n_param = 3

dev1 = qml.device('default.qubit', wires=n_param + 1 + n_count_qubits * n_param)
dev2 = qml.device('default.qubit', wires=n_param)

""""""


def S_gauss_strong(x):
    for i, x_ in enumerate(x):
        init_param = x_ - offset
        qml.RX(init_param, wires=i + 1)
        for hash_point in range(n_count_qubits):
            add = delta * (2 ** hash_point)
            qml.CRZ(add, wires=[1 + n_param + hash_point + n_count_qubits * i, i + 1])


def W_cl_strong(theta, x):
    """Trainable circuit block."""
    StronglyEntanglingLayers(theta, wires=range(1, len(x) + 1))


@qml.qnode(dev1)
def Hadamard_test_gauss(x):
    Hadamard_liste = [0] + [1 + n_param + i for i in range(n_param * n_count_qubits)]
    for i in Hadamard_liste:
        qml.Hadamard(i)
    for theta in weights[:-1]:
        W_cl_strong(theta, x)
        S_gauss_strong(x)

    W_cl_strong(weights[-1], x)

    for i in range(1, n_param + 1):
        qml.CZ(wires=[0, i])

    qml.Hadamard(0)
    return qml.expval(qml.PauliZ(0))


""""""

""""""


def W_strong(theta, x):
    """Trainable circuit block."""
    StronglyEntanglingLayers(theta, wires=range(len(x)))


def S_strong(x):
    """Data-encoding circuit block."""
    for i, x_ in enumerate(x):
        qml.RX(x_, wires=i)


@qml.qnode(dev2)
def my_serial_quantum_model(x):
    for theta in weights[:-1]:
        W_strong(theta, x)
        S_strong(x)

    # (L+1)'th unitary
    W_strong(weights[-1], x)
    op = qml.PauliZ(0)
    for i in range(1, len(x)):
        op = op @ qml.PauliZ(i)
    return qml.expval(op)


repetitions = 1
iterations = 1
lr = 0.05

random_seeds = np.random.randint(0, 1000, size=(repetitions))
seed_dict = {'seed'+ str(i): [seedie] for i,seedie in enumerate(random_seeds)}
df_seeds = pd.DataFrame(seed_dict)
#df_seeds.to_csv(r'./results0/seeds.csv')

opt0 = qml.GradientDescentOptimizer(lr)

L = 2 * lr
delta = L / (2 ** n_count_qubits)
offset = (delta / 2) * (2 ** n_count_qubits)
df_cost = pd.DataFrame()
for rep in range(repetitions):
    filepath = r'C:\Users\Constantin\Desktop\rep' + str(rep) + '.csv'
    #filepath = r'./results0/rep' + str(rep) + '.csv'

    np.random.seed(random_seeds[rep])

    xx = np.array([np.random.uniform(0, 2 * np.pi) for i in range(n_param)],
                  requires_grad=True)

    weights = 2 * np.pi * np.random.random(
        size=(r + 1, trainable_block_layers, n_param, 3),
        requires_grad=True)

    params = xx
    cl_params = xx

    for it in range(iterations):
        params = opt0.step(my_serial_quantum_model, params)
        cost = my_serial_quantum_model(params)
        diction = {'param' + str(i): [param] for i, param in enumerate(params)}
        diction_cost = {'cost': cost}
        diction.update(diction_cost)
        # params_list.append(normal_params)
        # cost_list.append(cost)

        cl_params = opt0.step(Hadamard_test_gauss, cl_params)
        cl_cost = my_serial_quantum_model(cl_params)
        cl_diction = {'cl_param' + str(i): [param] for i, param in enumerate(cl_params)}
        cl_diction_cost = {'cl_cost': cl_cost}
        diction.update(cl_diction)
        diction.update(cl_diction_cost)
        # cl_cost_list.append(cl_cost)
        # cl_params_list.append(cl_params)

        normal_cl_params = opt0.step(my_serial_quantum_model, cl_params)
        normal_cl_cost = my_serial_quantum_model(cl_params)
        normal_cl_diction = {'cl_param' + str(i): [param] for i, param in
                             enumerate(normal_cl_params)}
        normal_cl_diction_cost = {'cl_cost': normal_cl_cost}
        diction.update(normal_cl_diction)
        diction.update(normal_cl_diction_cost)
        # normal_cl_cost_list.append(normal_cl_cost)
        # normal_cl_params_list.append(normal_cl_params)

        new_df_cost = pd.DataFrame(diction)
        df_cost = pd.concat([df_cost, new_df_cost], ignore_index=True)
        df_cost.to_csv(filepath)

