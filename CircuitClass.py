import pennylane as qml
from pennylane import numpy as np
import itertools
from pennylane.templates import StronglyEntanglingLayers


class Distribution_func:
    def __init__(self, counting_qubits, qf_qubits, n_loss_param, training_length, r, seed, Hadamard_index, superpos_hyperparameter):
        np.random.seed(seed)

        self.seed = seed
        self.Hadamard_index = Hadamard_index
        self.superpos_hyperparameter = superpos_hyperparameter

        # self.ob_coeff = observable[0]
        # self.ob_observables = observable[1]
        # self.ob = qml.Hamiltonian(observable[0], observable[1])

        self.counting_qubits = counting_qubits
        self.training_length = training_length

        self.qf_qubits = qf_qubits
        self.r = r
        self.n_loss_param = n_loss_param
        self.qf_parameters = 2 * np.pi * np.random.random(size=(r + 1, 1, qf_qubits, 3), requires_grad=True)
        self.delta = 2*np.pi / ((2**counting_qubits)-1)
        self.angles = [(2 ** i)* self.delta for i in range(counting_qubits)]

        self.dev_qf_eval = qml.device('default.qubit', wires=qf_qubits)
        self.dev_search = qml.device('default.qubit', wires=counting_qubits * n_loss_param + qf_qubits + 1)
        self.dev_cc_eval = qml.device('default.qubit', wires=counting_qubits * n_loss_param, shots = 10000)
        self.dev = qml.device('default.qubit', wires=counting_qubits + qf_qubits)
        self.dev_total = qml.device('default.qubit', wires=counting_qubits*n_loss_param + qf_qubits)
        self.dev_fid = qml.device('default.qubit',
                                    wires=counting_qubits * n_loss_param)

        #list with wire mapping for each counting circuit
        self.sing_circuit_wire_map = {i: i + 1 + self.qf_qubits for i in range(
            self.counting_qubits*self.n_loss_param)}

        self.sep_circuit_wire_map = [{i: i + 1 + qf_qubits + self.counting_qubits*j for i in range(
            self.counting_qubits)} for j in range(self.n_loss_param)]

        self.qf_wire_map = {i: i + 1 for i in range(
            self.qf_qubits)}

        self.controlled_qf_wire_map = {i: i + 1  for i in range(
            self.qf_qubits+self.counting_qubits*self.n_loss_param)}

    def _W_strong(self, theta):#TODO finished this function and checked
        StronglyEntanglingLayers(theta, wires=range(self.qf_qubits))

    def _S_strong(self, x):#TODO finished this function and checked
        for i, x_ in enumerate(x):
            qml.RX(x_, wires=i)

    def _controlled_S_strong(self): #TODO finished this function and checked
        for j, rot_angl in enumerate(self.angles):
            for i in range(self.qf_qubits):
                qml.CRX(rot_angl, wires=[self.qf_qubits+i*self.counting_qubits+j, i])

    def _quantum_function(self, func_parameters): #TODO finished this function and checked
        random_weights = self.qf_parameters

        for theta in random_weights[:-1]:
            self._W_strong(theta)
            self._S_strong(func_parameters)

        self._W_strong(random_weights[-1])

    def _controlled_quantum_function(self):
        random_weights = self.qf_parameters

        for theta in random_weights[:-1]:
            self._W_strong(theta)
            self._controlled_S_strong()

        self._W_strong(random_weights[-1])

    def quantum_function_eval(self, func_parameters): #TODO finished this function and checked
        @qml.qnode(device=self.dev_qf_eval)

        def inside_quantum_function(func_parameters):
            self._quantum_function(func_parameters)
            # return qml.expval(qml.Hamiltonian)
            op = qml.PauliZ(0)
            for i in range(1, self.qf_qubits):
                op = op@qml.PauliZ(i)
            return qml.expval(op)
        #print(qml.draw(inside_quantum_function, expansion_strategy='device')(func_parameters))
        return inside_quantum_function(func_parameters)

    def sing_CH_search(self, counting_parameters):
        @qml.qnode(device=self.dev_search)
        def inside_sing_CH_search(counting_parameters):
            qml.Hadamard(0)

            mapped_H_counting_function = qml.map_wires(self.Circuit6,
                                                     self.sing_circuit_wire_map)
            mapped_H_counting_function(counting_parameters[0],
                                     self.counting_qubits * self.n_loss_param)

            mapped_counting_function = qml.map_wires(self.Circuit6,
                                                     self.sing_circuit_wire_map)
            mapped_counting_function(counting_parameters[1],
                                     self.counting_qubits * self.n_loss_param)

            mapped_quantum_function = qml.map_wires(self._controlled_quantum_function,
                                                    self.controlled_qf_wire_map)
            mapped_quantum_function()

            for wire in range(1, self.qf_qubits + 1):
                qml.ctrl(qml.PauliZ(wire), 0)

            # qml.ctr(qml.Hamiltonian....)
            qml.Hadamard(0)
            return qml.expval(qml.PauliZ(0))

        # qml.counts(wires=[int(self.qf_qubits + 1 + i) for i in range(self.n_loss_param * self.counting_qubits)])
        # print(qml.draw(inside_sing_search)(counting_parameters))
        return inside_sing_CH_search(counting_parameters)

    def sing_search(self, counting_parameters):
        @qml.qnode(device=self.dev_search)
        def inside_sing_search(counting_parameters):
            qml.Hadamard(0)
            mapped_counting_function = qml.map_wires(self.Circuit6, self.sing_circuit_wire_map)
            mapped_counting_function(counting_parameters, self.counting_qubits*self.n_loss_param)

            mapped_quantum_function = qml.map_wires(self._controlled_quantum_function, self.controlled_qf_wire_map)
            mapped_quantum_function()

            for wire in range(1, self.qf_qubits+1):
                qml.ctrl(qml.PauliZ(wire), 0)

            # qml.ctr(qml.Hamiltonian....)
            qml.Hadamard(0)
            return qml.expval(qml.PauliZ(0))

        #qml.counts(wires=[int(self.qf_qubits + 1 + i) for i in range(self.n_loss_param * self.counting_qubits)])
        #print(qml.draw(inside_sing_search)(counting_parameters))
        return inside_sing_search(counting_parameters)

    def sep_search(self, counting_parameters):
        @qml.qnode(device=self.dev_search)
        def inside_sep_search(counting_parameters):
            qml.Hadamard(0)
            for i, wire_map in enumerate(self.sep_circuit_wire_map):
                mapped_counting_function = qml.map_wires(self.Circuit6, wire_map)
                mapped_counting_function(counting_parameters[i], self.counting_qubits)

            mapped_quantum_function = qml.map_wires(self._controlled_quantum_function, self.controlled_qf_wire_map)
            mapped_quantum_function()

            for wire in range(1, self.qf_qubits+1):
                qml.ctrl(qml.PauliZ(wire), 0)

            # qml.ctrl(H, 0)(wires=[1,2])
            qml.Hadamard(0)
        # for ob_coeff, ob_oper in zip(self.ob_coeff, self.ob_observables):
            return qml.expval(qml.PauliZ(0))
        #print(qml.draw(inside_sep_search)(counting_parameters))
        return inside_sep_search(counting_parameters)
        #return print(qml.draw(inside_sep_search)(counting_parameters))

    def fidelity(self, counting_parameters, search_type):
        wires = range(self.counting_qubits*self.n_loss_param)
        @qml.qnode(device=self.dev_fid)
        def sing_circuit(counting_parameters):
            self.Circuit6(parameters=counting_parameters,n_counting_qubits=self.counting_qubits*self.n_loss_param)
            return qml.state()

        @qml.qnode(device=self.dev_fid)
        def sep_circuit(counting_parameters):
            wire_maps = [{i:i+self.n_loss_param*j for i in range(self.counting_qubits)} for j in range(self.n_loss_param)]
            for i, wire_map in enumerate(wire_maps):
                mapped_quantum_function = qml.map_wires(self.Circuit6, wire_map)
                mapped_quantum_function(counting_parameters[i], self.counting_qubits)
            return qml.state()

        @qml.qnode(device=self.dev_fid)
        def superposition(x):
            for wire in wires:
                qml.Hadamard(wires=wire)
            return qml.state()

        if search_type=='Sep':
            Fid = qml.qinfo.fidelity(sep_circuit, superposition,
                                 wires0=[i for i in wires],
                                 wires1=[i for i in wires])(counting_parameters, (3))
            return Fid
        elif search_type=='Sing':
            Fid = qml.qinfo.fidelity(sing_circuit, superposition,
                               wires0=[i for i in wires],
                               wires1=[i for i in wires])(counting_parameters, (3))
            return Fid

    def loss_func_sing(self, counting_parameters):
        evaluation = self.sing_search(counting_parameters)
        fid = self.fidelity(counting_parameters, 'Sing')
        return evaluation - self.superpos_hyperparameter * fid

    def loss_func_sep(self, counting_parameters):
        evaluation = self.sep_search(counting_parameters)
        fid = self.fidelity(counting_parameters,'Sep')
        return evaluation - self.superpos_hyperparameter*fid

    def Circuit6(self, parameters, n_counting_qubits): #TODO finished this function and checked
        # n_counting_qubits = self.counting_qubits

        if self.Hadamard_index:
            for i in range(n_counting_qubits):
                qml.Hadamard(wires=i)

        for l in range(self.training_length):
            for i, params in enumerate(parameters[l][:n_counting_qubits]):
                qml.RX(params, wires=i)
            for i, params in enumerate(
                    parameters[l][n_counting_qubits:2 * n_counting_qubits]):
                qml.RZ(params, wires=i)

            step = 0
            for i, j in itertools.product(range(n_counting_qubits),
                                          range(n_counting_qubits)):
                if i == j:
                    continue
                else:
                    qml.CRZ(parameters[l][2 * n_counting_qubits + step], wires=[i, j])
                    step += 1

            for i, params in enumerate(parameters[l][
                                       2 * n_counting_qubits + step:3 * n_counting_qubits + step]):
                qml.RX(params, wires=i)
            for i, params in enumerate(parameters[l][
                                       3 * n_counting_qubits + step:4 * n_counting_qubits + step]):
                qml.RZ(params, wires=i)

    def eval_sing(self, parameters, name): #TODO finished this function and checked
        @qml.qnode(device=self.dev_cc_eval)
        def inside_Circuit6_eval(parameters, counting_qubits, name):
            if name == '6':
                self.Circuit6(parameters, counting_qubits * self.n_loss_param)
            return qml.counts(all_outcomes=True)
        return inside_Circuit6_eval(parameters, self.counting_qubits, name)

    def eval_sep(self, parameters, name): #TODO finished this function and checked
        @qml.qnode(device=self.dev_cc_eval)
        def inside_Circuit6_eval(parameters, counting_qubits, name):
            if name == '6':
                for i in range(self.n_loss_param):
                    wire_map = {k:k+counting_qubits*i for k in range(counting_qubits)}
                    mapped_counting_function = qml.map_wires(self.Circuit6, wire_map)
                    mapped_counting_function(parameters[i], counting_qubits)
            return qml.counts(all_outcomes=True)
        return inside_Circuit6_eval(parameters, self.counting_qubits, name)


class Average_GradientDescent:
    def __init__(self, n_count_qubits, r, trainable_block_layers, n_param, L):
        self.n_count_qubits = n_count_qubits
        self.r = r
        self.trainable_block_layers = trainable_block_layers
        self.n_param = n_param

        self.delta = L / (2 ** n_count_qubits)
        #subtract 1 / (2 ** (n_count_qubits+1)) to center the interval of the average
        # around x
        self.offset = L/2 - (1 / (2 ** (n_count_qubits+1)))

        self.weights = 2 * np.pi * np.random.random(
        size=(r + 1, trainable_block_layers, n_param, 3),
        requires_grad=True)

        self.dev1 = qml.device('default.qubit', wires=n_param + 1 + n_count_qubits * n_param)
        self.dev2 = qml.device('default.qubit', wires=n_param)

    def S_entangling(self, x):
            for i, x_ in enumerate(x):
                init_param = x_ - self.offset
                qml.RX(init_param, wires=i + 1)
                for entangling_qubit in range(self.n_count_qubits):
                    add_angle = self.delta * (2 ** entangling_qubit)
                    qml.CRX(add_angle, wires=[1 + self.n_param + entangling_qubit + self.n_count_qubits * i,
                                              i + 1])

    def W_entangling_layer(self, theta, x):
            """Trainable circuit block."""
            StronglyEntanglingLayers(theta, wires=range(1, len(x) + 1))

    def Average_GradientDescent(self, x):

        @qml.qnode(self.dev1)
        def inside_Average_GradientDescent(x):
            Hadamard_liste = [0] + [1 + self.n_param + i for i in
                                    range(self.n_param * self.n_count_qubits)]
            for i in Hadamard_liste:
                qml.Hadamard(i)

            for theta in self.weights[:-1]:
                self.W_entangling_layer(theta, x)
                self.S_entangling(x)

            self.W_entangling_layer(self.weights[-1], x)

            for i in range(1, self.n_param + 1):
                qml.CZ(wires=[0, i])

            qml.Hadamard(0)
            return qml.expval(qml.PauliZ(0))

        return inside_Average_GradientDescent(x)

    def W_normal(self, theta, x):
        """Trainable circuit block."""
        StronglyEntanglingLayers(theta, wires=range(len(x)))

    def S_normal(self, x):
        """Data-encoding circuit block."""
        for i, x_ in enumerate(x):
            qml.RX(x_, wires=i)

    def quantum_function(self, x):

        @qml.qnode(self.dev2)
        def inside_quantum_function(x):
            for theta in self.weights[:-1]:
                self.W_normal(theta, x)
                self.S_normal(x)

            # (L+1)'th unitary
            self.W_normal(self.weights[-1], x)

            op = qml.PauliZ(0)
            for i in range(1, len(x)):
                op = op @ qml.PauliZ(i)
            return qml.expval(op)

        return inside_quantum_function(x)

# inst1 = Distribution_func(counting_qubits=3, qf_qubits=2, training_length=1, r=1, seed=10, n_loss_param=2)
# param_shape = (2, 1, (3 ** 2 + 3 * 3))
# counting_parameters = 2 * np.pi * np.zeros(shape=param_shape,
#                                                        requires_grad=True)
# print(inst1.sep_search(counting_parameters))
# print(inst1.quantum_function_eval([0,0]))
# print(inst1.Circuit6(counting_parameters, c6_n_qubits*n_loss_param))
# print(inst1.eval(counting_parameters, counting_qubits=c6_n_qubits, name ='6'))
