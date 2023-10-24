import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
import itertools
from pennylane.templates import StronglyEntanglingLayers
import seaborn as sns


class Distribution_func:
    def __init__(self, counting_qubits, qf_qubits, n_loss_param, training_length, r,
                 seed, Hadamard_index, superpos_hyperparameter):
        np.random.seed(seed)

        # defines the random function
        self.seed = seed

        # set hadamard index (initial state = total superposition, yes or no)
        self.Hadamard_index = Hadamard_index
        self.superpos_hyperparameter = superpos_hyperparameter

        # self.ob_coeff = observable[0]
        # self.ob_observables = observable[1]
        # self.ob = qml.Hamiltonian(observable[0], observable[1])

        # QUANTUM FUNCTION PARAMETERS
        # number of qubits of the counting register
        self.counting_qubits = counting_qubits
        # DETERMINE STILL WHAT THIS IS
        self.training_length = training_length
        # number of qbits used for the random quantum function
        self.qf_qubits = qf_qubits
        # max freq used in the f.t. representation of the quantum function
        self.r = r
        # number of paramaters in the loss function = quantum function
        self.n_loss_param = n_loss_param

        self.qf_parameters = 2 * np.pi * np.random.random(size=(r + 1, 1, qf_qubits, 3),
                                                          requires_grad=True)
        # Delta between angles of gate
        self.delta = 2 * np.pi / ((2 ** counting_qubits) - 1)
        self.angles = [(2 ** i) * self.delta for i in range(counting_qubits)]

        # device for simple evaluation of the quantum function
        self.dev_qf_eval = qml.device('default.qubit', wires=qf_qubits)
        # device for search algorithm
        self.dev_search = qml.device('default.qubit',
                                     wires=counting_qubits * n_loss_param + qf_qubits + 1)
        # Evaluation of counting circuit. Likewise to QAOA
        self.dev_cc_eval = qml.device('default.qubit',
                                      wires=counting_qubits * n_loss_param, shots=10000)
        # unsure about this one
        # self.dev_total = qml.device('default.qubit',
        #                             wires=counting_qubits * n_loss_param + qf_qubits)

        # device to calculate fidelity for superposition hyperparameter
        self.dev_fid = qml.device('default.qubit',
                                  wires=counting_qubits * n_loss_param)

        # list with wire mapping for each counting circuit
        self.sing_circuit_wire_map = {i: i + 1 + self.qf_qubits for i in range(
            self.counting_qubits * self.n_loss_param)}

        self.sep_circuit_wire_map = [
            {i: i + 1 + qf_qubits + self.counting_qubits * j for i in range(
                self.counting_qubits)} for j in range(self.n_loss_param)]

        self.qf_wire_map = {i: i + 1 for i in range(
            self.qf_qubits)}

        self.controlled_qf_wire_map = {i: i + 1 for i in range(
            self.qf_qubits + self.counting_qubits * self.n_loss_param)}

    # entangling layer for the random q-function
    def _W_strong(self, theta):  # TODO finished this function and checked
        StronglyEntanglingLayers(theta, wires=range(self.qf_qubits))

    # encoding layer of the variables for the random q-function
    def _S_strong(self, x):  # TODO finished this function and checked
        for i, x_ in enumerate(x):
            qml.RX(x_, wires=i)

    # encoding layer of the variables for the random q-function
    # now with controlled rotations and iterating over all self.angles
    # in order to encode all angles into each state of the counting register
    def _controlled_S_strong(self):  # TODO finished this function and checked
        for j, rot_angl in enumerate(self.angles):
            for i in range(self.qf_qubits):
                qml.CRX(rot_angl,
                        wires=[self.qf_qubits + i * self.counting_qubits + j, i])

    # create normal quantum function
    def _quantum_function(self, func_parameters):
        random_weights = self.qf_parameters

        for theta in random_weights[:-1]:
            self._W_strong(theta)
            self._S_strong(func_parameters)

        self._W_strong(random_weights[-1])

    # create controlled version of quantum function
    def _controlled_quantum_function(self):
        random_weights = self.qf_parameters

        for theta in random_weights[:-1]:
            self._W_strong(theta)
            self._controlled_S_strong()

        self._W_strong(random_weights[-1])

    # QNode for normal quantum function
    def quantum_function_eval(self, func_parameters):  # TODO finished this function and checked
        @qml.qnode(device=self.dev_qf_eval)
        def inside_quantum_function(func_parameters):
            self._quantum_function(func_parameters)
            op = qml.PauliZ(0)
            for i in range(1, self.qf_qubits):
                op = op @ qml.PauliZ(i)
            return qml.expval(op)

        # print(qml.draw(inside_quantum_function, expansion_strategy='device')(func_parameters))
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

    # QNode for single search (all counting qubits get one Big circuit)
    def sing_search(self, counting_parameters):
        @qml.qnode(device=self.dev_search)
        def inside_sing_search(counting_parameters):
            qml.Hadamard(0)
            mapped_counting_function = qml.map_wires(self.Circuit6,
                                                     self.sing_circuit_wire_map)
            mapped_counting_function(counting_parameters,
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
        return inside_sing_search(counting_parameters)

    # QNode for seperate search (each register of counting qubits representing one variable
    # of the quantum function, has one seperate Circuit. The registers do not share entanglement)
    def sep_search(self, counting_parameters):
        @qml.qnode(device=self.dev_search)
        def inside_sep_search(counting_parameters):
            qml.Hadamard(0)
            for i, wire_map in enumerate(self.sep_circuit_wire_map):
                mapped_counting_function = qml.map_wires(self.Circuit6, wire_map)
                mapped_counting_function(counting_parameters[i], self.counting_qubits)

            mapped_quantum_function = qml.map_wires(self._controlled_quantum_function,
                                                    self.controlled_qf_wire_map)
            mapped_quantum_function()

            for wire in range(1, self.qf_qubits + 1):
                qml.ctrl(qml.PauliZ(wire), 0)

            # qml.ctrl(H, 0)(wires=[1,2])
            qml.Hadamard(0)
            # for ob_coeff, ob_oper in zip(self.ob_coeff, self.ob_observables):
            return qml.expval(qml.PauliZ(0))

        # print(qml.draw(inside_sep_search)(counting_parameters))
        return inside_sep_search(counting_parameters)
        # return print(qml.draw(inside_sep_search)(counting_parameters))

    # Calculate fidelity between Circuit6 and the total superposition
    def fidelity(self, counting_parameters, search_type):
        wires = range(self.counting_qubits * self.n_loss_param)

        @qml.qnode(device=self.dev_fid)
        def sing_circuit(counting_parameters):
            self.Circuit6(parameters=counting_parameters,
                          n_counting_qubits=self.counting_qubits * self.n_loss_param)
            return qml.state()

        @qml.qnode(device=self.dev_fid)
        def sep_circuit(counting_parameters):
            wire_maps = [
                {i: i + self.n_loss_param * j for i in range(self.counting_qubits)} for
                j in range(self.n_loss_param)]
            for i, wire_map in enumerate(wire_maps):
                mapped_quantum_function = qml.map_wires(self.Circuit6, wire_map)
                mapped_quantum_function(counting_parameters[i], self.counting_qubits)
            return qml.state()

        @qml.qnode(device=self.dev_fid)
        def superposition(x):
            for wire in wires:
                qml.Hadamard(wires=wire)
            return qml.state()

        if search_type == 'Sep':
            Fid = qml.qinfo.fidelity(sep_circuit, superposition,
                                     wires0=[i for i in wires],
                                     wires1=[i for i in wires])(counting_parameters,
                                                                (3))
            return Fid
        elif search_type == 'Sing':
            Fid = qml.qinfo.fidelity(sing_circuit, superposition,
                                     wires0=[i for i in wires],
                                     wires1=[i for i in wires])(counting_parameters,
                                                                (3))
            return Fid

    # Define loss function for single search
    def loss_func_sing(self, counting_parameters):
        evaluation = self.sing_search(counting_parameters)
        fid = self.fidelity(counting_parameters, 'Sing')
        return evaluation - self.superpos_hyperparameter * fid

    # Define loss for seperate search
    def loss_func_sep(self, counting_parameters):
        evaluation = self.sep_search(counting_parameters)
        fid = self.fidelity(counting_parameters, 'Sep')
        return evaluation - self.superpos_hyperparameter * fid

    # Circuit 6
    def Circuit6(self, parameters,
                 n_counting_qubits):  # TODO finished this function and checked
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

    # Evaluate states from Circuit 6 of the single search (likewise to QAOA)
    def eval_sing(self, parameters, name):
        @qml.qnode(device=self.dev_cc_eval)
        def inside_Circuit6_eval(parameters, counting_qubits, name):
            if name == '6':
                self.Circuit6(parameters, counting_qubits * self.n_loss_param)
            return qml.counts(all_outcomes=True)

        return inside_Circuit6_eval(parameters, self.counting_qubits, name)

    # Evaluate states from Circuit 6 of the seperate search(likewise to QAOA)
    def eval_sep(self, parameters, name):
        @qml.qnode(device=self.dev_cc_eval)
        def inside_Circuit6_eval(parameters, counting_qubits, name):
            if name == '6':
                for i in range(self.n_loss_param):
                    wire_map = {k: k + counting_qubits * i for k in
                                range(counting_qubits)}
                    mapped_counting_function = qml.map_wires(self.Circuit6, wire_map)
                    mapped_counting_function(parameters[i], counting_qubits)
            return qml.counts(all_outcomes=True)

        return inside_Circuit6_eval(parameters, self.counting_qubits, name)



class Average_GradientDescent:
    def __init__(self, n_count_qubits, r, entangling_block_layers, n_param, L, seed):
        # number of counting qubits per variable
        self.n_count_qubits = n_count_qubits
        # max frequency of the random function
        self.r = r
        # number of entangling block layers
        self.entangling_block_layers = entangling_block_layers
        # number of parameters
        self.n_param = n_param

        self.delta = L / (2 ** n_count_qubits)
        # subtract 1 / (2 ** (n_count_qubits+1)) to center the interval of the average
        # around x
        self.offset = L / 2 - (1 / (2 ** (n_count_qubits + 1)))

        np.random.seed(seed)
        self.weights = 2 * np.pi * np.random.random(
            size=(r + 1, entangling_block_layers, n_param, 3),
            requires_grad=True)

        self.dev1 = qml.device('default.qubit',
                               wires=n_param + 1 + n_count_qubits * n_param)
        self.dev2 = qml.device('default.qubit', wires=n_param)

    def S_entangling(self, x):
        for i, x_ in enumerate(x):
            init_param = x_ - self.offset
            qml.RX(init_param, wires=i + 1)
            for entangling_qubit in range(self.n_count_qubits):
                add_angle = self.delta * (2 ** entangling_qubit)
                qml.CRX(add_angle, wires=[
                    1 + self.n_param + entangling_qubit + self.n_count_qubits * i,
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


class vFFT:
    def __init__(self, k_vector_dim, m, n, cos_or_sin=None):
        self.k_vector_dim = k_vector_dim
        self.m = m
        self.n = n
        self.log_N_1 = m + n
        self.num_wires = self.log_N_1 * k_vector_dim + 1
        self.total_wires = [i for i in range(self.num_wires)]
        self.dev = qml.device('default.qubit', wires=self.num_wires)

        function_list = {
            # "Rosenbrock": lambda x, y: 100 * (y ** 2 - y * (x ** 2) + x ** 4) + (
            #             x - 1) ** 2,
            "Rosenbrock": lambda x, y: x**y - y**x,
            "styblinski_tang": lambda x: (x ** 4 - 16 * (x ** 2) + 5 * x) / 2,
            "sin": lambda x: np.sin(x)
        }

        if self.k_vector_dim == 1:
            function_name = "styblinski_tang"
        elif self.k_vector_dim == 2:
            function_name = "Rosenbrock"
        else:
            exit("Dimension does not have a function")


        # print(f"function_name{function_name}")
        self.function = function_list[function_name]
        # self.observable__ = self.pauli_decomposition(self.function, matrix_or_pauli=None)
        self.observable__ = np.identity(n=2**self.log_N_1)

    def int_to_bit(self, integer):
        return bin(integer)[2:].zfill(self.k_vector_dim*(self.m + self.n))

    def bit_to_float(self, x_vector):
        float_vector = []
        for bit_string in x_vector:
            float = 0
            for i, bit in enumerate(list(bit_string)):
                float += 2 ** (self.n - i) * int(bit)
            float_vector.append(float)
        return float_vector
    def numpy_fft(self, plot):
        _shape = tuple(2**self.log_N_1 for _ in range(self.k_vector_dim))
        y = np.zeros(shape = _shape)
        for i in range(2 ** (self.log_N_1 * self.k_vector_dim)):
            biti = self.int_to_bit(i)
            md_biti = [biti[i * (self.m + self.n):(i + 1) * (self.m + self.n)] for i in
                       range(self.k_vector_dim)]
            x_vector = tuple([int(i,2) for i in md_biti])
            # float = tuple(self.bit_to_float(md_biti))
            float = self.bit_to_float(md_biti)

            y[x_vector] = self.function(x=float[0], y=float[1])

        fourier_amplitudes = np.fft.fftn(y)
        # print("y", y, "fourier ampl", fourier_amplitudes, np.shape(fourier_amplitudes), type(fourier_amplitudes), np.abs(fourier_amplitudes))
        a = np.abs(fourier_amplitudes)
        b = fourier_amplitudes.imag
        c = fourier_amplitudes.real

        # Create a figure and subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot heatmap for variable a
        axes[0].imshow(a, cmap='hot', interpolation='nearest')
        axes[0].set_title('Absolute')
        axes[0].set_xlabel("k_1")
        axes[0].set_ylabel("k_2")
        axes[0].axis('off')  # Turn off axis labels

        # Plot heatmap for variable b
        axes[1].imshow(b, cmap='hot', interpolation='nearest')
        axes[1].set_title('Imaginary')
        axes[1].set_xlabel("k_1")
        axes[1].set_ylabel("k_2")
        axes[1].axis('off')  # Turn off axis labels

        # Plot heatmap for variable c
        axes[2].imshow(c, cmap='hot', interpolation='nearest')
        axes[2].set_title('Real')
        axes[2].set_xlabel("k_1")
        axes[2].set_ylabel("k_2")
        axes[2].axis('off')  # Turn off axis labels

        # Add a colorbar for each subplot
        for ax in axes:
            cbar = fig.colorbar(ax.images[0], ax=ax, orientation='vertical', shrink=0.6,
                                aspect=20)
            cbar.set_label('Colorbar Label')

        plt.tight_layout()
        plt.show()
        # plt.figure(figsize=(8, 6))
        # # plt.imshow(y, cmap='viridis', origin='lower', extent=[0, 64, 0, 64])
        # plt.imshow(np.abs(fourier_amplitudes), cmap='hot', interpolation='nearest')
        #
        # # Add a colorbar
        # plt.colorbar()
        #
        # # Add labels and title
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.title("Matplotlib imshow Example")
        #
        # plt.show()

        # if plot:
        #     real_part = []
        #     img_part = []
        #     abs_part = []
        #     for i in fourier_amplitudes:
        #         real_part.append(i.real)
        #         img_part.append(i.imag)
        #         abs_part.append(abs(i))
        #     plt.plot(range(int(len(real_part))), real_part, label="Real")
        #     plt.plot(range(int(len(img_part))), img_part, label="Imaginary")
        #     plt.plot(range(int(len(abs_part))), abs_part, label="Absolute value")
        #     plt.legend()
        #     plt.title("numpy fft")
        #     plt.show()

        return fourier_amplitudes
    def normal_fft(self, plot):
        y = []
        for i in range(2 ** self.log_N_1):
            biti = self.int_to_bit(i)
            float = self.bit_to_float(biti)
            y.append(self.function(x=float[0], y=float[1]))

        fourier_amplitudes = []
        for k in range(2 ** self.log_N_1):
            fourier_amplitudes.append(
                sum([y[i] * complex(np.cos(-2 * np.pi * k * i / (2 ** self.log_N_1)),
                                    np.sin(-2 * np.pi * k * i / (2 ** self.log_N_1)))
                     for i in range(int(len(y)))]))
        if plot:
            real_part = []
            img_part = []
            abs_part = []
            for i in fourier_amplitudes:
                real_part.append(i.real)
                img_part.append(i.imag)
                abs_part.append(abs(i))
            plt.plot(range(int(len(real_part))), real_part, label="Real")
            plt.plot(range(int(len(img_part))), img_part, label="Imaginary")
            plt.plot(range(int(len(abs_part))), abs_part, label="Absolute value")
            plt.legend()
            plt.title("normal fft")
            plt.show()

        return fourier_amplitudes

    def pauli_decomposition(self, function, matrix_or_pauli=None):
        Hamiltonian = np.zeros(shape=(2 ** (self.log_N_1*self.k_vector_dim), 2 ** (self.log_N_1*self.k_vector_dim)))
        for i in range(2 ** (self.log_N_1*self.k_vector_dim)):
            biti = self.int_to_bit(i)
            md_biti = [biti[i*(self.m + self.n):(i+1)*(self.m + self.n)] for i in range(self.k_vector_dim)]
            float = self.bit_to_float(md_biti)
            # print("float", float)
            Hamiltonian[i, i] = function(x=float[0], y=float[1])
        if matrix_or_pauli == None:
            return Hamiltonian
        else:
            return qml.pauli_decompose(Hamiltonian)

    def QFT_half(self, k, wires):
        padded_binary = bin(k)[2:].zfill(self.log_N_1)
        binary_input = [int(bit) for bit in padded_binary]
        # print(f"k={k},  binary_input={binary_input}")
        # 1,2...,self.log_N_1+1
        shifts = [2*np.pi * 2 ** (-i) for i in range(1, self.log_N_1 + 2)]
        # print("shifts", shifts)
        for i, wire in enumerate(wires):
            # print(f"i, wire {i, wire}")
            qml.Hadamard(wire)
            # print(f"binary_input {binary_input}")
            # print(f"binary_input[-(i+2):] {binary_input[-(i+2):]}")
            for idx, bin_state in enumerate(binary_input[-(i+2):]):
                # print(f"idx, bin_state {idx, bin_state}")
                # print(f"i, len(wires) {i, len(wires)}")
                if i == len(wires)-1:
                    idx += 1
                if bin_state == 1:
                    # print(f"happened, idx, wire {idx, wire}")
                    qml.PhaseShift(phi=shifts[idx], wires=wire)

    def QFT_half_inverse(self, k, wires):
        padded_binary = bin(k)[2:].zfill(self.log_N_1)
        binary_input = [int(bit) for bit in padded_binary]
        # print(f"k={k},  binary_input={binary_input}")
        # 1,2...,self.log_N_1+1
        shifts = [-2*np.pi * 2 ** (-i) for i in range(1, self.log_N_1 + 2)]
        # print("shifts", shifts)
        for i, wire in enumerate(wires):
            # print(f"i, wire {i, wire}")
            qml.Hadamard(wire)
            # print(f"binary_input {binary_input}")
            # print(f"binary_input[-(i+2):] {binary_input[-(i+2):]}")
            for idx, bin_state in enumerate(binary_input[-(i+2):]):
                # print(f"idx, bin_state {idx, bin_state}")
                # print(f"i, len(wires) {i, len(wires)}")
                if i == len(wires)-1:
                    idx += 1
                if bin_state == 1:
                    # print(f"happened, idx, wire {idx, wire}")
                    qml.PhaseShift(phi=shifts[idx], wires=wire)
    def QFT_half_2(self, k, wires):
        padded_binary = bin(k)[2:].zfill(self.log_N_1)
        binary_input = [int(bit) for bit in padded_binary]
        # 1,2...,self.log_N_1+1
        shifts = [2*np.pi * 2 ** -(i+1) for i in range(1, self.log_N_1 + 2)]
        # print(f"shifts {shifts}")
        # print(f"shifts {shifts}")
        for i, wire in enumerate(wires):
            # print(f"i, wire {i, wire}")
            qml.Hadamard(wire)
            # print(f"binary_input {binary_input}")
            # print(f"binary_input[-(i+2):] {binary_input[self.log_N_1-2-i:]}")
            for idx, bin_state in enumerate(binary_input[self.log_N_1-2-i:]):
                # print(f"idx, bin_state {idx, bin_state}")
                # print(f"i, len(wires) {i, len(wires)}")
                if i == 0:
                    idx += 1
                if bin_state == 1:
                    # print(f"happened, idx, wire {idx, wire}")
                    qml.PhaseShift(phi=shifts[idx], wires=wire)

    def test_function(self, k_):
        @qml.qnode(qml.device('default.qubit', wires=self.num_wires - 1))
        def inside_test_function(k__=k_):
            self.QFT_half(k=k__, wires=[i for i in range(self.num_wires-1)])
            return qml.state()

        # wires=[i for i in range(self.num_wires-1)]
        # print(qml.draw(inside_test_function, expansion_strategy='device')())
        return inside_test_function(k__=k_)

    def v_FFT(self, k_vector_, sinus_=None, plot=None):
        @qml.qnode(self.dev)
        def inside_v_FFT(sinus, k_vector=k_vector_):
            """Set 1st qubit into superposition for cos and sin terms"""
            qml.Hadamard(0)

            if sinus == True:
                qml.PauliX(1)

                qml.ctrl(qml.PhaseShift, control=(0), control_values=(0))(
                    phi=-np.pi / 4, wires=1)
                qml.ctrl(qml.PhaseShift, control=(0), control_values=(1))(
                    phi=np.pi / 4, wires=1)
                qml.PauliX(1)

            # """Start basis embedding for the frequency vector"""
            # for idx_k, k_ in enumerate(k_vector):
            #     cell_wires = range(idx_k * self.log_N_1 + 1, (idx_k + 1) * self.log_N_1 + 1)
            #     padded_binary = bin(k_)[2:].zfill(len(cell_wires))
            #     binary_input = [int(bit) for bit in padded_binary]
            #     qml.BasisEmbedding(features=binary_input, wires=cell_wires)

            """Apply cell_QFT to each component of the k_vector"""
            # (0) open-control QFT
            for idx_cell, k__ in enumerate(k_vector):
                cell_wires = [i for i in range(idx_cell * self.log_N_1 + 1,
                                               (idx_cell + 1) * self.log_N_1 + 1)]
                qml.ctrl(self.QFT_half, control=(0), control_values=(0))(k=k__,
                                                                         wires=cell_wires)
                # qml.ctrl(qml.QFT, control=(0), control_values=(0))(wires=cell_wires)
            # (1) closed-control QFT
            for idx_cell, k__ in enumerate(k_vector):
                cell_wires = range(idx_cell * self.log_N_1 + 1,(idx_cell + 1) * self.log_N_1 + 1)
                qml.ctrl(self.QFT_half_inverse, control=(0), control_values=(1))(
                    k=k__, wires=cell_wires)
                # qml.ctrl(qml.adjoint(qml.QFT), control=(0), control_values=(1))(wires=cell_wires)


            qml.Hadamard(0)

            return qml.density_matrix([i for i in range(self.num_wires)])

        # print(qml.draw(inside_v_FFT, expansion_strategy='device')(sinus=sinus_))
        density_matrix = inside_v_FFT(sinus=sinus_)
        observable_ = np.kron(np.array([[1, 0], [0, -1]]), self.observable__)
        trace_operator = np.matmul(density_matrix, observable_)

        if plot:
            plot_list = []
            for i in range(2 ** (self.k_vector_dim * self.log_N_1 + 1)):
                plot_list.append(trace_operator[i, i])
            cosinus_list = plot_list[:int(len(plot_list) / 2)]
            sinus_list = plot_list[int(len(plot_list) / 2):]
            cos_2 = []
            for i in range(len(sinus_list)):
                cos_2.append(cosinus_list[i] + sinus_list[i])

            integer_string = ''.join(str(num) for num in k_vector_)
            if sinus_:
                trig_func = "sin"
            else:
                trig_func = "cos"

            titel_ = f"{trig_func}(({int(integer_string)})*x)"
            plt.plot(np.linspace(0, (1-1/len(cosinus_list))*2*np.pi, len(cosinus_list)), cosinus_list, label="cosinus")
            plt.plot(np.linspace(0, (1-1/len(sinus_list))*2*np.pi, len(sinus_list)), sinus_list, label="sinus")
            plt.plot(np.linspace(0, (1-1/len(cos_2))*2*np.pi, len(cos_2)), cos_2, label="cosinus double")
            plt.title(titel_)
            plt.legend()
            plt.show()
        return np.trace(trace_operator)

    def ft(self, plot):
        complex_number_list = []
        abs_complex_number_list = []
        fourier_amplitudes = np.zeros(shape=tuple(2**self.log_N_1 for _ in range(self.k_vector_dim)), dtype=np.complex128)
        for k_vectors in list(itertools.product([i for i in range(2 ** self.log_N_1)],
                                                repeat=self.k_vector_dim)):
            # print(f"k_vectors {k_vectors} / {2 ** self.log_N_1}")
            cos_2 = self.v_FFT(k_vector_=k_vectors, sinus_=False)
            sin_2 = self.v_FFT(k_vector_=k_vectors, sinus_=True)
            # complex_number_list.append(complex(cos_2, sin_2))
            # abs_complex_number_list.append(abs(complex_number_list[-1]))
            # print(f"cos_2 {cos_2.real} {float(cos_2)}, {type(cos_2)}, {cos_2.dtype}")
            real_part1 = cos_2.real
            real_part2 = sin_2.real

            # Create a new complex number using the extracted real parts
            new_complex_num = complex(real_part1, real_part2)
            # print(f"cos_2 {cos_2.real} {cos_2.real} {float(cos_2)}, {type(cos_2)}, {cos_2.dtype}")
            #
            # print(f"sin_2 {sin_2.real} {sin_2.real} {float(sin_2)}, {type(sin_2)}, {sin_2.dtype}")
            # print("np.complex128(cos_2.real + 1j * sin_2.real)",np.complex128(cos_2.real + 1j * sin_2.real))
            # print("new_complex_num", new_complex_num)
            fourier_amplitudes[tuple(k_vectors)] = new_complex_num
            # print("fourier_amplitudes[tuple(k_vectors)]", fourier_amplitudes, np.shape(fourier_amplitudes))

        a = np.abs(fourier_amplitudes)
        b = fourier_amplitudes.imag
        c = fourier_amplitudes.real

        # Create a figure and subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot heatmap for variable a
        axes[0].imshow(a, cmap='hot', interpolation='nearest')
        axes[0].set_title('Absolute')
        axes[0].set_xlabel("k_1")
        axes[0].set_ylabel("k_2")
        axes[0].axis('off')  # Turn off axis labels

        # Plot heatmap for variable b
        axes[1].imshow(b, cmap='hot', interpolation='nearest')
        axes[1].set_title('Imaginary')
        axes[1].set_xlabel("k_1")
        axes[1].set_ylabel("k_2")
        axes[1].axis('off')  # Turn off axis labels

        # Plot heatmap for variable c
        axes[2].imshow(c, cmap='hot', interpolation='nearest')
        axes[2].set_title('Real')
        axes[2].set_xlabel("k_1")
        axes[2].set_ylabel("k_2")
        axes[2].axis('off')  # Turn off axis labels

        # Add a colorbar for each subplot
        for ax in axes:
            cbar = fig.colorbar(ax.images[0], ax=ax, orientation='vertical', shrink=0.6,
                                aspect=20)
            cbar.set_label('Colorbar Label')

        plt.tight_layout()
        plt.show()
        # if plot:
        #     # real_part = []
        #     # img_part = []
        #     # for i in fourier_amplitudes:
        #     #     real_part.append(i.real)
        #     #     img_part.append(i.imag)
        #     plt.plot(range(int(len(fourier_amplitudes.real))), fourier_amplitudes.real, label="Real")
        #     plt.plot(range(int(len(fourier_amplitudes.imag))), fourier_amplitudes.imag, label="Imaginary")
        #     plt.plot(range(int(len(np.abs(fourier_amplitudes)))), np.abs(fourier_amplitudes),
        #              label="Absolute value")
        #     plt.legend()
        #     plt.title("quantum fft")
        #     plt.show()
        return complex_number_list, abs_complex_number_list

class qaoa:
    def __init__(self, function, m, n, n_layers, lr):
        if function=="Rosenbrock":
            self.k_vector_dim = 2
        elif function=="Styb" or function=="Sin":
            self.k_vector_dim = 1

        function_dict = {"Rosenbrock": lambda x, y: 100 * (y ** 2 - y * (x ** 2) + x ** 4) + (x - 1) ** 2,
                         "Styb": lambda x: (x ** 4 - 16 * (x ** 2) + 5 * x) / 2,
                         "Sin": lambda x: np.sin(x)}

        self.function = function_dict[function]
        self.m = m
        self.n = n
        self.n_layers = n_layers
        self.log_N_1 = m + n
        self.shots = 1000
        self.optimizer = qml.GradientDescentOptimizer(stepsize=lr)
        self.pauli_decomposition()

    def int_to_bit(self, integer):
        return bin(integer)[2:].zfill(self.k_vector_dim*(self.log_N_1))

    def bit_to_float(self, x_vector):
        float_vector = []
        for bit_string in x_vector:
            float = 0
            for i, bit in enumerate(list(bit_string)):
                float += 2 ** (self.n - i) * int(bit)
            float_vector.append(float)
        return float_vector

    def pauli_decomposition(self, matrix_or_pauli=None):
        self.Hamiltonian = np.zeros(shape=(
        2 ** (self.log_N_1 * self.k_vector_dim), 2 ** (self.log_N_1 * self.k_vector_dim)))
        for i in range(2 ** (self.log_N_1 * self.k_vector_dim)):
            # print(f"pauli done {i/2 ** (self.log_N_1 * self.k_vector_dim)}")
            biti = self.int_to_bit(i)
            md_biti = [biti[i * (self.m + self.n):(i + 1) * (self.m + self.n)] for i in
                       range(self.k_vector_dim)]
            float = self.bit_to_float(md_biti)
            if self.k_vector_dim==1:
                self.Hamiltonian[i, i] = self.function(x=float[0])
            elif self.k_vector_dim==2:
                self.Hamiltonian[i, i] = self.function(x=float[0], y=float[1])
        # print(f"doing this")
        self.H_P = qml.pauli_decompose(self.Hamiltonian)
        # print(f"done with this")

    def qaoa_func(self,qaoa_param):
        @qml.qnode(device=qml.device('default.qubit', wires=self.log_N_1*self.k_vector_dim))
        def inside_qaoa_circuit(params):
            # Initial state preparation (all in |+⟩ state)
            for i in range(self.log_N_1*self.k_vector_dim):
                qml.Hadamard(wires=i)

            # QAOA alternating layers
            for layer in range(self.n_layers):
                # Apply e^(-i*gamma*H_P) layer
                qml.qaoa.cost_layer(params[0][layer], self.H_P)

                # Apply e^(-i*beta*H_B) layer, here mixer is defined as pauli-X
                qml.qaoa.mixer_layer(params[1][layer],
                                     qml.Hamiltonian([1 for _ in range(self.log_N_1*self.k_vector_dim)], [qml.PauliX(wire) for wire in range(self.log_N_1*self.k_vector_dim)]))

            return qml.expval(self.H_P)

        return inside_qaoa_circuit(params=qaoa_param)

    def qaoa_func_readout(self, qaoa_param):
        @qml.qnode(device=qml.device('default.qubit',
                                     wires=self.log_N_1 * self.k_vector_dim, shots=self.shots))
        def inside_qaoa_circuit_readout(params):
            # Initial state preparation (all in |+⟩ state)
            for i in range(self.log_N_1*self.k_vector_dim):
                qml.Hadamard(wires=i)

            # QAOA alternating layers
            for layer in range(self.n_layers):
                # Apply e^(-i*gamma*H_P) layer
                qml.qaoa.cost_layer(params[0][layer], self.H_P)

                # Apply e^(-i*beta*H_B) layer, here mixer is defined as pauli-X
                qml.qaoa.mixer_layer(params[1][layer],
                                     qml.Hamiltonian([1 for _ in range(self.log_N_1*self.k_vector_dim)],
                                                     [qml.PauliX(wire) for wire in
                                                      range(self.log_N_1*self.k_vector_dim)]))

            return qml.counts()

        return inside_qaoa_circuit_readout(params=qaoa_param)

    def qaoa_optimization(self, param, max_it):
        parameter_list = []
        cost_list = []

        cost = self.qaoa_func(param)
        parameter_list.append(param)
        cost_list.append(cost)
        for it in range(max_it):
            param = self.optimizer.step(self.qaoa_func, param)
            cost = self.qaoa_func(param)
            parameter_list.append(param)
            cost_list.append(cost)
            # print(f"it: {it}, cost: {cost}")

        return parameter_list, cost_list

    def readout_qaoa(self, final_param):
        counts = self.qaoa_func_readout(qaoa_param=final_param)
        sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
        return sorted_counts
    # original function
    # def plot_function_w_result(self, sorted_dict):
    #     max_float = sum([2 ** (self.n - i) for i in range(self.log_N_1)])
    #     floats = []
    #     true_func_value = np.zeros(shape=tuple([2 ** self.log_N_1] * self.k_vector_dim))
    #     for i in range(2 ** (self.log_N_1 * self.k_vector_dim)):
    #         biti = self.int_to_bit(i)
    #         md_biti = [biti[i * (self.log_N_1):(i + 1) * (self.log_N_1)] for i in
    #                    range(self.k_vector_dim)]
    #         indices = tuple([int(bit, 2) for bit in md_biti])
    #         floatie = self.bit_to_float(md_biti)
    #         floats.append(floatie)
    #         true_func_value[indices] = self.function(*floatie)
    #
    #     if self.k_vector_dim == 1:
    #         plt.plot(np.linspace(0, max_float, 2 ** (self.log_N_1 * self.k_vector_dim)),
    #                  list(true_func_value), label="function")
    #         histogram_floats = [self.bit_to_float([bitties])[0] for bitties in
    #                             sorted_dict.keys()]
    #         histogram_values = [counties / self.shots for counties in
    #                             sorted_dict.values()]
    #         plt.bar(histogram_floats, histogram_values, width=0.1, alpha=0.5,
    #                 label="histogram")
    #         plt.legend()
    #         plt.show()
    #
    #     elif self.k_vector_dim == 2:
    #         extent = [0, max_float, 0, max_float]  # [left, right, bottom, top]
    #         plt.imshow(true_func_value, cmap='hot', interpolation='nearest',
    #                    origin='lower', extent=extent)
    #         plt.colorbar()
    #         plt.xlabel('X Coordinate')
    #         plt.ylabel('Y Coordinate')
    #         plt.show()
    #
    #         histogram_data = np.zeros(
    #             shape=tuple([2 ** self.log_N_1] * self.k_vector_dim))
    #         for bitties, values in sorted_dict.items():
    #             md_biti = [bitties[i * (self.log_N_1): (i + 1) * (self.log_N_1)] for i
    #                        in
    #                        range(self.k_vector_dim)]
    #             indices = tuple([int(bit, 2) for bit in md_biti])
    #             histogram_data[indices] = values / self.shots
    #
    #         plt.imshow(histogram_data, cmap='hot', interpolation='nearest',
    #                    origin='lower', extent=extent)
    #         plt.colorbar()
    #         plt.xlabel('X Coordinate')
    #         plt.ylabel('Y Coordinate')
    #         plt.show()


    def plot_function_w_result(self, sorted_dict):
        max_float = sum([2**(self.n-i) for i in range(self.log_N_1)])
        floats = []
        true_func_value = np.zeros(shape=tuple([2**self.log_N_1]*self.k_vector_dim))
        for i in range(2 ** (self.log_N_1 * self.k_vector_dim)):
            biti = self.int_to_bit(i)
            md_biti = [biti[i * (self.log_N_1):(i + 1) * (self.log_N_1)] for i in
                       range(self.k_vector_dim)]
            indices = tuple([int(bit,2) for bit in md_biti])
            floatie = self.bit_to_float(md_biti)
            floats.append(floatie)
            true_func_value[indices] = self.function(*floatie)

        if self.k_vector_dim == 1:
            histogram_floats = [self.bit_to_float([bitties])[0] for bitties in sorted_dict.keys()]
            histogram_values = [counties / self.shots for counties in
                                sorted_dict.values()]
            # Create main figure and axes
            joint_grid = sns.JointGrid(x=np.linspace(0,max_float,2**self.log_N_1), y=list(true_func_value), marginal_ticks=True)

            # Overlay the lineplot on the main plot
            joint_grid.ax_joint.plot(np.linspace(0,max_float,2**self.log_N_1), list(true_func_value), 'r-')

            # Plot the histogram on the top
            joint_grid.ax_marg_x.hist(histogram_floats, weights=histogram_values, color='b', alpha=0.6,
                                      bins=30)

            # Set labels
            joint_grid.ax_joint.set_xlabel('X values')
            joint_grid.ax_joint.set_ylabel('Function Value')

            plt.show()

        elif self.k_vector_dim == 2:
            histogram_data = np.zeros(shape=tuple([2 ** self.log_N_1] * self.k_vector_dim))
            for bitties, values in sorted_dict.items():
                md_biti = [bitties[i * (self.log_N_1): (i + 1) * (self.log_N_1)] for i
                           in
                           range(self.k_vector_dim)]
                indices = tuple([int(bit, 2) for bit in md_biti])
                histogram_data[indices] = values / self.shots

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            sns.heatmap(true_func_value, cmap='hot', ax=ax1, cbar=True)
            ax1.set_title("Function Heatmap")
            sns.heatmap(histogram_data, cmap='hot', ax=ax2, cbar=True)
            ax2.set_title("Distribution Heatmap")
            plt.show()
#
#
#
# m = 2
# n = 2
# O = m+n
# instance = vFFT(k_vector_dim=1, m=m, n=n)
# # state = instance.test_function(k_=1)
# # instance.v_FFT(k_vector_=[1], sinus_=False, plot=True)
# for i in range(0,O**2):
#     instance.v_FFT(k_vector_=[i], sinus_=True, plot=True)
#
# exit()
# # print(f"state {state}")
# # angles = []
# # for state_ in state:
# #     angles.append(from_complex_to_angle(state_))
# # print(f"angles {angles}")
# print(instance.numpy_fft(plot=True))
# print(instance.ft(plot=True))
#
# exit("Done with both")
# print(instance.ft(plot=True))
# print(instance.normal_fft(plot=True))
# # for i in range(2**O):
# exit()
#     print("i",i)
#     if i==2**4:
#         break
#     instance.v_FFT(k_vector_=[i], sinus_=True, plot=True)

# def v_FFT(self, k_vector_, sinus_=None):
#     observable__ = self.pauli_decomposition(self.function)
#
#     @qml.qnode(self.dev)
#     def inside_v_FFT(sinus, k_vector=k_vector_):
#         """Set 1st qubit into superposition for cos and sin terms"""
#         qml.Hadamard(0)
#
#         """Start basis embedding for the frequency vector"""
#         for idx_k, k_ in enumerate(k_vector):
#             cell_wires = range(idx_k * self.log_N_1 + 1, (idx_k + 1) * self.log_N_1 + 1)
#             padded_binary = bin(k_)[2:].zfill(len(cell_wires))
#             binary_input = [int(bit) for bit in padded_binary]
#             qml.BasisEmbedding(features=binary_input, wires=cell_wires)
#
#         """Apply phase shift to a qubit in state 1 before the qft"""
#         if sinus == True:
#             for idx_k, k_ in enumerate(k_vector):
#                 cell_wires = range(idx_k * self.log_N_1 + 1,
#                                    (idx_k + 1) * self.log_N_1 + 1)
#                 padded_binary = bin(k_)[2:].zfill(len(cell_wires))
#                 binary_input = [int(bit) for bit in padded_binary]
#                 for j, i in enumerate(binary_input):
#                     if i == 1:
#                         acting_wire = j + 1
#                         break
#                     else:
#                         acting_wire = 1
#
#                 qml.ctrl(qml.PhaseShift, control=(0), control_values=(0))(
#                     phi=-np.pi / 4, wires=acting_wire)
#                 qml.ctrl(qml.PhaseShift, control=(0), control_values=(1))(
#                     phi=np.pi / 4, wires=acting_wire)
#
#         """Apply cell_QFT to each component of the k_vector"""
#         # (0) open-control QFT
#         for idx_cell in range(len(k_vector)):
#             cell_wires = [i for i in range(idx_cell * self.log_N_1 + 1,
#                                            (idx_cell + 1) * self.log_N_1 + 1)]
#             qml.ctrl(qml.QFT, control=(0), control_values=(0))(wires=cell_wires)
#
#         # (1) closed-control QFT
#         for idx_cell in range(len(k_vector)):
#             cell_wires = range(idx_cell * self.log_N_1 + 1,
#                                (idx_cell + 1) * self.log_N_1 + 1)
#             qml.ctrl(qml.adjoint(qml.QFT), control=(0), control_values=(1))(
#                 wires=cell_wires)
#
#         qml.Hadamard(0)
#         # return qml.counts(wires=[i for i in range(self.num_wires - 1)])
#         return qml.density_matrix([i for i in range(self.num_wires)])
#         # return qml.probs(wires=[i for i in range(2,self.num_wires)])
#
#     # print(qml.draw(inside_v_FFT, expansion_strategy='device')(sinus=sinus_))
#     density_matrix = inside_v_FFT(sinus=sinus_)
#     # observable__ = np.kron(np.array([[1, 0], [0, -1]]), np.identity(n=2 ** 7))
#     observable__ = np.kron(np.array([[1, 0], [0, -1]]), observable__)
#     # print(f"observable__ {observable__}, type {type(observable__)}, {np.shape(observable__)}")
#     # print(f"observable__ {density_matrix}, type {type(density_matrix)}, {np.shape(density_matrix)}")
#     trace_operator = np.matmul(density_matrix, observable__)
#     return np.trace(trace_operator)

# density_matrix = inside_v_FFT(sinus = sinus_)
# print(f"density_matrix {density_matrix}, type {type(density_matrix)}, {np.shape(density_matrix)}")
# print(
#     f"observable__ {observable__}, type {type(observable__)}, {np.shape(observable__)}")
# observable__ = np.kron(np.array([[1,0],[0,-1]]), np.identity(n=2**7))
# print(
#     f"observable__ {observable__}, type {type(observable__)}, {np.shape(observable__)}")
# trace_operator = np.matmul(density_matrix, observable__)
# print(
#     f"trace_operator {trace_operator}, type {type(trace_operator)}, {np.shape(trace_operator)}")
# import matplotlib.pyplot as plt
# plot_liste = []
# for i in range(2**8):
#     plot_liste.append(trace_operator[i,i])
# print(f"plot_liste {plot_liste}")
# new_plot_liste = []
# for i in range(int(len(plot_liste)/2)):
#     new_plot_liste.append(plot_liste[i] + plot_liste[i+int(len(plot_liste) / 2)])
# plt.plot(range(int(len(plot_liste)/2)), plot_liste[:int(len(plot_liste)/2)], label="cos")
# plt.plot(range(int(len(plot_liste)/2)), plot_liste[int(len(plot_liste) / 2):],
#             label="sin")
# plt.plot(range(len(new_plot_liste)), new_plot_liste, label="new")
# plt.legend()
# plt.show()
# exit()
# counts = inside_v_FFT(sinus=sinus_)
# print(f"counts {counts}")
# decimal_counts_0 = {}
# decimal_counts_1 = {}
#
# for key, value in counts.items():
#     hadam_key = int(key[0])
#     bin_num = int(key[1:], 2)
#     print(f" hadam_key={hadam_key}, bin_num={bin_num}")
#     if hadam_key == 1:
#         decimal_counts_1[bin_num] = value / (10000 * (2 ** (self.log_N_1 + 1)))
#     elif hadam_key == 0:
#         decimal_counts_0[bin_num] = value / (10000 * (2 ** (self.log_N_1 + 1)))
#     else:
#         exit("WTF")
#
# print(decimal_counts_0)
# print(decimal_counts_1)
#
# decimal_list_0 = []
# x_list_0 = []
# decimal_list_1 = []
# x_list_1 = []
# for i in range(2 ** (self.log_N_1 + 1)):
#     if i in decimal_counts_0:
#         decimal_list_0.append(decimal_counts_0[i])
#         x_list_0.append(i * 2 * np.pi / 2 ** (self.log_N_1 + 1))
#     else:
#         decimal_list_0.append(0)
#         x_list_0.append(i * 2 * np.pi / 2 ** (self.log_N_1 + 1))
#
#     if i in decimal_counts_1:
#         decimal_list_1.append(decimal_counts_1[i])
#         x_list_1.append(i * 2 * np.pi / 2 ** (self.log_N_1 + 1))
#     else:
#         decimal_list_1.append(0)
#         x_list_1.append(i * 2 * np.pi / 2 ** (self.log_N_1 + 1))
#
# cos_half = []
# for i, j in zip(decimal_list_0, decimal_list_1):
#     cos_half.append(i - j)
# import matplotlib.pyplot as plt
# plt.plot(x_list_0, decimal_list_0, label="cos(x)")
# plt.plot(x_list_1, decimal_list_1, label="sin(x)")
# plt.plot(x_list_1, cos_half, label="cos(2x)")
# plt.legend()
# plt.show()
# exit()
# N_q = 5
# true_list = []
# pred_list = []
# xx_list = []
# import matplotlib.pyplot as plt
# for i in range(2**N_q):
#     true, pred, xx = instance.test_circuit(n=2,m=3, x_=[i])
#     true_list.append(true)
#     pred_list.append(pred)
#     xx_list.append(xx)
#
# plt.plot(xx_list, true_list, label="true")
# plt.plot(xx_list, pred_list, label="pred")
# plt.legend()
# plt.show()
# exit("finished")
