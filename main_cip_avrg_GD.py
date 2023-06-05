from CircuitClass import Average_GradientDescent
from pennylane import numpy as np
import pennylane as qml
import os
import sys
import multiprocessing
from multiprocessing import Process
import pickle

"Learning parameters"
repetitions = 2
iterations = 250
_lr = 0.02
def run_experiment(rep, l, n_q, r, lr,
                   entangling_block_layers, n_param,
                   seed, folder_path, gd_parameters):

    np.random.seed(seed)

    GD_parameters = gd_parameters[rep]

    opt0 = qml.AdamOptimizer(lr)

    filepath = os.path.join(folder_path, f"rep{rep}_l{l}")
    pickle_file = os.path.join(filepath, 'finish_status.pkl')
    info_file = os.path.join(filepath, "info.pkl")
    #continue unfinished experiment
    if os.path.exists(filepath):
        with open(pickle_file, 'rb') as file:
            finish_status = pickle.load(file)

        with open(info_file, 'rb') as file:
            info_dict = pickle.load(file)

        # Load the parameters
        normal_params_list = list(np.load(
            os.path.join(filepath, 'normal_params_list.npy')
            , allow_pickle=True))
        normal_cost_list = list(np.load(
            os.path.join(filepath, 'normal_cost_list.npy')
            , allow_pickle=True))
        average_params_list = list(np.load(
            os.path.join(filepath, 'average_params_list.npy')
            , allow_pickle=True))
        average_cost_list = list(np.load(
            os.path.join(filepath, 'average_cost_list.npy')
            , allow_pickle=True))
        average_point_cost_list = list(np.load(
            os.path.join(filepath, 'average_point_cost_list.npy')
            , allow_pickle=True))

        #load most recent parameters
        normal_params = normal_params_list[-1]
        average_params = average_params_list[-1]

    #create new experiment
    else:
        os.mkdir(filepath)
        finish_status = False
        with open(pickle_file, 'wb') as file:
            pickle.dump(finish_status, file)

        info_dict = {"rep": rep,
                     "l": l,
                     "n_q": n_q,
                     "r": r,
                     "entangling_block_layers": entangling_block_layers,
                     "n_param": n_param,
                     "seed": seed,
                     "lr": lr,
                     "Iteration": 0}
        with open(info_file, 'wb') as file:
            pickle.dump(info_dict, file)

        normal_params = GD_parameters
        average_params = GD_parameters

        normal_params_list = []
        normal_cost_list = []

        average_params_list = []
        # cost of the averge
        average_cost_list = []
        # cost of the point that average params lead to
        average_point_cost_list = []

    # skip the function if it was already completed
    if finish_status:
        return

    class_instance = Average_GradientDescent(n_count_qubits=n_q, r=r,
                                             entangling_block_layers=entangling_block_layers,
                                             n_param=n_param, L=l)


    start_it = info_dict["Iteration"]
    # training loop
    for it in range(start_it, iterations):
        normal_params_list.append(normal_params)
        normal_params, normal_cost = opt0.step_and_cost(class_instance.quantum_function,
                                                        normal_params)

        normal_cost_list.append(normal_cost)

        average_params_list.append(average_params)
        average_params, avrg_cost = opt0.step_and_cost(
            class_instance.Average_GradientDescent, average_params)
        average_cost_list.append(avrg_cost)

        avrg_point_cost = class_instance.quantum_function(average_params)
        average_point_cost_list.append(avrg_point_cost)

        # step_and_cost returns the cost before the optimisation step
        if it == iterations - 1:
            normal_cost = class_instance.quantum_function(normal_params)
            normal_cost_list.append(normal_cost)
            normal_params_list.append(normal_params)

            average_params_list.append(average_params)
            avrg_point_cost = class_instance.quantum_function(average_params)
            average_point_cost_list.append(avrg_point_cost)
            avrg_cost = class_instance.Average_GradientDescent(average_params)
            average_cost_list.append(avrg_cost)

        # Save the parameters after each iteration
        np.save(os.path.join(filepath, 'normal_params_list.npy'), normal_params_list)
        np.save(os.path.join(filepath, 'normal_cost_list.npy'),
                np.array(normal_cost_list))
        np.save(os.path.join(filepath, 'average_params_list.npy'), average_params_list)
        np.save(os.path.join(filepath, 'average_cost_list.npy'),
                np.array(average_cost_list))
        np.save(os.path.join(filepath, 'average_point_cost_list.npy'),
                np.array(average_point_cost_list))

        info_dict["Iteration"] = it + 1
        with open(info_file, 'wb') as file:
            pickle.dump(info_dict, file)

    finish_status = True
    with open(pickle_file, 'wb') as file:
        pickle.dump(finish_status, file)

def worker(experiment):
    # rep, l = experiment
    print(f"experiment {experiment}")
    run_experiment(*experiment)


if __name__ == "__main__":

    "Circuit parameters/Set-up - on CIP"
    n_count_qubits = int(sys.argv[1])
    _r = int(sys.argv[2])
    _entangling_block_layers = int(sys.argv[3])
    _n_param = int(sys.argv[4])
    _seed = int(sys.argv[5])
    # print(f"n_count_qubits {n_count_qubits}, {type(n_count_qubits)}"
    #       f"r {_r}"
    #       f"entangling_block_layers {_entangling_block_layers}"
    #       f"n_param {_n_param}"
    #       f"seed {_seed}")
    # L = [_lr, _lr * 2, _lr * 3, _lr * 4]
    L = int(sys.argv[6]) * 0.01

    # the function that is created is defined by r, entangling_block_layers, n_param and seed
    # folder_path = fr"C:\Users\Constantin\Desktop\avrg_GD_test_{seed}"
    _folder_path = fr"../Results/avrg_GD/seed{_seed}_r{_r}_n_param{_n_param}_ebl{_entangling_block_layers}_nq{n_count_qubits}"
    if not os.path.exists(_folder_path):
        os.mkdir(_folder_path)

    np.random.seed(_seed)
    "Circuit parameters/Set-up - on Laptop"
    # n_count_qubits = 5
    # _r = 6
    # _entangling_block_layers = 3
    # _n_param = 2
    # _seed = 333
    # L = _lr*2
    # # L = [_lr, _lr*2, _lr*3, _lr*4]
    #
    # #the function that is created is defined by r, entangling_block_layers, n_param and seed
    # _folder_path = fr"C:\Users\Constantin\Desktop\MA_test\avrg_GD\seed{_seed}_r{_r}_n_param{_n_param}_ebl{_entangling_block_layers}_nq{n_count_qubits}"
    # if not os.path.exists(_folder_path):
    #     os.mkdir(_folder_path)
    #
    # np.random.seed(_seed)

    _gd_parameters = np.array([np.random.uniform(0, 2 * np.pi, size=_n_param) for _ in range(repetitions)],
                                 requires_grad=True)


    experiments = []
    for rep_ in range(repetitions):
        experiments.append((rep_, L, n_count_qubits, _r, _lr,
               _entangling_block_layers, _n_param,
               _seed, _folder_path, _gd_parameters))

    # pool = multiprocessing.Pool(processes=num_processes)
    # pool.map(worker, experiments)

    processes = [Process(target=worker, args=(i,)) for i in experiments]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

