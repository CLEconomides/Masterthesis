from CircuitClass import Average_GradientDescent
from pennylane import numpy as np
import pennylane as qml
import pandas as pd
import os

"Learning parameters"
repetitions = 1
iterations = 10
lr = 0.05

random_seeds = np.random.randint(0, 1000, size=(repetitions))
seed_dict = {'seed'+ str(i): seedie for i, seedie in enumerate(random_seeds)}
print(f"seed_dict {seed_dict}")


opt0 = qml.GradientDescentOptimizer(lr)


"Circuit parameters"
n_count_qubits = 3
r = 11
trainable_block_layers = 4
n_param = 3
L = 2 * lr

folder_path = r"C:\Users\Constantin\Desktop\avrg_GD_test"

np.save(os.path.join(folder_path, 'seed_dict.npy'), seed_dict)

# Load the dictionary


for rep in range(repetitions):
    print(f"Rep= {rep}")
    filepath = os.path.join(folder_path,f"rep{str(rep)}")
    os.mkdir(filepath)

    np.random.seed(random_seeds[rep])

    gd_parameters = np.array([np.random.uniform(0, 2 * np.pi) for i in range(n_param)],
                             requires_grad=True)

    class_instance = Average_GradientDescent(n_count_qubits=n_count_qubits,
                                                               r=r,
                                                               trainable_block_layers=trainable_block_layers,
                                                               n_param=n_param,
                                                               L=L)

    normal_params = gd_parameters
    average_params = gd_parameters

    normal_params_list =[]
    normal_cost_list =[]

    average_params_list = []
    #cost of the averge
    average_cost_list = []
    #cost of the point that average params lead to
    average_point_cost_list = []


    for it in range(iterations):
        print(f"Iteration = {it}")
        normal_params_list.append(normal_params)
        normal_params, normal_cost = opt0.step_and_cost(class_instance.quantum_function, normal_params)

        normal_cost_list.append(normal_cost)

        average_params_list.append(average_params)
        average_params, avrg_cost = opt0.step_and_cost(class_instance.Average_GradientDescent, average_params)
        average_cost_list.append(avrg_cost)

        avrg_point_cost = class_instance.quantum_function(average_params)
        average_point_cost_list.append(avrg_point_cost)

        #step_and_cost returns the cost before the optimisation step
        if it == iterations-1:
            print(f"it:{it}")
            print("im here")
            normal_cost = class_instance.quantum_function(normal_params)
            normal_cost_list.append(normal_cost)
            normal_params_list.append(normal_params)

            average_params_list.append(average_params)
            avrg_point_cost = class_instance.quantum_function(average_params)
            average_point_cost_list.append(avrg_point_cost)
            avrg_cost = class_instance.Average_GradientDescent(average_params)
            average_cost_list.append(avrg_cost)

    print(f"normal_params_list: {normal_params_list}", f"normal_cost_list:{normal_cost_list}")
    # Save the parameters
    np.save(os.path.join(filepath,'normal_params_list.npy'), normal_params_list)
    np.save(os.path.join(filepath, 'normal_cost_list.npy'), np.array(normal_cost_list))

    np.save(os.path.join(filepath, 'average_params_list.npy'), average_params_list)
    np.save(os.path.join(filepath, 'average_cost_list.npy'),
            np.array(average_cost_list))
    np.save(os.path.join(filepath, 'avrg_point_cost.npy'),
            np.array(avrg_point_cost))

    # Load the parameters
    # loaded_parameters = np.load('parameters.npy', allow_pickle=True)

loaded_normal_params_list = np.load(os.path.join(filepath,'normal_params_list.npy'), allow_pickle=True)
loaded_normal_cost_list = np.load(os.path.join(filepath,'normal_cost_list.npy'), allow_pickle=True)
print(f"loaded_normal_params_list: {loaded_normal_params_list}")
print(f"loaded_normal_cost_list: {loaded_normal_cost_list}")
loaded_seed_dict = np.load(os.path.join(folder_path, 'seed_dict.npy'), allow_pickle=True).item()
print(f"loaded_seed_dict {loaded_seed_dict}")
