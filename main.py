import pandas as pd
import seaborn as sns
from CircuitClass import Distribution_func
from pennylane import numpy as np
import pennylane as qml
import operator
# from orqviz.pca import (get_pca, perform_2D_pca_scan, plot_pca_landscape,
#                             plot_optimization_trajectory_on_pca)
import matplotlib.pyplot as plt
# from plots import counts_2_df_heatmap, plot_heatmap, plot_evaluated_counts_and_percentages, plot_cumulative_averages, count_eval_and_distribution, plot_first_n_counts
from utils import normal_GD, string_2_angles, string_2_bin
# string_2_angles, cumulative_average
import os
import sys
import time
from pennylane._grad import grad as get_gradient



"parameters"
seed = 44
n_qbts = [5]
training_lengths = [1]
n_loss_param = 2
rr = 5
iterations = 240
lrs = [0.05]
hyperparameters = [0.2]

N_gd = 10

sing = True
zero_param = True
Hadamard_index = True

inst2 = Distribution_func(counting_qubits=n_qbts[0],
                          qf_qubits=n_loss_param,
                          training_length=training_lengths[0],
                          r=rr, seed=seed,
                          n_loss_param=n_loss_param,
                          Hadamard_index=Hadamard_index,
                          superpos_hyperparameter=hyperparameters[0])

# count_eval_dict = {}
# angle_list = []
# for count in [bin(i)[2:].zfill(n_loss_param * 5) for i in
#               range(2 ** (n_loss_param * 5))]:
#     angles = string_2_angles(count, 5)
#     angle_list.append(angles)
#     count_eval_dict[count] = float(inst2.quantum_function_eval(angles))
# df_results = np.zeros(shape=(2 ** (5), 2 ** (5)))
# for count, value in count_eval_dict.items():
#     pos = string_2_bin(count, 5)
#     print('pos', pos)
#     print('value',value)
#     df_results[pos[0]][pos[1]] = value
# sns.heatmap(df_results)
# # plt.title(file_specific_path)
# plt.show()

for hyperparameter_a in hyperparameters:
    for lr in lrs:
        for c6_n_qubits in n_qbts:
            for training_length in training_lengths:
                # parent_directory = r'./results'
                # directory_name = str(c6_n_qubits) + '_' + str(n_loss_param) + '_' + str(
                #     seed) + '_' + str(rr) + '_' + 'S' + str(sing)[0] + '_' + 'Z' + \
                #                  str(zero_param)[
                #                      0] + '_' + 'H' + str(Hadamard_index)[
                #                      0] + '_' + 'lr' + str(
                #     lr) + '_' + 'h' + str(hyperparameter_a)
                # directory = os.path.join(parent_directory, directory_name)
                # os.mkdir(directory)

                # df_count_eval = pd.DataFrame(count_eval_dict)
                # df_count_eval.to_csv(os.path.join(directory, 'count_eval.csv'))

                inst1 = Distribution_func(counting_qubits=c6_n_qubits,
                                          qf_qubits=n_loss_param,
                                          training_length=training_lengths, r=rr,
                                          seed=seed,
                                          n_loss_param=n_loss_param,
                                          Hadamard_index=Hadamard_index,
                                          superpos_hyperparameter=hyperparameter_a)


                def loss_func_sing(counting_parameters):
                    costie = inst1.sing_search(counting_parameters)
                    fid = inst1.fidelity(counting_parameters, 'Sing')
                    print('costie', costie)
                    print('fid', fid)
                    return costie - hyperparameter_a * fid


                if sing:
                    list_eval_list = []
                    list_value_list = []
                    list_counting_parameters = []
                    list_gd_cost_list = []
                    list_gd_parameters_list = []
                    list_counts = []
                    list_cost = []
                    list_fid = []

                    n_ = c6_n_qubits * n_loss_param
                    n_count_params = (n_ ** 2 + 3 * n_)
                    param_shape = (training_length, n_count_params)
                    print(param_shape)

                    if zero_param:
                        counting_parameters = 2 * np.pi * np.zeros(shape=param_shape,
                                                                   requires_grad=True)
                    else:
                        counting_parameters = 2 * np.pi * np.random.uniform(
                            size=param_shape,
                            requires_grad=True)

                    start_GD_parameters = 2 * np.pi * np.random.uniform(
                        size=(N_gd, n_loss_param),
                        requires_grad=True)

                    '''Gradient Descent'''
                    gd_lr = lr

                    gd_opt = qml.AdamOptimizer(gd_lr)
                    opt = qml.AdamOptimizer(lr)

                    for start_GD_parameter in start_GD_parameters:
                        gd_cost_list, gd_parameter_list = normal_GD(opt=gd_opt,
                                                                    iterations=iterations,
                                                                    instance=inst1,
                                                                    parameters=start_GD_parameter)
                        list_gd_cost_list.append(gd_cost_list)
                        print(gd_cost_list)
                        list_gd_parameters_list.append(gd_parameter_list)
                    exit()

                    df_cost_list = pd.DataFrame(np.array(list_gd_cost_list),
                                                columns=['cost_it' + str(i) for i in
                                                         range(len(
                                                             list_gd_cost_list[0]))])
                    df_cost_list.to_csv(os.path.join(directory, 'df_gd_cost.csv'))
                    names = ['n_gds', 'iteration', 'dimension']
                    np_gd_parameters_list = np.array(list_gd_parameters_list)
                    index = pd.MultiIndex.from_product(
                        [range(s) for s in np_gd_parameters_list.shape],
                        names=names)
                    df = \
                    pd.DataFrame({'A': np_gd_parameters_list.flatten()}, index=index)[
                        'A']

                    df_parameter_list = df.unstack(
                        level='dimension').swaplevel().sort_index()
                    df_parameter_list.to_csv(os.path.join(directory, 'df_gd_param.csv'))

                    '''Gradient Descent of Circuit6'''
                    starttime = time.time()
                    if True:
                        # if hyperparameter_a == 0:
                        for it in range(iterations):
                            print(it)
                            counts = inst1.eval_sing(counting_parameters, '6')
                            counts = dict(
                                sorted(counts.items(), key=operator.itemgetter(1),
                                       reverse=True))
                            list_counts.append(counts)

                            cost = inst1.sing_search(counting_parameters)
                            list_cost.append(cost)

                            counting_parameters = opt.step(inst1.sing_search,
                                                           counting_parameters)
                            list_counting_parameters.append(
                                counting_parameters.flatten())

                    else:
                        for it in range(iterations):
                            counts = inst1.eval_sing(counting_parameters, '6')
                            counts = dict(
                                sorted(counts.items(), key=operator.itemgetter(1),
                                       reverse=True))
                            list_counts.append(counts)
                            cost = inst1.sing_search(counting_parameters)
                            list_cost.append(cost)

                            fid = inst1.fidelity(counting_parameters, 'Sing')
                            list_fid.append(fid)

                            prev_count_param = counting_parameters
                            counting_parameters, cost2 = opt.step_and_cost(
                                inst1.loss_func_sing,
                                counting_parameters)
                            list_cost.append(cost2)

                            list_counting_parameters.append(
                                counting_parameters.flatten())

                    df_counting_parameters = pd.DataFrame(
                        np.array(list_counting_parameters))

                    df_counting_parameters.to_csv(
                        os.path.join(directory, 'df_counting_param.csv'))

                    df_cost = pd.DataFrame(np.array(list_cost))
                    df_cost.to_csv(os.path.join(directory, 'df_cost.csv'))

                    df_counts = pd.DataFrame(list_counts)
                    df_counts.to_csv(os.path.join(directory, 'df_counts.csv'))

                    df_info = pd.DataFrame(
                        {'n_c': [c6_n_qubits], 'n_loss_params': [n_loss_param],
                         'seed': [seed],
                         'rr': [rr]})
                    df_info.to_csv(os.path.join(directory, 'df_info.csv'))

                else:
                    list_eval_list = []
                    list_value_list = []
                    list_counting_parameters = []
                    list_gd_cost_list = []
                    list_gd_parameters_list = []
                    list_counts = []
                    list_cost = []

                    n_count_params = (c6_n_qubits ** 2 + 3 * c6_n_qubits)
                    param_shape = (n_loss_param, training_lengths, n_count_params)

                    if zero_param:
                        counting_parameters = 2 * np.pi * np.zeros(shape=param_shape,
                                                                   requires_grad=True)
                    else:
                        counting_parameters = 2 * np.pi * np.random.uniform(
                            size=param_shape,
                            requires_grad=True)

                    start_GD_parameters = 2 * np.pi * np.random.uniform(
                        size=(N_gd, n_loss_param),
                        requires_grad=True)

                    '''Gradient Descent'''
                    for start_GD_parameter in start_GD_parameters:
                        gd_cost_list, gd_parameter_list = normal_GD(opt=gd_opt,
                                                                    iterations=iterations,
                                                                    instance=inst1,
                                                                    parameters=start_GD_parameter)
                        list_gd_cost_list.append(gd_cost_list)
                        list_gd_parameters_list.append(gd_parameter_list)

                    df_cost_list = pd.DataFrame(np.array(list_gd_cost_list),
                                                columns=['cost_it' + str(i) for i in
                                                         range(len(
                                                             list_gd_cost_list[0]))])
                    df_cost_list.to_csv(os.path.join(directory, 'df_gd_cost.csv'))
                    names = ['n_gds', 'iteration', 'dimension']
                    np_gd_parameters_list = np.array(list_gd_parameters_list)
                    index = pd.MultiIndex.from_product(
                        [range(s) for s in np_gd_parameters_list.shape],
                        names=names)
                    df = \
                    pd.DataFrame({'A': np_gd_parameters_list.flatten()}, index=index)[
                        'A']

                    df_parameter_list = df.unstack(
                        level='dimension').swaplevel().sort_index()
                    df_parameter_list.to_csv(os.path.join(directory, 'df_gd_param.csv'))

                    '''Gradient Descent of Circuit6'''
                    starttime = time.time()
                    for it in range(iterations):
                        counts = inst1.eval_sep(counting_parameters, '6')
                        counts = dict(
                            sorted(counts.items(), key=operator.itemgetter(1),
                                   reverse=True))
                        list_counts.append(counts)

                        fid = inst1.fidelity(counting_parameters, search_type='Sep')

                        cost = inst1.sep_search(counting_parameters)
                        list_cost.append(cost)

                        counting_parameters = opt.step(inst1.sep_search,
                                                       counting_parameters)
                        list_counting_parameters.append(counting_parameters.flatten())

                    df_counting_parameters = pd.DataFrame(
                        np.array(list_counting_parameters))

                    df_counting_parameters.to_csv(
                        os.path.join(directory, 'df_counting_param.csv'))

                    df_cost = pd.DataFrame(np.array(list_cost))
                    df_cost.to_csv(os.path.join(directory, 'df_cost.csv'))

                    df_counts = pd.DataFrame(list_counts)
                    df_counts.to_csv(os.path.join(directory, 'df_counts.csv'))

                    df_info = pd.DataFrame(
                        {'n_c': [c6_n_qubits], 'n_loss_params': [n_loss_param],
                         'seed': [seed],
                         'rr': [rr]})
                    df_info.to_csv(os.path.join(directory, 'df_info.csv'))



