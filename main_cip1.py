import pandas as pd
import seaborn as sns
from CircuitClass import Distribution_func
from pennylane import numpy as np
import pennylane as qml
import operator
# from orqviz.pca import (get_pca, perform_2D_pca_scan, plot_pca_landscape,
#                             plot_optimization_trajectory_on_pca)
# from plots import counts_2_df_heatmap, plot_heatmap, plot_evaluated_counts_and_percentages, plot_cumulative_averages, count_eval_and_distribution, plot_first_n_counts
from utils import normal_GD, string_2_angles
import os
import sys
import time
import shutil

"""folder_string = "./resultsss" + str(int(sys.argv[8]))
folder_string = r"C:\Users\Constantin\Desktop\resultsss11"

for string in [x[0] for x in os.walk(folder_string)][1:]:
    # c6_n_qubits = 3
    # n_loss_param=2
    # seed=1
    # rr=5
    # lr=0.05
    # hyperparameter_a = 0
    # training_length = 2

    k = 0
    for j, i in enumerate(string):
        if i == "/":
            k += 1
        if k == 2:
            string = string[(j + 1):]
            break
    c6_n_qubits = int(string[0])

    n_loss_param = int(string[2])

    seed = int(string[4])

    rr = int(string[6])

    lr = ""
    Letter = False
    for j, letter in enumerate(string):
        if letter == "r":
            Letter = True
        if letter == "_" and Letter == True:
            break
        if Letter:
            lr += string[j + 1]

    lr= lr[:-1]
    lr = float(lr)

    hyp = ""
    Letter = False
    for j, letter in enumerate(string):
        if letter == "h":
            Letter = True
        if letter == "_" and Letter == True:
            break
        if Letter:
            hyp += string[j + 1]

    hyperparameter_a = float(hyp[:-1])

    training_length = int(string[-1])

    # inst2 = Distribution_func(counting_qubits=n_qbts[-1], qf_qubits=n_loss_param,
    #                                   training_length=1, r=rr, seed=seed,
    #                                   n_loss_param=n_loss_param,
    #                                   Hadamard_index=Hadamard_index,
    #                                   superpos_hyperparameter=0)
    #
    # count_eval_dict = {}
    # len_bin_string = n_qbts[-1]*n_loss_param
    # for count in [bin(i)[2:].zfill(len_bin_string) for i in
    #               range(2 ** (len_bin_string))]:
    #     angles = string_2_angles(count, n_qbts[-1])
    #     count_eval_dict[count] = [float(inst2.quantum_function_eval(angles))]
    sing = True
    zero_param = True
    Hadamard_index = True
    N_gd = 10
    iterations = 240

    parent_directory = r'./results4'
    directory_name = str(c6_n_qubits) + '_' + str(n_loss_param) + '_' + str(seed) + \
                     '_' + str(rr) + '_' + 'S' + str(sing)[0] + '_' + 'Z' + \
                     str(zero_param)[0] + '_' + 'H' + str(Hadamard_index)[0] + \
                     '_' + 'lr' + str(lr) + '_' + 'h' + str(hyperparameter_a) + \
                     '_' + 'trl' + str(training_length)

    directory = os.path.join(parent_directory, directory_name)
    os.mkdir(directory)

    inst1 = Distribution_func(counting_qubits=c6_n_qubits,
                              qf_qubits=n_loss_param,
                              training_length=training_length, r=rr,
                              seed=seed,
                              n_loss_param=n_loss_param,
                              Hadamard_index=Hadamard_index,
                              superpos_hyperparameter=hyperparameter_a)


    def loss_func_sing(counting_parameters):
        costie = inst1.sing_search(counting_parameters)
        fid = inst1.fidelity(counting_parameters, 'Sing')
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
        if True:
            # if hyperparameter_a == 0:
            for it in range(iterations):

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
        df_info.to_csv(os.path.join(directory, 'df_info.csv'))"""

"parameters"
seed = int(sys.argv[5])

n_qbts = sys.argv[1]
n_qbts = n_qbts.strip('][').split(',')
n_qbts = [int(n_qbt) for n_qbt in n_qbts]

training_lengths = sys.argv[2]
training_lengths = training_lengths.strip('][').split(',')
training_lengths = [int(training_length) for training_length in training_lengths]

n_loss_param = int(sys.argv[6])

rr = int(sys.argv[7])

job = int(sys.argv[8])

"check for resume mode"
resume_directory = os.path.join(r'./resume_job', 'job' + str(job))
resume_mode = False
if os.path.exists(resume_directory):
    loaded_resume = pd.read_csv(os.path.join(resume_directory, "df_resume.csv"))
    loaded_resume.drop('Unnamed: 0', inplace=True, axis=1)  # delete 'unnamed 0' row
    loaded_resume = loaded_resume.iloc[0].to_dict()

    previouse_training_length = loaded_resume["training_length"]
    previouse_lr = loaded_resume["learning_rate"]
    previouse_counting_qubits = loaded_resume["counting_qubits"]
    previouse_hyperparameter = loaded_resume["hyperparameter"]

    resume_mode = True

iterations = 180
lrs = sys.argv[3]
lrs = lrs.strip('][').split(',')
lrs = [float(Lrs) for Lrs in lrs]

hyperparameters = sys.argv[4]
hyperparameters = hyperparameters.strip('][').split(',')
hyperparameters = [float(hyp) for hyp in hyperparameters]

N_gd = 10

sing = True
zero_param = True
Hadamard_index = True

"create highest resolution cost function landscape"
inst2 = Distribution_func(counting_qubits=n_qbts[-1], qf_qubits=n_loss_param,
                                      training_length=1, r=rr, seed=seed,
                                      n_loss_param=n_loss_param,
                                      Hadamard_index=Hadamard_index,
                                      superpos_hyperparameter=0)

count_eval_dict = {}
len_bin_string = n_qbts[-1]*n_loss_param
for count in [bin(i)[2:].zfill(len_bin_string) for i in range(2 ** (len_bin_string))]:
    angles = string_2_angles(count, n_qbts[-1])
    count_eval_dict[count] = [float(inst2.quantum_function_eval(angles))]


Resume = True
for hyperparameter_a in hyperparameters:
    for lr in lrs:
        for c6_n_qubits in n_qbts:
            for training_length in training_lengths:

                if resume_mode and hyperparameter_a == previouse_hyperparameter and lr == previouse_lr and c6_n_qubits == previouse_counting_qubits and training_length == previouse_training_length:
                    resume_mode = False
                    continue

                elif resume_mode:
                    continue

                parent_directory = r'./results5'
                directory_name = str(c6_n_qubits) + '_' + str(n_loss_param) + '_' + str(
                    seed) + \
                                 '_' + str(rr) + '_' + 'S' + str(sing)[0] + '_' + 'Z' + \
                                 str(zero_param)[0] + '_' + 'H' + str(Hadamard_index)[
                                     0] + \
                                 '_' + 'lr' + str(lr) + '_' + 'h' + str(
                    hyperparameter_a) + \
                                 '_' + 'trl' + str(training_length)


                directory = os.path.join(parent_directory, directory_name)
                os.mkdir(directory)

                df_count_eval = pd.DataFrame(count_eval_dict)
                df_count_eval.to_csv(os.path.join(directory, 'count_eval.csv'))

                inst1 = Distribution_func(counting_qubits=c6_n_qubits,
                                          qf_qubits=n_loss_param,
                                          training_length=training_length, r=rr,
                                          seed=seed,
                                          n_loss_param=n_loss_param,
                                          Hadamard_index=Hadamard_index,
                                          superpos_hyperparameter=hyperparameter_a)


                def loss_func_sing(counting_parameters):
                    costie = inst1.sing_search(counting_parameters)
                    fid = inst1.fidelity(counting_parameters, 'Sing')
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
                    # print(param_shape)

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
                        pd.DataFrame({'A': np_gd_parameters_list.flatten()},
                                     index=index)[
                            'A']

                    df_parameter_list = df.unstack(
                        level='dimension').swaplevel().sort_index()
                    df_parameter_list.to_csv(os.path.join(directory, 'df_gd_param.csv'))

                    '''Gradient Descent of Circuit6'''
                    starttime = time.time()
                    if hyperparameter_a == 0:
                        for it in range(iterations):
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
                    param_shape = (n_loss_param, training_length, n_count_params)

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
                        pd.DataFrame({'A': np_gd_parameters_list.flatten()},
                                     index=index)[
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

                resume_dict = {"counting_qubits": [c6_n_qubits],
                               "hyperparameter": [hyperparameter_a],
                               "learning_rate": [lr],
                               "training_length": [training_length],
                               "Resume": [Resume]}

                if not os.path.exists(resume_directory):
                    os.mkdir(resume_directory)
                df_resume = pd.DataFrame(resume_dict)
                df_resume.to_csv(os.path.join(resume_directory, 'df_resume.csv'))

resume_dict["Resume"] = [False]
df_resume = pd.DataFrame(resume_dict)
df_resume.to_csv(os.path.join(resume_directory, 'df_resume.csv'))
