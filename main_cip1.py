import pandas as pd
from CircuitClass import Distribution_func
from pennylane import numpy as np
import pennylane as qml
import operator
from utils import normal_GD, string_2_angles
import os
import sys

"parameters"
seed = int(sys.argv[5])

# the number of counting qubits used per variable
n_qbts = sys.argv[1]
n_qbts = n_qbts.strip('][').split(',')
n_qbts = [int(n_qbt) for n_qbt in n_qbts]

# repetitions of the circuit that is to be trained
training_lengths = sys.argv[2]
training_lengths = training_lengths.strip('][').split(',')
training_lengths = [int(training_length) for training_length in training_lengths]

# dimension of loss function
n_loss_param = int(sys.argv[6])

# maximum frequency of random loss function
rr = int(sys.argv[7])

# number of the job to be executed
job = int(sys.argv[8])

"check for resume mode"
resume_directory = os.path.join(r'../resume_job', 'job' + str(job))
resume_mode = False
if os.path.exists(resume_directory):
    loaded_resume = pd.read_csv(os.path.join(resume_directory, "df_resume.csv"))
    loaded_resume.drop('Unnamed: 0', inplace=True, axis=1)  # delete 'unnamed 0' row
    loaded_resume = loaded_resume.iloc[0].to_dict()

    previouse_training_length = loaded_resume["training_length"]
    previouse_lr = loaded_resume["learning_rate"]
    previouse_counting_qubits = loaded_resume["counting_qubits"]
    previouse_hyperparameter = loaded_resume["hyperparameter"]
    resume_mode = loaded_resume["hyperparameter"]
    print(f"loaded_resume {resume_mode}")

# optimisation parameters
iterations = 180

lrs = sys.argv[3]
lrs = lrs.strip('][').split(',')
lrs = [float(Lrs) for Lrs in lrs]

hyperparameters = sys.argv[4]
hyperparameters = hyperparameters.strip('][').split(',')
hyperparameters = [float(hyp) for hyp in hyperparameters]

N_gd = 10

# use single or seperate circuit method
sing = True
# if circuit parameters are initialised randomly or all as zero
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

"""Loop for all parameter combinations"""
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

                parent_directory = r'../Results/distr_search'
                directory_name = str(c6_n_qubits) + '_' + str(n_loss_param) + '_' + str(
                    seed) + \
                                 '_' + str(rr) + '_' + 'S' + str(sing)[0] + '_' + 'Z' + \
                                 str(zero_param)[0] + '_' + 'H' + str(Hadamard_index)[
                                     0] + \
                                 '_' + 'lr' + str(lr) + '_' + 'h' + str(
                    hyperparameter_a) + \
                                 '_' + 'trl' + str(training_length)

                directory = os.path.join(parent_directory, directory_name)

                if os.path.exists(directory):
                    continue

                print(f"Doing experiment {hyperparameter_a}, {lr}, {c6_n_qubits}, {training_length}")

                os.mkdir(directory)

                # save the highest resolution cost function landscape
                df_count_eval = pd.DataFrame(count_eval_dict)
                df_count_eval.to_csv(os.path.join(directory, 'count_eval.csv'))

                inst1 = Distribution_func(counting_qubits=c6_n_qubits,
                                          qf_qubits=n_loss_param,
                                          training_length=training_length, r=rr,
                                          seed=seed,
                                          n_loss_param=n_loss_param,
                                          Hadamard_index=Hadamard_index,
                                          superpos_hyperparameter=hyperparameter_a)

                if sing:
                    list_counting_parameters = []
                    list_gd_cost_list = []
                    list_gd_parameters_list = []
                    list_counts = []
                    list_cost = []

                    n_ = c6_n_qubits * n_loss_param
                    n_count_params = (n_ ** 2 + 3 * n_)
                    param_shape = (training_length, n_count_params)

                    if zero_param:
                        counting_parameters = 2 * np.pi * np.zeros(shape=param_shape,
                                                                   requires_grad=True)
                    else:
                        counting_parameters = 2 * np.pi * np.random.uniform(
                            size=param_shape, requires_grad=True)

                    start_GD_parameters = 2 * np.pi * np.random.uniform(
                        size=(N_gd, n_loss_param), requires_grad=True)

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
                    df = pd.DataFrame({'A': np_gd_parameters_list.flatten()},
                                     index=index)['A']

                    df_parameter_list = df.unstack(
                        level='dimension').swaplevel().sort_index()
                    df_parameter_list.to_csv(os.path.join(directory, 'df_gd_param.csv'))

                    '''Gradient Descent of Circuit6'''
                    if hyperparameter_a == 0:
                        for it in range(iterations):
                            # get current state
                            counts = inst1.eval_sing(counting_parameters, '6')
                            counts = dict(
                                sorted(counts.items(), key=operator.itemgetter(1),
                                       reverse=True))
                            list_counts.append(counts)

                            #calculate loss function for current parameters
                            cost = inst1.sing_search(counting_parameters)
                            list_cost.append(cost)

                            counting_parameters = opt.step(inst1.sing_search,
                                                           counting_parameters)
                            list_counting_parameters.append(counting_parameters.flatten())

                    else:
                        list_cost_2 = []
                        list_fid = []
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

                            counting_parameters, cost2 = opt.step_and_cost(
                                inst1.loss_func_sing, counting_parameters)

                            list_cost_2.append(cost2)

                            list_counting_parameters.append(
                                counting_parameters.flatten())

                        df_cost_2 = pd.DataFrame(np.array(list_cost_2))
                        df_cost_2.to_csv(os.path.join(directory, 'df_cost2.csv'))

                        df_fid = pd.DataFrame(np.array(list_fid))
                        df_fid.to_csv(os.path.join(directory, 'df_fid.csv'))

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
                    if hyperparameter_a == 0:
                        for it in range(iterations):
                            # get current state
                            counts = inst1.eval_sep(counting_parameters, '6')
                            counts = dict(
                                sorted(counts.items(), key=operator.itemgetter(1),
                                       reverse=True))
                            list_counts.append(counts)

                            # calculate loss function for current parameters
                            cost = inst1.sep_search(counting_parameters)
                            list_cost.append(cost)

                            counting_parameters = opt.step(inst1.sep_search,
                                                           counting_parameters)
                            list_counting_parameters.append(
                                counting_parameters.flatten())

                    else:
                        list_cost_2 = []
                        list_fid = []
                        for it in range(iterations):
                            counts = inst1.eval_sep(counting_parameters, '6')
                            counts = dict(
                                sorted(counts.items(), key=operator.itemgetter(1),
                                       reverse=True))
                            list_counts.append(counts)

                            cost = inst1.sep_search(counting_parameters)
                            list_cost.append(cost)

                            fid = inst1.fidelity(counting_parameters, 'Sep')
                            list_fid.append(fid)

                            counting_parameters, cost2 = opt.step_and_cost(
                                inst1.loss_func_sep, counting_parameters)

                            list_cost_2.append(cost2)

                            list_counting_parameters.append(
                                counting_parameters.flatten())

                        df_cost_2 = pd.DataFrame(np.array(list_cost_2))
                        df_cost_2.to_csv(os.path.join(directory, 'df_cost2.csv'))

                        df_fid = pd.DataFrame(np.array(list_fid))
                        df_fid.to_csv(os.path.join(directory, 'df_fid.csv'))

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
                print(f"resume_dict {resume_dict}")

                if not os.path.exists(resume_directory):
                    os.mkdir(resume_directory)
                df_resume = pd.DataFrame(resume_dict)
                df_resume.to_csv(os.path.join(resume_directory, 'df_resume.csv'))

resume_dict["Resume"] = [False]
df_resume = pd.DataFrame(resume_dict)
df_resume.to_csv(os.path.join(resume_directory, 'df_resume.csv'))