import numpy as np
import pennylane as qml

def string_inverter(string):
    return string[::-1]


def filename(n, n_loss_param, seed, rr, sing, zero_param, Hadamard_index, lr, superpos_hyperparameter, trl):
    return f"{n}_{n_loss_param}_{seed}_{rr}_S{str(sing)[0]}_Z{str(zero_param)[0]}_H{str(Hadamard_index)[0]}_lr{lr}_h{superpos_hyperparameter}_trl{trl}"
    # return str(n) + '_' + str(n_loss_param) + '_' + str(
    #     seed) + '_' + str(rr) + '_' + 'S' + str(sing)[
    #     0] + '_' + 'Z' + str(zero_param)[0] + '_' + 'H' + \
    # str(Hadamard_index)[0] + '_' + 'lr' + str(float(lr)) \
    # + '_' + 'h' + str(superpos_hyperparameter) \
    # + '_' + 'trl' + str(trl)

# print(f"filename={filename(4,2,3,1,True,True,True,0.5,0,4)} and filename1 = {filename1(4,2,3,1,True,True,True,0.5,0,4)}")
# print(f"filenames are equal = {filename(4,2,3,1,True,True,True,0.5,0,4) == filename1(4,2,3,1,True,True,True,0.5,0,4)}")



def find_from_n(count_dict, n_qbts_to, n):
    #finds the minimum evaluation from the n first entries of the ordered instance.count_list[-1] ( starting from largest)
    evaluations = []
    values =  []
    new_angle_list = []
    angle_list=[]
    k=0
    for string, value in eval_instance.count_list[-1].items():
        if k==n:
            break
        new_string = bin_converter(string, eval_instance.instance, n_qbts_to)
        evaluations.append(eval_instance.eval_count[new_string])
        new_angle_list.append(string_2_angles(new_string,8))
        angle_list.append(string_2_angles(string,eval_instance.instance.counting_qubits))
        values.append(value)
        k+=1
    position = evaluations.index(min(evaluations))
    print('min eval', min(evaluations))
    print(new_angle_list)
    min_new_angle = new_angle_list[position]
    print('min_original angle', min_new_angle)
    print('orig eval', eval_instance.instance.quantum_function_eval(min_new_angle))
    min_angle = angle_list[position]
    print('new angle', min_angle)
    print('old eval', eval_instance.instance.quantum_function_eval(min_angle))


    return min_angle


def bin_converter(string, n_loss_param, n_qbts_to):
    #instance parameter was taken out
    z_fill_parameter = int(n_qbts_to/n_loss_param)
    n_qbts_from = len(string) / n_loss_param
    # print("n_qbts from",n_qbts_from)
    qbt_diff = n_qbts_to/n_loss_param - n_qbts_from
    # print("qbt diff", qbt_diff)

    strings = [string[int(i * n_qbts_from):int((i + 1) * n_qbts_from)] for i in
               range(n_loss_param)]
    strings = [string_inverter(string) for string in strings]

    new_string = ''
    if qbt_diff>0:
        for coord in strings:
            new_coord = bin(int(coord, 2) * int(2 ** qbt_diff))[2:].zfill(z_fill_parameter)
            new_string += string_inverter(new_coord)
    elif qbt_diff==0:
        for coord in strings:
            new_string += string_inverter(coord)
        print("Qbt diff =", 0)
    else:
        for coord in strings:
            old_coord = int(coord,2)*2*np.pi / (2 ** n_qbts_from)
            i = old_coord * (2**n_qbts_to) / (2*np.pi)
            i = round(i)
            i = bin(int(i))[2:].zfill(z_fill_parameter)
            new_string += string_inverter(i)

            # differences = [abs(old_coord-i*2*np.pi/(2**n_qbts_to)) for i in range(2**n_qbts_to)]
            # new_coord = bin(int(differences.index(min(differences))))[2:].zfill(n_qbts_to)
            # new_string += string_inverter(new_coord)

    return new_string

def GD_from_selected_point(instancee, lr, min_key, iterations):
    opt = qml.AdamOptimizer(lr)
    print('min key in GD func',min_key)

    costs=[]
    parameters = []
    for it in range(iterations):
        cost = instancee.instance.quantum_function_eval(min_key)
        print('cost',cost)
        min_key = opt.step(instancee.instance.quantum_function_eval, min_key)
        cost = instancee.instance.quantum_function_eval(min_key)
        costs.append(cost)
        parameters.append(min_key)

    return costs, parameters


def distance_to_GlMin(instance, local_min, new_gd_cost, min_param):
    dictionary = instance.eval_count
    key_list = list(dictionary.keys())
    eval_list = list(dictionary.values())

    position = eval_list.index(min(eval_list))
    global_min = key_list[position]
    gloabl_min = string_2_angles(global_min, 8)

    return np.linalg.norm(np.array(gloabl_min) - np.array(local_min)),\
           np.linalg.norm(np.array(gloabl_min) - np.array(new_gd_cost)),\
           np.linalg.norm(np.array(gloabl_min) - np.array(min_param))
#evaluates all counts once so that they can be accessed aech time
def evaluate_counts(counts, instance):
    count_eval_dict = {}
    for count in counts.key():
        angles = string_2_angles(count)
        count_eval_dict[count]=instance.quantum_function_eval(angles)
    return count_eval_dict

#gradient descent function
def normal_GD(opt, iterations, instance, parameters):
    costs = []
    parameterS = []
    for it in range(iterations):
        costs.append(instance.quantum_function_eval(parameters))
        parameters = opt.step(instance.quantum_function_eval, parameters)
        parameterS.append(parameters)
    return costs, parameterS

def count_eval_and_distribution2(eval_counts, distributed_counts, n):
    step = 0

    tot_val = 0
    for value in distributed_counts.values():
        tot_val += value

    evaluations = []
    percentages = []

    for key, value in distributed_counts.items():

        if step == n:
            break

        evaluation = eval_counts[key]

        evaluations.append(evaluation)
        percentages.append(value / tot_val)
        step += 1

    return evaluations, percentages

def count_eval_and_distribution(counts, counting_qubits, instance, n):
    #returns a list of the evaluated counts and their respective percentage
    step=0

    tot_val = 0
    for value in counts.values():
        tot_val += value

    evaluations = []
    pre_norm_percentages = []

    for key, value in counts.items():

        if step == n:
            break

        angles = string_2_angles(key, counting_qubits)
        evaluation = instance.quantum_function_eval(angles)


        evaluations.append(evaluation)
        pre_norm_percentages.append(value / tot_val)
        step += 1

    # norm_percentages = pre_norm_percentages / tot_val
    norm_percentages = pre_norm_percentages
    print('evaluations', evaluations)
    evaluations = [float(evals) for evals in evaluations]

    return evaluations, norm_percentages



def cumulative_average(counts, counting_qubits, instance, n):
    #calculates the cumulative average of the evaluated counts for up to n counts
    #counts have to be pre-ordered and start by the count with the highest value
    step=0
    tot_val = 0
    avrg = 0
    temp_list = []

    for key, value in counts.items():
        if step == n:
            break

        angles = string_2_angles(key, counting_qubits)
        evaluation = instance.quantum_function_eval(angles)
        avrg += value*evaluation
        tot_val += value
        temp_list.append(avrg/tot_val)

        step+=1
    return avrg/tot_val, temp_list

def string_2_bin(string, counting_qubits):
    diff_strings = [string[i * counting_qubits:(i + 1) * counting_qubits] for i in
                    range(int(len(string) / counting_qubits))]
    diff_strings = [string_inverter(string) for string in diff_strings]

    return [int(ind_strings, 2) for ind_strings in diff_strings]

def string_2_angles(string, counting_qubits):
    #maps the binary string to its representing angle
    delta = 2 * np.pi / ((2 ** counting_qubits) - 1)

    diff_strings = [string[i * counting_qubits:(i + 1) * counting_qubits] for i in
               range(int(len(string) / counting_qubits))]

    diff_strings = [string_inverter(string) for string in diff_strings]

    return [int(ind_strings,2)*delta for ind_strings in diff_strings]

####################################################################################
def bin2angle_check(counts, angles, counting_qubits):
    liste = []
    for key, value in counts.items():
        angle = [0 for _ in range(int(len(key)/counting_qubits))]
        for idx, string in enumerate(key):
            if int(string)==1:
                angle[int(idx/counting_qubits)] += angles[idx%counting_qubits]
            else:
                continue
        liste.append((angle, value))
    return liste

def bin2angle_scatter(counts, angles, counting_qubits):
    liste = []
    for key, value in counts.items():
        n_parameters = int(len(key)/counting_qubits)
        parameter_angles = [0 for _ in range(n_parameters)]
        for idx, string in enumerate(key):
            if int(string)==1:
                parameter_angles[int(idx/counting_qubits)] += angles[idx%counting_qubits]
            else:
                continue
        liste.append((parameter_angles, value))
    scatter_data = [[] for _ in range(n_parameters)]
    for param_angle, value in liste:
        # print('anlges', angle)
        for ind_indx, indiv_angle in enumerate(param_angle):
            scatter_data[ind_indx]+=[indiv_angle for _ in range(value)]

    return scatter_data

def bin2angle_heatmap(counts, angles, counting_qubits):
    n_parameters_per_dimension = 2 ** len(angles)
    shape=()
    for i in range(2):
        shape += (n_parameters_per_dimension,)
    heatmap = np.zeros(shape=shape)
    for key, value in counts.items():
        n_parameters = int(len(key)/counting_qubits)
        parameter_angles = [int(key[i*counting_qubits:(i+1)*counting_qubits],2) for i in range(n_parameters)]
        heatmap[parameter_angles[0], parameter_angles[1]] += value

    return heatmap

def counts2plot_check(counts, angles):
    liste = []
    for key, value in counts.items():
        angle = 0
        for idx, string in enumerate(key):
            if int(string)==1:
                angle += angles[idx]
            else:
                continue
        liste += [angle for _ in range(value)]
    return liste

