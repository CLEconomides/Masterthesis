import seaborn as sns
import pandas as pd
from CircuitClass import Distribution_func
from utils import string_inverter, count_eval_and_distribution, cumulative_average, count_eval_and_distribution2
import numpy as np
import operator
import matplotlib.pyplot as plt

def plot_gds(instance):
    for gd_list in instance.gd_cost_list:
        plt.plot(range(len(gd_list)), gd_list, color='blue')


def plot_best_n_2_count(instance):
    min_eval_list, cummulative_value_list, min_eval_index = min_eval_AND_percentages(
        instance)
    min_dic_value = min(instance.eval_count.values())
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 8), sharex=True,
                                   sharey=True)
    sns.set_style('darkgrid')
    for gd_list in instance.gd_cost_list:
        sns.lineplot(x=range(len(gd_list)), y=gd_list, ax=ax1, color='orange')
    sns.lineplot(x=range(len(min_eval_list)), y=min_eval_list, ax=ax1, color='blue')
    sns.lineplot(x=range(len(cummulative_value_list)), y=cummulative_value_list, ax=ax2,
                 color='green')
    sns.lineplot(x=range(len(instance.df_cost)), y=instance.df_cost, ax=ax2)
    sns.lineplot(x=range(len(instance.df_cost)),
                 y=[min_dic_value for _ in range(len(instance.df_cost))], ax=ax1)
    #sns.lineplot(x=range(len(min_eval_index)), y=min_eval_index, ax=ax2)
    ax1.set_title('Gradient Descents')
    ax2.set_title('DF Cost' + str(instance.seed) + str(instance.rr))
    plt.tight_layout()
    plt.show()

def min_eval_AND_percentages(instance):
    n_cq = instance.n * instance.n_loss_param
    cummulative_value_list = []
    min_eval_list = []
    min_eval_index = []
    for i, counts in enumerate(instance.count_list):
        eval, value = count_eval_and_distribution2(eval_counts=instance.eval_count,
                                                   distributed_counts=counts,
                                                   n=n_cq ** 2)

        cummulative_value_list.append(sum(value))
        min_eval = min(eval)
        min_eval_list.append(min_eval)
        min_eval_index.append(value[eval.index(min_eval)])

    return min_eval_list, cummulative_value_list, min_eval_index

def plot_cumulative_averages(counts, counting_qubits, instance, n):
    #plots the cumulative averages of counts
    avrg, temp_list = cumulative_average(counts=counts, counting_qubits=counting_qubits,
                                         instance=instance, n=n)
    plt.plot(range(len(temp_list)), temp_list)
    plt.plot(range(len(temp_list)), [avrg for _ in range(len(temp_list))])
    # plt.show()


def plot_first_n_counts(list_of_eval_lists, normal_gd_list, n):
    arr = np.array(list_of_eval_lists)
    arr = np.transpose(arr)

    for i, eval_list in enumerate(arr):
        plt.plot(range(len(eval_list)), eval_list, label=i)

    plt.plot(range(len(normal_gd_list)), normal_gd_list, label='GD')
    plt.legend()
    # plt.show()


def plot_evaluated_counts_and_percentages(counts, counting_qubits, instance, n, save):
    eval_list, value_list = count_eval_and_distribution(counts=counts, counting_qubits=counting_qubits, instance=instance, n=n)
    eval_list = [float(eval_list[i]) for i in range(len(eval_list))]

    ax = sns.lineplot(data=eval_list, label='model evaluations', color='red')
    ax2 = ax.twinx()
    sns.lineplot(data=value_list, label='fidelity perc', color='blue', ax=ax2)
    #ax.figure.legend()
    plt.show()


def plot_heatmap(counts, counting_qubits, df_result=None, save=None):
    dataframe_counts = counts_2_df_heatmap(counts, counting_qubits)

    # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 8), sharex=True,
    #                                     sharey=True)
    sns.set_style('darkgrid')
    sns.heatmap(data=dataframe_counts)
    #, ax=ax1
    #sns.heatmap(data=df_result, ax=ax2)
    #ax1.set_title('counts')
    #ax2.set_title('function')
    # plt.title('Iteration'+str(df_result))
    plt.tight_layout()
    plt.show()

    # if save: TODO add savefig feature
    #     plt.savefig()

def counts_2_heatmap(counts, counting_qubits):
    heatmap = np.zeros(shape=(2**counting_qubits,2**counting_qubits))
    for key, value in counts.items():
        strings = [key[i*counting_qubits:(i+1)*counting_qubits] for i in range(int(len(key)/counting_qubits))]
        strings = [string_inverter(string) for string in strings]
        heatmap[int(strings[0],2)][int(strings[1],2)] = value

    return heatmap

def counts_2_df_heatmap(counts, counting_qubits):
    heatmap = counts_2_heatmap(counts, counting_qubits)
    delta = 2 * np.pi / ((2 ** counting_qubits) - 1)
    columns = [round(i*delta,2) for i in range(2**counting_qubits)]
    df = pd.DataFrame(heatmap)
    df.columns = columns
    df.index = columns
    return df

# def first_count_averages(counts, counting_qubits, function):
#     counts = dict(sorted(counts.items(), key=operator.itemgetter(1), reverse=True))
#     evaluation = []
#     values = []
#     for key, value in counts.items():
#         angle = key_2_angles(key, counting_qubits)
#         evaluation.append(function(angle))
#         values.append(value)
#     plt.plot(range(len(evaluation)), evaluation, label = 'evaluations')
#     plt.plot(range(len(values)), values, label='values')
#     plt.legend()
#     plt.show()




# def get_min_count(counts, counting_qubits, n_loss_param):
#     for key, value in counts.items():
#         parameters =