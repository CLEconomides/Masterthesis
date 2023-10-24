import pandas as pd
from pennylane import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plots
from CircuitClass import Distribution_func
import utils
import operator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle


class Average_GradientDescent_file_opener:
    def __init__(self, file_path):
        self.file_path = file_path
        with open(os.path.join(file_path,"info.pkl"), 'rb') as file:
            info_dict = pickle.load(file)

        self.rep = info_dict["rep"]
        # {"rep": rep,
        #  "l": l,
        #  "n_q": n_q,
        #  "r": r,
        #  "entangling_block_layers": entangling_block_layers,
        #  "n_param": n_param,
        #  "seed": seed,
        #  "lr": lr}
class Distribution_func_file_opener:
    def __init__(self, n, n_loss_param, seed, rr, sing, zero_param, Hadamard_index, lr, superpos_hyperparameter, trl, name=None, save_directory=None):
        self.n = n
        self.n_loss_param = n_loss_param
        self.seed = seed
        self.rr = rr
        self.sing = sing
        self.zero_param = zero_param
        self.Hadamard_index = Hadamard_index
        self.lr = lr
        self.superpos_hyperparameter = superpos_hyperparameter
        self.trl = trl

        self.save_directory = save_directory

        self.instance = Distribution_func(counting_qubits=n, qf_qubits=n_loss_param,
                                          training_length=1, r=rr, seed=seed,
                                          n_loss_param=n_loss_param,
                                          Hadamard_index=True,
                                          superpos_hyperparameter=superpos_hyperparameter)
        n_gds = 10


        filepath = r'C:\Users\Constantin\Desktop\results5\results5'
        self.file_specific_path = utils.filename(n=n, n_loss_param=n_loss_param, seed=seed,
                                            rr=rr, sing=sing, zero_param=zero_param,
                                            Hadamard_index=Hadamard_index,lr=lr,
                                            superpos_hyperparameter=superpos_hyperparameter,
                                            trl=trl)

        if name is not None:
            filepath = name
            info_filepath = r'df_info.csv'
            df_info = pd.read_csv(os.path.join(filepath, info_filepath))
            df_info.drop('Unnamed: 0', inplace=True,
                               axis=1)  # delete 'unnamed 0' row
            self.info = df_info.iloc[
                0].to_dict()  # get number of rows for for-loop
            self.n = self.info['n_c']
            self.n_loss_param = self.info['n_loss_params']
            self.seed = self.info['seed']
            self.rr = self.info['rr']
            self.trl = int(name[-1])
            prev_letter = None
            _lr=""
            _superpos_hyperparameter=""
            for idx, letter in enumerate(name):
                if prev_letter=="l" and letter=="r":
                    for i in range(5):
                        letter = name[idx+1+i]
                        if letter == "_":
                            break
                        _lr+=letter
                    self.lr = float(_lr)
                elif prev_letter=="_" and letter=="h":
                    for i in range(5):
                        letter = name[idx+1+i]
                        if letter == "_":
                            break
                        _superpos_hyperparameter+=letter
                    self.superpos_hyperparameter = float(_superpos_hyperparameter)

                prev_letter = letter
            print(f"lr={self.lr}, trl={self.trl}, h={self.superpos_hyperparameter}")

            self.instance = Distribution_func(counting_qubits=self.n, qf_qubits=self.n_loss_param,
                                              training_length=1, r=self.rr, seed=self.seed,
                                              n_loss_param=self.n_loss_param,
                                              Hadamard_index=True,
                                              superpos_hyperparameter=superpos_hyperparameter)
            self.file_specific_path = utils.filename(n=self.n, n_loss_param=self.n_loss_param,
                                                     seed=self.seed,
                                                     rr=self.rr, sing=self.sing,
                                                     zero_param=self.zero_param,
                                                     Hadamard_index=self.Hadamard_index,
                                                     lr=self.lr,
                                                     superpos_hyperparameter=self.superpos_hyperparameter,
                                                     trl=self.trl)

        self.filepath = os.path.join(filepath, self.file_specific_path)


        "evaluation dictionary"
        count_eval_filepath = r'count_eval.csv'
        df_count_eval = pd.read_csv(os.path.join(filepath, count_eval_filepath))
        df_count_eval.drop('Unnamed: 0', inplace=True, axis=1)  # delete 'unnamed 0' row
        self.eval_count = df_count_eval.iloc[
            0].to_dict()  # get number of rows for for-loop

        self.n_qbts_to = len(list(self.eval_count.keys())[0]) #get the number of qubits used for the eval dict

        "average cost"
        cost_filepath = r'df_cost.csv'
        df_cost = pd.read_csv(os.path.join(filepath, cost_filepath))
        self.df_cost = df_cost[df_cost.columns[1]].to_numpy()

        "counting circuit parameters"
        counting_param_filepath = r'df_counting_param.csv'
        df_counting_param = pd.read_csv(os.path.join(filepath, counting_param_filepath))
        df_counting_param.drop('Unnamed: 0', inplace=True,
                               axis=1)  # delete 'unnamed 0' row
        self.df_counting_param = df_counting_param.to_numpy()

        "counts per iteration"
        counts_filepath = r'df_counts.csv'
        df_counts = pd.read_csv(os.path.join(filepath, counts_filepath))
        df_counts.drop('Unnamed: 0', inplace=True, axis=1)  # delete 'unnamed 0' row
        self.count_list = [dict(
            sorted(df_counts.iloc[row].to_dict().items(), key=operator.itemgetter(1),
                   reverse=True)) for row in
                           range(df_counts.shape[0])]  # get number of rows for for-loop

        "Classical Gradient Descent costs"
        df_gd_cost_filepath = r'df_gd_cost.csv'
        df_gd_cost = pd.read_csv(os.path.join(filepath, df_gd_cost_filepath))
        df_gd_cost.drop('Unnamed: 0', inplace=True, axis=1)  # delete 'unnamed 0' row
        self.gd_cost_list = [df_gd_cost.iloc[row].to_list() for row in
                             range(df_gd_cost.shape[0])]

        "Classical Gradient Descent parameters"
        df_gd_param_filepath = r'df_gd_param.csv'
        df_gd_param = pd.read_csv(os.path.join(filepath, df_gd_param_filepath))
        self.gd_param = [[df_gd_param.iloc[it * n_gds + gd][2:].to_list() for it in
                          range(int(df_gd_param.shape[0] / n_gds))] for gd in range(
            n_gds)]
        # list of all gradient descents, and a list for each with all iterations

        self.eval_count_loss_dict = self.count_eval_loss()
        print("Last count list", self.count_list[-1])
        #best evaluation from reduced dictionary
        self.key_with_smallest_value, self.value_with_smallest_value = self.smallest_eval_from_dict()
        #best evaluation of n2 most probable solutions
        self.best_eval_from_n2, self.best_string_from_n2, self.count_of_best_in_n2 = self._find_from_n2(self.count_list[-1])
        #most probable evaluation
        self.max_eval, self.string_max_eval = self.most_probable(self.count_list[-1])
        #create figure 1
        #self.plot_GD_v_Avg()
        #create figure 2
        #self.plot_eval_v_count(iteration=-1)


        print(f"Attention, int(self.n_qbts_to/self.n_loss_param)={int(self.n_qbts_to/self.n_loss_param)}, self.n={self.n}")
        # for it in range(0,240,10):
        #     self.plot_eval_v_count(iteration=it)
        # print("first")
        # plots.plot_heatmap(self.eval_count, self.n)
        # plots.plot_heatmap(self.eval_count_loss_dict, int(self.n_qbts_to/self.n_loss_param))

    #we have a dictionary containing a grid with evaluated points of the cost function.
    #we also have a dictionary containing for each iteration the count of all points,
    #each point represents a point on a grid which is more sparese than the dictionary
    # containing the grid with evaluated points of the cost function.
    #Finally we would like to use the same keys from the count_dictionary and assign to
    #each string=point its evaluation from the closest point from the evaluation_dict
    def count_eval_loss(self):
        count_eval_loss_dict = {}
        for key in self.count_list[0].keys():
            # print("key", key)
            # print(self.n_qbts_to)
            new_key = utils.bin_converter(key,self.n_loss_param, self.n_qbts_to)
            # print(f"new_key={new_key} / key ={key}")
            if new_key==key:
                count_eval_loss_dict = self.eval_count
                break
            # print("new key", new_key)
            # print("len new key", len(new_key))
            count_eval_loss_dict[key] = self.eval_count[new_key]
            # print("eval", self.eval_count[new_key])
        return count_eval_loss_dict

    def smallest_eval_from_dict(self):
        key_with_smallest_value = min(self.eval_count_loss_dict, key=self.eval_count_loss_dict.get)
        value_with_smallest_value = self.eval_count_loss_dict[key_with_smallest_value]
        return key_with_smallest_value, value_with_smallest_value

    #plot evaluation v count/probability
    def plot_eval_v_count(self, iteration):
        total_shots=0
        for shots in self.count_list[iteration].values():
            total_shots += shots
        eval_v_count = [[eval, self.count_list[iteration][key]/total_shots] for key, eval in self.eval_count_loss_dict.items()]
        plt.scatter([evaluation[0] for evaluation in eval_v_count], [string_count[1] for string_count in eval_v_count])
        plt.yscale('log')
        plt.title(f"Figure 2 ,iteration: {iteration}, seed: {self.seed}, n_l: {self.n_loss_param}, r: {self.rr}")
        plt.legend()

        if self.save_directory is not None:
            fig2_path = self.file_specific_path + "_fig2.png"
            figure_name = os.path.join(self.save_directory, fig2_path)
            plt.savefig(figure_name)

        plt.figure()


    def last_count_evaluation(self):
        return [[key, value] for key, value in self.count_list[-1].items()]

    def most_probable(self, dictionary):
        #gets the string with the highest evalutation associated
        evaluations = [value for value in dictionary.values()]
        keys = [key for key in dictionary.keys()]
        max_eval = max(evaluations)
        string_max_eval = keys[evaluations.index(max_eval)]
        print(len(string_max_eval))
        most_probable_eval = self.eval_count_loss_dict[string_max_eval]
        return most_probable_eval, string_max_eval

    def _find_from_n2(self, count_dict):
        N = (self.n * self.n_loss_param)**2
        sorted_count_dict = dict(sorted(count_dict.items(),
                                        key = lambda x: x[1],
                                        reverse=True)[:N])
        details = [[key, self.eval_count_loss_dict[key], count] for key, count in sorted_count_dict.items()]
        evaluations = [detail[1] for detail in details]
        best_eval_from_n2 = min(evaluations)
        best_string_from_n2 = details[evaluations.index(best_eval_from_n2)][0]
        count_of_best_in_n2 = details[evaluations.index(best_eval_from_n2)][2]

        return best_eval_from_n2, best_string_from_n2, count_of_best_in_n2



    def find_from_n(self, count_dict, n):
        # finds the minimum evaluation from the n first entries of the ordered instance.count_list[-1] ( starting from largest)
        evaluations = []
        values = []
        new_angle_list = []
        angle_list = []
        k = 0
        for string, value in count_dict.items():
            if k == n:
                break
            new_string = utils.bin_converter(string, self.instance, self.n_qbts_to)
            evaluations.append(self.eval_count[new_string])
            new_angle_list.append(utils.string_2_angles(new_string, 8))
            angle_list.append(
                utils.string_2_angles(string, self.instance.counting_qubits))
            values.append(value)
            k += 1
        position = evaluations.index(min(evaluations))
        # print('min eval', min(evaluations))
        print(new_angle_list)
        min_new_angle = new_angle_list[position]
        # print('min_original angle', min_new_angle)
        # print('orig eval', self.instance.quantum_function_eval(min_new_angle))
        min_angle = angle_list[position]
        # print('new angle', min_angle)
        # print('old eval', self.instance.quantum_function_eval(min_angle))

        return min_angle

    def sorted_classical_gd(self):
        #takes the last value of the GDs and sorts out the smallest of all tries
        smallest_gd_list = [gd_list[-1] for gd_list in self.gd_cost_list]
        return min(smallest_gd_list), smallest_gd_list

    def plot_GD_v_Avg(self):
        "plot the cost function of the average v each GD v the best of n**2 for each iteration"
        max_iteration = 50

        max_eval_per_iteration = [self.eval_count_loss_dict[
                                      self.most_probable(dictionary=x)[1]] for x in self.count_list][:max_iteration]
        best_of_n2 = [self.eval_count_loss_dict[self._find_from_n2(x)[1]] for x in self.count_list][:max_iteration]

        sns.lineplot(y=self.df_cost[:max_iteration], x=range(len(self.df_cost[:max_iteration])), label="Average cost")
        step_dict = {}
        for j, gd_cost in enumerate(self.gd_cost_list):
            for k in [-0.8,-0.6,-0.4,-0.2]:
                if  k - 0.2 < gd_cost[-1] and gd_cost[-1] < k:
                    if str(k) not in step_dict:
                        step_dict[str(k)] = 0
                    if step_dict[str(k)] >= 2:
                        break
                    step_dict[str(k)] += 1
                    sns.lineplot(y=gd_cost[:max_iteration], x=range(len(gd_cost[:max_iteration])),
                             label="Gradient Descent")

        sns.lineplot(y=max_eval_per_iteration, x=range(len(max_eval_per_iteration)), label= "Most probable")
        sns.lineplot(y=best_of_n2, x=range(len(best_of_n2)), label= "Best from n^2 most prob.")
        sns.lineplot(y=[self.value_with_smallest_value for _ in range(max_iteration)], x=range(max_iteration),
                     label="Smallest evaluation of grid")
        plt.legend()

        if self.save_directory is not None:
            fig1_path = self.file_specific_path + "_fig1.png"
            figure_name = os.path.join(self.save_directory, fig1_path)
            plt.savefig(figure_name)

        plt.figure()

    def _do_DimRed(self, data, DimRed_type=None):
        print(f"data={data}, type={type(data)}")
        data = np.array(data)
        if DimRed_type == "t-SNE":

            tsne = TSNE(n_components=2)
            data_tsne = tsne.fit_transform(data)
            print(f"data_tsne: {data_tsne}, shape= {data_tsne.shape}")
            return data_tsne
        elif DimRed_type == "PCA":
            pca = PCA(n_components=2)
            data_pca = pca.fit_transform(data)
            print(f"data_pca: {data_pca}, shape= {data_pca.shape}")
            print(f"PCA components: {pca.components_}")
            return data_pca
        else:
            raise ValueError("Type of dimensionality reduction is not t-SNE or PCA")

    def DimRed_of_iteration(self, DimRed_type):
        for iteration, iteration_dict in enumerate(self.count_list):
            if not iteration%5==0:
                continue
            self.do_DimRed_on_prob_distr(dict_of_iter=iteration_dict, iteration=iteration, DimRed_type=DimRed_type)
        print(f"finished {str(type)}")

    def do_DimRed_on_prob_distr(self, dict_of_iter, iteration, DimRed_type):
        points = [(utils.string_2_angles(key, self.n), evaluated_shots, self.eval_count_loss_dict[key]) for key, evaluated_shots in dict_of_iter.items()]

        point_list = []
        costs = []
        for angle, eval_shots, cost in points:
            point_list += [angle]*(eval_shots)
            costs += [cost]*eval_shots

        reduced_data = self._do_DimRed(point_list, DimRed_type=DimRed_type)

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=costs)
        cbar = plt.colorbar(sc)
        cbar.set_label('Cost function value')
        plt.xlabel('axis-1')
        plt.ylabel('axis-2')
        plt.title(f"{str(type)} of: {self.file_specific_path}, iteration={iteration}")
        png_filename = self.save_directory + f"{str(DimRed_type)}_iteration={iteration}.png"
        print(f"png filepath: {png_filename}")
        plt.savefig(png_filename, dpi=800)
        plt.show()









from_directory = r"C:\Users\Constantin\Desktop\results5\results5"
result_directory = r"C:\Users\Constantin\Desktop\Masterarbeit_results_new"

dir = [x for x in os.walk(from_directory)][0][0]
var_dict = {}
red_type = "t-SNE"
for jj, filepath in enumerate([x for x in os.walk(from_directory)][0][1]):
    print(f"Process {round(jj/len([x for x in os.walk(from_directory)][0][1]),3)}%")
    # print(f"dir={dir}")
    # print(f"filepath={filepath}")
    print(f"total_path = {os.path.join(dir,filepath)}")

    instance = Distribution_func_file_opener(5, 3, 2, 5, True, True, True, 0.05, 0.0, 4, name=r"C:\Users\Constantin\Desktop\results5\results5\5_3_1_5_ST_ZT_HT_lr0.05_h0.95_trl8", save_directory=result_directory)
    print(f"N_q:{instance.n}, N_loss: {instance.n_loss_param}, seed: {instance.seed}, r: {instance.rr}")
    instance.DimRed_of_iteration(DimRed_type=red_type)
    exit("finished after one iteration PCA")
    temp_dict_name = f"{instance.seed}_{instance.n_loss_param}_{instance.rr}"

    if temp_dict_name not in var_dict:
        var_dict[temp_dict_name] = {"seed":[],
                                    "n_loss_param" : [],
                                    "r" : [],
                                    "n_q": [],
                                    "lr": [],
                                    "h": [],
                                    "trl": [],
                                    "evaluation":[],
                                    "type_of_eval":[],
                                    "best_value":[]
                                    }

    # add the best from n^2
    var_dict[temp_dict_name]["seed"].append(instance.seed)
    var_dict[temp_dict_name]["n_loss_param"].append(instance.n_loss_param)
    var_dict[temp_dict_name]["r"].append(instance.rr)
    var_dict[temp_dict_name]["n_q"].append(instance.n)
    var_dict[temp_dict_name]["lr"].append(instance.lr)
    var_dict[temp_dict_name]["h"].append(instance.superpos_hyperparameter)
    var_dict[temp_dict_name]["trl"].append(instance.trl)
    var_dict[temp_dict_name]["evaluation"].append(instance.max_eval)
    var_dict[temp_dict_name]["type_of_eval"].append("most probable")
    var_dict[temp_dict_name]["best_value"].append(instance.value_with_smallest_value)

    # add most probable solution
    var_dict[temp_dict_name]["seed"].append(instance.seed)
    var_dict[temp_dict_name]["n_loss_param"].append(instance.n_loss_param)
    var_dict[temp_dict_name]["r"].append(instance.rr)
    var_dict[temp_dict_name]["n_q"].append(instance.n)
    var_dict[temp_dict_name]["lr"].append(instance.lr)
    var_dict[temp_dict_name]["h"].append(instance.superpos_hyperparameter)
    var_dict[temp_dict_name]["trl"].append(instance.trl)
    var_dict[temp_dict_name]["evaluation"].append(instance.best_eval_from_n2)
    var_dict[temp_dict_name]["type_of_eval"].append("best of n^2")
    var_dict[temp_dict_name]["best_value"].append(instance.value_with_smallest_value)

    # add the best gradient evaluations for all "N_gds" Gradient Descents
    for last_gd_value in [gd_list[-1] for gd_list in instance.gd_cost_list]:
        var_dict[temp_dict_name]["seed"].append(instance.seed)
        var_dict[temp_dict_name]["n_loss_param"].append(instance.n_loss_param)
        var_dict[temp_dict_name]["r"].append(instance.rr)
        var_dict[temp_dict_name]["n_q"].append("GD")
        var_dict[temp_dict_name]["lr"].append(instance.lr)
        var_dict[temp_dict_name]["h"].append(instance.superpos_hyperparameter)
        var_dict[temp_dict_name]["trl"].append(instance.trl)
        var_dict[temp_dict_name]["evaluation"].append(last_gd_value)
        var_dict[temp_dict_name]["type_of_eval"].append("Gradient Descent")
        var_dict[temp_dict_name]["best_value"].append(instance.value_with_smallest_value)

for key, value in var_dict.items():
    fig_title = key + "_fig3.png"
    df = pd.DataFrame(value)
    sns.boxplot(x="n_q", y="evaluation", hue="type_of_eval", data=df)
    sns.boxplot(x="n_q", y="best_value", data=df)

    plt.title(str(key))
    plt.savefig(os.path.join(result_directory,fig_title))
    plt.figure()

    csv_title = key + "DataFrame_fig3.csv"
    df.to_csv(os.path.join(result_directory,csv_title))

exit()

test_instance = Distribution_func_file_opener(5, 3, 2, 5, 'T', 'T', 'T', 0.05, 0.0, 4, save_directory=result_directory)
# print(test_instance.maximum_string(test_instance.count_list[138]))
# print(max(test_instance.count_list[0], test_instance.count_list[0].get))
exit()
# instance1 = File_opener(5, 2, 555, 5, 'T', 'T', 'T', 0.05, 0.95)
# df_results = np.zeros(shape=(2 ** instance1.n, 2 ** instance1.n))
# for count, value in instance1.eval_count.items():
#     pos = utils.string_2_bin(count, instance1.n)
#     df_results[pos[0]][pos[1]] = value
# sns.heatmap(df_results)
# plt.show()
# for i,count in enumerate(instance1.count_list):
#     if i%10==0:
#         print(i)
#         # if i==50:
#         #     break
#         print(count)
#         plots.plot_heatmap(counts=count, counting_qubits=instance1.n, df_result=i, save=None )
# exit()
filepath = r'C:\Users\Constantin\PycharmProjects\MasterArbeit\results2'
file_specific_path = '4_2_2_7_ST_ZT_HT_lr0.05_h0.0_trl8'
filepath = os.path.join(filepath, file_specific_path)
print(filepath)
instance1 = Distribution_func_file_opener(1, 1, 1, 1, 'T', 'T', 'T',
                                          1, 1, 1, filepath)

df_results = np.zeros(shape=(2 ** instance1.n, 2 ** instance1.n))

for count, value in instance1.eval_count.items():
    pos = utils.string_2_bin(count, instance1.n)
    df_results[pos[0]][pos[1]] = value
sns.heatmap(df_results)
plt.title(file_specific_path)
plt.show()
exit()

seeds = [1,2]
rs = [5,7]
lrs = [0.05,0.2]
hs = [0.0, 0.95]
trls = [1,4,8]
for seed in seeds:
    for r_ in rs:
        for lr in lrs:
            for h_ in hs:
                for trl_ in trls:
                    filepath = r'C:\Users\Constantin\PycharmProjects\MasterArbeit\results2'
                    file_specific_path = str(6) + '_' + str(2) + '_' + str(seed) + '_' + str(r_) + '_' + 'S' + 'T' + '_' + 'Z' + 'T' + '_' + 'H' + 'T' + '_' + 'lr' + str(float(lr)) + '_' + 'h' + str(h_) + '_' + 'trl' + str(trl_)
                    filepath = os.path.join(filepath, file_specific_path)
                    print(filepath)
                    instance1 = Distribution_func_file_opener(1, 1, 1, 1, 'T', 'T', 'T',
                                                              1, 1, 1, filepath)

                    df_results = np.zeros(shape=(2 ** instance1.n, 2 ** instance1.n))

                    for count, value in instance1.eval_count.items():
                        pos = utils.string_2_bin(count, instance1.n)
                        df_results[pos[0]][pos[1]] = value
                    sns.heatmap(df_results)
                    plt.show()
                    # for i, count in enumerate(instance1.count_list):
                    #     if i % 20 == 0:
                    #         print(i)
                    #         # if i==50:
                    #         #     break
                    #         plots.plot_heatmap(counts=count,
                    #                            counting_qubits=instance1.n, df_result=i,
                    #                            save=None)

file_names = [x[0] for x in os.walk(r'C:\Users\Constantin\PycharmProjects\MasterArbeit\results')]

distance_2_GlMin=[]
gd_distance_2_GlMin=[]
distance_from_new_gd=[]
g=0
for file_n in file_names[1:]:
    if False:
        instance1 = File_opener(1, 1, 1, 1, 'T', 'T', 'T',
                                1, 1, 1, file_n)
        print(file_n)
        print(instance1.eval_count['00000000'])
        g+=1
        continue
        if g==20:
            exit()

    # print(file_n)
    # trl = float(file_n[-1])
    # h = float(file_n[-8:-5])
    # lr = float(file_n[-14:-10])
    # rr = float(file_n[-27])
    # seed = float(file_n[-29])
    # n_loss_p = float(file_n[-31])
    # n_c6 = float(file_n[-33])
    # print(h,lr,rr,seed, n_loss_p, n_c6)

    print(f"file {file_names.index(file_n)}, {file_names.index(file_n)/len(file_names)}% done ")
    instance1 = Distribution_func_file_opener(1, 1, 1, 1, 'T', 'T', 'T',
                                              1, 1, 1, file_n)

    #find best evaluation by the standard gradient descent
    gd_eval_list = [gd_cost_list[-1] for gd_cost_list in instance1.gd_cost_list]
    min_trial = gd_eval_list.index(min(gd_eval_list))
    min_param = instance1.gd_param[min_trial][-1]
    min_trial_cost = min(gd_eval_list)
    print(file_n)
    print(instance1.instance.counting_qubits)
    print(instance1.instance.n_loss_param)

    #find the best evaluation from n squared points
    min_key = utils.find_from_n(eval_instance=instance1, n_qbts_to=8, n=(instance1.n*instance1.n_loss_param)**2)
    print(f"min_key={min_key}, with length {len(min_key)}")
    gd_angle = np.array(utils.string_2_angles(min_key,8), requires_grad=True)

    print('gd angle',gd_angle)

    print('qf eval', instance1.instance.quantum_function_eval(gd_angle))

    #perform gd from the best evaluation found
    new_gd_cost ,new_gd_param = utils.GD_from_selected_point(instance1, 0.02, gd_angle, 240)
    exit()
    #get distance to global min for all three variables
    min_key_dist, new_gd_cost_dist, gd_cost_dist = utils.distance_to_GlMin(instance1, utils.string_2_angles(min_key, instance1.n), new_gd_param[-1], min_param)
    distance_2_GlMin.append(min_key_dist)
    distance_from_new_gd.append(new_gd_cost_dist)
    gd_distance_2_GlMin.append(gd_cost_dist)

    dictionary = {'min_key':min_key,
                  'min_trial_param':min_param,
                  'min_trial_cost':min_trial_cost,
                  'new_gd_param': new_gd_param,
                  'new_gd_cost':new_gd_cost,
                  'min_key_dist':min_key_dist,
                  'new_gd_cost_dist':new_gd_cost_dist,
                  'gd_cost_dist':gd_cost_dist}

    pandas_df = pd.DataFrame(dictionary)
    pandas_df.to_csv(os.path.join(file_n, 'analysis.csv'))

dictionar = {'distance_2_GlMin':distance_2_GlMin,
                  'distance_from_new_gd':distance_from_new_gd,
                  'gd_distance_2_GlMin':gd_distance_2_GlMin}

pandas_df = pd.DataFrame(dictionar)
pandas_df.to_csv(os.path.join(r'C:\Users\Constantin\PycharmProjects\MasterArbeit\results', 'analysis.csv'))


exit()
file_names = [x[0] for x in os.walk(r'C:\Users\Constantin\PycharmProjects\MasterArbeit\results')]
print(file_names)
print(file_names[1:])
i=0
for file_n in file_names[1:]:
    i+=1
    print('file_n', file_n)
    instance1 = Distribution_func_file_opener(1, 1, 1, 1, 'T', 'T', 'T',
                                              1, 1, 1, file_n)

    plt.title(file_n[len(file_names[0]):])
    plt.plot([i for i in range(len(instance1.df_cost))],
             instance1.df_cost)
    plt.savefig(str(file_n)+str(file_n[len(file_names[0]):])+'.png')
    plt.savefig(r'C:\Users\Constantin\PycharmProjects\MasterArbeit\results\figures' + str(file_n[len(file_names[0]):]) + '.png')
    plt.clf()

exit()
for nnn in [3,4,5,6,7,8,9,12,15]:
    for n_loss_par in [1,2]:
        for rr in [5,7]:
            for lr_ in [0.05, 0.2]:
                for trl_ in [1,4,8]:
                    for h_ in [0.0,0.95]:
                        for seed in [1,2]:
                            print(nnn)
                            if nnn == 8 or nnn == 10 or nnn == 12 or nnn==6:
                                continue
                            instance1 = Distribution_func_file_opener(nnn, n_loss_par, seed, rr, 'T', 'T', 'T',
                                                                      lr_, h_, trl_)

                            plt.title(f'n={nnn} ,n_loss_par={n_loss_par},r={rr} ,lr={lr_} ,trl={trl_} ,H={h_} ,seed={seed}')
                            plt.plot([i for i in range(len(instance1.df_cost))],
                                     instance1.df_cost)
                            plt.show()

exit()
# for seed in [31, 40, 60]:
#     for lr in [0.01, 0.05, 0.1, 0.2, 0.5]:
#         instance1 = File_opener(5, 3, seed, 7, 'T', 'T', 'T', lr)
#         plt.plot(range(len(instance1.df_cost)), instance1.df_cost, label=str(lr))
#     plt.legend()
#     plt.show()
#
# exit()

for count, value in instance1.eval_count.items():
    pos = utils.string_2_bin(count, instance1.n)
    df_results[pos[0]][pos[1]] = value
sns.heatmap(df_results)
plt.show()
for i,count in enumerate(instance1.count_list):
    if i%10==0:
        print(i)
        # if i==50:
        #     break
        print(count)
        plots.plot_heatmap(counts=count, counting_qubits=instance1.n, df_result=i, save=None )
