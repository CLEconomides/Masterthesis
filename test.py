import pandas as pd
import sys
import numpy as np
import os
import os
import shutil

var_dict = {}
for i in range(0,10):
    ii = i%4
    if "var{0}".format(ii) in var_dict:
        print(i)
    else:
        var_dict["var{0}".format(ii)] = {}

print(var_dict)
exit()

def copy_files(from_directory, to_directory):
    # from_directory = r"C:\Users\Constantin\Desktop\results6\results6"
    # to_directory = r"C:\Users\Constantin\Desktop\results5\results5"
    dir_files = [x for x in os.walk(from_directory)][0]
    overwritten = 0
    files = 0
    for file in dir_files[1]:
        files += 1
        temp_from_path = os.path.join(from_directory, file, "count_eval.csv")
        if not os.path.exists(temp_from_path):
            print(f"count_eval.csv does not exists in Path = {file}")
            exit()
        temp_to_path = os.path.join(to_directory, file, "count_eval.csv")
        if os.path.exists(temp_to_path):
            print(f"old count_eval.csv has been overwritten")
            overwritten += 1
        shutil.copyfile(temp_from_path, temp_to_path)

    print(files)
    print(overwritten)
    print(f"{files} many Files have been used",
          f"{overwritten} have files have been overwritten")


exit()

list1= sys.argv[0]
list2= sys.argv[1]
print(list1)
print(list2)
Liste=[]

for i in list1:
    for j in list2:
        Liste.append(i+j)

df = pd.DataFrame(np.array(Liste))
df.to_csv(os.path.join('./results', 'df_counts.csv'))