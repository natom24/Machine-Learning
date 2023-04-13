import os
import pandas as pd
import re
import matplotlib.pyplot as plt
from glob import glob


def temp_count(files_loc): 
    # Specify where 
    
    # Create empty dataframes for recording values
    temp_count = []
    
    img_list = glob(os.path.join(files_loc,'*.jpg')) # Create list of image names
    
    lab_list = os.listdir(os.path.join(files_loc, 'labels'))
    
    for f in lab_list:
        o_txt = open(os.path.join(files_loc,'labels', f), 'r').read()
        
        
        
        multi = int(re.findall('\d\-\d', f)[0][-1]) # Pull the ratio of hemocytes to anticoag
        
        temp_count.append((o_txt.count('\n'))*multi) # Count number of lines and record
    
    
    zeros= len(img_list) - len(lab_list)
    
    temp_count.extend([0]*zeros)
    
    return temp_count

temp26 = temp_count('C:/School/Project/Haemocyte Counting LD95 SLAV Purdue March 2022/26.16')

temp31 = temp_count('C:/School/Project/Haemocyte Counting LD95 SLAV Purdue March 2022/31.21')
#def temp_compare(temp1_count, temp2_count):
    
#seaborn.violinplot(data = [temp26,temp31])

data = [temp26,temp31]
plt.boxplot(data)

import scipy.stats as stats

stats.ttest_ind(a = temp26, b = temp31)

mean26 = sum(temp26)/len(temp26)
mean31 = sum(temp31)/len(temp31)

plt.boxplot(data, labels = ['26','31'])

inf_coev_26 = temp_count("C:/School/Project/Machine-Learning/Hemo_data/Haemocyte_Virus_Split/26_Inf_Coev")

inf_control_26 = temp_count("C:/School/Project/Machine-Learning/Hemo_data/Haemocyte_Virus_Split/26_Inf_Control")

uninf_control_26 = temp_count("C:/School/Project/Machine-Learning/Hemo_data/Haemocyte_Virus_Split/26_Uninf_Control")

uninf_coev_26 = temp_count("C:/School/Project/Machine-Learning/Hemo_data/Haemocyte_Virus_Split/26_Uninf_Coev")

inf_coev_31 = temp_count("C:/School/Project/Machine-Learning/Hemo_data/Haemocyte_Virus_Split/31_Inf_Coev")

inf_control_31 = temp_count("C:/School/Project/Machine-Learning/Hemo_data/Haemocyte_Virus_Split/31_Inf_Control")

uninf_control_31 = temp_count("C:/School/Project/Machine-Learning/Hemo_data/Haemocyte_Virus_Split/31_Uninf_Control")

uninf_coev_31 = temp_count("C:/School/Project/Machine-Learning/Hemo_data/Haemocyte_Virus_Split/31_Uninf_Coev")

data2 = [inf_coev_26, inf_control_26, uninf_control_26, uninf_coev_26, inf_coev_31, inf_control_31, uninf_control_31, uninf_coev_31] 

plt.xticks(rotation=90)
plt.boxplot(data2, labels = ['inf_coev_26', 'inf_control_26', 'uninf_control_26', 'uninf_coev_26', 'inf_coev_31', 'inf_control_31', 'uninf_control_31', 'uninf_coev_31'])

#from statsmodels.stats.multicomp import pairwise_tukeyhsd


