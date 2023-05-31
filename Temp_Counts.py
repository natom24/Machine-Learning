import os
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

temp26 = temp_count('C:/School/Project/Data Splits/Temp_crop/26.16_notreat')

temp31 = temp_count('C:/School/Project/Data Splits/Temp_crop/31.21_notreat')
#def temp_compare(temp1_count, temp2_count):
    
#seaborn.violinplot(data = [temp26,temp31])

data = [temp26,temp31]
plt.boxplot(data)

import scipy.stats as stats

stats.ttest_ind(a = temp26, b = temp31)

mean26 = sum(temp26)/len(temp26)
mean31 = sum(temp31)/len(temp31)

plt.boxplot(data, labels = ['26','31'])

inf_coev_26 = temp_count("C:/School/Project/Data Splits/Haemocyte_Virus_Split/26_Inf_Coev")

inf_control_26 = temp_count("C:/School/Project/Data Splits/Haemocyte_Virus_Split/26_Inf_Control")

uninf_control_26 = temp_count("C:/School/Project/Data Splits/Haemocyte_Virus_Split/26_Uninf_Control")

uninf_coev_26 = temp_count("C:/School/Project/Data Splits/Haemocyte_Virus_Split/26_Uninf_Coev")

inf_coev_31 = temp_count("C:/School/Project/Data Splits/Haemocyte_Virus_Split/31_Inf_Coev")

inf_control_31 = temp_count("C:/School/Project/Data Splits/Haemocyte_Virus_Split/31_Inf_Control")

uninf_control_31 = temp_count("C:/School/Project/Data Splits/Haemocyte_Virus_Split/31_Uninf_Control")

uninf_coev_31 = temp_count("C:/School/Project/Data Splits/Haemocyte_Virus_Split/31_Uninf_Coev")

data2 = [inf_coev_26, inf_control_26, uninf_control_26, uninf_coev_26, inf_coev_31, inf_control_31, uninf_control_31, uninf_coev_31] 

plt.xticks(rotation=90)
plt.boxplot(data2, labels = ['inf_coev_26', 'inf_control_26', 'uninf_control_26', 'uninf_coev_26', 'inf_coev_31', 'inf_control_31', 'uninf_control_31', 'uninf_coev_31'])

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

inf_coev_26_12 = temp_count("C:/School/Project/Data Splits/Last_Gen_Cropped/26.16/Infected_Coevolved")

inf_control_26_12 = temp_count("C:/School/Project/Data Splits/Last_Gen_Cropped/26.16/Infected_Control")

uninf_control_26_12 = temp_count("C:/School/Project/Data Splits/Last_Gen_Cropped/26.16/Uninfected_Control")

uninf_coev_26_12 = temp_count("C:/School/Project/Data Splits/Last_Gen_Cropped/26.16/Uninfected_Coevolved")

inf_coev_31_12 = temp_count("C:/School/Project/Data Splits/Last_Gen_Cropped/31.21/Infected_Coevolved")

inf_control_31_12 = temp_count("C:/School/Project/Data Splits/Last_Gen_Cropped/31.21/Infected_Control")

uninf_control_31_12 = temp_count("C:/School/Project/Data Splits/Last_Gen_Cropped/31.21/Uninfected_Control")

df_inf_coev_26 = pd.DataFrame({'counts': inf_coev_26_12, 'temp': np.repeat('26',len(inf_coev_26_12)),'treatment': np.repeat('coevolved',len(inf_coev_26_12)),'inf':np.repeat('inf',len(inf_coev_26_12))})

df_inf_cont_26 = pd.DataFrame({'counts': inf_control_26_12, 'temp': np.repeat('26',len(inf_control_26_12)),'treatment': np.repeat('control',len(inf_control_26_12)),'inf':np.repeat('inf',len(inf_control_26_12))})

df_uninf_cont_26 = pd.DataFrame({'counts': uninf_control_26_12, 'temp': np.repeat('26',len(uninf_control_26_12)),'treatment': np.repeat('control',len(uninf_control_26_12)),'inf':np.repeat('uninf',len(uninf_control_26_12))})

#df_uninf_coev_26 = pd.DataFrame({'counts': uninf_coev_26_12, 'temp': np.repeat('26',len(uninf_coev_26_12)),'treatment': np.repeat('coevolved',len(uninf_coev_26_12)),'inf':np.repeat('uninf',len(uninf_coev_26_12))})

df_inf_coev_31 = pd.DataFrame({'counts': inf_coev_31_12, 'temp': np.repeat('31',len(inf_coev_31_12)),'treatment': np.repeat('coevolved',len(inf_coev_31_12)),'inf':np.repeat('inf',len(inf_coev_31_12))})

df_inf_cont_31 = pd.DataFrame({'counts': inf_control_31_12, 'temp': np.repeat('31',len(inf_control_31_12)),'treatment': np.repeat('control',len(inf_control_31_12)),'inf':np.repeat('inf',len(inf_control_31_12))})

df_uninf_cont_31 = pd.DataFrame({'counts': uninf_control_31_12, 'temp': np.repeat('31',len(uninf_control_31_12)),'treatment': np.repeat('control',len(uninf_control_31_12)),'inf':np.repeat('uninf',len(uninf_control_31_12))})

df_comb = [df_inf_coev_26, df_inf_cont_26, df_uninf_cont_26, df_inf_coev_31, df_inf_cont_31, df_uninf_cont_31]

df_comb = pd.concat(df_comb)

df_comb.to_csv("C:/Users/natom/Downloads/hemocytes.csv")

stat_sum = ols("counts~C(temp)+C(treatment)+C(inf)+C(temp):C(treatment)+C(temp):C(inf) +C(treatment):C(inf)+C(temp):C(treatment):C(inf)", data=df_comb).fit()
#sm.stats.anova_lm(stat_sum, typ=3)

