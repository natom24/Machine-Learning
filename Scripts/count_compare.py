import os
import pandas as pd
import matplotlib.pyplot as plt

def count_compare(count_loc, mac_count_loc): 
    # Specify where 
    
    # Create empty dataframes for recording values
    o_count = []
    mac_count = []
    
    img_list = os.listdir(count_loc)
    

    for f in img_list:
        o_txt = open(os.path.join(count_loc, f), 'r').read()
        
        o_count.append(o_txt.count('\n')) # Count number of lines and record
        
        mac_txt = open(os.path.join(mac_count_loc, f), 'r').read()
    
        mac_count.append(mac_txt.count('\n'))
    
        #print(o_count)
        plt.scatter(o_count,mac_count)
        plt.plot([0,50,100,110],[0,50,100,110])

        plt.show()
        plt.xlabel('True Counts'); plt.ylabel('Machine Counts')
    
count_compare(count_loc = 'C:/School/Project/Machine-Learning/Yolo/datasets/hemo/labels/test', 
              mac_count_loc = 'C:/School/Project/Machine-Learning/Yolo/yolov8/runs/detect/predict/labels')


#c_count = 0
#i_count = 0

#for i in range(o_count):
    
#    if(mac_txt[i] == o_txt[i]):
#        c_count = c_count+1
#   else:
#        i_count= i_count+1