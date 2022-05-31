# Import packages
import os
import json
import PIL
import numpy as np
import tensorflow as tf
#import torch
#import torchvision
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

##################### Create Labelmap ######################## 

label_list = ["hemo","cell"]
path = 'Data\\'

# Function to create pbtext file containing labels
def create_labelmap(labels = label_list, path = path):
    
    with open(path + 'label_map.pbtxt', 'w') as f: # Creates the labelmap 
        x = 1 # Initial ID value
        
        # Goes through each label writing the appropriate labelmap format
        for label in label_list:
            f.write ("item {\n")
            f.write ("\tid: %s\n"%(x))
            f.write ("\tname: '"+ label_list[x-1] + "'\n")
            f.write ("}\n")
            
            x = x+1 #Increments 
            
#create_labelmap()

################### Loads Images In ######################


img = [] # Blank list for images

for file in os.listdir('../Code/TestData/train'): # Loop to open and save each file to a list
    if file.endswith(".jpg"):  
        img_data = PIL.Image.open('../Code/TestData/train/%s'%(file)) # Reads in file
        img.append(img_data) # Adds photo data to list

plt.imshow(img[2]) #Plots one photo

    
    



########################### Split Data ##########################

'''
Come back to once tfrecords established and labels. 
Currently splitting through conversion to csv. Best method?
Also do I need validation?
'''

# Splits the data into the train and test sets
#train, test = train_test_split(img, test_size = .2)


# Adds a validation set (DO I NEED THIS!!!!!!)
#train, val = train_test_split(train, test_size = .2)


#plt.imshow(test[0])


#labels = [{'name':'Hemo', 'id':1}, {'name':'Cell', 'id':2}] #Creates the labels for the two objects being tested


########################### Model ##########################

model_path = ''
model_name = 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz'
link = 'https://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz'
