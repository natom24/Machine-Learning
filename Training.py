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

########################### Paths ###########################



##################### Create Labelmap ######################## 

label_list = ["hemo","cell"] # Label list used here
path = 'Data\\' # Default Path

# Function to create pbtext file containing labels
def create_labelmap(labels = label_list, path = path):
    
    with open(path + 'label_map.pbtxt', 'w') as f: # Creates the labelmap 
        ID = 1 # Initial ID value
        
        # Goes through each label writing the appropriate labelmap format
        for label in label_list:
            f.write ("item {\n")
            f.write ("\tid: %s\n"%(ID))
            f.write ("\tname: '"+ label_list[ID-1] + "'\n")
            f.write ("}\n")
            
            ID = ID+1 #Increments ID value


################### Loads Images In ######################


img = [] # Blank list for images

for file in os.listdir('Data/Images/test'): # Loop to open and save each file to a list
    if file.endswith(".JPG"): # Only open .jpg images 
        img_data = PIL.Image.open('Data/Images/test/%s'%(file)) # Reads in file
        img.append(img_data) # Adds photo data to list

plt.imshow(img[1]) #Plots one photo

    
    
########################### Model ##########################

model_path = 'Models/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/saved_model'
model_name = 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz'
link = 'https://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz'

model = tf.saved_model.load(model_path)
