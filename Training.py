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

################### Loads Images In ######################


img = [] # Blank list for images

for file in os.listdir('../Code/TestData/train'): # Loop to open and save each file to a list
    if file.endswith(".jpg"):  
        img_data = PIL.Image.open('../Code/TestData/train/%s'%(file)) # Reads in file
        img.append(img_data) # Adds photo data to list

plt.imshow(img[2]) #Plots one photo



##################### Load Labels ######################## 

'''
Multiple other programs have been written to convert
data to a tf records format and convert json/xml to CSV.
Should I still write my own version??
'''



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

model = tf.keras.models.Sequential()
