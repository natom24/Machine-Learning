# Import packages
import os
import json
import PIL
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


################### Loads Images In ######################


img = [] # Blank list for images

for file in os.listdir('Data/Images'): # Loop to open and save each file
    img_data = tf.io.read_file('Data/Images/%s'%(file)) # Reads in file
    img_data = tf.io.decode_jpeg(img_data) # Decodes images to pixel data
    img.append(img_data) # Adds photo data to list

plt.imshow(img[4]) #Plots photo


################## Augment? ##################

'''Not sure best package to use -- needs further research'''

##################### Load Labels ######################## 

'''This may not be correct, I know I'm going to convert json files to tfrecords so this may end up obsolete'''

label = [] # Blank list for images

for file in os.listdir('Data/Labels'): # Loop to open and save each file
    text = open('Data/Labels/%s'%(file)) #Opens Json file
    label.append(json.load(text)) # Adds json data to list


########################### Split Data ##########################

'''Come back to once tfrecords established/ laebls and '''

# Splits the data into the train and test sets
train, test = train_test_split(img, test_size = .2)


# Adds a validation set (DO I NEED THIS!!!!!!)
#train, val = train_test_split(train, test_size = .2)


plt.imshow(test[0])


#labels = [{'name':'Hemo', 'id':1}, {'name':'Cell', 'id':2}] #Creates the labels for the two objects being tested


