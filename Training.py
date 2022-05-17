import os
import json
import PIL
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt





# Loads data into 
img = []

for file in os.listdir('Data/Images'):
    img_data = tf.io.read_file('Data/Images/%s'%(file))
    img_data = tf.io.decode_jpeg(img_data)
    img.append(img_data)


labels = [{'name':'Hemo', 'id':1}, {'name':'Cell', 'id':2}] #Creates the labels for the two objects being tested

# Limit GPU Consumption 
# Doesn't work on windows
#x = tf.config.experimental.list_physical_devices('')

