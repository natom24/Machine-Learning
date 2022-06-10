# Import packages
import os
import PIL
import numpy as np
#import tensorflow as tf
import torch
#import torchvision
import matplotlib.pyplot as plt

########################### test for GPU ###########################

device = (torch.device('cuda') if torch.cuda.is_available()
                  else torch.device('cpu'))

print(f"Training on device {device}.")

##################### Create Labelmap ######################## 


################### Loads Images In ######################


img = [] # Blank list for images

for file in os.listdir('Data/Images/test'): # Loop to open and save each file to a list
    if file.endswith(".JPG"): # Only open .jpg images 
        img_data = PIL.Image.open('Data/Images/test/%s'%(file)) # Reads in file
        img.append(img_data) # Adds photo data to list

plt.imshow(img[1]) #Plots one photo

    
    
########################### Model Config ##########################

