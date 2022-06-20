# Import packages
import os
import PIL
import numpy as np
import pandas as pd
import datetime
import torch
import torchvision
from datasets import HemocyteDataset
import matplotlib.pyplot as plt

# Test for GPU, run on CPU if GPU is not available
device = (torch.device('cuda') if torch.cuda.is_available()
                  else torch.device('cpu'))

print(f"Training on device {device}.")


hemo_dataset = HemocyteDataset(root='Data/Images',annot='Data/Labels_data.csv')

dataset_loader = torch.utils.data.DataLoader(hemo_dataset, batch_size = 2, shuffle = True)


indices = torch.randperm(len(dataset_loader)).tolist()
dataset = torch.utils.data.Subset(dataset_loader, indices)


########################### Model ###########################

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

for e in range(epoch):
    print("add stuff")    
    
################### Loads Images In ######################


    img = [] # Blank list for images

    for file in os.listdir('Data/Images/test'): # Loop to open and save each file to a list
        if file.endswith(".JPG"): # Only open .jpg images 
            img_data = PIL.Image.open('Data/Images/test/%s'%(file)) # Reads in file
            img.append(img_data) # Adds photo data to list

    plt.imshow(img[1]) #Plots one photo