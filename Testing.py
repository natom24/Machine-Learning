# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:14:08 2022

@author: natom
"""
import torch
import torchvision
import Training
from PIL import Image

import matplotlib.pyplot as plt

test_loader = torch.utils.data.DataLoader(test_set, shuffle = True, collate_fn=collate_fn)

#boxes = x[0]['boxes']

def draw_bbox(data_loader):
    
    Images = []
    
    Outputs = []
    
    model.eval()
    for image,targets in test_loader:
        
        with torch.no_grad():
            Images.append(image)
            Outputs.append(model(image))
    
    img_int8 = torch.tensor(image[0]*255, dtype = torch.uint8)
    img_box = torchvision.utils.draw_bounding_boxes(image = img_int8, boxes = Outputs[0][0]['boxes'])
    img_box = img_box.permute(1,2,0) # change layering for plotting
    
    return img_box

ex_img = draw_bbox(test_loader)

plt.imshow(ex_img)
