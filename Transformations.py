# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:54:36 2022

@author: natom
"""
from torchvision import transforms

def toTensor(img):
    to_tensor= transforms.ToTensor() # May need to be changed, just a quick method of converting to tensor for testing
    img = to_tensor(img) # Calls conversion