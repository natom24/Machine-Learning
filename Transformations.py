# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:54:36 2022

@author: natom
"""
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

def toTensor(img):
    to_tensor= transforms.ToTensor() # May need to be changed, just a quick method of converting to tensor for testing
    img = to_tensor(img) # Calls conversion



def collate_fn(batch):
    
    """
    Formats data in lists since each image often has a different number of objects 
    """
    
    image = list()
    target = list()
    
    for b in batch:
        image.append(b[0])
        target.append(b[1])
    
    return image,target

def train_transform():
    t_img = A.Compose([
        A.RandomCrop(512,512),
        A.HorizontalFlip(.5),
        ToTensorV2(),
        ], bbox = A.BboxParams(format = 'pascal_voc'))
    
    return t_img