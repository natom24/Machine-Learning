# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 09:28:13 2022

@author: natom
"""
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def create_model(num_classes):
    backbone = torchvision.models.resnet152(weights=torchvision.models.resnet.ResNet152_Weights)
    
    backbone.out_channels = 1024
    
    anchor_generator = AnchorGenerator(sizes = ((8,16,32,64,128),),aspect_ratios = ((0.5,1.0,2.0),))
    
    roi_pool = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7,sampling_ratio=2)
    
    model = FasterRCNN(backbone = backbone, num_classes = num_classes,
                       rpn_anchor_generator = anchor_generator, box_roi_pool = roi_pool)
    
    return model

