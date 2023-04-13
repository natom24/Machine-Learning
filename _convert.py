# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:34:36 2023

@author: natom
"""

import os

img_list = 'C:/School/Project/Haemocyte Counting LD95 SLAV Purdue March 2022/26.16'

for f in os.listdir(img_list):
    if f.find('26-16')>-1:
        os.rename(os.path.join(img_list, f), os.path.join(img_list, f.replace("26-16", "26.16")))
        
    