# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 13:14:27 2022

@author: natom
"""

import cv2
import matplotlib.pyplot as plt

path = 'C:/School/Project/Code/Hemo_data/full_Images/34.26C Virus 2.B 1-2.JPG'

image = cv2.imread(path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


image_edge = cv2.Canny(gray_image, threshold1=00, threshold2=50)

plt.imshow(image_edge)
