# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 12:40:12 2023

@author: natom
"""
import os
import re
import xml.etree.ElementTree as ET

def fix_filename(annot_path):
    
    for file in os.listdir(annot_path):
        tree = ET.parse(os.path.join(annot_path, file))
        root = tree.getroot()
        
        path = root.find('path')
        
        filename = root.find('filename')
        
        path.text = filename.text
        
        Data = ET.tostring(root)
        
        with open(os.path.join(annot_path, file), 'wb') as f:
            f.write(Data)

        
