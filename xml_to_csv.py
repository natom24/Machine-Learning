# Converts xml annotations to csv files for use in training.
#
#
# Can be run using the following command at the command line:
#
# python xml_to_csv.py Code\Data\Labels
#

import sys
import os
import numpy as np
import glob
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    
    box_list = []
    
    for f in glob.glob(path+'\*.xml'):
        tree = ET.parse(f)
        root = tree.getroot()
        
        size = root.find('size')
        width = size.find('width').text
        height = size.find('height').text
        
        
        for obj in root.findall('object'):
            box_size = obj.find('bndbox')
            box_data = os.path.basename(f), width, height, obj.find('name').text, box_size.find('xmin').text, box_size.find('ymin').text, box_size.find('xmax').text, box_size.find('ymax').text
            box_list.append(box_data)

    return box_list

def main():
    path = sys.argv[1]
    data = xml_to_csv(path = path)
    np.savetxt(path+'_data.csv', data, delimiter =", ",header = 'File Name,Width,Length,Label,xmin,ymin,xmax,ymax', fmt ='% s')

main()