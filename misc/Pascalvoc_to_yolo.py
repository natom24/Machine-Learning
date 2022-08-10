import os
import sys
import glob
import xml.etree.ElementTree as ET

def pascalvoc_to_yolo(file_dir, save_path, width = 1600, height = 1200):
    
    annot_path = os.path.join(file_dir, "Hemo_data","Boxes")
    
    for file in os.listdir(annot_path):
        
        tree = ET.parse(os.path.join(annot_path, file))
        root = tree.getroot()
        
        file_name = file
        file_name = file_name[:-4] # Drop the xml ending
        
        #bo
        
        for obj in root.findall('object'):
            box_size = obj.find('bndbox')
            
            
            xmin = int(box_size.find('xmin').text)
            ymin = int(box_size.find('ymin').text)
            xmax = int(box_size.find('xmax').text)
            ymax = int(box_size.find('ymax').text)

            x = ((xmin+xmax)/2)/width
            y = ((ymin+ymax)/2)/height
            x_length = (xmax-xmin)/width
            y_length = (ymax-ymin)/height
            
            
            with open(os.path.join(save_path, file_name + '.txt') ,'w') as f:
                f.write('0 ' + str(round(x,6)) + ' ' + str(round(y,6)) + ' ' + str(round(x_length,6)) + ' ' + str(round(y_length,6)) + '/n')
            
            #with open ()

def main():
    
    pascalvoc_to_yolo(file_dir = 'C:\Machine-Learning', save_path = 'C:\Machine-Learning\Hemo_data\Yolo_Boxes')
    
main()