import os
import sys
import glob
import xml.etree.ElementTree as ET

def pascalvoc_to_yolo(file_dir, save_path, width = 1600, height = 1200):
    
    annot_path = os.path.join(file_dir, "Hemo_data","Small_Grid","VOC_Boxes")
    
    for file in os.listdir(annot_path):
        
        tree = ET.parse(os.path.join(annot_path, file))
        root = tree.getroot()
        
        file_name = file
        file_name = file_name[:-4] # Drop the xml ending
        
        #file_path = os.path.join(save_path, file_name + '.txt')
        
        boxes = []
        
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
            
            boxes.append('0 ' + str(round(x,6)) + ' ' + str(round(y,6)) + ' ' + str(round(x_length,6)) + ' ' + str(round(y_length,6)) + '\n')
            

            with open(file_path ,'w') as f:
                for i in boxes:
                    f.write(i)
                    
    print('Done!')
            
def main():
    
    pascalvoc_to_yolo(file_dir = 'C:\School\Project\Machine-Learning', save_path = 'C:\School\Project\Machine-Learning\Hemo_data\Small_Grid\Yolo_Boxes')
    
    
main()