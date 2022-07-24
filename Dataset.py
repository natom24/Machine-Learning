import os
import re
import torch
from torchvision import transforms
import xml.etree.ElementTree as ET
from PIL import Image
#from torchvision import utils

class HemocyteDataset(torch.utils.data.Dataset):
    
    """Hemocyte dataset"""

    def __init__(self,file_dir, transforms = None):
        """
        root: the path to the images
        annot: the name of the csv file containing all label data
        transform: any transformations that 
        """
        
        self.file_dir = file_dir
        
        self.img_list = list(os.listdir(os.path.join(file_dir,'Hemo_data',"Images")))
        
        self.transforms = transforms
        
    
    def __getitem__(self,idx):
        """
        Load in annotations and images
        """
        # Pulls a string with the path to the selected image
        
        img_path = os.path.join(self.file_dir,"Hemo_data","Images",self.img_list[idx]) 
        
        annot_name = re.sub('\.JPG','.xml',self.img_list[idx]) # Get name of file with xml ending
        annot_path = os.path.join(self.file_dir,"Hemo_data","Boxes",annot_name)
        
        img = Image.open(img_path)
        img = img.convert('RGB')
        
        boxes = []
        #iscrowd = []
        
        tree = ET.parse(annot_path)
        root = tree.getroot()
            
        for obj in root.findall('object'):
            box_size = obj.find('bndbox')
            
            xmin = int(box_size.find('xmin').text)
            ymin = int(box_size.find('ymin').text)
            xmax = int(box_size.find('xmax').text)
            ymax = int(box_size.find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])
            


        if self.transforms is not None:
            #t_data = self.transforms(img, boxes)
            #img = t_data['images']
            #boxes = t_data['boxes']
            
            to_tensor= transforms.ToTensor() # May need to be changed, just a quick method of converting to tensor for testing
            imgs = to_tensor(img) # Calls conversion
        

        target = {}
        target['boxes'] = torch.as_tensor(boxes,dtype=torch.float32)
        target['labels'] = torch.ones((len(boxes),), dtype=torch.int64)
        target['image_id'] = torch.as_tensor([idx])
        

        
        
        return imgs,target
        
    def __len__(self,):
        """
        Return the total number of images
        """
        return len(self.img_list)

