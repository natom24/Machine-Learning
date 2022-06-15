import os
import re
import torch
import pandas as pd
#import torch.utils.data.Dataset
from PIL import Image
from torchvision import utils

torch.__version__

class HemocyteDataset(torch.utils.data.Dataset):
    
    """Hemocyte dataset"""

    def __init__(self,root,annot):
        """
        root: the path to the images
        annot: the name of the csv file containing all label data
        transform: any transformations that 
        """
        
        self.root = root
        self.annot = pd.read_csv(os.path.join(root, 'Data', annot))
        
        self.img = list(os.listdir(os.path.join(root,'Data',"Images")))
        
    
    def __getitem__(self,idx):
        """
        Load in annotations and images
        """
        # Pulls a string with the path to the selected image
        img_path = os.path.join(self.root,'Data',"Images",self.img[idx]) 
        annot_name = re.sub('\.JPG','.xml',self.img[idx]) # Get name of file with xml ending
        
        img = Image.open(img_path)
        img = img.convert('RGB')
        
        label_data = self.annot[self.annot.iloc[:,0] == annot_name]
        
        boxes = []
        labels = []
        area = []
        iscrowd = []
        
        label_data[:,3] = pd.factorize(label_data[:,3])
        
        for l in label_data:
            
            
            xmin = label_data[l,4]
            ymin = label_data[l,5]
            xmax = label_data[l,6]
            ymax = label_data[l,7]
            labels.append(label_data[l,3])
            boxes.append([[xmin, ymin, xmax, ymax]])
            area.append((xmax-xmin)*(ymax-ymin))
            

        target = {}
        target['boxes'] = torch.as_tensor(boxes,dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        target['iscrowd'] = torch.zeros(boxes.shape[0], dtype=torch.int64)
        target['image_id'] = torch.as_tensor([idx])
        
        
        return img,target
        
    def __len__(self,):
        """
        Return the total number of images
        """
        return len(self.img)