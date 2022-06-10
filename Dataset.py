import os
import re
import torch
import pandas as pd
#import torch.utils.data.Dataset
from PIL import Image
from torchvision import transforms, utils

torch.__version__

class HemocyteDataset(torch.utils.data.Dataset):
    
    """Hemocyte dataset"""

    def __init__(self,root,annot,transform=None):
        """
        root: the path to the images
        annot: the name of the csv file containing all label data
        transform: any transformations that 
        """
        
        self.root = root
        self.annot = pd.read_csv(os.path.join(root, 'Data', 'Boxes', annot), header = None)
        self.transform = transform
        
        self.img = list(os.listdir(os.path.join(root,'Data',"Images")))
        
    
    def __getitem__(self,idx):
        
        img_path = os.path.join(self.root,'Data',"Images",self.img[idx])
        annot_name = re.sub('\.JPG','.xml',self.img[idx])
        
        img = Image.open(img_path)
        img = img.convert('RGB')
        
        target = {}
        
        
        return img,target
        
    def __len__(self,):
        """
        Return the total number of images
        """
        return len(self.img)