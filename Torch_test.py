import torch
#import torch.utils.data.Dataset
from PIL import Image
from torchvision import transforms, utils

torch.__version__

class HemocyteDataset(torch.utils.data.Dataset):
    
    """Hemocyte dataset"""

    def __init__(self,root,annot,transform=None):
        """
        root: the path to the images
        annot: the csv file with all annotations
        transform: any transformations that 
        """
        self.root = root
        self.annot = annot
        self.transform = transform
    
    def __getitem__(self,id):
        print("finish")