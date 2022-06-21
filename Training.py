# Import packages
import torch
import torchvision
import utils
from datasets import HemocyteDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

########################### Model ############################################

def create_model(num_classes):
    
    # load model 
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True) 
    
    # Define the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    #replace the pre-trained head
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

################################################################################
   
# Test for GPU, run on CPU if GPU is not available
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

print(f"Training on device {device}.")

model = create_model(1)

model = model.to(device) # runs model to GPU if available

hemo_dataset = HemocyteDataset(file_dir='C:\School\Project\Code', transforms = True)

dataset_loader = torch.utils.data.DataLoader(hemo_dataset, batch_size = 2, shuffle = True, collate_fn=collate_fn)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)



epoch = 10

def train_one_epoch():
    model = model.to(device) # runs model to GPU if available
    
    model.train()

for e in range(epoch):
    print('hello')