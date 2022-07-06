# Import packages
import torch
import torchvision
import torch.nn as nn
#from datasets import HemocyteDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes

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

model = create_model(num_classes = 2)
model = model.to(device) # runs model on GPU if available

hemo_dataset = HemocyteDataset(file_dir='C:\School\Project\Code', transforms = True)

data_loader = torch.utils.data.DataLoader(hemo_dataset, batch_size = 2, shuffle = True, collate_fn=collate_fn)

params = [p for p in model.parameters() if p.requires_grad]

#box_loss = nn.MSEloader()
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

#hemo_datax

num_epoch = 5


for e in range(num_epoch):
    
    train_one_epoch(model,optimizer,data_loader,device,e)


def train(model, optimizer, data_loader, device, epochs = num_epoch):

    for e in range(epochs):
        model.to(device)
        model.train()
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        
        losses.backward()
        optimizer.step()
            
            
model.eval()