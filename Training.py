# Import packages
import torch
import torchvision
import torch.nn as nn
import tqdm
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



####### Load data #######
hemo_dataset = HemocyteDataset(file_dir='C:\School\Project\Code', transforms = True) # Loads hemocyte data in

test_size = int(.5*len(hemo_dataset)) # Generate the size of the test set
train_size = len(hemo_dataset)-test_size # Generate the size of the train set

train_set, test_set = torch.utils.data.random_split(hemo_dataset, [test_size,train_size]) # Split the data into train and test

train_loader = torch.utils.data.DataLoader(train_set, shuffle = True, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 2, shuffle = True, collate_fn=collate_fn)
##########################



params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

loss_function = nn.BCELoss()

def train(model, optimizer, data_loader, device, epochs = 5):
    
    for e in range(epochs):
        
        model.to(device)
        model.train()
        
        for images, targets in data_loader:
            optimizer.zero_grad()
        
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
        
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
            losses.backward()
            optimizer.step()
            
        print("Epoch {} has a loss rate of {}".format(e,losses.item()))
            
        
#def eval(model,images)
#    model.eval()
    
    
    
#    return()