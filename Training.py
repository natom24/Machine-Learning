# Import packages
import torch
import torchvision
import torch.nn as nn
import tqdm
import Dataset
import engine
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes

########################### Model ############################################

def create_model(num_classes = 2):
    
    # load model 
    #model = torchvision.models.detection.faster_rcnn.FasterRCNN(backbone = weights = True) 
    
    # Define the number of input features
    #in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    
      ############Delete?##############
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True, num_classes = 2)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    #model.roi_heads.maskrcnn_loss = nn.BCELoss()
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)
    ############Delete?##############
    
    #replace the pre-trained head
    #model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
################################################################################
   
# Test for GPU, run on CPU if GPU is not available
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

print(f"Training on device {device}.")

model = create_model(num_classes = 2)
model = model.to(device) # runs model on GPU if available

####### Load data #######
hemo_dataset = HemocyteDataset(file_dir='C:\Machine-Learning', transforms = True) # Loads hemocyte data in

test_size = int(.99*len(hemo_dataset)) # Generate the size of the test set
val_size = 0 #int((len(hemo_dataset)-test_size)*.4)
train_size = len(hemo_dataset)-test_size-val_size # Generate the size of the train set


test_set, train_set, val_set = torch.utils.data.random_split(hemo_dataset, [test_size,train_size, val_size]) # Split the data into train and test


train_loader = torch.utils.data.DataLoader(train_set, batch_size = 23,shuffle = False, collate_fn=collate_fn)
#val_loader = torch.utils.data.DataLoader(val_set, shuffle = True, collate_fn=collate_fn)
##########################



params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.00005)
#optimizer = torch.optim.Adam(params, lr=0.001,  weight_decay=0.0001)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 3, gamma = .1)

loss_function = nn.BCELoss()

def train(model, optimizer, train_loader, device, epochs = 20):
    
    loss_iter = []
    
    for e in range(epochs):
        
        model.to(device)
        model.train()
        
        for images, targets in tqdm.tqdm(train_loader):
            
        
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
        
            loss_dict = model(images, targets)
            #losses = loss_function(loss_dict, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            losses.backward()
            
            optimizer.zero_grad()
            optimizer.step()
            
            loss_iter.append(losses.item())
        print("Epoch {} has a loss rate of {}".format(e,losses.item()))
        
        #lr_scheduler.step()
        #valid_loss = 0
        #model.eval()
        #for images, target in val_loader:
        #    model(images)
        
#def eval(model,images)
#    model.eval()
train(model, optimizer, train_loader, device, epochs = 10)
