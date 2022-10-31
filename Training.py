# Import packages
import torch
import torchvision
import torch.nn as nn
import tqdm
from Dataset import HemocyteDataset, collate_fn, get_transforms
from engine import train_one_epoch, evaluate
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
#import matplotlib.pyplot as plt
#from torchvision.utils import draw_bounding_boxes

########################### Model ############################################

def create_model(num_classes):
    backbone = torchvision.models.resnet152(weights=torchvision.models.resnet.ResNet152_Weights)
    
    backbone.out_channels = 1024
    
    anchor_generator = AnchorGenerator(sizes = ((8,16,32,64,128),),aspect_ratios = ((0.5,1.0,2.0),))
    
    roi_pool = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7,sampling_ratio=2)
    
    model = FasterRCNN(backbone = backbone, num_classes = num_classes,
                       rpn_anchor_generator = anchor_generator, box_roi_pool = roi_pool)
    
    return model

################################################################################
   
# Test for GPU, run on CPU if GPU is not available
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

print(f"Training on device {device}.")

model = create_model(num_classes = 2)
model = model.to(device) # runs model on GPU if available

####### Load data #######
hemo_dataset = HemocyteDataset(file_dir='C:\School\Project\Machine-Learning', transforms = get_transforms(train = True)) # Loads hemocyte data in

test_size = 1#int(.99*len(hemo_dataset)) # Generate the size of the test set
train_size = len(hemo_dataset)-test_size # Generate the size of the train set


test_set, train_set = torch.utils.data.random_split(hemo_dataset, [test_size,train_size]) # Split the data into train and test


train_loader = torch.utils.data.DataLoader(train_set, batch_size = 8,shuffle = False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_set, shuffle = True, collate_fn=collate_fn)
##########################



params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.002)# weight_decay=0.00005)
#optimizer = torch.optim.Adam(params, lr=0.001,  weight_decay=0.0001)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 3, gamma = .1)

loss_function = nn.BCELoss()

def train(model, optimizer, train_loader, test_loader, device, epochs = 20):
    
    loss_iter = []
    
    for e in range(epochs):
       
        train_one_epoch(model, optimizer, train_loader, device, epochs, print_freq=10)
        
        lr_scheduler.step()
        
        evaluate(model, test_loader, device=device)
        
#def eval(model,images)
#    model.eval()
train(model, optimizer, train_loader, test_loader, device, epochs = 10)
