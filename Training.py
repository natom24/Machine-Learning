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

model = create_model(num_classes = 1)
model = model.to(device) # runs model on GPU if available

hemo_dataset = HemocyteDataset(file_dir='C:\School\Project\Code', transforms = True)

data_loader = torch.utils.data.DataLoader(hemo_dataset, batch_size = 4, shuffle = True, collate_fn=collate_fn)

params = [p for p in model.parameters() if p.requires_grad]

#box_loss = nn.MSEloader()
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

#hemo_datax

epoch = 1

for e in range(epoch):
    
    train_one_epoch(model,optimizer,data_loader,device,e)

model.eval()
torch.cuda.empty_cache()

img, _ = hemo_dataset[1]
img_int = torch.tensor(img, dtype=torch.uint8)
with torch.no_grad():
    prediction = model([img.to(device)])
    pred = prediction[0]

fig = plt.figure(figsize=(14, 10))
plt.imshow(draw_bounding_boxes(img_int,
    pred['boxes'][pred['scores'] > 0.8], width=4).permute(1, 2, 0))