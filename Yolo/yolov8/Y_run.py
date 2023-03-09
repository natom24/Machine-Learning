import os
from ultralytics import YOLO


#os.chdir('C:\School\Project\Machine-Learning\Yolo\yolov8')

model =  YOLO("C:/School/Project/Machine-Learning/Yolo_v/yolov8/best.pt")

results = model.predict(data="C:/School/Project/Machine-Learning/Yolo/datasets/hemo/images/test", save = True, save_txt = True)
