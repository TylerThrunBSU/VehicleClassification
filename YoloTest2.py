from ultralytics import YOLO
from roboflow import Roboflow
import os
from IPython.display import display

model = YOLO("yolo11n.pt")



model.train(
    data=r'C:\Users\tthrun\Desktop\AI Coding Section\Yolo Test\stanford-car-yolov5s-1\data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    workers=0,
    amp=False
)
