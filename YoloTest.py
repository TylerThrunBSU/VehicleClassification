from ultralytics import YOLO
from roboflow import Roboflow
import os
from IPython.display import display

model = YOLO("yolo11n.pt")


rf = Roboflow(api_key="21RCuRJpDrRXGR6CYUIS")
project = rf.workspace("nyankosensei").project("highway-cctv-images-for-vehicle-detection-dataset")
version = project.version(6)
dataset = version.download("yolov11")
                
                                

data_yaml_path = os.path.join(dataset.location, "data.yaml")
print(f"Dataset configuration file located at: {data_yaml_path}")

model.train(
    data=data_yaml_path,
    epochs=10,
    imgsz=640,
    batch=-1,
    workers=0,
    amp=False
)
