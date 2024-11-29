from ultralytics import YOLO


model = YOLO(r'C:\Users\tthrun\Desktop\AI Coding Section\Yolo Test\runs\detect\train40\weights\best.pt')

results = model(
    source=r'C:\Users\tthrun\Pictures\VehiclesTestingSet',  
    save=True,
    imgsz=640,
    conf=0.5, 
    project=r'C:\Users\tthrun\Desktop\AI Coding Section\Yolo Test\output',
    name='VehiclesTestingSetResults'
)
