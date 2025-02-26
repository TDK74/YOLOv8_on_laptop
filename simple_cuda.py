import ultralytics
from ultralytics import YOLO


model = YOLO('yolov8n.pt')  # load an official model
results = model(source='DiskName (C, D, E...):\path-to-folder\file_name.jpg', device='cuda')  # use of GPU for processing
