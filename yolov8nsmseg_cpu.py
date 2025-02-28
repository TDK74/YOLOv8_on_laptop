import cv2
from time import time
# import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np


# load YOLOv8n-seg
# model = YOLO('yolov8n-seg.pt')
# load YOLOv8s-seg
# model = YOLO('yolov8s-seg.pt')
# load YOLOv8m-seg
model = YOLO('yolov8m-seg.pt')
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error opening camera.")
    exit()

width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = 0

def fps_calculation(fps, start_time, end_time):
    loop_time = end_time - start_time
    fps_func = 1 / loop_time

    return fps_func


def objects_processing(frame):
    results = model(frame, device='cpu')

    for result in results:
        pil_image = Image.fromarray(results[0].plot()[ : , : , : : -1])
        frame = np.array(pil_image)

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box[ : 4])
            confidence = confidences[i]
            class_id = class_ids[i]
            label = result.names[class_id]

    return frame


try:
    while True:
        start_time = time()
        ret, frame = cam.read()

        if not ret:
            print("Error reading frame.")
            break

        frame = objects_processing(frame)

        end_time = time()
        fps = fps_calculation( fps, start_time, end_time)

        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        #cv2.imshow('YOLOv8n-seg Real-time', frame)
        #cv2.imshow('YOLOv8s-seg Real-time', frame)
        cv2.imshow('YOLOv8m-seg Real-time', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("User's Ctrl+C detected.")

finally:
    cam.release()
    cv2.destroyAllWindows()
