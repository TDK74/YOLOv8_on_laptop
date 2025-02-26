import cv2
from time import time
import torch
from ultralytics import YOLO


# Load YOLOv8n
# model = YOLO('yolov8n.pt').to('cuda')  # transfer the model to GPU
# Load YOLOv8s
# model = YOLO('yolov8s.pt').to('cuda')  # transfer the model to GPU
# load YOLOv8m
model = YOLO('yolov8m.pt').to('cuda')  # transfer the model to GPU
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("error")
    exit()

width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = 0

def fps_calculation(fps, start_time, end_time):
    loop_time = end_time - start_time
    fps_func = 1 / loop_time

    return fps_func


def objects_processing(frame):
    results = model(frame, device='cuda')  # use of GPU for processing

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box[ : 4])
            confidence = confidences[i]
            class_id = class_ids[i]
            label = result.names[class_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

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
        #cv2.imshow('YOLOv8n Real-time', frame)
        #cv2.imshow('YOLOv8s Real-time', frame)
        cv2.imshow('YOLOv8m Real-time', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("User's Ctrl+C detected.")

finally:
    cam.release()
    cv2.destroyAllWindows()
