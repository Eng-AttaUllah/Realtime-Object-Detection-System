# realtime object detection with the webcam
# usage: python real_time.py
# press q in the window to quit

import time

import cv2
from ultralytics import YOLO

# conf 0.4 works ok for webcam since frames are blurry compared to photos
CONF = 0.4

# my custom model, swap for model/yolov8n.pt if you want the 80 coco classes
MODEL = "model/object_model.pt"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("could not open the webcam")
    raise SystemExit

model = YOLO(MODEL)

# using 480p and imgsz 480 to keep the fps reasonable on cpu
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
fps = 0.0
t_last = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        print("failed to grab frame")
        break

    res = model(frame, conf=CONF, imgsz=480, verbose=False)[0]

    annotated = res.plot()
    for b in res.boxes:
        pass  # could print detections here but it spams the terminal

    # rough fps, updated once a second so the number is not jumping around
    frame_count += 1
    now = time.time()
    if now - t_last >= 1.0:
        fps = frame_count / (now - t_last)
        frame_count = 0
        t_last = now

    cv2.putText(annotated, f"fps: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("real time detection (q to quit)", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
