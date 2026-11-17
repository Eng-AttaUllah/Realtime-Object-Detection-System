# AI-Based Object Recognition System

A real time object detection system that finds everyday objects (person, laptop, bottle, phone, book, etc) in images, videos and a live webcam feed, draws a box around each one and shows the object name with a confidence score.

Built with Python, OpenCV and a pretrained YOLOv8 model.

## What it does

Show it an image (or point the webcam at your desk) and it:

1. detects every object it knows (80 classes from the COCO dataset)
2. draws a box around each object
3. writes the class name + confidence score on the box
4. runs fast enough for live video (yolov8n nano model)

Example output on a street photo (`dataset/bus.jpg`):

```
detected 4 object(s) in 1.84s:
  - bus: 0.87
  - person: 0.87
  - person: 0.85
  - person: 0.83
```

Annotated images get saved to `dataset/results/`.

## Tech stack

| Technology | Purpose |
|---|---|
| Python | main language |
| OpenCV | image / video processing, drawing boxes |
| YOLOv8 (ultralytics) | object detection model |
| PyTorch | deep learning backend (yolo runs on it) |
| NumPy | array operations |
| Matplotlib | showing results in the notebook |
| Streamlit | web app interface |

## Project structure

```
Object_Recognition_Project/
│
├── dataset/
│   ├── bus.jpg            test images
│   ├── zidane.jpg
│   └── results/           annotated outputs (not committed)
│
├── model/
│   └── yolov8n.pt         model weights (auto downloaded on first run)
│
├── notebooks/
│   └── object_recognition.ipynb    phases 1-4, also runs on Kaggle
│
├── detect.py              run detection on a single image
├── real_time.py           live webcam detection (opencv window)
├── app.py                 streamlit web app
│
├── requirements.txt
└── README.md
```

## How to run

```bash
# needs python 3.11 or 3.12 (torch does not support 3.13+ yet)
python -m venv .venv
source .venv/bin/activate        # windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Detect objects in one image:

```bash
python detect.py --image dataset/bus.jpg --conf 0.4
```

Live webcam detection (press q to quit):

```bash
python real_time.py
```

Web app with image upload, video upload and webcam modes:

```bash
streamlit run app.py
```

On a cpu-only laptop expect roughly 5-10 fps with the nano model. If it feels slow, the app and real_time.py already use smaller `imgsz=480` for the camera.

## Notebook on Kaggle

`notebooks/object_recognition.ipynb` runs top to bottom on Kaggle as-is (the downloads and pip install work there the same way). It covers:

- phase 1: how object detection works (classification vs detection, why YOLO)
- phase 2: getting and exploring test images
- phase 3: loading the pretrained yolov8n model and its 80 COCO classes
- phase 4: detecting objects in images, drawing boxes manually with OpenCV

## Notes

- The model weights (`yolov8n.pt`, ~6 MB) download automatically the first time you run anything, so they are not in this repo.
- Confidence threshold of 0.4 is a good default for photos, webcam frames are blurrier so some objects need a lower threshold.
- Things I want to add later: custom training on my own objects (books and phone are sometimes missed), and a faster fps counter in the streamlit webcam mode.

## What I learned

- how YOLO predicts all boxes in a single pass and why that makes it realtime capable
- what the YOLO result object contains (class ids, confidences, xyxy boxes) and how to parse it
- OpenCV is BGR, matplotlib is RGB, forget that and your images look blue
- running detection on every video frame on cpu is slow, so smaller input size matters
