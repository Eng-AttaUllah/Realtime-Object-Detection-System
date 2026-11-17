# AI-Based Object Recognition System

A real time object detection system that finds everyday objects (person, laptop, bottle, phone, book, etc) in images, videos and a live webcam feed, draws a box around each one and shows the object name with a confidence score.

Built with Python, OpenCV and a pretrained YOLOv8 model.

## What it does

Show it an image (or point the webcam at your desk) and it:

1. detects every object it knows, draws a box around each one
2. writes the class name + confidence score on the box
3. runs fast enough for live video (yolov8n nano model)

It has two models:

- **`model/object_model.pt`** — fine tuned by me on 9 custom classes: bottle, calculator, cube, flower, gamepad, keyboard, mouse, mug, stapler (the default everywhere)
- **`model/yolov8n.pt`** — the pretrained COCO model, 80 everyday classes (person, laptop, phone, etc)

Example output with the custom model on a test photo it never saw:

```
stapler     -> predicted stapler     conf=0.79
mug         -> predicted mug         conf=0.99
calculator  -> predicted calculator  conf=0.86
```

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
│   ├── bus.jpg, zidane.jpg       coco test images
│   ├── sample_mug.jpg            custom model test images
│   ├── sample_stapler.jpg
│   └── results/                  annotated outputs (not committed)
│
├── dataset_custom/               my fine tuning dataset (357 images, 9 classes)
│   ├── train/ valid/ test/
│   └── data.yaml
│
├── model/
│   ├── object_model.pt           my fine tuned weights (train it, see below)
│   └── yolov8n.pt                pretrained coco weights (auto downloaded)
│
├── notebooks/
│   └── object_recognition.ipynb    phases 1-4, also runs on Kaggle
│
├── detect.py              run detection on a single image
├── real_time.py           live webcam detection (opencv window)
├── app.py                 streamlit web app
├── train_custom.py        fine tune yolov8n on dataset_custom
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

The pretrained weights download by themselves, but the custom model has to be trained once (see "training the custom model" below).

Detect objects in one image (uses my custom 9 class model by default):

```bash
python detect.py --image dataset/sample_mug.jpg
python detect.py --image dataset/bus.jpg --model model/yolov8n.pt   # coco classes
```

Live webcam detection (press q to quit):

```bash
python real_time.py
```

Web app with image upload, video upload and webcam modes, plus a model switcher in the sidebar:

```bash
streamlit run app.py
```

On a cpu-only laptop expect roughly 5-10 fps with the nano model. If it feels slow, the app and real_time.py already use smaller `imgsz=480` for the camera.

## Training the custom model

The pretrained yolov8 only knows 80 COCO classes, so things like a stapler or a mug (mine) come out as wrong labels. Fine tuning fixes that:

1. images were collected and annotated on Roboflow (357 images, one object each, 9 classes)
2. `dataset_custom/` has a clean 305/35/17 train-valid-test split
3. `python train_custom.py` fine tunes yolov8n (40 epochs, imgsz 416, batch 4, ~1 hour on cpu)
4. best weights get copied to `model/object_model.pt`

Final metrics on the held out test split (17 unseen images): mAP50 **0.995**, mAP50-95 **0.73**. Per class (mAP50-95) the strongest is calculator at 0.90 and the weakest is stapler at 0.60, it just needs more stapler photos.

Training results (plots, confusion matrix) are not committed, they regenerate in `runs/`.

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
