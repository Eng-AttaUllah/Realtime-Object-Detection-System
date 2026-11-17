# fine tune yolov8n on my custom 9 class dataset
# usage: python train_custom.py
# takes a while on cpu, run it and leave it

from ultralytics import YOLO

model = YOLO("model/yolov8n.pt")

# imgsz 416 because the roboflow export is already 416x416
# epochs 40 is enough for 305 images, val mAP stops improving after that
# batch 4 because batch 8 got oom killed, this laptop has 8gb ram
# results (weights + plots) end up in runs/detect/custom_v1
model.train(
    data="dataset_custom/data.yaml",
    epochs=40,
    imgsz=416,
    batch=4,
    device="cpu",
    workers=2,
    cache=False,
    save_period=10,
    name="custom_v1",
)

# quick sanity check on the test split with the best weights
metrics = model.val(split="test")
print(metrics)
