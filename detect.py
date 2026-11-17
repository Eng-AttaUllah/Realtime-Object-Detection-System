# simple script to run yolo on a single image
# usage: python detect.py --image dataset/bus.jpg

import argparse
import os
import time

import cv2
from ultralytics import YOLO


def draw_boxes(img, boxes, names):
    # draw boxes manually so I can control how the label looks
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        label = f"{names[cls]} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        # put label above the box, if it goes off the top just draw it inside
        ty = y1 - 6 if y1 - 6 > 10 else y1 + 15
        cv2.putText(img, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
    return img


def main():
    parser = argparse.ArgumentParser(description="run yolov8 on an image")
    parser.add_argument("--image", required=True, help="path to the image")
    parser.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    # default is my custom trained model, pass --model model/yolov8n.pt for the 80 class coco one
    parser.add_argument("--model", default="model/object_model.pt", help="which weights to use")
    parser.add_argument("--show", action="store_true", help="open a window with the result (needs a display)")
    args = parser.parse_args()

    model = YOLO(args.model)

    img = cv2.imread(args.image)
    if img is None:
        print("could not read the image, check the path")
        return

    t0 = time.time()
    res = model(img, conf=args.conf)[0]
    dt = time.time() - t0

    names = model.names
    print(f"\ndetected {len(res.boxes)} object(s) in {dt:.2f}s:")
    for b in res.boxes:
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        print(f"  - {names[cls]}: {conf:.2f}")

    out_img = draw_boxes(img.copy(), res.boxes, names)

    # always save the result so there is a record of what was detected
    out_dir = "dataset/results"
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]
    out_path = os.path.join(out_dir, base + "_detected.jpg")
    cv2.imwrite(out_path, out_img)
    print(f"\nsaved to {out_path}")

    if args.show:
        # show it, press any key to close the window
        cv2.imshow("detections (press any key)", out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
