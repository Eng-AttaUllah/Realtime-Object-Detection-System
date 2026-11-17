# streamlit app for the object recognition project
# run with: streamlit run app.py

import os
import tempfile

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


# load the model only once instead of on every rerun, app was slow without this
@st.cache_resource
def load_model(weights):
    return YOLO(weights)


# sidebar controls
conf = st.sidebar.slider("confidence threshold", 0.05, 0.95, 0.4, 0.05)
which = st.sidebar.radio("model", ["custom (9 classes)", "pretrained (80 classes)"])
weights = "model/object_model.pt" if which.startswith("custom") else "model/yolov8n.pt"
model = load_model(weights)

st.title("Real Time Object Recognition")
st.write("detect objects in images, videos or the webcam using yolov8")

source = st.sidebar.radio("choose input", ["Image", "Video", "Webcam"])
st.sidebar.write(f"loaded: {weights}")

# ------------------------------------------------ image mode
if source == "Image":
    file = st.file_uploader("upload an image", type=["jpg", "jpeg", "png"])
    if file is not None:
        # decode the uploaded bytes into an opencv image
        bytes_data = file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        res = model(img, conf=conf)[0]

        st.subheader("result")
        st.image(res.plot()[:, :, ::-1], use_container_width=True)

        st.subheader("detected objects")
        if len(res.boxes) == 0:
            st.write("nothing found, try lowering the confidence threshold")
        for b in res.boxes:
            cls = int(b.cls[0])
            c = float(b.conf[0])
            st.write(f"- {model.names[cls]}: **{c:.2f}**")

# ------------------------------------------------ video mode
elif source == "Video":
    file = st.file_uploader("upload a video", type=["mp4", "avi", "mov"])
    if file is not None:
        # ultralytics cant read uploaded bytes directly so save to a temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(file.getvalue())

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            # some codecs dont report the frame count
            total = 1

        # this runs the whole video before showing anything, can take a while on cpu
        st.write(f"processing {total} frames... this might take a bit on cpu")
        out_path = os.path.join(tempfile.gettempdir(), "detected_video.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        progress = st.progress(0.0)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = model(frame, conf=conf, imgsz=480, verbose=False)[0]
            writer.write(res.plot())
            i += 1
            progress.progress(min(i / total, 1.0))

        cap.release()
        writer.release()

        st.subheader("result video")
        st.video(out_path)
        st.caption("processing every frame on cpu is slow, higher res videos take longer")

# ------------------------------------------------ webcam mode
else:
    # option 1: take a single photo with the browser camera (works everywhere)
    st.subheader("take a snapshot")
    snap = st.camera_input("take a photo and detect objects in it")
    if snap is not None:
        img = cv2.imdecode(np.frombuffer(snap.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        res = model(img, conf=conf)[0]
        st.image(res.plot()[:, :, ::-1], use_container_width=True)
        for b in res.boxes:
            cls = int(b.cls[0])
            st.write(f"- {model.names[cls]}: **{float(b.conf[0]):.2f}**")

    # option 2: live feed with the local webcam (only works when app runs locally)
    st.subheader("live feed")
    st.write("uses the webcam through opencv, needs the app running on your machine")
    go = st.checkbox("start camera")
    frame_slot = st.empty()
    stop = st.button("stop")

    if go:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("could not open the webcam")
        else:
            # 480p keeps it less laggy on cpu
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            while go and not stop:
                ok, frame = cap.read()
                if not ok:
                    st.error("failed to grab frame")
                    break
                res = model(frame, conf=conf, imgsz=480, verbose=False)[0]
                frame_slot.image(res.plot()[:, :, ::-1], use_container_width=True)
            cap.release()
            frame_slot.empty()
