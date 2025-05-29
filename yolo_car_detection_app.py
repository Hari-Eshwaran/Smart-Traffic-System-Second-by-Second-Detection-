import streamlit as st
import torch
import cv2
import tempfile
from PIL import Image
import numpy as np

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

model = load_model()

st.title("🚗 Smart Traffic Detection with YOLOv5")
st.markdown("Upload a traffic video and detect cars in real-time.")

# Upload video
video_file = st.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'mkv'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLOv5 inference
        results = model(img_rgb)

        # Filter for cars, trucks, buses
        labels = results.pandas().xyxy[0]
        vehicles = labels[labels['name'].isin(['car', 'truck', 'bus'])]

        for _, row in vehicles.iterrows():
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            conf = row['confidence']
            label = row['name']
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_rgb, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show processed frame in Streamlit
        stframe.image(img_rgb, channels="RGB")

    cap.release()
