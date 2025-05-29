import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

model = load_model()

st.title("🚦 Smart Traffic System (Second-by-Second Detection)")
st.markdown("""
- Detects traffic using AI (YOLOv5) every second.
- Assigns signal priority based on vehicle counts.
- Handles emergency vehicles.
- Generates a report at the end.
""")

video_file = st.file_uploader("Upload Traffic Video", type=['mp4', 'avi', 'mov'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames // fps
    stframe = st.empty()

    # Initialize stats
    report_data = []
    total_counts = {'North': 0, 'South': 0, 'East': 0, 'West': 0}
    emergency_events = 0

    for second in range(duration):
        cap.set(cv2.CAP_PROP_POS_FRAMES, second * fps)
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        detections = results.pandas().xyxy[0]

        # Count per lane
        lane_counts = {'North': 0, 'South': 0, 'East': 0, 'West': 0}
        emergency_detected = False

        for _, row in detections.iterrows():
            label = row['name']
            if label not in ['car', 'truck', 'bus', 'motorcycle', 'ambulance', 'fire truck']:
                continue

            if 'ambulance' in label.lower() or 'fire' in label.lower():
                emergency_detected = True

            x_center = int((row['xmin'] + row['xmax']) / 2)
            y_center = int((row['ymin'] + row['ymax']) / 2)

            if x_center < width // 2 and y_center < height // 2:
                lane = 'North'
            elif x_center < width // 2:
                lane = 'West'
            elif y_center < height // 2:
                lane = 'East'
            else:
                lane = 'South'

            lane_counts[lane] += 1
            total_counts[lane] += 1

            # Draw bounding box
            cv2.rectangle(img_rgb, (int(row['xmin']), int(row['ymin'])),
                          (int(row['xmax']), int(row['ymax'])), (0,255,0), 2)
            cv2.putText(img_rgb, label, (int(row['xmin']), int(row['ymin']) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Decide signal
        timestamp = f"{second:02d} sec"
        if emergency_detected:
            signal_status = "ALL RED - Emergency 🚨"
            emergency_events += 1
        else:
            max_lane = max(lane_counts, key=lane_counts.get)
            signal_status = f"{max_lane} GREEN"

        report_data.append({
            "Time (sec)": timestamp,
            **lane_counts,
            "Signal": signal_status
        })

        stframe.image(img_rgb, channels="RGB", caption=f"Time: {timestamp}")
        st.markdown(f"### Signal: `{signal_status}`")
        st.write("Lane Counts:", lane_counts)

    cap.release()

    # Final Report
    st.success("✅ Finished! Generating second-by-second traffic report...")
    st.markdown(f"**Emergency Events:** {emergency_events}")
    st.write("**Total Vehicles by Lane:**", total_counts)

    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df)

    csv = report_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Report", csv, "traffic_signal_report.csv", "text/csv")
