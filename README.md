# 🚦 Smart Traffic Management System (AI-Powered)

A Streamlit web app that simulates a smart traffic light controller using AI to detect traffic density and prioritize lanes accordingly. Emergency vehicles are given highest priority with complete stop signals.

---

## 💡 Features

- ✅ Vehicle detection using **YOLOv5** (cars, trucks, bikes, buses, etc.).
- ✅ **Emergency vehicle detection** (ambulance, fire truck).
- ✅ Dynamic signal control based on real-time traffic density.
- ✅ **One-frame-per-second** analysis for efficient and clear control.
- ✅ **Visual video display** with bounding boxes and current signal.
- ✅ Downloadable **CSV report** of the traffic situation and decisions.

---

## 📦 Tech Stack

- **Streamlit** – for interactive UI
- **OpenCV** – for video processing
- **YOLOv5** – object detection
- **Pandas** – for report generation

---

## 🚀 How to Run

### 1. Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/smart-traffic-ai.git
cd smart-traffic-ai
pip install -r requirements.txt
