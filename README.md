# Real-Time Alertness Detection Dashboard (Python + FastAPI)

A fast, lightweight Computer Vision web application for detecting drowsiness and sleep states in real-time.

Built with **Python**, **FastAPI**, **OpenCV**, and a **Single-Page Dashboard**.
**No Docker and No Node.js required!**

## ⚙️ System Requirements & Python Version

* **Python Version:** **Python 3.8 to 3.12** *(Recommended: **Python 3.10** or **3.11**)*
* **OS:** Windows, macOS, or Linux

### Install Dependencies:
```bash
pip install -r requirements.txt
```
---

## 🚀 Quick Start (Single Command)

To run the entire web application and open the dashboard:
```bash
python main.py
```
This automatically starts the FastAPI server and opens **http://localhost:5000** 

---

## 📋 What Was Implemented

1. **FastAPI Backend (`main.py`)**:
   - `POST /upload` - Accepts MP4 video file, saves to `uploads/`, returns `video_id`.
   - `GET /detect/{video_id}` - Analyzes video frames, applies temporal alertness tracking, returns JSON scores and statuses.
   - `GET /stream/{video_id}` - Streams video with range-support for HTML5 video player.
   - `GET /video_feed` - Real-time MJPEG live webcam stream with detection overlay.
   - `GET /` - Interactive modern dashboard with synchronized video playback, bounding box overlay, live alert badges, and alert history table.

2. **Core Alertness Logic (`detector.py`)**:
   - Continuous closed-eye duration tracking:
     - **Awake (Green)**: Normal eye blinks ($< 1.0\text{ sec}$). Score: 90–100%.
     - **Drowsy (Yellow)**: Eyes closed continuously $\ge 1.0\text{ sec}$ and $< 3.0\text{ sec}$. Score: 40–65%.
     - **Sleeping (Red)**: Eyes closed continuously $\ge 3.0\text{ sec}$. Score: 5–20%.

3. **Performance Optimization**:
   - Samples video frames at 4 FPS for quick processing (processes full test video in 1–2 seconds).
   - In-memory result caching to disk (`{video_id}_results.json`) for instant repeat loads.

---
