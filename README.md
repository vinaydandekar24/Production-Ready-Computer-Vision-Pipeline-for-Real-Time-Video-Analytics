# Production-Ready-Computer-Vision-Pipeline-for-Real-Time-Video-Analytics
A real-time vehicle speed detection and monitoring system that uses YOLOv8 object detection, perspective transformation, and multi-object tracking to estimate vehicle speeds from CCTV/IP camera feeds. Detected vehicles are captured, uploaded to AWS S3, and logged to MongoDB — with live annotated video streamed via RTSP.
# 🚗 Vehicle Speed Estimator

A real-time vehicle speed detection and monitoring system that uses YOLOv8 object detection, perspective transformation, and multi-object tracking to estimate vehicle speeds from CCTV/IP camera feeds. Detected vehicles are captured, uploaded to AWS S3, and logged to MongoDB — with live annotated video streamed via RTSP.

---

## 📋 Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Calibration](#calibration)
- [Running the System](#running-the-system)
- [How It Works](#how-it-works)
- [MongoDB Schema](#mongodb-schema)
- [Environment Variables Reference](#environment-variables-reference)
- [Security Notice](#security-notice)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

- **Real-time vehicle detection** using YOLOv8 (supports cars, motorcycles, buses, trucks)
- **Speed estimation** via perspective-corrected homography and frame-to-frame displacement
- **Multi-object tracking** using ByteTrack (Supervision) with IoU-based fallback tracker
- **Overspeed alerting** — bounding boxes turn red when a vehicle exceeds the configured speed limit
- **Vehicle snapshot capture** — collects 5 clear (non-blurry) cropped frames per vehicle
- **AWS S3 upload** — vehicle crops are uploaded asynchronously in background threads
- **MongoDB logging** — each detected vehicle produces a structured record with speed, type, S3 image URLs, and timestamp
- **RTSP live streaming** — annotated video is forwarded via a C++ companion binary (`live_streaming.exe`) through FFmpeg to a MediaMTX RTSP server
- **GPU/CPU auto-detection** — uses YOLOv8 `.pt` on CUDA if available, falls back to OpenVINO on CPU
- **Blur filtering** — skips blurry frames and crops to maintain data quality

---

## 🏗️ System Architecture

```
IP Camera (RTSP)
        │
        ▼
speed_estimator.py
  ├── YOLOv8 detection (GPU / OpenVINO CPU)
  ├── ByteTrack / SimpleTracker
  ├── ViewTransformer (homography)
  ├── Speed calculation (m/s → km/h)
  ├── VehicleFrameBuffer
  │     ├── Collects 5 clear crops per vehicle
  │     └── Background thread → AWS S3 upload → MongoDB insert
  └── Annotated frames via TCP socket
              │
              ▼
        live_streaming.exe  (C++)
              │
              ▼
        FFmpeg  →  H.264 encode
              │
              ▼
     MediaMTX RTSP Server (EC2)
              │
              ▼
      RTSP consumers / dashboards
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Tracking | [Supervision ByteTrack](https://github.com/roboflow/supervision) |
| Computer Vision | OpenCV, NumPy |
| Deep Learning | PyTorch (CUDA) / OpenVINO (CPU) |
| Database | MongoDB Atlas (PyMongo) |
| Object Storage | AWS S3 (boto3) |
| Streaming | FFmpeg + MediaMTX |
| Streaming bridge | Custom C++ TCP→RTSP forwarder (`live_streaming.cpp`) |
| Config | python-dotenv |

---

## 📁 Project Structure

```
vehicle-speed-estimator/
├── speed_estimator.py       # Main application — detection, tracking, speed estimation
├── live_streaming.cpp       # C++ RTSP forwarder (compile to live_streaming.exe)
├── live_streaming.exe       # Pre-compiled Windows binary (or compile yourself)
├── calibration.json         # Perspective calibration points for the camera
├── .env                     # Environment variables (DO NOT COMMIT — see Security)
├── .env.example             # Template for environment variables
└── README.md
```

---

## ✅ Prerequisites

- Python 3.10+
- FFmpeg installed and available on `PATH`
- A running [MediaMTX](https://github.com/bluenviron/mediamtx) RTSP server (or any RTSP server)
- MongoDB Atlas cluster (or local MongoDB)
- AWS S3 bucket with appropriate IAM permissions
- (Optional) NVIDIA GPU with CUDA for hardware-accelerated inference

**Python packages:**

```
ultralytics
supervision
opencv-python
numpy
torch
pymongo
boto3
python-dotenv
```

---

## 🚀 Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/vehicle-speed-estimator.git
cd vehicle-speed-estimator
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

3. **Install Python dependencies**

```bash
pip install ultralytics supervision opencv-python numpy torch pymongo boto3 python-dotenv
```

> For GPU inference, install the CUDA-enabled version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

4. **Configure environment variables**

```bash
cp .env.example .env
# Edit .env with your actual credentials and settings
```

5. **(Optional) Compile the C++ streaming bridge**

```bash
# Windows (MinGW / MSYS2)
g++ -std=c++17 -o live_streaming.exe live_streaming.cpp -lws2_32 -lpthread

# Linux
g++ -std=c++17 -o live_streaming live_streaming.cpp -lpthread
```

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and fill in your values:

```env
# AWS S3
AWS_ACCESS_KEY=YOUR_ACCESS_KEY
AWS_SECRET_KEY=YOUR_SECRET_KEY
AWS_REGION=us-east-1
AWS_BUCKET_NAME=your-bucket-name

# MongoDB
MONGO_URI=mongodb+srv://user:password@cluster.mongodb.net/
MONGO_DB_NAME=vehicle

# Camera / Stream
SOURCE_MODE=python
RUN_SOURCE=rtsp://admin:password@192.168.1.2:554/stream
RUN_CAMERA_ID=CAM_001
RUN_SPEED_LIMIT=80

# Live Streaming (C++ bridge → FFmpeg → MediaMTX)
FRAME_PORT=9000
FRAME_WIDTH=1920
FRAME_HEIGHT=1080
FRAME_FPS=25
EC2_SERVER=your.ec2.ip.address
STREAM_NAME=mycamera
MEDIAMTX_USERNAME=publisher
MEDIAMTX_PASSWORD=yourSecurePassword

# Quality: LOW / MEDIUM / HIGH
QUALITY=HIGH

# App flags
RUN_SHOW_PREVIEW=False
RUN_SAVE_OUTPUT=False
RUN_FRAME_STEP=1
RUN_RTSP_OUTPUT=enabled
RUN_LS_EXE=live_streaming.exe
```

---

## 📐 Calibration

Calibration maps a trapezoidal region of the camera image to a real-world rectangular area (in metres). This is used to compute accurate distances and therefore speeds.

**`calibration.json` format:**

```json
{
  "source_points": [
    [943, 216],
    [1748, 272],
    [1484, 699],
    [254, 471]
  ],
  "real_width_m": 6.0,
  "real_length_m": 7.0,
  "frame_width": 2880,
  "frame_height": 1620
}
```

| Field | Description |
|---|---|
| `source_points` | Four pixel coordinates (top-left, top-right, bottom-right, bottom-left) forming the road region of interest |
| `real_width_m` | Actual width of the road segment in metres |
| `real_length_m` | Actual length of the road segment in metres |
| `frame_width` / `frame_height` | Resolution at which the points were measured (auto-scaled if your stream differs) |

The calibration can be stored either locally as `calibration.json` or in the MongoDB `calibration` collection with a matching `camera_id` field.

---

## ▶️ Running the System

```bash
python speed_estimator.py
```

The script will:
1. Connect to MongoDB and load calibration data
2. Auto-detect GPU (CUDA) or fall back to OpenVINO on CPU
3. Launch `live_streaming.exe` (if `RUN_RTSP_OUTPUT=enabled`)
4. Open the configured camera source
5. Begin detection, tracking, speed estimation, and streaming

**Stop with `Ctrl+C`** — all resources (camera, writer, socket, subprocess) are cleaned up gracefully.

---

## ⚙️ How It Works

### Speed Estimation Pipeline

1. **Frame capture** — frames are read from an RTSP stream (or video file / webcam)
2. **Blur check** — frames with a Laplacian variance below the threshold are discarded
3. **YOLOv8 inference** — detects vehicles (cars, motorcycles, buses, trucks)
4. **Tracking** — ByteTrack assigns persistent IDs across frames
5. **Perspective transform** — bounding box bottom-center points are projected from pixel space to real-world metres using homography
6. **Speed calculation** — displacement over time (metres per second → km/h) is computed from the coordinate history sliding window
7. **Snapshot buffering** — `VehicleFrameBuffer` collects 5 sharp, spaced-out crops per vehicle
8. **Async upload** — once 5 frames are collected, a background thread uploads them to S3 and inserts a record into MongoDB

### Live Streaming

Annotated frames are sent as raw BGR bytes over a local TCP socket to `live_streaming.exe`, which pipes them into FFmpeg for H.264 encoding and RTSP forwarding to a MediaMTX server.

---

## 🗄️ MongoDB Schema

**Collection: `speedEstimates`**

```json
{
  "vehicleID":   "1863",
  "camera_id":   "CAM_001",
  "vehicleType": "truck",
  "speed":       "54.2 KM/H",
  "limit":       80,
  "numPlate":    null,
  "frameImgs":   [
    "https://bucket.s3.region.amazonaws.com/vehicles/1863/..._frame_120.jpg",
    "..."
  ],
  "capturedAt":  "2026-06-08T10:23:45.000Z",
  "status":      false
}
```

---

## 🔐 Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `AWS_ACCESS_KEY` | Yes | AWS IAM access key ID |
| `AWS_SECRET_KEY` | Yes | AWS IAM secret access key |
| `AWS_REGION` | Yes | S3 bucket region |
| `AWS_BUCKET_NAME` | Yes | S3 bucket name for vehicle images |
| `MONGO_URI` | Yes | MongoDB connection string |
| `MONGO_DB_NAME` | Yes | Database name |
| `RUN_SOURCE` | Yes | Camera RTSP URL, video file path, or webcam index |
| `RUN_CAMERA_ID` | Yes | Logical camera identifier (must match calibration) |
| `RUN_SPEED_LIMIT` | No | Speed limit in km/h (default: 80) |
| `SOURCE_MODE` | No | `python` (TCP frames) or `rtsp` (direct forward) |
| `EC2_SERVER` | No | MediaMTX server IP or hostname |
| `STREAM_NAME` | No | RTSP stream name on MediaMTX |
| `MEDIAMTX_USERNAME` | No | MediaMTX publisher username |
| `MEDIAMTX_PASSWORD` | No | MediaMTX publisher password |
| `QUALITY` | No | FFmpeg encode quality: `LOW`, `MEDIUM`, `HIGH` |
| `FRAME_PORT` | No | Local TCP port for C++ bridge (default: 9000) |
| `FRAME_WIDTH/HEIGHT/FPS` | No | Output stream dimensions and frame rate |
| `RUN_SHOW_PREVIEW` | No | Show local OpenCV preview window (default: False) |
| `RUN_SAVE_OUTPUT` | No | Save annotated video to MP4 (default: False) |
| `RUN_FRAME_STEP` | No | Process every Nth frame (default: 1) |
| `RUN_RTSP_OUTPUT` | No | Enable RTSP output via live_streaming.exe (default: enabled) |
| `RUN_LS_EXE` | No | Path to live_streaming.exe (default: `./live_streaming.exe`) |

---

## 🔒 Security Notice

> **⚠️ IMPORTANT — Never commit your `.env` file to version control.**

Your `.env` file contains sensitive credentials including AWS access keys and your MongoDB connection string. Exposing these publicly can lead to unauthorized cloud usage and data breaches.

**Before pushing to GitHub:**

1. Add `.env` to `.gitignore`:
   ```
   .env
   *.log
   __pycache__/
   *.pt
   yolov8n_openvino_model/
   output_*.mp4
   ```

2. Provide an `.env.example` with placeholder values (no real credentials).

3. If credentials were ever committed, **rotate them immediately**:
   - AWS: generate new keys in IAM and deactivate the old ones
   - MongoDB: reset the database user password

---

## 🐛 Troubleshooting

**`No calibration found for camera 'CAM_001'`**
→ Make sure `calibration.json` is in the same directory, or that a matching document exists in the MongoDB `calibration` collection.

**`live_streaming.exe not found`**
→ Either compile from `live_streaming.cpp` or set `RUN_RTSP_OUTPUT=disabled` in `.env` to disable RTSP streaming.

**`ultralytics not installed`**
→ Run `pip install ultralytics`

**`boto3 not installed — S3 upload disabled`**
→ Run `pip install boto3`. S3 upload is optional; the system runs without it.

**Low speed accuracy**
→ Verify that `source_points` in `calibration.json` correctly covers a measurable road section, and that `real_width_m` / `real_length_m` reflect accurate real-world dimensions.

**No GPU detected / falling back to CPU**
→ Ensure CUDA-enabled PyTorch is installed and the NVIDIA driver is up to date. The system will use OpenVINO automatically if no GPU is found.
