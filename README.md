# 🚨 Intelligence System Hub — ANPR & Helmet Detection

> **Real-Time Automated Helmet Detection and License Plate Recognition System Using YOLOv11 and OCR**

[![GitHub Pages](https://img.shields.io/badge/Live%20Dashboard-GitHub%20Pages-blue?logo=github)](https://abs6187.github.io/helmet-detection-dashboard-yolov11-ocr-clean/)
[![Render Deploy](https://img.shields.io/badge/Flask%20App-Render-green?logo=render)](https://helmet-detection-dashboard-yolov11-ocr.onrender.com/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-orange)](LICENSE)
[![Author](https://img.shields.io/badge/Author-Abs6187-lightgrey?logo=github)](https://github.com/Abs6187)

---

## 🌐 Live Links

| Resource | Link |
|---|---|
| 🖥️ **GitHub Pages Dashboard** | [abs6187.github.io/helmet-detection-dashboard-yolov11-ocr-clean](https://abs6187.github.io/helmet-detection-dashboard-yolov11-ocr-clean/) |
| ☁️ **Flask Review Dashboard (Render)** | [helmet-detection-dashboard-yolov11-ocr.onrender.com](https://helmet-detection-dashboard-yolov11-ocr.onrender.com/) |
| 🤗 **Helmet & License Plate (HF Space)** | [Abs6187/Helmet-License-Plate-Detection](https://huggingface.co/spaces/Abs6187/Helmet-License-Plate-Detection) |
| 🤗 **Road Sentinel (HF Space)** | [Abs6187/roadsentinel-helmet-monitor](https://huggingface.co/spaces/Abs6187/roadsentinel-helmet-monitor) |
| 🤗 **Vehicle Speed Estimation (HF Space)** | [Abs6187/Vehicle_Speed_Estimation_and_Counting](https://huggingface.co/spaces/Abs6187/Vehicle_Speed_Estimation_and_Counting) |
| 📂 **GitHub Repository** | [Abs6187/helmet-detection-dashboard-yolov11-ocr-clean](https://github.com/Abs6187/helmet-detection-dashboard-yolov11-ocr-clean) |

---

## 📋 Project Overview

This project is a unified traffic surveillance suite that detects:

1. **No-Helmet Riders** — using a custom YOLOv11 model (`best.pt`)
2. **Triple-Riding Violations** — using Ultralytics YOLO (`yolov8n.pt`)
3. **License Plate Recognition** — via OCR pipeline
4. **Vehicle Speed Estimation & Counting** — real-time vector tracking

Captured offence images are saved into timestamped folders under `static/`, and a Flask dashboard is provided for case review and fine tracking.

---

## 🗂️ Repository Structure

```
helmet-detection-dashboard-yolov11-ocr-clean/
├── app/                          # Flask app modules
├── dataset_samples/              # Class-wise sample images
│   ├── double_riding/
│   ├── single_rider/
│   └── triple_riding/
├── static/                       # Saved offence images
├── templates/                    # Flask HTML templates
├── tests/                        # Automated tests
├── index.html                    # GitHub Pages dashboard
├── run_all.bat                   # Launch all local services
├── helmets.py                    # No-helmet detection script
├── triples.py                    # Triple-riding detection script
├── offender.py                   # Flask dashboard app
├── hf_space_client.py            # HuggingFace Space client
├── best.pt                       # Custom YOLOv11 model
├── yolov8n.pt                    # Ultralytics base model
├── requirements.txt              # Python dependencies
├── render.yaml                   # Render deployment config
├── wsgi.py                       # Production WSGI entry
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Detection Models | YOLOv11 (custom `best.pt`), YOLOv8n |
| Web Framework | Flask 3.1+ |
| Deep Learning | PyTorch 2.6+, Ultralytics 8.4+ |
| Computer Vision | OpenCV 4.10+, NumPy 2.0+ |
| OCR | EasyOCR / PaddleOCR |
| UI | Gradio (HF Spaces), HTML/CSS (dashboard) |
| Deployment | Render (Flask), GitHub Pages (dashboard), HuggingFace Spaces |

---

## 🚀 Local Setup

### Prerequisites
- Python 3.10+
- Windows (for `run_all.bat`)

### Install & Run

```bash
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Run Flask Dashboard
```bash
.\.venv\Scripts\python.exe -m flask --app offender run --host 127.0.0.1 --port 5000 --no-reload
```
Open: http://127.0.0.1:5000

### Run No-Helmet Detection
```bash
.\.venv\Scripts\python.exe helmets.py --source 1
```
Options: `--source 0` (OBS virtual cam), `--model best.pt`, `--conf 0.25`, `--session-seconds 10`

### Run Triple-Riding Detection
```bash
.\.venv\Scripts\python.exe triples.py --source 1
```

### Launch All (Windows)
```bash
run_all.bat
```

---

## 🌍 Local Service Ports

| Service | Port |
|---|---|
| Helmet & License Plate Detection | 7861 |
| Road Sentinel | 7862 |
| Vehicle Speed Estimation & Counting | 7863 |

---

## ☁️ Render Deployment

This repo includes `render.yaml` for one-click Render deployment.

**Production Start Command:**
```
gunicorn wsgi:app --workers 2 --threads 4 --timeout 120
```

**Health Check Endpoint:** `/healthz`

---

## 🧪 Testing

```bash
# Smoke test
.\.venv\Scripts\python.exe test.py

# Full automated tests
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m pytest -q
```

---

## 📄 Research Publication

- [IJRAR25B3370.pdf](https://helmet-detection-dashboard-yolov11-ocr.onrender.com/resources/IJRAR25B3370.pdf) — Published research paper on this system

---

## 👥 Contributors

| Name | Role |
|---|---|
| **Abhay Gupta** ([@Abs6187](https://github.com/Abs6187)) | Lead Engineer |
| Aditi Lakhera | Research & Dev |
| Balraj Patel | Research & Dev |
| Bhumika Patel | Research & Dev |

*Shri Ram Institute of Technology, Jabalpur — Computer Engineering*

---

## 📅 Project Timeline

| Semester | Milestone |
|---|---|
| 5th Sem | Minor Project: YOLOv11 Helmet Detection foundation |
| 6th Sem | First complete dashboard: Helmet + ANPR + OCR |
| 7th Sem | Vehicle Speed Estimation & Counting module |
| 8th Sem | Road Sentinel / Triple-Riding Detection — full suite |

---

## 📚 References

- [Ultralytics Docs](https://docs.ultralytics.com)
- [Flask Docs](https://flask.palletsprojects.com)
- [Render Flask Deployment](https://render.com/docs/deploy-flask)
- [Bootstrap 5.3](https://getbootstrap.com/docs/5.3/)

---

## License

This project is licensed under [AGPL-3.0](LICENSE).

---

*Created by **[Abs6187](https://github.com/Abs6187)** — Abhay Gupta*
