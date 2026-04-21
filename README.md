# Real-Time Automated Helmet Detection and License Plate Recognition System Using YOLOv11 and OCR

## Project Created By

Abhay Gupta  
Student, Computer Engineering  
Shri Ram Institute of Technology, Jabalpur

## Project Overview

This project detects:

1. No-helmet riders using a custom YOLO model (`best.pt`)
2. Triple-riding violations using Ultralytics YOLO (`yolov8n.pt` by default)

Captured offence images are saved into timestamped folders under `static/`, and a Flask dashboard is provided for case review and fine tracking.

## Dataset Samples

Class-wise sample files are stored in `dataset_samples/`:

- `dataset_samples/double_riding`
- `dataset_samples/single_rider`
- `dataset_samples/triple_riding`

Each class contains a few example image/label pairs copied from:
`static/Riding.v1i.yolov8/train`.

## Updated Tech Stack

- Flask `3.1+`
- Ultralytics `8.4+`
- Torch `2.6+`
- OpenCV Python `4.10+`
- NumPy `2.0+`

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run Dashboard

```powershell
.\.venv\Scripts\python.exe -m flask --app offender run --host 127.0.0.1 --port 5000 --no-reload
```

Open: `http://127.0.0.1:5000`

## Run No-Helmet Detection

```powershell
.\.venv\Scripts\python.exe helmets.py --source 1
```

Common options:

- `--source 0` for OBS virtual camera
- `--model best.pt`
- `--conf 0.25`
- `--snapshot-interval 0.25`
- `--session-seconds 10`

## Run Triple-Riding Detection

```powershell
.\.venv\Scripts\python.exe triples.py --source 1
```

Common options:

- `--source 0` for OBS virtual camera
- `--model yolov8n.pt`
- `--conf 0.25`
- `--distance-threshold 20`
- `--session-seconds 10`

## Smoke Test

```powershell
.\.venv\Scripts\python.exe test.py
```

## Automated Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m pytest -q
```

## Render Deployment

This repo includes a Render Blueprint file:

- `render.yaml`

Production app start command:

- `gunicorn wsgi:app --workers 2 --threads 4 --timeout 120`

Health check endpoint:

- `/healthz`

### Deploy from Render Dashboard

1. In Render, create a new Blueprint or Web Service from this GitHub repo.
2. Use:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn wsgi:app`
3. Set health check path to `/healthz` (if not automatically picked up from `render.yaml`).

### Deploy with Render CLI

Install (Linux/macOS):

```bash
curl -fsSL https://raw.githubusercontent.com/render-oss/cli/refs/heads/main/bin/install.sh | sh
```

Login:

```bash
render login
```

Deploy:

```bash
render deploys create <RENDER_SERVICE_ID> --wait --output json --confirm
```

### GitHub Workflow for Render Deploy

Workflow file:

- `.github/workflows/render-deploy.yml`

Set these repository secrets:

- `RENDER_API_KEY`
- `RENDER_SERVICE_ID`

## Docs Used For This Update

- Ultralytics docs: https://docs.ultralytics.com
- Ultralytics Python usage: https://docs.ultralytics.com/usage/python
- Flask docs: https://flask.palletsprojects.com
- Flask development server notes: https://flask.palletsprojects.com/en/stable/server
- Bootstrap docs (dashboard UI update): https://getbootstrap.com/docs/5.3/getting-started/introduction/
- Render Flask deployment docs: https://render.com/docs/deploy-flask
- Render CLI docs: https://render.com/docs/cli
- Render health checks docs: https://render.com/docs/health-checks
