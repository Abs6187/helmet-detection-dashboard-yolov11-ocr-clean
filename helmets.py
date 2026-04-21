from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
from ultralytics import YOLO

WITHOUT_HELMET_LABEL = "Without Helmet"


def create_session_folder(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = base_dir / f"NO_HELMET_session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def save_snapshot(frame, folder: Path) -> None:
    timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
    image_path = folder / f"snapshot_{timestamp}.jpg"
    success = cv2.imwrite(str(image_path), frame)
    if success:
        print(f"Saved snapshot: {image_path}")
    else:
        print("Failed to save snapshot.")


def run_helmet_detection(
    model_path: str,
    source: int,
    confidence: float,
    image_size: int,
    snapshot_interval: float,
    session_seconds: int,
) -> None:
    output_dir = Path("static") / "no_helmet"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Could not open camera source.")
        return

    print("Press 'q' to quit.")
    session_folder: Path | None = None
    session_end_time: datetime | None = None
    last_snapshot_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from source.")
            break

        now = datetime.now()
        timestamp_now = time.time()
        raw_frame = frame.copy()
        result = model.predict(source=frame, conf=confidence, imgsz=image_size, verbose=False)[0]

        detected_labels = set()
        if result.boxes is not None and result.boxes.cls is not None:
            for class_id in result.boxes.cls.int().tolist():
                detected_labels.add(result.names[int(class_id)])

        has_violation = WITHOUT_HELMET_LABEL in detected_labels
        if has_violation:
            if session_folder is None or (session_end_time is not None and now > session_end_time):
                session_folder = create_session_folder(output_dir)
                session_end_time = now + timedelta(seconds=session_seconds)
                print(f"Started no-helmet session: {session_folder.name}")

            if timestamp_now - last_snapshot_time >= snapshot_interval and session_folder is not None:
                save_snapshot(raw_frame, session_folder)
                last_snapshot_time = timestamp_now
        elif session_end_time is not None and now > session_end_time:
            session_folder = None
            session_end_time = None

        annotated_frame = result.plot()
        cv2.imshow("Helmet Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Helmet detection using Ultralytics YOLO.")
    parser.add_argument("--model", default="best.pt", help="Model checkpoint path.")
    parser.add_argument("--source", type=int, default=1, help="Camera source id (0 for OBS, 1 for webcam).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--snapshot-interval", type=float, default=0.25, help="Seconds between saved snapshots.")
    parser.add_argument("--session-seconds", type=int, default=10, help="Duration of one offence capture session.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_helmet_detection(
        model_path=args.model,
        source=args.source,
        confidence=args.conf,
        image_size=args.imgsz,
        snapshot_interval=args.snapshot_interval,
        session_seconds=args.session_seconds,
    )
