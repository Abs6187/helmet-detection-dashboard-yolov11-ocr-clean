from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def get_boxes_by_class(result, target_class: str) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    if result.boxes is None or result.boxes.cls is None:
        return boxes

    xyxy = result.boxes.xyxy.int().tolist()
    classes = result.boxes.cls.int().tolist()
    for box, class_id in zip(xyxy, classes):
        if result.names[int(class_id)] == target_class:
            x1, y1, x2, y2 = box
            boxes.append((x1, y1, x2, y2))
    return boxes


def box_distance(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return float(np.hypot(dx, dy))


def are_three_boxes_close(boxes: list[tuple[int, int, int, int]], threshold: float) -> bool:
    if len(boxes) < 3:
        return False
    for triplet in combinations(boxes, 3):
        distances = (
            box_distance(triplet[0], triplet[1]),
            box_distance(triplet[0], triplet[2]),
            box_distance(triplet[1], triplet[2]),
        )
        if all(distance <= threshold for distance in distances):
            return True
    return False


def is_motorcycle_near_any_person(
    motorcycle_boxes: list[tuple[int, int, int, int]],
    person_boxes: list[tuple[int, int, int, int]],
    threshold: float,
) -> bool:
    return any(box_distance(motorcycle_box, person_box) <= threshold for motorcycle_box in motorcycle_boxes for person_box in person_boxes)


def create_session_folder(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = base_dir / f"TRIPLES_session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def save_snapshot(raw_frame, folder: Path) -> None:
    timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
    image_path = folder / f"snapshot_{timestamp}.jpg"
    success = cv2.imwrite(str(image_path), raw_frame)
    if success:
        print(f"Saved snapshot: {image_path}")
    else:
        print("Failed to save snapshot.")


def detect_from_webcam(
    model_path: str,
    source: int,
    confidence: float,
    image_size: int,
    box_distance_threshold: float,
    session_seconds: int,
) -> None:
    output_dir = Path("static") / "triples"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Could not open camera source.")
        return

    session_folder: Path | None = None
    session_end_time: datetime | None = None
    print("Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        raw_frame = frame.copy()
        result = model.predict(source=frame, conf=confidence, imgsz=image_size, verbose=False)[0]

        person_boxes = get_boxes_by_class(result, "person")
        motorcycle_boxes = get_boxes_by_class(result, "motorcycle")
        person_count = len(person_boxes)

        condition = (
            person_count >= 3
            and are_three_boxes_close(person_boxes, box_distance_threshold)
            and len(motorcycle_boxes) >= 1
            and is_motorcycle_near_any_person(motorcycle_boxes, person_boxes, box_distance_threshold)
        )

        now = datetime.now()
        if condition:
            if session_folder is None or (session_end_time is not None and now > session_end_time):
                session_folder = create_session_folder(output_dir)
                session_end_time = now + timedelta(seconds=session_seconds)
                print(f"Started triple-riding session: {session_folder.name}")
            if session_folder is not None:
                save_snapshot(raw_frame, session_folder)
        elif session_end_time is not None and now > session_end_time:
            session_folder = None
            session_end_time = None

        annotated = result.plot()
        cv2.imshow("Triple Riding Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triple-riding detection with Ultralytics YOLO.")
    parser.add_argument("--model", default="yolov8n.pt", help="Model checkpoint path.")
    parser.add_argument("--source", type=int, default=1, help="Camera source id (0 for OBS, 1 for webcam).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--distance-threshold", type=float, default=20.0, help="Maximum distance between detected boxes.")
    parser.add_argument("--session-seconds", type=int, default=10, help="Duration of one offence capture session.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detect_from_webcam(
        model_path=args.model,
        source=args.source,
        confidence=args.conf,
        image_size=args.imgsz,
        box_distance_threshold=args.distance_threshold,
        session_seconds=args.session_seconds,
    )
