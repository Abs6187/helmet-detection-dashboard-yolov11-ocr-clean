from __future__ import annotations

from ultralytics import YOLO

from offender import build_summary, get_offenders


def run_smoke_test() -> None:
    for model_path in ("best.pt", "yolov8n.pt"):
        model = YOLO(model_path)
        print(f"Loaded model: {model_path} ({type(model.model).__name__})")

    offenders = get_offenders()
    summary = build_summary(offenders)
    print(f"Detected offender folders: {summary['total_cases']}")
    print(f"Fine applied: {summary['fine_applied']}")
    print(f"Fine pending: {summary['fine_pending']}")


if __name__ == "__main__":
    run_smoke_test()
