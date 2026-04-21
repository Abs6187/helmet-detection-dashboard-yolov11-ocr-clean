from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "offenders_data.json"
STATIC_DIR = BASE_DIR / "static"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

CATEGORY_META = {
    "triples": {"label": "Triple Riding", "default_offence": "Triple Riding"},
    "no_helmet": {"label": "No Helmet", "default_offence": "No Helmet"},
}
EDITABLE_FIELDS = {"name", "number_plate", "fine", "location", "fine_applied"}


def load_data() -> dict[str, dict[str, dict[str, Any]]]:
    if not DATA_FILE.exists():
        return {}

    try:
        with DATA_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def save_data(data: dict[str, dict[str, dict[str, Any]]]) -> None:
    with DATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def parse_session_time(folder_name: str) -> str:
    prefixes = ("TRIPLES_session_", "NO_HELMET_session_", "THRIPLES_session_", "session_")

    for prefix in prefixes:
        if folder_name.startswith(prefix):
            raw_timestamp = folder_name[len(prefix) : len(prefix) + 15]
            try:
                parsed = datetime.strptime(raw_timestamp, "%Y%m%d_%H%M%S")
                return parsed.strftime("%d %b %Y, %H:%M:%S")
            except ValueError:
                return "Unknown"

    return "Unknown"


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def default_record(category: str) -> dict[str, Any]:
    return {
        "name": "",
        "offence": CATEGORY_META[category]["default_offence"],
        "number_plate": "",
        "fine": 0,
        "location": "",
        "fine_applied": False,
    }


def get_offenders() -> dict[str, list[dict[str, Any]]]:
    data = load_data()
    updated = False
    offenders: dict[str, list[dict[str, Any]]] = {}

    for category in CATEGORY_META:
        category_dir = STATIC_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)
        offenders[category] = []
        data.setdefault(category, {})

        folders = sorted((path for path in category_dir.iterdir() if path.is_dir()), key=lambda path: path.name, reverse=True)

        for folder in folders:
            if folder.name not in data[category]:
                data[category][folder.name] = default_record(category)
                updated = True

            record = data[category][folder.name]
            images = sorted([file.name for file in folder.iterdir() if file.suffix.lower() in IMAGE_EXTENSIONS])

            offenders[category].append(
                {
                    "folder": folder.name,
                    "images": images,
                    "session_time": parse_session_time(folder.name),
                    "fine_applied": parse_bool(record.get("fine_applied")),
                    "name": record.get("name", ""),
                    "offence": record.get("offence", CATEGORY_META[category]["default_offence"]),
                    "number_plate": record.get("number_plate", ""),
                    "fine": int(record.get("fine", 0) or 0),
                    "location": record.get("location", ""),
                }
            )

    if updated:
        save_data(data)

    return offenders


def build_summary(offenders: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary = {
        "total_cases": 0,
        "fine_applied": 0,
        "fine_pending": 0,
        "total_fine_amount": 0,
        "categories": {},
    }

    for category, items in offenders.items():
        total_cases = len(items)
        fine_applied = sum(1 for offender in items if offender["fine_applied"])
        fine_pending = total_cases - fine_applied
        total_fine_amount = sum(int(offender.get("fine", 0) or 0) for offender in items)

        summary["categories"][category] = {
            "total_cases": total_cases,
            "fine_applied": fine_applied,
            "fine_pending": fine_pending,
            "total_fine_amount": total_fine_amount,
        }

        summary["total_cases"] += total_cases
        summary["fine_applied"] += fine_applied
        summary["fine_pending"] += fine_pending
        summary["total_fine_amount"] += total_fine_amount

    return summary


@app.route("/")
def index():
    offenders = get_offenders()
    summary = build_summary(offenders)
    return render_template("index.html", offenders=offenders, summary=summary, category_meta=CATEGORY_META)


@app.route("/update_offender", methods=["POST"])
def update_offender():
    payload = request.get_json(silent=True) or {}
    category = payload.get("category")
    folder = payload.get("folder")
    field = payload.get("field")
    value = payload.get("value")

    if not all([category, folder, field]):
        return jsonify({"status": "error", "message": "Missing data"}), 400

    if category not in CATEGORY_META:
        return jsonify({"status": "error", "message": "Invalid category"}), 400

    if field not in EDITABLE_FIELDS:
        return jsonify({"status": "error", "message": f"Field '{field}' is not editable"}), 400

    data = load_data()
    data.setdefault(category, {})
    data[category].setdefault(folder, default_record(category))

    if field == "fine":
        try:
            value = int(value)
        except (TypeError, ValueError):
            return jsonify({"status": "error", "message": "Fine must be a whole number"}), 400
        if value < 0:
            return jsonify({"status": "error", "message": "Fine cannot be negative"}), 400
    elif field == "fine_applied":
        value = parse_bool(value)
    else:
        value = str(value or "").strip()

    data[category][folder][field] = value
    save_data(data)

    updated_record = {
        **default_record(category),
        **data[category][folder],
        "fine_applied": parse_bool(data[category][folder].get("fine_applied")),
    }
    return jsonify({"status": "success", "record": updated_record})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
