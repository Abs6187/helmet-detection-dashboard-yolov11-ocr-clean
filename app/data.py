"""app/data.py — Data-access helpers for the offenders JSON store."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from app.config import (
    CATEGORY_META,
    DATA_FILE,
    EDITABLE_FIELDS,
    IMAGE_EXTENSIONS,
    STATIC_DIR,
)


# ── JSON persistence ─────────────────────────────────────────────────────────

def load_data() -> dict[str, dict[str, dict[str, Any]]]:
    """Return the full offenders dict; {} on missing/corrupt file."""
    if not DATA_FILE.exists():
        return {}
    try:
        with DATA_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def save_data(data: dict[str, dict[str, dict[str, Any]]]) -> None:
    with DATA_FILE.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4, ensure_ascii=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_session_time(folder_name: str) -> str:
    prefixes = (
        "TRIPLES_session_",
        "NO_HELMET_session_",
        "THRIPLES_session_",
        "session_",
    )
    for prefix in prefixes:
        if folder_name.startswith(prefix):
            raw = folder_name[len(prefix) : len(prefix) + 15]
            try:
                return datetime.strptime(raw, "%Y%m%d_%H%M%S").strftime(
                    "%d %b %Y, %H:%M:%S"
                )
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


# ── Business logic ────────────────────────────────────────────────────────────

def get_offenders() -> dict[str, list[dict[str, Any]]]:
    """Sync filesystem folders → JSON store and return enriched offender list."""
    data = load_data()
    updated = False
    offenders: dict[str, list[dict[str, Any]]] = {}

    for category in CATEGORY_META:
        category_dir = STATIC_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)
        offenders[category] = []
        data.setdefault(category, {})

        folders = sorted(
            (p for p in category_dir.iterdir() if p.is_dir()),
            key=lambda p: p.name,
            reverse=True,
        )

        for folder in folders:
            if folder.name not in data[category]:
                data[category][folder.name] = default_record(category)
                updated = True

            record = data[category][folder.name]
            images = sorted(
                f.name
                for f in folder.iterdir()
                if f.suffix.lower() in IMAGE_EXTENSIONS
            )
            offenders[category].append(
                {
                    "folder":       folder.name,
                    "images":       images,
                    "session_time": parse_session_time(folder.name),
                    "fine_applied": parse_bool(record.get("fine_applied")),
                    "name":         record.get("name", ""),
                    "offence":      record.get("offence", CATEGORY_META[category]["default_offence"]),
                    "number_plate": record.get("number_plate", ""),
                    "fine":         int(record.get("fine", 0) or 0),
                    "location":     record.get("location", ""),
                }
            )

    if updated:
        save_data(data)
    return offenders


def build_summary(offenders: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_cases": 0,
        "fine_applied": 0,
        "fine_pending": 0,
        "total_fine_amount": 0,
        "categories": {},
    }
    for category, items in offenders.items():
        total   = len(items)
        applied = sum(1 for o in items if o["fine_applied"])
        amount  = sum(int(o.get("fine", 0) or 0) for o in items)
        summary["categories"][category] = {
            "total_cases": total,
            "fine_applied": applied,
            "fine_pending": total - applied,
            "total_fine_amount": amount,
        }
        summary["total_cases"]       += total
        summary["fine_applied"]      += applied
        summary["fine_pending"]      += total - applied
        summary["total_fine_amount"] += amount
    return summary


def update_record(category: str, folder: str, field: str, value: Any) -> dict[str, Any]:
    """Validate and persist a single field update. Returns the updated record."""
    if category not in CATEGORY_META:
        raise ValueError(f"Invalid category '{category}'")
    if field not in EDITABLE_FIELDS:
        raise ValueError(f"Field '{field}' is not editable")

    data = load_data()
    data.setdefault(category, {})
    if folder not in data[category]:
        data[category][folder] = default_record(category)

    record = data[category][folder]

    if field == "fine":
        int_value = int(value)
        if int_value < 0:
            raise ValueError("Fine must be ≥ 0")
        record["fine"] = int_value
    elif field == "fine_applied":
        record["fine_applied"] = parse_bool(value)
    else:
        record[field] = str(value).strip()

    save_data(data)
    return record
