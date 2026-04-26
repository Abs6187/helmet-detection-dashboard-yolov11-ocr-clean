"""app/config.py — Central constants and environment configuration."""
from __future__ import annotations

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parents[1]   # project root
DATA_FILE      = BASE_DIR / "offenders_data.json"
STATIC_DIR     = BASE_DIR / "static"
DATASET_DIR    = BASE_DIR / "dataset_samples"
LOCAL_MODEL    = BASE_DIR / "best.pt"

# ── File handling ────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

# ── Offender categories ──────────────────────────────────────────────────────
CATEGORY_META: dict[str, dict[str, str]] = {
    "triples":   {"label": "Triple Riding",  "default_offence": "Triple Riding"},
    "no_helmet": {"label": "No Helmet",      "default_offence": "No Helmet"},
}

EDITABLE_FIELDS = {"name", "number_plate", "fine", "location", "fine_applied"}

# ── HF Space settings ────────────────────────────────────────────────────────
HF_TOKEN            = os.environ.get("HF_TOKEN", "")
REMOTE_TIMEOUT      = int(os.environ.get("HF_REMOTE_TIMEOUT", "60"))
SPEED_REMOTE_TIMEOUT = int(os.environ.get("HF_SPEED_TIMEOUT", "300"))
