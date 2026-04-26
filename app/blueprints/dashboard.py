"""app/blueprints/dashboard.py — Dashboard and case-management routes."""
from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from app.config import CATEGORY_META, EDITABLE_FIELDS
from app.data import build_summary, get_offenders, update_record

bp = Blueprint("dashboard", __name__)


@bp.get("/")
def index():
    offenders = get_offenders()
    summary   = build_summary(offenders)
    return render_template(
        "dashboard.html",
        offenders=offenders,
        summary=summary,
        category_meta=CATEGORY_META,
    )


@bp.get("/healthz")
def health_check():
    return jsonify({"status": "ok"}), 200


@bp.post("/update_offender")
def update_offender():
    payload  = request.get_json(silent=True) or {}
    category = payload.get("category")
    folder   = payload.get("folder")
    field    = payload.get("field")
    value    = payload.get("value")

    if not all([category, folder, field]):
        return jsonify({"status": "error", "message": "Missing required fields: category, folder, field"}), 400

    try:
        record = update_record(category, folder, field, value)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    return jsonify({"status": "success", "record": record})
