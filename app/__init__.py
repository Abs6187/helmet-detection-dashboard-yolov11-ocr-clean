"""app/__init__.py — Flask application factory."""
from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask


def create_app() -> Flask:
    """Create and configure the Flask application.

    All blueprints are registered here.
    The static folder is the project-root ``static/`` directory.
    The templates folder is the project-root ``templates/`` directory.
    """
    root = Path(__file__).resolve().parents[1]

    app = Flask(
        __name__,
        static_folder=str(root / "static"),
        template_folder=str(root / "templates"),
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ── Blueprints ────────────────────────────────────────────────────────────
    from app.blueprints.dashboard import bp as dashboard_bp
    from app.blueprints.detection import bp as detection_bp
    from app.blueprints.speed     import bp as speed_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(detection_bp)
    app.register_blueprint(speed_bp)

    return app
