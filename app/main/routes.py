from flask import render_template, send_from_directory, current_app
from app.main import bp
import os

@bp.route("/")
def index():
    return render_template("index.html")

@bp.route("/healthz")
def healthz():
    return "OK", 200

@bp.route("/resources/<path:filename>")
def serve_resource(filename):
    resources_dir = os.path.join(current_app.root_path, '..', 'Resources')
    return send_from_directory(resources_dir, filename)
