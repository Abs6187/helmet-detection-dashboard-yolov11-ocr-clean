"""
tests/test_env.py
=================
Environment and dependency checks — runs without any heavy dependencies (torch, cv2, etc.)
These tests are safe to run on CI (requirements-test.txt only).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestEnvironment:
    """Basic env checks that every deploy target should satisfy."""

    def test_python_version_is_310_or_newer(self):
        assert sys.version_info >= (3, 10), f"Python 3.10+ required, got {sys.version_info}"

    def test_repo_root_exists(self):
        assert REPO_ROOT.exists()

    def test_required_source_files_exist(self):
        required = [
            # Legacy detector scripts (kept for direct use)
            "helmets.py",
            "triples.py",
            "hf_space_client.py",
            # App package
            "app/__init__.py",
            "app/config.py",
            "app/data.py",
            "app/blueprints/dashboard.py",
            "app/blueprints/detection.py",
            "app/blueprints/speed.py",
            # Entrypoints
            "wsgi.py",
            # Assets
            "requirements.txt",
            "requirements-test.txt",
            "render.yaml",
            # Templates
            "templates/base.html",
            "templates/dashboard.html",
            # Static
            "static/css/app.css",
            "static/js/dashboard.js",
            "static/js/demo.js",
            "static/vendor/css/bootstrap.min.css",
            "static/vendor/js/bootstrap.bundle.min.js",
        ]
        missing = [f for f in required if not (REPO_ROOT / f).exists()]
        assert not missing, f"Missing files: {missing}"

    def test_no_github_ci_workflows(self):
        """CI/CD workflows were intentionally removed — ensure they stay gone."""
        workflow_dir = REPO_ROOT / ".github" / "workflows"
        if workflow_dir.exists():
            yml_files = list(workflow_dir.glob("*.yml"))
            assert not yml_files, (
                f"Unexpected workflow file(s): {[f.name for f in yml_files]}. "
                "These should have been removed."
            )

    def test_context_md_is_gitignored(self):
        gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "context.md" in gitignore, "context.md must be in .gitignore (contains secrets)"

    def test_huggingfacehub_md_is_gitignored(self):
        gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "huggingfacehub.md" in gitignore, "huggingfacehub.md must be in .gitignore"

    def test_integrate_md_is_gitignored(self):
        gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "integrate.md" in gitignore, "integrate.md must be in .gitignore"

    def test_flask_importable(self):
        assert importlib.import_module("flask") is not None

    def test_pytest_importable(self):
        assert importlib.import_module("pytest") is not None

    def test_app_factory_importable(self):
        """The app package must expose create_app() without crashing."""
        from app import create_app
        flask_app = create_app()
        assert flask_app is not None

    def test_render_yaml_has_health_check(self):
        content = (REPO_ROOT / "render.yaml").read_text(encoding="utf-8")
        assert "healthCheckPath" in content or "healthz" in content, (
            "render.yaml should define a health check pointing to /healthz"
        )

    def test_wsgi_uses_factory(self):
        """wsgi.py must use create_app() and expose an 'app' object."""
        content = (REPO_ROOT / "wsgi.py").read_text(encoding="utf-8")
        assert "create_app" in content
        assert "app" in content

    def test_all_blueprints_registered(self):
        """The factory must register all three blueprints, providing 9 routes."""
        from app import create_app
        flask_app = create_app()
        rule_map  = {str(r) for r in flask_app.url_map.iter_rules()}
        assert "/healthz"          in rule_map
        assert "/detect"           in rule_map
        assert "/demo_samples"     in rule_map
        assert "/speed_estimate"   in rule_map

    def test_requirements_contains_gradio_client(self):
        content = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
        assert "gradio_client" in content, (
            "gradio_client must be in requirements.txt for HF Space API calls"
        )

    def test_hf_token_env_var_documented(self):
        """Check context.md mentions HF_TOKEN so the deploy team knows to set it."""
        ctx = REPO_ROOT / "context.md"
        if ctx.exists():
            assert "HF_TOKEN" in ctx.read_text(encoding="utf-8")
