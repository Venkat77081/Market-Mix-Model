"""Shared test helpers: every new run belongs to a project."""
from __future__ import annotations

from pathlib import Path

from mmm_tool import projects


def _runs_dir() -> Path:
    from mmm_tool.api import app as app_mod

    return app_mod.RUNS_DIR                    # monkeypatched per test


def project_in(name: str = "Tests", settings: dict | None = None) -> str:
    """The slug of a project for the test's runs folder, made on first use. It reads files with
    the test environment's settings (conftest: the india and insurance packs) unless given others."""
    from mmm_tool.settings import machine_wide

    runs = _runs_dir()
    hit = next((p for p in projects.list_all(runs) if p["name"] == name), None)
    return hit["slug"] if hit else projects.create(runs, name, settings=settings if settings is not None else machine_wide())["slug"]


def run_path(run_id: str) -> Path:
    d = projects.find_run(_runs_dir(), run_id)
    assert d is not None, f"no run {run_id}"
    return d
