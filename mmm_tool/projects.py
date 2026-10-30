"""Projects: one business's files, how they're read, every run and its model registry.

A project is a folder you can copy or back up whole:

    runs/projects/<slug>/project.json     name, description, settings ({packs, calendar}), created
    runs/projects/<slug>/registry.json    the project's product lines (step 57)
    runs/projects/<slug>/runs/<run_id>/   each run, as before

New runs always belong to a project. Runs made before projects (runs/<run_id>/, with the
old runs/registry.json) stay where they are until moved into a project; a registered run moves
with every version of its product line, so a line's versions stay together.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,59}$")
RUN_ID = re.compile(r"^[0-9a-f]{12}$")


class ProjectError(ValueError):
    pass


def root(runs_dir: Path) -> Path:
    return Path(runs_dir) / "projects"


def folder(runs_dir: Path, slug: str) -> Path:
    if not SLUG.match(slug or ""):
        raise ProjectError("No such project.")
    return root(runs_dir) / slug


def runs_folder(runs_dir: Path, slug: str) -> Path:
    return folder(runs_dir, slug) / "runs"


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)


def slug_for(runs_dir: Path, name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:50] or "project"
    slug, n = base, 2
    while (root(runs_dir) / slug).exists():
        slug, n = f"{base}-{n}", n + 1
    return slug


def _check_name(runs_dir: Path, name: str, own: str | None = None) -> str:
    name = (name or "").strip()
    if not 1 <= len(name) <= 80 or any(ord(c) < 32 for c in name):
        raise ProjectError("Give the project a name (up to 80 characters).")
    clash = next((p for p in list_all(runs_dir) if p["name"].lower() == name.lower() and p["slug"] != own), None)
    if clash:
        raise ProjectError(f"There's already a project called '{clash['name']}'.")
    return name


def create(runs_dir: Path, name: str, description: str = "", settings: dict | None = None) -> dict:
    from .settings import clean

    name = _check_name(runs_dir, name)
    p = {"slug": slug_for(runs_dir, name), "name": name, "description": (description or "").strip()[:300],
         "settings": clean(settings or {}), "created": time.time()}
    (folder(runs_dir, p["slug"]) / "runs").mkdir(parents=True)
    _write(folder(runs_dir, p["slug"]) / "project.json", p)
    return p


def load(runs_dir: Path, slug: str) -> dict:
    f = folder(runs_dir, slug) / "project.json"
    if not f.is_file():
        raise ProjectError("No such project.")
    return json.loads(f.read_text(encoding="utf-8"))


def update(runs_dir: Path, slug: str, name: str | None = None, description: str | None = None,
           settings: dict | None = None) -> dict:
    """Change a project's name, description or settings; runs already made keep theirs."""
    from .settings import clean

    p = load(runs_dir, slug)
    if name is not None:
        p["name"] = _check_name(runs_dir, name, own=slug)
    if description is not None:
        p["description"] = description.strip()[:300]
    if settings is not None:
        p["settings"] = clean(settings)
    _write(folder(runs_dir, slug) / "project.json", p)
    return p


def list_all(runs_dir: Path) -> list[dict]:
    r = root(runs_dir)
    if not r.is_dir():
        return []
    out = []
    for d in sorted(r.iterdir()):
        f = d / "project.json"
        if SLUG.match(d.name) and f.is_file():
            out.append(json.loads(f.read_text(encoding="utf-8")))
    return out


def run_dirs(runs_dir: Path, slug: str | None) -> list[Path]:
    """A project's run folders, or (slug None) the runs not in a project."""
    base = runs_folder(runs_dir, slug) if slug else Path(runs_dir)
    if not base.is_dir():
        return []
    return [d for d in base.iterdir() if RUN_ID.match(d.name) and (d / "meta.json").is_file()]


def find_run(runs_dir: Path, run_id: str) -> Path | None:
    if not RUN_ID.match(run_id or ""):
        return None
    d = Path(runs_dir) / run_id
    if (d / "meta.json").is_file():
        return d
    return next((p for p in root(runs_dir).glob(f"*/runs/{run_id}") if (p / "meta.json").is_file()), None)


def project_of(run_dir: Path) -> str | None:
    """The slug of the project a run folder is in, or None for a run made before projects."""
    d = Path(run_dir)
    if d.parent.name == "runs" and d.parent.parent.parent.name == "projects" and SLUG.match(d.parent.parent.name):
        return d.parent.parent.name
    return None


def registry_dir(run_dir: Path, runs_dir: Path) -> Path:
    """Where the run's registry.json is: its project's folder, or the runs folder for an older run."""
    slug = project_of(run_dir)
    return folder(runs_dir, slug) if slug else Path(runs_dir)


def trash(runs_dir: Path) -> Path:
    return Path(runs_dir) / "trash"


def delete(runs_dir: Path, slug: str, confirm: str, busy=lambda d: False) -> dict:
    """Delete a project: its folder (every run, model and its registry) moves to runs/trash/, so a
    mistake can be undone by moving it back. The project's name must be typed to confirm, and a
    project with a fit or calculation in progress isn't deleted."""
    p = load(runs_dir, slug)
    if (confirm or "").strip() != p["name"]:
        raise ProjectError(f"Type the project's name, {p['name']}, to delete it.")
    runs = run_dirs(runs_dir, slug)
    stuck = [d.name for d in runs if busy(d)]
    if stuck:
        raise ProjectError(f"Run(s) {', '.join(stuck)} have a fit or a calculation in progress; delete the project once it finishes.")
    dest = trash(runs_dir) / f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(folder(runs_dir, slug)), str(dest))
    return {"slug": slug, "name": p["name"], "runs": len(runs), "moved_to": str(dest)}


def move(runs_dir: Path, run_id: str, slug: str, busy=lambda d: False) -> list[str]:
    """Move a run into a project. A registered run brings every version of its product line
    (and their runs), so the line stays whole; a line the project already has is refused."""
    from .export import registry as reg

    load(runs_dir, slug)
    src = find_run(runs_dir, run_id)
    if src is None:
        raise ProjectError("No such run.")
    if project_of(src) == slug:
        return []
    src_reg_dir = registry_dir(src, runs_dir)
    src_reg = reg.load(src_reg_dir)
    hit = reg.find_run(src_reg, run_id)
    line = hit[0] if hit else None
    moving = [src]
    if line:
        if line in reg.load(folder(runs_dir, slug))["lines"]:
            raise ProjectError(f"The project already has a product line called '{line}'; rename one first.")
        for v in src_reg["lines"][line]["versions"]:
            d = find_run(runs_dir, v["run_id"])
            if d is not None and d != src:
                moving.append(d)
    stuck = [d.name for d in moving if busy(d)]
    if stuck:
        raise ProjectError(f"Run(s) {', '.join(stuck)} have a fit or a calculation in progress; move them once it finishes.")
    dest = runs_folder(runs_dir, slug)
    dest.mkdir(parents=True, exist_ok=True)
    for d in moving:
        shutil.move(str(d), str(dest / d.name))
    if line:
        target = reg.load(folder(runs_dir, slug))
        target["lines"][line] = src_reg["lines"].pop(line)
        reg.save(folder(runs_dir, slug), target)
        reg.save(src_reg_dir, src_reg)
    return [d.name for d in moving]
