"""How a run's file is read: its keyword packs and calendar preset.

Every run keeps its own settings in meta.json ("settings": {"packs": [...], "calendar": "..."}),
chosen on the upload screen, so two runs can be read differently side by side and a bundle or
refit reads its file the same way again. A new run takes the remembered default
(runs/settings.json, "Use these for my new runs"); with none, today's machine-wide settings
(MMM_KEYWORD_PACKS or `packs:` in roles.yaml, MMM_CALENDARS or ./calendars.yaml), which
for a fresh install is generic: no packs and no calendar.

While a run is read, use(settings) makes its packs and calendar the ones every step sees.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
from pathlib import Path
from typing import Any, Iterator

_ACTIVE: contextvars.ContextVar[dict | None] = contextvars.ContextVar("run_settings", default=None)
DEFAULT_FILE = "settings.json"


class SettingsError(ValueError):
    pass


def active() -> dict | None:
    """The settings of the run being read, or None (the machine-wide settings apply)."""
    return _ACTIVE.get()


@contextlib.contextmanager
def use(settings: dict | None) -> Iterator[None]:
    token = _ACTIVE.set(clean(settings) if settings is not None else None)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def clean(settings: dict) -> dict:
    """{"packs": [names], "calendar": preset name}; unknown packs or presets are refused."""
    from .configure.context import load_calendars
    from .ingest.keywords import available_packs

    if not isinstance(settings, dict):
        raise SettingsError("Settings are {'packs': [...], 'calendar': '...'}.")
    packs = settings.get("packs") or []
    if isinstance(packs, str):
        packs = [p.strip() for p in packs.split(",")]
    packs = [str(p) for p in packs if str(p).strip()]
    known = available_packs()
    bad = [p for p in packs if p not in known and not p.endswith((".yaml", ".yml"))]
    if bad:
        raise SettingsError(f"Unknown keyword pack(s) {bad}; there are {', '.join(known) or 'none'}.")
    cal = settings.get("calendar") or "none"
    with _no_override():
        presets = load_calendars()["presets"]
    if cal != "none" and cal not in presets:
        raise SettingsError(f"Unknown calendar preset '{cal}'; there are none, {', '.join(presets)}.")
    return {"packs": list(dict.fromkeys(packs)), "calendar": cal}


@contextlib.contextmanager
def _no_override() -> Iterator[None]:
    token = _ACTIVE.set(None)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def from_args(packs: str | None, calendar: str | None) -> dict | None:
    """--packs india,insurance / --calendar us_retail; None when neither is given."""
    if packs is None and calendar is None:
        return None
    return clean({"packs": packs or "", "calendar": calendar or "none"})


ARGS_HELP = {"packs": "keyword packs to read the file with, comma-separated (e.g. india,insurance; '' for none). "
                      "Default: MMM_KEYWORD_PACKS or packs: in roles.yaml",
             "calendar": "calendar preset step 21 suggests (e.g. us_retail, or none). Default: calendars.yaml"}


def machine_wide() -> dict:
    """What the environment and the project's files turn on, with no run's settings."""
    from .configure.context import load_calendars
    from .ingest.keywords import _read, enabled_packs, user_file

    with _no_override():
        uf = user_file()
        packs = enabled_packs(_read(uf) if uf.is_file() else {})
        return {"packs": packs, "calendar": load_calendars()["default_preset"]}


def remembered(runs_dir: Path) -> dict | None:
    p = Path(runs_dir) / DEFAULT_FILE
    if not p.is_file():
        return None
    try:
        return clean(json.loads(p.read_text(encoding="utf-8")))
    except (ValueError, SettingsError):
        return None


def remember(runs_dir: Path, settings: dict) -> dict:
    s = clean(settings)
    p = Path(runs_dir) / DEFAULT_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(s), encoding="utf-8")
    tmp.replace(p)
    return s


def for_new_run(runs_dir: Path) -> dict:
    return remembered(runs_dir) or machine_wide()


def for_run(run_dir: Path, meta: dict | None = None) -> tuple[dict, str]:
    """(settings, where they come from: "run", "default" or "machine") for a run's folder.
    A run made before runs kept their settings reads with the remembered default."""
    meta = meta if meta is not None else json.loads((Path(run_dir) / "meta.json").read_text(encoding="utf-8"))
    if isinstance(meta.get("settings"), dict):
        return clean(meta["settings"]), "run"
    r = remembered(Path(run_dir).parent)
    return (r, "default") if r else (machine_wide(), "machine")


def describe(settings: dict) -> str:
    """'India · Insurance · India insurance calendar', or 'Generic'."""
    from .configure.context import load_calendars
    from .ingest.keywords import pack_info

    parts = [pack_info(p)["title"] for p in settings.get("packs") or []]
    cal = settings.get("calendar") or "none"
    if cal != "none":
        with _no_override():
            label = (load_calendars()["presets"].get(cal) or {}).get("label", cal)
        parts.append(f"{label} calendar")
    return " · ".join(parts) or "Generic"


def catalog() -> dict[str, Any]:
    """Every pack and calendar preset, with what each one changes, for the settings panel."""
    from .configure.context import load_calendars
    from .ingest.keywords import available_packs, pack_info

    with _no_override():
        presets = load_calendars()["presets"]
    return {"packs": [pack_info(p) for p in available_packs()],
            "calendars": [{"name": "none", "label": "None", "events": []}]
            + [{"name": k, "label": v.get("label", k), "events": [e.get("label", e.get("name", "")) for e in v.get("events") or []]}
               for k, v in presets.items()]}
