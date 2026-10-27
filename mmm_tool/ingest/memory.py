"""Remembers column roles the user confirmed in step 3.

When a later file has a column with the same name (compared as words, so
"TV Spend" matches "tv_spend"), its earlier role is suggested first, with
high confidence. This is what lets a refit (step 58) reuse saved answers.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .keywords import words_of
from .types import ROLES


def default_path() -> Path:
    return Path(os.environ.get("MMM_ROLE_MEMORY", ".mmm/role_memory.json"))


class RoleMemory:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else default_path()
        self.entries: dict[str, dict] = {}
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.entries = {k: v for k, v in data.get("columns", {}).items() if v.get("role") in ROLES}
            except (json.JSONDecodeError, AttributeError):
                self.entries = {}  # a damaged file is ignored, then rewritten on the next save

    @staticmethod
    def key(column: str) -> str:
        return words_of(column)

    def recall(self, column: str) -> dict | None:
        return self.entries.get(self.key(column))

    def remember(self, roles: dict[str, str], source_file: str) -> int:
        """Store confirmed roles; returns how many entries changed."""
        changed = 0
        now = time.time()
        for col, role in roles.items():
            k = self.key(col)
            if not k:
                continue
            old = self.entries.get(k)
            if old is None or old["role"] != role:
                self.entries[k] = {"role": role, "column": col, "file": source_file, "confirmed_at": now}
                changed += 1
        if changed:
            self.save()
        return changed

    def forget(self, column: str | None = None) -> int:
        if column is None:
            n = len(self.entries)
            self.entries = {}
        else:
            n = int(self.entries.pop(self.key(column), None) is not None)
        self.save()
        return n

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"columns": self.entries}, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def to_list(self) -> list[dict]:
        return sorted(({"key": k, **v} for k, v in self.entries.items()), key=lambda e: e["key"])
