"""Step 3's role guide (the i next to "Role") describes every role, each once, with a label."""
from __future__ import annotations

import re
from pathlib import Path

from mmm_tool.ingest.types import ROLES

PAGE = (Path(__file__).parents[1] / "mmm_tool" / "web" / "index.html").read_text(encoding="utf-8")


def _block(name: str) -> str:
    start = PAGE.index(f"const {name} = ")
    return PAGE[start:PAGE.index("};" if name == "ROLE_LABEL" else "];\n", start)]


def test_the_guide_describes_every_role_once():
    described = re.findall(r'\["([a-z_]+)", "[a-z_]*", "', _block("ROLE_HELP"))
    print(f"\n{len(described)} roles described: {described}")
    assert sorted(described) == sorted(ROLES) and len(described) == len(set(described))


def test_every_role_has_a_label():
    labels = set(re.findall(r"\b([a-z_]+): \"", _block("ROLE_LABEL")))
    assert set(ROLES) <= labels


def test_the_icon_sits_in_the_role_column_heading():
    assert '<th><span class="th-i">Role <button type="button" class="ibtn" id="role-i"' in PAGE
    assert 'aria-describedby="rolehelp"' in PAGE and 'id="rolehelp" role="tooltip"' in PAGE
