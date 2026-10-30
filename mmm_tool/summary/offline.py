"""Make Meridian's HTML reports work offline.

Meridian's reports (model results summary, optimization summary) load vega, vega-lite and
vega-embed from www.gstatic.com; where that host can't be reached the page is blank. This
puts the tool's own copies (mmm_tool/web/vendor/, BSD-3-Clause) inside the page instead, so a
report opens anywhere, including as a downloaded file or inside a bundle.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "web" / "vendor"
# Meridian's script tags, by the library each one loads.
_TAG = re.compile(r'<script[^>]*\bsrc="https?://www\.gstatic\.com/external_hosted/(vega|vega_lite|vega_embed)/[^"]*"[^>]*>\s*</script>')
_FILE = {"vega": "vega.min.js", "vega_lite": "vega-lite.min.js", "vega_embed": "vega-embed.min.js"}


@lru_cache(maxsize=None)
def _library(name: str) -> str:
    # "</script" inside the code would end the inline tag early.
    return (VENDOR / _FILE[name]).read_text(encoding="utf-8").replace("</script", "<\\/script")


# Meridian's "insights" icon, also on gstatic: a small line-chart icon of our own instead.
_ICON = re.compile(r"https?://www\.gstatic\.com/images/icons/material/system/svg/insights_24px\.svg")
_ICON_SVG = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
             "<path d='M3 17l5-5 4 4 8-8' fill='none' stroke='%235f6368' stroke-width='2'/></svg>")


def inline_charts(html: str) -> str:
    """The page with Meridian's remote chart scripts replaced by inline copies (unchanged if it has none)."""
    html = _TAG.sub(lambda m: f"<script>/* {m.group(1)} (inlined for offline use) */\n{_library(m.group(1))}\n</script>", html)
    return _ICON.sub(_ICON_SVG, html)


def make_offline(path: Path) -> Path:
    """Rewrite a saved report in place so it works offline; returns the path."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    new = inline_charts(text)
    if new != text:
        tmp = p.with_suffix(".tmp")
        tmp.write_text(new, encoding="utf-8")
        tmp.replace(p)
    return p
