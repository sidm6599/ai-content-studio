"""Pure-function exporters: convert a list of flat item dicts to JSON, CSV, or Markdown.

Each item dict has the shape produced by ``app.py``::

    item.fields | {"brand_match": item.brand_match}

Values may be strings, numbers, None, or lists of strings (e.g. hashtags, bullets).
"""
from __future__ import annotations

import csv
import io
import json
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _str_value(value: Any) -> str:
    """Coerce any field value to a plain string suitable for CSV cells."""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    if value is None:
        return ""
    return str(value)


def _all_keys(items: list[dict]) -> list[str]:
    """Return the union of all keys across all items, preserving first-seen order."""
    seen: dict[str, None] = {}
    for item in items:
        for key in item:
            seen[key] = None
    return list(seen)


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------

def to_json(items: list[dict], indent: int = 2) -> str:
    """Serialise *items* to a JSON string.

    Args:
        items:  List of flat item dicts (string and list values).
        indent: Indentation level for pretty-printing. Defaults to 2.

    Returns:
        A JSON-encoded string representing the list.
    """
    return json.dumps(items, indent=indent, ensure_ascii=False)


def to_csv(items: list[dict]) -> str:
    """Serialise *items* to CSV.

    List-valued fields are flattened by joining elements with ``"; "``.
    The header row is the union of all keys across all items.

    Args:
        items: List of flat item dicts.

    Returns:
        A CSV string with a header row followed by one data row per item.
    """
    buf = io.StringIO()
    fieldnames = _all_keys(items)
    writer = csv.DictWriter(
        buf,
        fieldnames=fieldnames,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for item in items:
        row = {key: _str_value(item.get(key)) for key in fieldnames}
        writer.writerow(row)
    return buf.getvalue()


def to_markdown(items: list[dict]) -> str:
    """Render *items* as a Markdown document.

    Each item becomes a numbered section. String fields are rendered as
    ``**field**: value`` lines; list fields are rendered as a bullet sub-list
    under the bold field name.

    Args:
        items: List of flat item dicts.

    Returns:
        A Markdown string with one block per item.
    """
    if not items:
        return ""

    blocks: list[str] = []
    for idx, item in enumerate(items, start=1):
        lines: list[str] = [f"## Item {idx}", ""]
        for key, value in item.items():
            if isinstance(value, list):
                lines.append(f"**{key}**:")
                for element in value:
                    lines.append(f"- {element}")
            else:
                lines.append(f"**{key}**: {_str_value(value)}")
        lines.append("")          # blank line between blocks
        blocks.append("\n".join(lines))

    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Maps a format name to ``(exporter_function, file_extension)``.
#: Useful for UI wiring — iterate this dict to populate export menus.
EXPORTERS: dict[str, tuple[Callable[..., str], str]] = {
    "json": (to_json, "json"),
    "csv": (to_csv, "csv"),
    "markdown": (to_markdown, "md"),
}
