"""Offline tests for src/exporters — no network, no API keys required."""
from __future__ import annotations

import csv
import io
import json

import pytest

from src.exporters import EXPORTERS, to_csv, to_json, to_markdown

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_ITEMS: list[dict] = [
    {
        "caption": "Excited to share our new product!",
        "hashtags": ["#launch", "#innovation", "#tech"],
        "tone": "enthusiastic",
        "brand_match": 0.87,
    },
    {
        "caption": "Delivering value every day.",
        "hashtags": ["#growth"],
        "tone": "professional",
        "brand_match": None,
    },
    {
        # Third item adds an extra key absent from the others.
        "caption": "Check out this feature.",
        "hashtags": [],
        "tone": "casual",
        "brand_match": 0.55,
        "extra_field": "bonus info",
    },
]

SINGLE_ITEM: list[dict] = [
    {
        "title": "Resume bullet",
        "bullets": ["Built a RAG pipeline", "Reduced latency by 40 %"],
        "brand_match": 0.9,
    }
]

EMPTY: list[dict] = []


# ---------------------------------------------------------------------------
# to_json
# ---------------------------------------------------------------------------

class TestToJson:
    def test_round_trips_via_json_loads(self):
        result = to_json(SAMPLE_ITEMS)
        parsed = json.loads(result)
        assert parsed == SAMPLE_ITEMS

    def test_empty_list_produces_valid_json(self):
        result = to_json(EMPTY)
        assert json.loads(result) == []

    def test_indent_parameter_is_respected(self):
        compact = to_json(SAMPLE_ITEMS, indent=0)
        pretty = to_json(SAMPLE_ITEMS, indent=4)
        # Pretty version must be longer (has newlines / spaces).
        assert len(pretty) > len(compact)
        # Both must round-trip identically.
        assert json.loads(compact) == json.loads(pretty)

    def test_none_values_are_preserved(self):
        items = [{"brand_match": None, "caption": "hello"}]
        parsed = json.loads(to_json(items))
        assert parsed[0]["brand_match"] is None

    def test_list_fields_are_preserved(self):
        parsed = json.loads(to_json(SINGLE_ITEM))
        assert parsed[0]["bullets"] == SINGLE_ITEM[0]["bullets"]


# ---------------------------------------------------------------------------
# to_csv
# ---------------------------------------------------------------------------

class TestToCsv:
    def _parse(self, csv_str: str) -> tuple[list[str], list[dict]]:
        """Return (fieldnames, rows) using csv.DictReader."""
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        return reader.fieldnames or [], rows  # type: ignore[return-value]

    def test_has_header_row_and_one_row_per_item(self):
        result = to_csv(SAMPLE_ITEMS)
        lines = [line for line in result.splitlines() if line]
        # header + 3 data rows
        assert len(lines) == 1 + len(SAMPLE_ITEMS)

    def test_parseable_by_dict_reader(self):
        fieldnames, rows = self._parse(to_csv(SAMPLE_ITEMS))
        assert len(rows) == len(SAMPLE_ITEMS)
        assert "caption" in fieldnames
        assert "hashtags" in fieldnames

    def test_list_fields_are_joined_with_semicolon(self):
        _, rows = self._parse(to_csv(SAMPLE_ITEMS))
        # First item has three hashtags.
        assert rows[0]["hashtags"] == "#launch; #innovation; #tech"

    def test_empty_list_field_becomes_empty_string(self):
        _, rows = self._parse(to_csv(SAMPLE_ITEMS))
        # Third item's hashtags list is empty.
        assert rows[2]["hashtags"] == ""

    def test_none_field_becomes_empty_string(self):
        _, rows = self._parse(to_csv(SAMPLE_ITEMS))
        # Second item has brand_match=None.
        assert rows[1]["brand_match"] == ""

    def test_union_of_keys_forms_header(self):
        fieldnames, _ = self._parse(to_csv(SAMPLE_ITEMS))
        # extra_field only exists on the third item but must appear in header.
        assert "extra_field" in fieldnames

    def test_missing_key_in_item_yields_empty_cell(self):
        fieldnames, rows = self._parse(to_csv(SAMPLE_ITEMS))
        # First two items lack extra_field.
        assert rows[0].get("extra_field", "") == ""
        assert rows[1].get("extra_field", "") == ""

    def test_empty_list_produces_only_header_or_empty(self):
        result = to_csv(EMPTY)
        # No data rows — either empty string or just a header with no fields.
        lines = [line for line in result.splitlines() if line]
        assert len(lines) <= 1

    def test_single_item_round_trips(self):
        fieldnames, rows = self._parse(to_csv(SINGLE_ITEM))
        assert rows[0]["title"] == "Resume bullet"
        assert rows[0]["bullets"] == "Built a RAG pipeline; Reduced latency by 40 %"


# ---------------------------------------------------------------------------
# to_markdown
# ---------------------------------------------------------------------------

class TestToMarkdown:
    def test_contains_each_item_heading(self):
        result = to_markdown(SAMPLE_ITEMS)
        for i in range(1, len(SAMPLE_ITEMS) + 1):
            assert f"## Item {i}" in result

    def test_contains_each_caption(self):
        result = to_markdown(SAMPLE_ITEMS)
        for item in SAMPLE_ITEMS:
            assert item["caption"] in result

    def test_list_fields_rendered_as_bullet_lines(self):
        result = to_markdown(SAMPLE_ITEMS)
        # Each hashtag should appear as a bullet.
        assert "- #launch" in result
        assert "- #innovation" in result
        assert "- #tech" in result

    def test_bold_field_names_present(self):
        result = to_markdown(SAMPLE_ITEMS)
        assert "**caption**" in result
        assert "**hashtags**" in result
        assert "**brand_match**" in result

    def test_extra_field_from_third_item_present(self):
        result = to_markdown(SAMPLE_ITEMS)
        assert "extra_field" in result
        assert "bonus info" in result

    def test_single_item_markdown(self):
        result = to_markdown(SINGLE_ITEM)
        assert "## Item 1" in result
        assert "Built a RAG pipeline" in result
        assert "Reduced latency by 40 %" in result

    def test_empty_list_returns_empty_string(self):
        assert to_markdown(EMPTY) == ""


# ---------------------------------------------------------------------------
# EXPORTERS registry
# ---------------------------------------------------------------------------

class TestExportersRegistry:
    def test_all_three_formats_present(self):
        assert set(EXPORTERS) == {"json", "csv", "markdown"}

    def test_each_entry_is_tuple_of_callable_and_string(self):
        for name, (fn, ext) in EXPORTERS.items():
            assert callable(fn), f"EXPORTERS[{name!r}][0] must be callable"
            assert isinstance(ext, str), f"EXPORTERS[{name!r}][1] must be a str"

    def test_extensions_are_correct(self):
        assert EXPORTERS["json"][1] == "json"
        assert EXPORTERS["csv"][1] == "csv"
        assert EXPORTERS["markdown"][1] == "md"

    def test_functions_match_named_exports(self):
        assert EXPORTERS["json"][0] is to_json
        assert EXPORTERS["csv"][0] is to_csv
        assert EXPORTERS["markdown"][0] is to_markdown

    def test_registry_functions_produce_non_empty_output_for_sample(self):
        for name, (fn, _) in EXPORTERS.items():
            result = fn(SAMPLE_ITEMS)
            assert isinstance(result, str), f"{name} exporter must return str"
            assert result, f"{name} exporter must not return empty string for non-empty input"

    def test_registry_functions_handle_empty_input(self):
        """No exporter should raise on an empty list."""
        for name, (fn, _) in EXPORTERS.items():
            try:
                fn(EMPTY)
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"EXPORTERS[{name!r}] raised {exc!r} on empty input")
