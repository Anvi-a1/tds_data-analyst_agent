import json
from unittest.mock import patch, MagicMock

import pytest

from app.orchestrator import Orchestrator


@pytest.fixture
def orch():
    with patch("app.providers.ProviderFactory.create_provider") as mock_create:
        mock_provider = MagicMock()
        mock_provider.provider_name = "mock"
        mock_create.return_value = mock_provider
        yield Orchestrator()


def test_placeholder_top_level_as_json(orch):
    shared = {"data": {"a": 1, "b": 2}}
    rendered = orch._render_placeholders("{{shared_results.data.as_json}}", shared)
    assert json.loads(rendered) == {"a": 1, "b": 2}


def test_placeholder_top_level_as_python(orch):
    shared = {"nums": [1, 2, 3]}
    rendered = orch._render_placeholders("{{shared_results.nums.as_python}}", shared)
    # as_python returns Python literal using repr
    assert rendered.strip().startswith("[") and rendered.strip().endswith("]")
    assert "1" in rendered and "3" in rendered


def test_placeholder_nested_subkey_as_json(orch):
    shared = {"html": {"results": [{"text": "First"}, {"text": "Second"}]}}
    rendered = orch._render_placeholders(
        "{{shared_results.html.results.as_json}}", shared
    )
    parsed = json.loads(rendered)
    assert isinstance(parsed, list)
    assert parsed[0]["text"] == "First"


def test_placeholder_list_index_and_field(orch):
    shared = {"html": {"results": [{"text": "First"}, {"text": "Second"}]}}
    rendered = orch._render_placeholders(
        "{{shared_results.html.results.1.text.as_json}}", shared
    )
    assert json.loads(rendered) == "Second"


def test_placeholder_simple_form_defaults_to_json(orch):
    shared = {"obj": {"x": 10}}
    rendered = orch._render_placeholders("prefix {{shared_results.obj}} suffix", shared)
    assert rendered.startswith("prefix ") and rendered.endswith(" suffix")
    middle = rendered[len("prefix ") : -len(" suffix")]
    assert json.loads(middle) == {"x": 10}


def test_placeholder_missing_key_and_path(orch):
    shared = {"a": {"b": 1}, "lst": [1, 2]}
    # Missing key -> null for as_json
    assert (
        orch._render_placeholders("{{shared_results.missing.as_json}}", shared)
        == "null"
    )
    # Missing key -> None for as_python
    assert (
        orch._render_placeholders("{{shared_results.missing.as_python}}", shared)
        == "None"
    )
    # Missing path segment -> null
    assert (
        orch._render_placeholders("{{shared_results.a.missing.as_json}}", shared)
        == "null"
    )
    # Out of range index -> null
    assert (
        orch._render_placeholders("{{shared_results.lst.5.as_json}}", shared) == "null"
    )


def test_placeholder_multiple_and_mixed(orch):
    shared = {
        "a": {"b": 1},
        "html": {"results": [{"text": "First"}, {"text": "Second"}]},
    }
    s = "A={{shared_results.a.as_json}} B={{shared_results.html.results.0.text.as_json}} C={{shared_results.missing.as_json}}"
    rendered = orch._render_placeholders(s, shared)
    # Extract segments by markers to avoid splitting inside JSON
    b_marker = " B="
    c_marker = " C="
    assert rendered.startswith("A=")
    a_str = rendered[len("A=") : rendered.find(b_marker)]
    b_str = rendered[rendered.find(b_marker) + len(b_marker) : rendered.find(c_marker)]
    c_str = rendered[rendered.find(c_marker) + len(c_marker) :]
    assert json.loads(a_str) == {"b": 1}
    assert json.loads(b_str) == "First"
    assert c_str == "null"


def test_placeholder_non_digit_index_and_negative_index(orch):
    shared = {"lst": [10, 20]}
    assert (
        orch._render_placeholders("{{shared_results.lst.x.as_json}}", shared) == "null"
    )
    assert (
        orch._render_placeholders("{{shared_results.lst.-1.as_json}}", shared) == "null"
    )


def test_placeholder_path_into_primitive(orch):
    shared = {"x": 5}
    # Attempt to traverse into primitive -> null
    assert orch._render_placeholders("{{shared_results.x.y.as_json}}", shared) == "null"


def test_placeholder_key_with_dash_and_simple_nested(orch):
    shared = {"key-with-dash": {"val": 42}, "obj": {"arr": [1, 2, 3]}}
    assert (
        orch._render_placeholders(
            "{{shared_results.key-with-dash.val.as_json}}", shared
        )
        == "42"
    )
    # Simple form nested path defaults to JSON
    out = orch._render_placeholders("{{shared_results.obj.arr.1}}", shared)
    assert out == "2"


def test_placeholder_non_serializable_as_json_fallback_and_as_python_repr(orch):
    shared = {"weird": {"s": set([1, 2])}, "d": {"k": {"x": 1}}}
    out_json = orch._render_placeholders("{{shared_results.weird.s.as_json}}", shared)
    # Should be a JSON string of the set's repr like "{1, 2}"
    assert (
        out_json.startswith('"')
        and out_json.endswith('"')
        and "{" in out_json
        and "}" in out_json
    )
    out_py = orch._render_placeholders("{{shared_results.d.k.as_python}}", shared)
    assert out_py.strip().startswith("{") and out_py.strip().endswith("}")
