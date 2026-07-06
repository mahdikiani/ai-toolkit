"""
Property-based tests for the PromptEngine.

Property 2: Prompt engine rendering determinism
Property 6: YAML to JSON schema field preservation
Validates: Requirements 4.1, 4.6, 9.2, 9.4
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from apps.executions.engine.engine import PromptEngine, _yaml_schema_to_json_schema
from hypothesis import given, settings
from hypothesis import strategies as st

# Strategy for valid identifier-like keys (no special chars)
identifier_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz_",
    min_size=1,
    max_size=20,
)

# Strategy for YAML schema type values
type_strategy = st.sampled_from(["string", "number", "integer", "boolean"])

# Strategy for simple YAML schemas (dict of identifier -> type)
simple_schema_strategy = st.dictionaries(
    keys=identifier_strategy,
    values=type_strategy,
    min_size=1,
    max_size=10,
)


@pytest.mark.property
class TestPromptEngineRenderingDeterminism:
    """Property 2: Prompt engine rendering determinism."""

    @given(
        st.dictionaries(
            keys=identifier_strategy,
            values=st.one_of(st.text(max_size=50), st.integers(), st.booleans()),
            max_size=5,
        )
    )
    @settings(max_examples=50)
    def test_rendering_is_deterministic(self, input_data: dict) -> None:
        """
        Property 2: Prompt engine rendering determinism.

        Rendering the same template with the same data should always produce
        identical output.
        """
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            prompt_file = tmp_path / "test_prompt.yaml"
            prompt_file.write_text(
                "task:\n"
                "  system:\n"
                "    persona: You are a helpful assistant\n"
                "  user: 'Process the input'\n"
            )

            engine = PromptEngine(base_dir=tmp_path)

            system1, user1, format1 = engine.generate(prompt_file, input_data)
            system2, user2, format2 = engine.generate(prompt_file, input_data)

        assert system1 == system2, (
            f"System prompt is not deterministic for input {input_data!r}"
        )
        assert user1 == user2, (
            f"User prompt is not deterministic for input {input_data!r}"
        )
        assert format1 == format2, (
            f"Response format is not deterministic for input {input_data!r}"
        )

    @given(st.text(max_size=100))
    @settings(max_examples=50)
    def test_text_variable_rendering_is_deterministic(
        self, text_value: str
    ) -> None:
        """Rendering with text variables should be deterministic."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            prompt_file = tmp_path / "test_prompt.yaml"
            prompt_file.write_text(
                "task:\n  system:\n    persona: You are helpful\n  user: '{{ text }}'\n"
            )

            engine = PromptEngine(base_dir=tmp_path)

            _, user1, _ = engine.generate(prompt_file, {"text": text_value})
            _, user2, _ = engine.generate(prompt_file, {"text": text_value})

        assert user1 == user2


@pytest.mark.property
class TestYamlToJsonSchemaFieldPreservation:
    """Property 6: YAML to JSON schema field preservation."""

    @given(simple_schema_strategy)
    @settings(max_examples=100)
    def test_all_fields_preserved_in_properties(self, yaml_schema: dict) -> None:
        """
        Property 6: YAML to JSON schema field preservation.

        All fields in the YAML schema should appear in the JSON schema properties.
        """
        json_schema = _yaml_schema_to_json_schema(yaml_schema)

        yaml_fields = set(yaml_schema.keys())
        json_fields = set(json_schema.get("properties", {}).keys())

        assert yaml_fields == json_fields, (
            f"Fields not preserved in JSON schema:\n"
            f"  YAML fields: {yaml_fields}\n"
            f"  JSON fields: {json_fields}\n"
            f"  Missing: {yaml_fields - json_fields}\n"
            f"  Extra: {json_fields - yaml_fields}"
        )

    @given(simple_schema_strategy)
    @settings(max_examples=100)
    def test_all_fields_marked_as_required(self, yaml_schema: dict) -> None:
        """All YAML schema fields should be marked as required in JSON schema."""
        json_schema = _yaml_schema_to_json_schema(yaml_schema)

        yaml_fields = set(yaml_schema.keys())
        required_fields = set(json_schema.get("required", []))

        assert yaml_fields == required_fields, (
            f"Not all fields marked as required:\n"
            f"  YAML fields: {yaml_fields}\n"
            f"  Required fields: {required_fields}"
        )

    @given(simple_schema_strategy)
    @settings(max_examples=100)
    def test_output_is_valid_json_schema_structure(self, yaml_schema: dict) -> None:
        """JSON schema output should have valid structure."""
        json_schema = _yaml_schema_to_json_schema(yaml_schema)

        assert "type" in json_schema
        assert json_schema["type"] == "object"
        assert "properties" in json_schema
        assert "required" in json_schema
        assert "additionalProperties" in json_schema
        assert json_schema["additionalProperties"] is False

    @given(
        st.dictionaries(
            keys=identifier_strategy,
            values=type_strategy,
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=100)
    def test_type_mapping_is_correct(self, yaml_schema: dict) -> None:
        """Each YAML type should map to the correct JSON schema type."""
        type_map = {
            "string": "string",
            "number": "number",
            "integer": "integer",
            "boolean": "boolean",
        }

        json_schema = _yaml_schema_to_json_schema(yaml_schema)

        for field, yaml_type in yaml_schema.items():
            json_type = json_schema["properties"][field]["type"]
            expected_type = type_map.get(yaml_type, "string")
            assert json_type == expected_type, (
                f"Type mismatch for field '{field}': "
                f"YAML type '{yaml_type}' -> JSON type '{json_type}' "
                f"(expected '{expected_type}')"
            )

    @given(simple_schema_strategy)
    @settings(max_examples=100)
    def test_idempotent_conversion(self, yaml_schema: dict) -> None:
        """Converting the same schema twice should produce identical results."""
        result1 = _yaml_schema_to_json_schema(yaml_schema)
        result2 = _yaml_schema_to_json_schema(yaml_schema)

        assert result1 == result2
