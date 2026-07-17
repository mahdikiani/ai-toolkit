"""Prompt Engine."""

import json
import tomllib
from pathlib import Path

import yaml
from jinja2 import Template


class PromptDataError(ValueError):
    """Raised when prompt data is invalid or unsupported."""


class PromptDataTypeError(TypeError):
    """Raised when prompt data has an unexpected type."""


class PromptDataNotFoundError(FileNotFoundError):
    """Raised when a prompt data file does not exist."""


def _yaml_schema_to_json_schema(schema: dict) -> dict:
    """Convert simplified YAML output_schema to OpenRouter-compatible JSON Schema."""

    def _convert_value(val: object) -> dict:
        if isinstance(val, dict):
            props = {}
            required = []
            for k, v in val.items():
                if isinstance(k, str) and k.startswith("#"):
                    continue
                props[k] = _convert_value(v)
                required.append(k)
            return {
                "type": "object",
                "properties": props,
                "required": required,
                "additionalProperties": False,
            }
        if isinstance(val, list):
            item_schema = _convert_value(val[0]) if val else {"type": "string"}
            return {"type": "array", "items": item_schema}
        if isinstance(val, str):
            if "|" in val:
                parts = [p.strip() for p in val.split("|") if p.strip()]
                if parts:
                    return {"type": "string", "enum": parts}
            type_map = {
                "string": "string",
                "number": "number",
                "integer": "integer",
                "boolean": "boolean",
            }
            return {"type": type_map.get(val, "string")}
        return {"type": "string"}

    return _convert_value(schema)


def _render_value(value: object) -> str:
    """Convert a value to string for template rendering."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def load_data(path: Path | str) -> dict | str:
    """Load data from a file."""
    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        error = PromptDataNotFoundError(f"{path} not found")
        raise error

    with open(path, encoding="utf-8") as f:
        raw = f.read()

    if path.suffix.lower() in (".json",):
        return json.loads(raw)
    if path.suffix.lower() in (".yaml", ".yml"):
        return yaml.safe_load(raw) or {}
    if path.suffix.lower() in (".toml",):
        return tomllib.loads(raw)
    if path.suffix.lower() in (".txt",):
        return raw.strip()
    if path.suffix.lower() in (".md",):
        return raw.strip()
    if path.suffix.lower() in (".html",):
        error = PromptDataError("HTML input is not supported")
        raise error
    error = PromptDataError(f"Unsupported file type: {path.suffix}")
    raise error


class PromptEngine:
    """Prompt Engine."""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        """Initialize the Prompt Engine."""
        if base_dir is None:
            base_dir = Path(".")
        elif isinstance(base_dir, str):
            base_dir = Path(base_dir)
        self.base_dir = base_dir
        self._prompt_dir: Path = base_dir

    def _load_yaml(self, file_path: str | Path) -> dict:
        """Load YAML from path, resolved relative to the current prompt directory."""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        path = file_path
        if not path.is_absolute():
            path = (self._prompt_dir / file_path).resolve()
        data = load_data(path)
        if isinstance(data, dict):
            return data
        error = PromptDataError(
            f"Expected YAML dict from {file_path}, got {type(data)}"
        )
        raise error

    def resolve_component(
        self, comp_name: str, comp_data: dict[str, str], context: dict
    ) -> dict:
        """Resolve a component recursively: inline, or from a separate file."""
        # If a reference to another file: file: path.yaml#key
        if "file" in comp_data:
            file_ref = comp_data["file"]
            if "#" in file_ref:
                file_path, comp_key = file_ref.split("#", 1)
            else:
                file_path, comp_key = file_ref, comp_name

            prev_dir = self._prompt_dir
            full_path = (self._prompt_dir / file_path).resolve()
            self._prompt_dir = full_path.parent
            try:
                external_data = self._load_yaml(str(full_path))
                if not isinstance(external_data, dict):
                    error = PromptDataTypeError(f"File '{file_path}' must yield a dict")
                    raise error
                external_components = external_data.get("components", {})
                target_comp = external_components.get(comp_key)
                if not target_comp:
                    error = PromptDataError(
                        f"Component '{comp_key}' not found in file '{file_path}'."
                    )
                    raise error
                return self.resolve_component(comp_key, target_comp, context)
            finally:
                self._prompt_dir = prev_dir

        # Render the task with Jinja2
        task_template = Template(comp_data.get("task", ""))
        rendered_task = task_template.render(**context)

        return {
            "task": rendered_task,
            "output_schema": comp_data.get("output_schema", {}),
        }

    def _prepare_render_context(self, input_data: dict) -> dict:
        """Build render context with dict/list values as strings."""
        ctx = {
            k: _render_value(v) if isinstance(v, (dict, list)) else v
            for k, v in input_data.items()
        }
        for key in (
            "transcript",
            "document_text",
            "text",
            "metadata",
            "language",
            "question",
            "context",
            "conversation",
            "broken_json",
            "target_schema",
            "user_correction",
        ):
            ctx.setdefault(key, "")
        return ctx

    def _process_components(
        self,
        comp_names: list,
        components_cfg: dict,
        render_ctx: dict,
    ) -> tuple[list[str], dict[str, list[str]], dict]:
        """Resolve components; return XML parts, extra keys, and merged schema."""
        component_xml_parts: list[str] = []
        component_key_parts: dict[str, list[str]] = {}
        merged_output_schema: dict = {}
        for comp_name in comp_names or []:
            comp_data = components_cfg.get(comp_name)
            if not comp_data:
                error = PromptDataError(
                    f"Component '{comp_name}' not found in components"
                )
                raise error
            resolved = self.resolve_component(comp_name, comp_data, render_ctx)
            resolved_schema = resolved.get("output_schema", {})
            if isinstance(resolved_schema, dict) and resolved_schema:
                merged_output_schema.update(resolved_schema)
            component_xml_parts.append(
                f"<{comp_name}>\n{resolved['task'].strip()}\n</{comp_name}>"
            )
            for comp_key, comp_val in comp_data.items():
                if comp_key in ("task", "file"):
                    continue
                rendered = (
                    Template(comp_val).render(**render_ctx)
                    if isinstance(comp_val, str)
                    else _render_value(comp_val)
                )
                component_key_parts.setdefault(comp_key, []).append(rendered.strip())
        return component_xml_parts, component_key_parts, merged_output_schema

    def _render_system_key(
        self, key: str, value: object, render_ctx: dict
    ) -> str | None:
        """Render a system key value to XML string."""
        tag = "persona" if key == "personas" else key
        rendered = (
            Template(value).render(**render_ctx)
            if isinstance(value, str)
            else _render_value(value)
        )
        if rendered or value == "":
            return f"<{tag}>\n{rendered.strip()}\n</{tag}>"
        return None

    def generate(
        self, prompt_path: Path | str, input_data: dict
    ) -> tuple[str, str, dict | None]:
        """Generate prompt and OpenRouter-compatible response_format from schema."""
        prompt_path = Path(prompt_path)
        self._prompt_dir = prompt_path.resolve().parent

        main_config = load_data(prompt_path)
        render_ctx = self._prepare_render_context(input_data)
        if isinstance(main_config, str):
            return "", Template(main_config).render(**render_ctx), None
        if not isinstance(main_config, dict):
            error = PromptDataTypeError("Prompt file must yield a dict or string")
            raise error

        task_cfg = main_config.get("task", {})
        system_cfg = task_cfg.get("system") or main_config.get("system", {})
        components_cfg = main_config.get("components", {})
        system_parts: list[str] = []
        component_key_parts: dict[str, list[str]] = {}
        merged_output_schema: dict = {}

        (
            system_parts,
            component_key_parts,
            merged_output_schema,
        ) = self._render_system(system_cfg, components_cfg, render_ctx)

        for comp_key, parts in component_key_parts.items():
            if parts:
                system_parts.append(
                    f"<{comp_key}>\n" + "\n\n".join(parts) + f"\n</{comp_key}>"
                )

        system_xml = "\n\n".join(system_parts)
        user_template = (
            task_cfg.get("user")
            or main_config.get("user")
            or main_config.get("prompt")
            or ""
        )
        user_xml = Template(user_template).render(**render_ctx)

        response_format: dict | None = None
        if merged_output_schema:
            json_schema = _yaml_schema_to_json_schema(merged_output_schema)
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": json_schema,
                },
            }

        return system_xml, user_xml, response_format

    def _render_system(
        self, system_cfg: str | dict, components_cfg: dict, render_ctx: dict
    ) -> tuple[list[str], dict[str, list[str]], dict]:
        """Render system configuration and its referenced components."""
        if isinstance(system_cfg, str):
            part = self._render_system_key("system", system_cfg, render_ctx)
            return ([part] if part else []), {}, {}

        system_parts: list[str] = []
        component_key_parts: dict[str, list[str]] = {}
        merged_output_schema: dict = {}
        for key, value in system_cfg.items():
            if key == "components":
                parts, component_key_parts, schemas = self._process_components(
                    value, components_cfg, render_ctx
                )
                system_parts.extend(parts)
                merged_output_schema.update(schemas)
                continue
            part = self._render_system_key(key, value, render_ctx)
            if part:
                system_parts.append(part)
        return system_parts, component_key_parts, merged_output_schema
