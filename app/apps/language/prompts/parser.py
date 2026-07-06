"""Prompt parser for extracting schemas from YAML prompt templates with Jinja2."""

import re
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, meta


def extract_jinja2_variables(template_str: str) -> set[str]:
    """
    Extract all Jinja2 variables from a template string.

    Args:
        template_str: The template string to parse

    Returns:
        Set of variable names found in the template
    """
    env = Environment()
    try:
        ast = env.parse(template_str)
        return meta.find_undeclared_variables(ast)
    except Exception:
        # Fallback to regex if Jinja2 parsing fails
        return set(re.findall(r"\{\{\s*(\w+)", template_str))


def infer_field_type(variable_name: str, context: dict[str, Any]) -> str:
    """
    Infer the type of a field based on its name and context.

    Args:
        variable_name: The name of the variable
        context: Context dictionary with example values

    Returns:
        The inferred type as a string
    """
    # Check if we have an example value in context
    if variable_name in context:
        value = context[variable_name]
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"

    # Infer from variable name patterns
    if variable_name.endswith("_id"):
        return "string"
    elif variable_name.startswith("is_") or variable_name.startswith("has_"):
        return "boolean"
    elif variable_name.endswith("_count") or variable_name.endswith("_number"):
        return "integer"
    elif variable_name.endswith("_list") or variable_name.endswith("_items"):
        return "array"

    return "string"


def parse_prompt_file(prompt_path: Path) -> dict[str, Any]:
    """
    Parse a YAML prompt file and extract schema information.

    Args:
        prompt_path: Path to the YAML prompt file

    Returns:
        Dictionary containing prompt metadata, input fields, and output schema

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
        yaml.YAMLError: If the YAML is invalid
    """
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, encoding="utf-8") as f:
        content = yaml.safe_load(f)

    if not isinstance(content, dict):
        raise ValueError(f"Invalid prompt file format: {prompt_path}")

    # Extract metadata
    name = content.get("name", prompt_path.stem)
    description = content.get("description", "")
    tags = content.get("tags", [])
    model = content.get("model", "google/gemini-3.0-flash-preview")
    config = content.get("config", {})

    # Extract messages
    messages = content.get("messages", [])

    # Extract input fields from Jinja2 templates in messages
    input_fields = _extract_input_fields(messages, content.get("examples", {}))

    # Extract output schema
    output_schema = content.get("output_schema")

    return {
        "name": name,
        "description": description,
        "tags": tags,
        "model": model,
        "config": config,
        "messages": messages,
        "input_fields": input_fields,
        "output_schema": output_schema,
    }


def _extract_input_fields(
    messages: list[dict[str, Any]], examples: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Extract input fields from message templates.

    Args:
        messages: List of message blocks
        examples: Example values for context

    Returns:
        List of input field definitions
    """
    all_variables: set[str] = set()

    # Extract variables from all message content
    for message in messages:
        if isinstance(message, dict):
            content = message.get("content", "")
            if isinstance(content, str):
                variables = extract_jinja2_variables(content)
                all_variables.update(variables)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text", "")
                        if isinstance(text, str):
                            variables = extract_jinja2_variables(text)
                            all_variables.update(variables)

    # Build input field definitions
    input_fields = []
    for var_name in sorted(all_variables):
        # Skip common Jinja2 keywords and loop variables
        if var_name in {"loop", "self", "super", "range", "dict", "lipsum"}:
            continue

        field_type = infer_field_type(var_name, examples)

        input_fields.append({
            "name": var_name,
            "type": field_type,
            "required": True,
            "description": f"Input field: {var_name}",
        })

    return input_fields


def render_prompt(prompt_path: Path, context: dict[str, Any]) -> dict[str, Any]:
    """
    Render a prompt template with the given context.

    Args:
        prompt_path: Path to the YAML prompt file
        context: Dictionary of variables to render

    Returns:
        Dictionary with rendered messages and metadata

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
        jinja2.TemplateError: If template rendering fails
    """
    prompt_data = parse_prompt_file(prompt_path)

    env = Environment()
    rendered_messages = []

    for message in prompt_data["messages"]:
        rendered_message = {
            "role": message.get("role", "user"),
        }

        content = message.get("content", "")
        if isinstance(content, str):
            template = env.from_string(content)
            rendered_message["content"] = template.render(context)
        elif isinstance(content, list):
            rendered_content = []
            for part in content:
                if isinstance(part, dict):
                    rendered_part = {"type": part.get("type", "text")}
                    if "text" in part:
                        template = env.from_string(part["text"])
                        rendered_part["text"] = template.render(context)
                    if "file_url" in part:
                        rendered_part["file_url"] = part["file_url"]
                    rendered_content.append(rendered_part)
            rendered_message["content"] = rendered_content

        rendered_messages.append(rendered_message)

    return {
        "name": prompt_data["name"],
        "description": prompt_data["description"],
        "model": prompt_data["model"],
        "config": prompt_data["config"],
        "messages": rendered_messages,
        "output_schema": prompt_data["output_schema"],
    }
