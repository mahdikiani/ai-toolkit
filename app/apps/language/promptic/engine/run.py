"""Run Prompt Engine CLI."""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

import aiofiles
import dotenv
import httpx
import yaml
from engine import PromptEngine, load_data

from server.config import Settings

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


async def call_openrouter(
    system: str,
    user: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.2,
    response_format: dict | None = None,
) -> str:
    """Call OpenRouter API."""
    openrouter_url = f"{Settings.openrouter_base_url}/chat/completions"
    api_key = api_key or Settings.openrouter_api_key
    model = model or os.getenv("PROMPT_MODEL", "google/gemini-3-flash-preview")

    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }
    if max_tokens:
        body["max_tokens"] = max_tokens
    if response_format:
        body["response_format"] = response_format

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/prompt-library",
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                openrouter_url,
                json=body,
                headers=headers,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"OpenRouter HTTP {e.response.status_code}: {e.response.text}"
        ) from e
    except httpx.RequestError as e:
        raise RuntimeError(f"OpenRouter request failed: {e}") from e

    choices = data.get("choices")
    if not choices:
        raise RuntimeError(
            "پاسخی از مدل نیامد؛ پاسخ خام: "
            + json.dumps(data, ensure_ascii=False)[:500]
        )
    content = choices[0].get("message", {}).get("content") or ""
    return content.strip()


def extract_json_from_content(content: str) -> dict | list:
    """خروجی مدل گاهی داخل markdown code block است؛ استخراج JSON خام."""
    content = content.strip()
    # حذف ```json ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if m:
        content = m.group(1).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # شاید فقط یک آرایه یا آبجکت بدون wrapper باشد
        content = content.strip()
        if content.startswith("["):
            return json.loads(content)
        if content.startswith("{"):
            return json.loads(content)
        raise ValueError("خروجی مدل JSON معتبر نبود.") from None


async def _main() -> None:
    """Run the prompt engine."""

    parser = argparse.ArgumentParser(description="Modular Prompt Engine CLI")
    parser.add_argument(
        "-p", "--prompt", help="Path to the prompt YAML file", required=True
    )
    parser.add_argument("-i", "--input", help="Path to the input data file (YAML/JSON)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file. stream to stdout if not provided",
    )
    args = parser.parse_args()

    try:
        engine = PromptEngine()
        input_data = load_data(args.input)
        if not isinstance(input_data, dict):
            logger.error("Input file must yield a dict (YAML/JSON)")
            sys.exit(1)

        system_p, user_p, response_format = engine.generate(args.prompt, input_data)

        content = await call_openrouter(
            system_p, user_p, response_format=response_format
        )
        result = extract_json_from_content(content)

        if args.output:
            async with aiofiles.open(args.output, "w", encoding="utf-8") as f:
                if args.output.suffix == ".json":
                    logger.info("Writing JSON to %s", args.output)
                    await f.write(json.dumps(result, ensure_ascii=False, indent=2))
                elif args.output.suffix in [".yml", ".yaml"]:
                    logger.info("Writing YAML to %s", args.output)
                    yaml_content = yaml.dump(
                        result,
                        allow_unicode=True,
                        default_flow_style=False,
                        sort_keys=False,
                        width=100,
                    )
                    await f.write(yaml_content)
                else:
                    logger.info("Writing to %s", args.output)
                    for k, v in result.items():
                        await f.write(f"\n{'#' * 20}\n")
                        await f.write(f"# {k}\n")
                        await f.write(f"{'#' * 20}\n\n")
                        await f.write(str(v))
                        await f.write("\n\n")
        else:
            for k, v in result.items():
                logger.info("\n %s", "#" * 20)
                logger.info("# %s", k)
                logger.info("%s\n", "#" * 20)
                logger.info(v)

    except Exception:
        logger.exception("Error")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    asyncio.run(_main())
