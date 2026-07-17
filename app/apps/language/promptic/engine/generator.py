"""Run Prompt Engine CLI."""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from .engine import PromptEngine, load_data

logger = logging.getLogger(__name__)


def _main() -> None:
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
            return

        system_p, user_p, _ = engine.generate(args.prompt, input_data)
        output = {
            "system": system_p,
            "user": user_p,
        }

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                if args.output.suffix == ".json":
                    logger.info("Writing JSON to %s", args.output)
                    json.dump(output, f, ensure_ascii=False, indent=2)
                elif args.output.suffix in [".yml", ".yaml"]:
                    logger.info("Writing YAML to %s", args.output)
                    yaml.dump(
                        output,
                        f,
                        allow_unicode=True,
                        default_flow_style=False,
                        sort_keys=False,
                        width=100,
                    )
                else:
                    logger.info("Writing to %s", args.output)
                    for k, v in output.items():
                        f.write(f"\n{'#' * 20}\n")
                        f.write(f"# {k}\n")
                        f.write(f"{'#' * 20}\n\n")
                        f.write(v)
                        f.write("\n\n")

        else:
            for k, v in output.items():
                logger.info("\n %s", "#" * 100)
                logger.info("# %s", k)
                logger.info("%s\n", "#" * 100)
                logger.info(v)

    except Exception:
        logger.exception("Error")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    _main()
