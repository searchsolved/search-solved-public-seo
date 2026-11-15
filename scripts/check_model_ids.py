#!/usr/bin/env python3
"""
Model ID Checker

Scans every .py and .ipynb file in the repository (sample_data folders are
skipped) for Claude and OpenAI model ID strings and fails if any ID is not
in the approved lists below. Local model names such as qwen2.5:7b or llama3
are not Claude or OpenAI IDs, so they are ignored.

Usage:
    python scripts/check_model_ids.py

Exits 0 when every model ID found is approved. Otherwise prints one
file:line entry per unapproved ID and exits 1.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import json
import re
import sys
from pathlib import Path

# Approved model IDs: the cheap and mid tier for each provider. Tools offer
# both, cheap first for simple jobs and mid first for heavy ones. Update these
# lists (and the tools) together.
APPROVED_CLAUDE_MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]
APPROVED_OPENAI_MODELS = ["gpt-6-luna", "gpt-6.1-terra"]

# Family names that are not model IDs: tools check `model.startswith("gpt-6")`
# to decide whether to send reasoning parameters, and docs say "gpt-6 models".
FAMILY_NAMES = {"gpt-6"}

APPROVED = set(APPROVED_CLAUDE_MODELS) | set(APPROVED_OPENAI_MODELS) | FAMILY_NAMES

SKIP_DIRS = {"sample_data", ".git", "__pycache__", ".venv", "venv", "node_modules", ".ruff_cache"}
THIS_FILE = Path(__file__).resolve()

# Claude IDs (claude-...), OpenAI chat IDs (gpt-..., chatgpt-...), OpenAI
# o-series IDs in quotes ("o3", 'o4-mini') and older OpenAI families.
MODEL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_-])("
    r"claude-[a-z0-9][a-z0-9.-]*"
    r"|(?:chat)?gpt-[a-z0-9][a-z0-9.-]*"
    r"|text-(?:davinci|curie|babbage|ada|embedding)-[a-z0-9.-]+"
    r"|(?:dall-e|whisper|tts)-[a-z0-9.-]+"
    r")"
    r"|(?<=[\"'])(o[1-9](?:-(?:mini|pro|preview))?)(?=[\"'])"
)


def clean(model_id):
    """Drop trailing punctuation picked up from prose, e.g. 'gpt-4o-mini.'."""
    return model_id.rstrip(".-")


def iter_files(root):
    """Yield every .py and .ipynb file under root, skipping SKIP_DIRS."""
    for path in sorted(root.rglob("*")):
        if path.suffix not in (".py", ".ipynb") or not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.resolve() == THIS_FILE:
            continue
        yield path


def notebook_lines(path):
    """Return (line number, text) pairs for a notebook.

    Line numbers point at the line in the .ipynb JSON file, so they can be
    opened directly. If the notebook is not valid JSON, the raw file is used.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        json.loads(text)
    except json.JSONDecodeError:
        print(f"warning: {path} is not valid JSON, scanning raw text", file=sys.stderr)
    return enumerate(text.splitlines(), start=1)


def source_lines(path):
    """Return (line number, text) pairs for a .py file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return enumerate(text.splitlines(), start=1)


def find_unapproved(root):
    """Return a list of (relative path, line number, model ID) for unapproved IDs."""
    problems = []
    for path in iter_files(root):
        lines = notebook_lines(path) if path.suffix == ".ipynb" else source_lines(path)
        for number, line in lines:
            for match in MODEL_PATTERN.finditer(line):
                model_id = clean(match.group(1) or match.group(2))
                if model_id not in APPROVED:
                    problems.append((path.relative_to(root), number, model_id))
    return problems


def main():
    root = Path(__file__).resolve().parent.parent
    problems = find_unapproved(root)
    if not problems:
        print("All Claude and OpenAI model IDs are approved.")
        return 0
    print(f"Found {len(problems)} unapproved model ID(s):")
    for rel_path, number, model_id in problems:
        print(f"{rel_path}:{number}: {model_id}")
    print("")
    print("Approved Claude models: " + ", ".join(APPROVED_CLAUDE_MODELS))
    print("Approved OpenAI models: " + ", ".join(APPROVED_OPENAI_MODELS))
    return 1


if __name__ == "__main__":
    sys.exit(main())
