#!/usr/bin/env python3
# Author   : Lee Foot
# Website  : https://leefoot.com
"""
Auto CSS Selector Detector - CLI Version

Automatically identifies the best CSS selector for a page's main content
using an LLM, then extracts and converts the content to Markdown.

Usage:
    export OPENAI_API_KEY=sk-...
    python auto_css_selector_detector_cli.py --url https://example.com/page
    python auto_css_selector_detector_cli.py --url https://example.com/page --model gpt-4o
    python auto_css_selector_detector_cli.py --url https://example.com/page --base-url http://localhost:1234/v1

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import os
import sys

from auto_css_selector_detector import detect_and_extract


def main():
    parser = argparse.ArgumentParser(
        description="Detect the main content CSS selector of a web page using an LLM."
    )
    parser.add_argument(
        "--url",
        required=True,
        help="URL of the page to analyse.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model name to use (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.openai.com/v1",
        help="API base URL (default: https://api.openai.com/v1). Change for local LLMs.",
    )

    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    try:
        result = detect_and_extract(
            url=args.url,
            api_key=api_key,
            model=args.model,
            base_url=args.base_url,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"URL: {result.get('url')}")
    print(f"H1: {result.get('h1', 'N/A')}")
    print(f"Initial Selector: {result.get('selector')}")
    print(f"Specific Selector: {result.get('specific_selector')}")
    print(f"Reasoning: {result.get('reasoning')}")
    print()
    print("--- Extracted Content ---")
    print(result.get("extracted_text", ""))
    print()
    print("--- Internal Links ---")
    print(result.get("links", "[]"))


if __name__ == "__main__":
    main()
