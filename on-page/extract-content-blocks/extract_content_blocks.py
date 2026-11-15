"""
Content Block Extractor - Command Line Script

Uses Claude to identify the major content blocks on a list of pages, with an
XPath for each, then counts how often each XPath appears across the pages.
The Streamlit app (extract_content_blocks_app.py) does the same job in a
browser; both use content_blocks_core.py.

Usage:
    export ANTHROPIC_API_KEY=your-key
    python extract_content_blocks.py --input urls.txt --output-dir output
    python extract_content_blocks.py --input urls.txt --limit 2

Author: Lee Foot
Website: https://www.leefoot.com
"""

import argparse
import os
import sys
import time
import uuid
from datetime import datetime

from anthropic import Anthropic

from content_blocks_core import (
    CLAUDE_MODELS,
    COLUMNS,
    MAX_HTML_CHARS,
    MODEL,
    add_frequency,
    blocks_to_frame,
    extract_from_url,
)

COMBINED_FILENAME = "combined_output_with_frequency.csv"


def read_urls(filepath):
    """Read URLs from a text file, one per line."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def save_csv_batch(rows, output_dir):
    """Save a batch of rows to a CSV with a unique filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"content_blocks_{timestamp}_{str(uuid.uuid4())[:8]}.csv"
    filepath = os.path.join(output_dir, filename)
    blocks_to_frame(rows).to_csv(filepath, index=False, encoding="utf-8")
    print(f"  Saved {len(rows)} blocks to {filename}")
    return filepath


def parse_args():
    parser = argparse.ArgumentParser(
        description="Identify content blocks and XPaths on a list of pages with Claude."
    )
    parser.add_argument("--input", "-i", required=True, help="Text file with one URL per line")
    parser.add_argument("--output-dir", "-o", default="output", help="Folder for the CSV files (default: output)")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N URLs (use 2 for a test run)")
    parser.add_argument("--save-every", type=int, default=50, help="Save a batch file every N rows (default: 50)")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait before each page fetch (default: 1)")
    parser.add_argument("--model", default=MODEL, choices=CLAUDE_MODELS, help=f"Claude model (default: {MODEL})")
    parser.add_argument(
        "--max-html-chars", type=int, default=MAX_HTML_CHARS,
        help=f"Cut the filtered HTML at this many characters (default: {MAX_HTML_CHARS})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: set the ANTHROPIC_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)
    client = Anthropic(api_key=api_key)

    urls = read_urls(args.input)
    if args.limit:
        urls = urls[:args.limit]
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Processing {len(urls)} URLs. Batch save every {args.save_every} rows.\n")

    all_rows = []
    unsaved = []
    failed = []

    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url}")
        time.sleep(args.delay)
        try:
            rows, info = extract_from_url(client, url, args.model, args.max_html_chars)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(url)
            continue

        note = " (HTML was cut to the character limit)" if info["truncated"] else ""
        print(f"  {len(rows)} blocks. Tokens in {info['input_tokens']}, out {info['output_tokens']}{note}")
        all_rows.extend(rows)
        unsaved.extend(rows)

        if len(unsaved) >= args.save_every:
            save_csv_batch(unsaved, args.output_dir)
            unsaved = []

    if unsaved:
        save_csv_batch(unsaved, args.output_dir)

    # Combine this run's rows only, so an earlier run in the same folder does
    # not inflate the frequency counts.
    combined = add_frequency(blocks_to_frame(all_rows))
    combined_path = os.path.join(args.output_dir, COMBINED_FILENAME)
    combined.to_csv(combined_path, index=False, encoding="utf-8")

    print(f"\nDone: {len(urls) - len(failed)} of {len(urls)} URLs, {len(all_rows)} blocks.")
    if failed:
        print(f"Failed URLs ({len(failed)}):")
        for url in failed:
            print(f"  {url}")
    print(f"Combined file: {combined_path} (columns: {', '.join(COLUMNS + ['frequency'])})")

    if not combined.empty:
        print("\nTop 10 XPaths by frequency:")
        top = combined.drop_duplicates("xpath").head(10)
        for _, row in top.iterrows():
            print(f"  {row['frequency']:3d}x  {row['name']}")
            print(f"        {row['xpath'][:100]}")


if __name__ == "__main__":
    main()
