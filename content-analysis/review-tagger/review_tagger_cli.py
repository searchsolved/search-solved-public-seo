#!/usr/bin/env python3
"""
Review Tagger - CLI Version

Use OpenAI to tag each review in a CSV with a one or two-word descriptive
label capturing its primary topic.

Usage:
    export OPENAI_API_KEY=your-key
    python review_tagger_cli.py --input reviews.csv --output tagged_reviews.csv

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import os
import sys

import pandas as pd

try:
    from openai import OpenAI  # noqa: F401
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)

from review_tagger import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLUMN,
    DEFAULT_MODEL,
    tag_reviews,
)


def main():
    parser = argparse.ArgumentParser(
        description="Use OpenAI to tag reviews with one or two-word descriptive labels"
    )
    parser.add_argument("--input", required=True, help="Input CSV with reviews")
    parser.add_argument(
        "--output",
        default="tagged_reviews.csv",
        help="Output CSV path (default: tagged_reviews.csv)",
    )
    parser.add_argument(
        "--column",
        default=DEFAULT_COLUMN,
        help=f"Name of the review text column (default: {DEFAULT_COLUMN})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of reviews per API call (default: {DEFAULT_BATCH_SIZE})",
    )

    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("  export OPENAI_API_KEY=your-key")
        sys.exit(1)

    # Load data
    print(f"Loading: {args.input}")
    try:
        df = pd.read_csv(args.input, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(args.input, encoding="latin-1")

    print(f"  Loaded {len(df):,} rows")
    print(f"  Model: {args.model}")
    print(f"  Batch size: {args.batch_size}")

    total_batches = (len(df) + args.batch_size - 1) // args.batch_size
    print(f"  Total batches: {total_batches}")

    def print_progress(processed, total):
        print(f"  Processed batch {processed}/{total}")

    df_tagged = tag_reviews(
        df,
        api_key=api_key,
        review_column=args.column,
        model=args.model,
        batch_size=args.batch_size,
        progress_callback=print_progress,
    )

    # Save results
    df_tagged.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"\nResults saved to: {args.output}")
    print(f"  Reviews processed: {len(df_tagged)}")
    print(f"  Unique tags: {df_tagged['Tag'].nunique()}")

    untagged = df_tagged["Tag"].isna().sum()
    if untagged > 0:
        print(f"  Warning: {untagged} reviews could not be tagged")


if __name__ == "__main__":
    main()
