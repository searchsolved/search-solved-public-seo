#!/usr/bin/env python3
"""
Tag Consolidator - CLI Version

Use OpenAI to consolidate granular secondary tags into broader generic
categories, grouped by primary tag.

Usage:
    export OPENAI_API_KEY=your-key
    python tag_consolidator_cli.py --input tags.csv --output consolidated_tags.csv

Author: Lee Foot
Website: https://www.leefoot.com
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

from tag_consolidator import (
    DEFAULT_MODEL,
    DEFAULT_PRIMARY_COLUMN,
    DEFAULT_SECONDARY_COLUMN,
    GENERIC_TAG_COLUMN,
    consolidate_tags,
)


def main():
    parser = argparse.ArgumentParser(
        description="Use OpenAI to consolidate granular tags into broader generic categories"
    )
    parser.add_argument("--input", required=True, help="Input CSV with tags")
    parser.add_argument(
        "--output",
        default="consolidated_tags.csv",
        help="Output CSV path (default: consolidated_tags.csv)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--primary-column",
        default=DEFAULT_PRIMARY_COLUMN,
        help=f"Name of the primary tag column (default: {DEFAULT_PRIMARY_COLUMN})",
    )
    parser.add_argument(
        "--secondary-column",
        default=DEFAULT_SECONDARY_COLUMN,
        help=f"Name of the secondary tag column (default: {DEFAULT_SECONDARY_COLUMN})",
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

    total_groups = df[args.primary_column].nunique() if args.primary_column in df.columns else 0
    print(f"Total groups (batches) to process: {total_groups}")

    def print_progress(processed, total, group_name):
        print(f"Processed {processed}/{total} groups.")

    def save_checkpoint(partial_df):
        # Save progress after each group so a failed run can be inspected
        partial_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    df_final = consolidate_tags(
        df,
        api_key=api_key,
        model=args.model,
        primary_column=args.primary_column,
        secondary_column=args.secondary_column,
        progress_callback=print_progress,
        checkpoint_callback=save_checkpoint,
    )

    # Save final results
    df_final.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"\nResults saved to: {args.output}")
    print(f"  Rows processed: {len(df_final)}")
    print(f"  Generic categories: {df_final[GENERIC_TAG_COLUMN].nunique()}")

    unmapped = df_final[GENERIC_TAG_COLUMN].isna().sum()
    if unmapped > 0:
        print(f"  Warning: {unmapped} rows could not be mapped to a generic category")


if __name__ == "__main__":
    main()
