# Author: Lee Foot
# Website: https://leefoot.com
"""
Product Attribute Extractor - CLI Version

Extract structured product attributes from a CSV using an OpenAI-compatible LLM.

Usage:
    python product_attribute_extractor_cli.py --input products.csv --column "title" --output enriched.csv

    # With a local LLM (e.g. LM Studio):
    python product_attribute_extractor_cli.py --input products.csv --column "H1" \\
        --base-url http://localhost:1234/v1 --model local-model

    API key is read from OPENAI_API_KEY environment variable.

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import os
import sys
import pandas as pd
from tqdm import tqdm
from product_attribute_extractor import (
    create_client,
    extract_attributes,
    sort_columns_by_frequency,
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured product attributes from a CSV using an LLM.",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output", "-o",
        default="product_attributes_extracted.csv",
        help="Path for the output CSV file (default: product_attributes_extracted.csv).",
    )
    parser.add_argument(
        "--column", "-c",
        required=True,
        help="Name of the column containing product text (title, H1, or description).",
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-4o-mini",
        help="Model identifier (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.openai.com/v1",
        help="Base URL for OpenAI-compatible API (default: https://api.openai.com/v1).",
    )

    args = parser.parse_args()

    # Read API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    # Load input CSV
    try:
        df = pd.read_csv(args.input, on_bad_lines="skip")
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)

    if args.column not in df.columns:
        print(
            f"Error: Column '{args.column}' not found. "
            f"Available columns: {', '.join(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loaded {len(df):,} rows from {args.input}")
    print(f"Using model: {args.model}")
    print(f"Base URL: {args.base_url}")
    print(f"Text column: {args.column}")
    print()

    client = create_client(api_key=api_key, base_url=args.base_url)
    known_attributes = set()
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting attributes"):
        product_text = str(row[args.column])
        if not product_text or product_text == "nan":
            results.append({})
            continue

        attrs = extract_attributes(client, args.model, product_text, known_attributes)

        # Update known attributes iteratively
        for attr_name in attrs:
            known_attributes.add(attr_name)

        record = {"product_text": product_text}
        record.update(attrs)
        results.append(record)

    # Build output DataFrame
    output_df = pd.DataFrame(results)
    output_df = sort_columns_by_frequency(output_df)

    # Ensure product_text is first
    cols = ["product_text"] + [c for c in output_df.columns if c != "product_text"]
    output_df = output_df[cols]

    # Save output
    output_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nDone. {len(output_df):,} products processed.")
    print(f"{len(known_attributes)} unique attributes discovered.")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
