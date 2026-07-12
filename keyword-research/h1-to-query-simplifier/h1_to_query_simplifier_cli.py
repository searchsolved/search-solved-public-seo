#!/usr/bin/env python3
"""
H1 to Query Simplifier - CLI Version

Convert marketing-heavy H1 headings into clean, natural search queries
using the Anthropic API. Useful as a pre-processing step before SERP
clustering or keyword matching, where promotional H1s make poor queries.

Supports resuming: if the output file already exists, rows that have
already been simplified are skipped, and progress is saved periodically
so an interrupted run can pick up where it left off.

Usage:
    export ANTHROPIC_API_KEY=your-key-here
    python h1_to_query_simplifier_cli.py --input h1s.csv --output h1s_simplified.csv

    # Custom column and model
    python h1_to_query_simplifier_cli.py --input h1s.csv --output out.csv \
        --column "H1-1" --model claude-haiku-4-5

Author: Lee Foot
Website: https://www.leefoot.com
"""

import argparse
import logging
import os
import sys
import time

import anthropic
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5"
OUTPUT_COLUMN = "H1_Simplified"

PROMPT_TEMPLATE = """Convert this H1 into a natural Google search query. Remove marketing language, step counts, and promotional words. Focus on the core search intent.

H1: "{h1_text}"

Search query:"""


def simplify_h1(client, h1_text, model):
    """Simplify a single H1 into a natural search query. Returns None on failure."""
    try:
        message = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[
                {"role": "user", "content": PROMPT_TEMPLATE.format(h1_text=h1_text)}
            ],
        )
        simplified = message.content[0].text.strip()
        # Remove any remaining quotation marks or extra formatting
        simplified = simplified.replace('"', "").replace("'", "").strip()
        return simplified

    except anthropic.RateLimitError:
        logger.warning("Rate limited, waiting 10 seconds before retrying...")
        time.sleep(10)
        return simplify_h1(client, h1_text, model)
    except anthropic.APIStatusError as e:
        logger.error(f"API error ({e.status_code}) processing H1 '{h1_text}': {e.message}")
        return None
    except anthropic.APIConnectionError as e:
        logger.error(f"Connection error processing H1 '{h1_text}': {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing H1 '{h1_text}': {e}")
        return None


def process_csv(client, input_path, output_path, column, model, save_frequency=10, delay=0.1):
    """Process a CSV of H1s and add a simplified search query column."""
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path}")

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in CSV file. Available columns: {', '.join(df.columns)}"
        )

    # Resume from a previous run if the output file already exists
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        if OUTPUT_COLUMN in existing_df.columns and column in existing_df.columns:
            logger.info("Found existing output file. Resuming from where we left off...")
            df = existing_df.copy()
            already_processed = df[OUTPUT_COLUMN].notna().sum()
            logger.info(f"Already processed {already_processed} rows")

    if OUTPUT_COLUMN not in df.columns:
        df[OUTPUT_COLUMN] = ""

    # Work out which rows still need processing
    items_to_process = []
    for idx in range(len(df)):
        h1_text = df.iloc[idx][column]
        if (
            pd.notna(df.iloc[idx][OUTPUT_COLUMN])
            and str(df.iloc[idx][OUTPUT_COLUMN]).strip() != ""
        ):
            continue
        if pd.isna(h1_text) or str(h1_text).strip() == "":
            continue
        items_to_process.append(idx)

    logger.info(f"Found {len(items_to_process)} items to process")

    if not items_to_process:
        logger.info("All items already processed!")
        return

    processed_count = 0
    output_col_loc = df.columns.get_loc(OUTPUT_COLUMN)

    try:
        with tqdm(total=len(items_to_process), desc="Simplifying H1s", unit="item") as pbar:
            for i, idx in enumerate(items_to_process):
                h1_text = df.iloc[idx][column]

                simplified = simplify_h1(client, str(h1_text), model)
                if simplified:
                    df.iloc[idx, output_col_loc] = simplified
                    processed_count += 1
                    pbar.set_postfix({"row": idx + 1, "processed": processed_count})
                else:
                    logger.warning(f"Failed to simplify H1 at row {idx + 1}: '{h1_text}'")

                pbar.update(1)

                # Save every N items to avoid data loss
                if (i + 1) % save_frequency == 0:
                    df.to_csv(output_path, index=False)
                    tqdm.write(f"Saved progress: {processed_count} items processed")

                # Small delay to respect rate limits
                time.sleep(delay)

    except (KeyboardInterrupt, Exception):
        # Save progress before exiting so the run can be resumed
        df.to_csv(output_path, index=False)
        logger.info(f"Progress saved to {output_path} before exit")
        raise

    # Final save
    df.to_csv(output_path, index=False)
    logger.info(f"Final results saved to {output_path}")

    total_processed = (
        df[OUTPUT_COLUMN].notna() & (df[OUTPUT_COLUMN].astype(str).str.strip() != "")
    ).sum()
    logger.info(f"Total processed: {total_processed} out of {len(df)} H1 tags")
    logger.info(f"New in this session: {processed_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert marketing-heavy H1 headings into clean, natural search queries using the Anthropic API."
    )
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--output", required=True, help="Path to the output CSV file (also used to resume)")
    parser.add_argument(
        "--column",
        default="H1-1",
        help="Name of the column containing H1 text (default: H1-1, as exported by Screaming Frog)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Anthropic model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=10,
        help="Save progress every N processed rows (default: 10)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay in seconds between API requests (default: 0.1)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    try:
        process_csv(
            client,
            input_path=args.input,
            output_path=args.output,
            column=args.column,
            model=args.model,
            save_frequency=args.save_frequency,
            delay=args.delay,
        )
        print(f"\nProcessing complete! Check the output file: {args.output}")
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
