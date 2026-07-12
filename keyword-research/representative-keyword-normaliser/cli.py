"""
Representative Keyword Normaliser - CLI
Read keywords from a CSV and append an LLM-suggested representative keyword for each.

Usage:
    python cli.py --input keywords.csv --output normalised.csv
    python cli.py --input keywords.csv --output normalised.csv --column Keyword \
        --base-url http://127.0.0.1:11434/v1 --model qwen2.5:7b

Author: Lee Foot
Website: https://www.leefoot.com
"""

import argparse
import logging
import os

from tqdm import tqdm

from core import (DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL,
                  build_client, process_csv)


def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='representative_keyword_normaliser.log',
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def main():
    parser = argparse.ArgumentParser(
        description="Suggest a cleaner representative keyword for each keyword in a CSV "
                    "using any OpenAI-compatible endpoint (Ollama by default).")
    parser.add_argument('-i', '--input', required=True, help='Input CSV file path')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file path')
    parser.add_argument('-c', '--column', default=None,
                        help='Name of the keyword column (defaults to the first column)')
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL,
                        help=f'OpenAI-compatible API base URL (default: {DEFAULT_BASE_URL})')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help=f'Model name (default: {DEFAULT_MODEL})')
    parser.add_argument('--api-key', default=os.environ.get('OPENAI_API_KEY', DEFAULT_API_KEY),
                        help='API key (defaults to the OPENAI_API_KEY environment variable; '
                             'not needed for local servers)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase output verbosity')
    args = parser.parse_args()

    setup_logging(args.verbose)
    client = build_client(base_url=args.base_url, api_key=args.api_key)

    progress_bar = None

    def progress_callback(done, total):
        nonlocal progress_bar
        if progress_bar is None:
            progress_bar = tqdm(total=total, desc="Processing keywords")
        progress_bar.update(1)

    try:
        process_csv(args.input, args.output, client, model=args.model,
                    column=args.column, progress_callback=progress_callback)
    except Exception as e:
        logging.error(f"Error processing CSV file: {str(e)}")
    finally:
        if progress_bar is not None:
            progress_bar.close()


if __name__ == "__main__":
    main()
