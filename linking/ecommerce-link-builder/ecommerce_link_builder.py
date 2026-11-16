# Author: Lee Foot
# Website: https://leefoot.com

"""Brand Stockists SERP Scraper using DataForSEO API

Searches Google for "<brand> Stockists" for every brand in brands.txt and keeps the
position 1 organic result (homepages excluded), to find 'where to buy' pages.

Searches in the DataForSEO Standard queue can take 25 minutes or more to finish, so the
script works in two steps. `send` queues the searches and writes a job file; `collect`
reads the job file, fetches the finished searches and writes the results CSV. Add --wait
to `collect` to keep checking until every search is done.

Usage:
    python ecommerce_link_builder.py send
    python ecommerce_link_builder.py collect
    python ecommerce_link_builder.py collect --wait
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from dataforseo_queue import (
    QueueError,
    collect_results,
    estimate_cost,
    is_unresolved,
    job_from_csv,
    job_to_csv,
    organic_items,
    send_searches,
)

# Constants - DataForSEO credentials
# Set these here or use environment variables DATAFORSEO_LOGIN / DATAFORSEO_PASSWORD
DATAFORSEO_LOGIN = "your-login-here"
DATAFORSEO_PASSWORD = "your-password-here"

# Search settings, used when the searches are sent
# See https://docs.dataforseo.com/v3/serp/google/locations/ for all codes
LOCATION_CODE = 2826       # 2826 = United Kingdom, 2840 = United States
LANGUAGE_CODE = "en"
DEVICE = "desktop"
SERP_DEPTH = 10

# Analysis settings, used when the results are collected
STOCKISTS_SUFFIX = " Stockists"
TARGET_POSITION = 1
HOMEPAGE_DEPTH = 2
OUTPUT_FILENAME = "brand_links_output.csv"
BRANDS_FILENAME = "brands.txt"
JOB_FILENAME = "brand_links_job.csv"

WAIT_CHECK_SECONDS = 60
WAIT_LIMIT_MINUTES = 90


def read_file(filepath: Path) -> str:
    """Read content from a file."""
    try:
        return filepath.read_text(encoding='utf-8').strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Required file not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error reading file {filepath}: {e}")


def get_credentials() -> Tuple[str, str]:
    """Return DataForSEO login and password from env vars or constants."""
    login = os.environ.get('DATAFORSEO_LOGIN', DATAFORSEO_LOGIN)
    password = os.environ.get('DATAFORSEO_PASSWORD', DATAFORSEO_PASSWORD)
    if login == "your-login-here" or password == "your-password-here":
        print("Warning: Using placeholder credentials. Set DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD "
              "environment variables or update the constants at the top of the script.")
    return login, password


def load_brands(base_path: Path) -> List[str]:
    """Load brand names from file and return as list."""
    content = read_file(base_path / BRANDS_FILENAME)
    return [line.strip() for line in content.splitlines() if line.strip()]


def create_search_queries(brands: List[str]) -> List[str]:
    """Create search queries by appending 'Stockists' to each brand."""
    return [f"{brand}{STOCKISTS_SUFFIX}" for brand in brands]


def extract_organic_result(result: Dict, query: str, position: int) -> Optional[Dict[str, str]]:
    """Extract data from a single organic search result."""
    try:
        return {
            'query': query,
            'url': result.get('url') or 'MISSING',
            'title': result.get('title') or 'MISSING',
            'description': result.get('description') or 'MISSING',
            'position': position
        }
    except Exception as e:
        print(f"Error extracting result data: {e}")
        return None


def process_search_results(items: List[Dict], query: str) -> Tuple[List[Dict[str, str]], int]:
    """Extract organic data from one search's SERP items.

    Returns (results, unresolved). Organic results that came back as unresolved
    google.com/goto links keep their position but are left out and counted.
    """
    results = []
    unresolved = 0

    position = 0
    for item in organic_items(items):
        position += 1
        if is_unresolved(item.get('url')):
            unresolved += 1
            continue
        extracted_result = extract_organic_result(item, query, position)
        if extracted_result:
            results.append(extracted_result)

    return results, unresolved


def calculate_url_depth(url: str) -> int:
    """Calculate the depth of a URL by counting forward slashes."""
    if url == 'MISSING':
        return 0

    # Normalise URL by removing trailing slashes
    normalized_url = url.rstrip('/')
    return normalized_url.count('/')


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and filter the dataframe according to business rules."""
    # Filter to position 1 results only
    df = df[df['position'] == TARGET_POSITION].copy()

    # Remove rows with missing data
    df = df[~df['url'].isin(['MISSING'])]
    df = df[~df['description'].isin(['MISSING'])]
    df = df[~df['title'].isin(['MISSING'])]

    # Calculate URL depth and remove homepages
    df['url_depth'] = df['url'].apply(calculate_url_depth)
    df = df[df['url_depth'] != HOMEPAGE_DEPTH]

    # Select final columns and remove duplicates
    final_columns = ['query', 'url', 'title', 'description']
    df = df[final_columns].copy()
    df.drop_duplicates(subset=['url'], keep='first', inplace=True)

    return df


def build_results(ready: Dict[str, List[Dict]]) -> Tuple[pd.DataFrame, int, int]:
    """Turn collected searches into the cleaned results table.

    Returns (df, unresolved, unresolved_top), where unresolved counts every organic
    google.com/goto link and unresolved_top those at the target position.
    """
    all_results = []
    unresolved = 0
    unresolved_top = 0
    for query, items in ready.items():
        results, skipped = process_search_results(items, query)
        all_results.extend(results)
        unresolved += skipped
        organic = organic_items(items)
        if len(organic) >= TARGET_POSITION and is_unresolved(organic[TARGET_POSITION - 1].get('url')):
            unresolved_top += 1

    if not all_results:
        return pd.DataFrame(columns=['query', 'url', 'title', 'description']), unresolved, unresolved_top
    return clean_dataframe(pd.DataFrame(all_results)), unresolved, unresolved_top


def save_results(df: pd.DataFrame, output_path: Path) -> None:
    """Save results to CSV file with UTF-8-BOM encoding."""
    try:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Results saved to: {output_path}")
        print(f"Total records: {len(df)}")
        print("Encoding: UTF-8 with BOM (utf-8-sig)")
    except Exception as e:
        print(f"Error saving results: {e}")


def send(args) -> None:
    """Step 1: queue one search per brand and write the job file."""
    base_path = Path.cwd()
    print(f"Working directory: {base_path}")
    print(f"Using DataForSEO API (location code: {LOCATION_CODE}, language: {LANGUAGE_CODE})")

    login, password = get_credentials()
    try:
        brands = load_brands(base_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    search_queries = create_search_queries(brands)
    if not search_queries:
        print(f"Error: No brands found in {BRANDS_FILENAME}.")
        sys.exit(1)

    print(f"Brand queries: {len(search_queries)}")
    print(f"Estimated DataForSEO cost: ${estimate_cost(len(search_queries), SERP_DEPTH):.4f}")

    if not args.yes:
        if not sys.stdin.isatty():
            print("Run with --yes to confirm the cost when not running interactively.")
            sys.exit(1)
        if input("Send the searches? [y/N] ").strip().lower() not in ('y', 'yes'):
            print("Cancelled.")
            sys.exit(0)

    try:
        job_rows, warnings, cost = send_searches(search_queries, login, password, LOCATION_CODE, LANGUAGE_CODE,
                                                 device=DEVICE, depth=SERP_DEPTH)
    except QueueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    for warning in warnings:
        print(f"Warning: {warning}")
    if not job_rows:
        print("No searches were queued.")
        sys.exit(1)

    Path(args.job).write_text(job_to_csv(job_rows), encoding='utf-8')
    print(f"\nQueued {len(job_rows)} search(es). Cost reported by DataForSEO: ${cost:.4f}")
    print(f"Job file saved to: {args.job}")
    print("DataForSEO can take 25 minutes or more to finish. Then run:")
    print(f"  python {Path(__file__).name} collect --job {args.job}")


def collect(args) -> None:
    """Step 2: fetch the finished searches for a job and write the results CSV."""
    login, password = get_credentials()

    job_path = Path(args.job)
    if not job_path.is_file():
        print(f"Error: Job file does not exist: {job_path}")
        sys.exit(1)
    try:
        job_rows = job_from_csv(job_path.read_text(encoding='utf-8'))
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    started = time.time()
    while True:
        try:
            result = collect_results(job_rows, login, password)
        except QueueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        print(f"{len(result['ready'])} of {len(job_rows)} search(es) finished, "
              f"{len(result['pending'])} still running, {len(result['errors'])} failed.")
        if not result['pending'] or not args.wait:
            break
        if time.time() - started > WAIT_LIMIT_MINUTES * 60:
            print(f"Stopped waiting after {WAIT_LIMIT_MINUTES} minutes. Run collect again later.")
            break
        time.sleep(WAIT_CHECK_SECONDS)

    for error in result['errors']:
        print(f"Warning: {error}")
    if result['pending']:
        print(f"{len(result['pending'])} search(es) are still running, so the results below leave them out. "
              "Run collect again later for the full results.")
    if not result['ready']:
        print("No valid results to save.")
        return

    df, unresolved, unresolved_top = build_results(result['ready'])
    if unresolved:
        print(f"Warning: {unresolved} organic result(s) came back as unresolved google.com/goto links "
              f"and were left out ({unresolved_top} of them at position {TARGET_POSITION}, "
              "so those brands have no result).")

    if df.empty:
        print("No valid results to save.")
        return
    save_results(df, Path(args.output))


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Find 'where to buy' pages: the top Google result for '<brand> Stockists' "
                    f"for every brand in {BRANDS_FILENAME}"
    )
    sub = parser.add_subparsers(dest='command', required=True)

    p_send = sub.add_parser('send', help=f'Queue one search per brand in {BRANDS_FILENAME} and write a job file')
    p_send.add_argument('--job', default=JOB_FILENAME, help=f'Job file to write (default: {JOB_FILENAME})')
    p_send.add_argument('--yes', action='store_true', help='Skip the cost confirmation prompt')
    p_send.set_defaults(func=send)

    p_collect = sub.add_parser('collect', help='Collect finished searches and write the results CSV')
    p_collect.add_argument('--job', default=JOB_FILENAME, help=f'Job file written by send (default: {JOB_FILENAME})')
    p_collect.add_argument('--output', default=OUTPUT_FILENAME,
                           help=f'Output CSV path (default: {OUTPUT_FILENAME})')
    p_collect.add_argument('--wait', action='store_true',
                           help=f'Check every {WAIT_CHECK_SECONDS}s until all searches finish '
                                f'(up to {WAIT_LIMIT_MINUTES} minutes)')
    p_collect.set_defaults(func=collect)

    args = parser.parse_args()
    start_time = time.time()
    try:
        args.func(args)
    except Exception as e:
        print(f"Script failed: {e}")
        return

    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
