#!/usr/bin/env python3
"""
SERP Appearance Report - CLI Version

Fetch live Google results for a keyword list from the DataForSEO SERP API
and report every organic appearance of a domain.

Usage:
    python serp_appearance_cli.py --input keywords.txt --domain example.com
    python serp_appearance_cli.py --input keywords.csv --domain example.com --output report.csv \\
        --location "United States" --device mobile --depth 20

The keyword file is either a text file with one keyword per line, or a CSV
with a keyword column (keyword, keywords, query, queries, search term or
top queries; otherwise the first column is used).

Environment variables:
    DATAFORSEO_LOGIN    - DataForSEO account login
    DATAFORSEO_PASSWORD - DataForSEO account password

Author: Lee Foot
Website: https://www.leefoot.com
"""

import argparse
import os
import sys
from pathlib import Path

from serp_appearance_core import (
    DEVICES,
    LOCATIONS,
    estimate_cost,
    normalise_domain,
    read_keywords_file,
    results_to_dataframe,
    run_report,
)


def main():
    parser = argparse.ArgumentParser(
        description='Fetch live Google results for a keyword list from DataForSEO '
                    'and report organic appearances of a domain'
    )
    parser.add_argument('--input', required=True,
                        help='Keyword file: .txt with one keyword per line, or .csv with a keyword column')
    parser.add_argument('--domain', required=True,
                        help='Domain to report, e.g. example.com (subdomains are included)')
    parser.add_argument('--output', default='serp_appearance_report.csv',
                        help='Output CSV path (default: serp_appearance_report.csv)')
    parser.add_argument('--location', default='United Kingdom', choices=list(LOCATIONS.keys()),
                        help='Search location (default: United Kingdom)')
    parser.add_argument('--device', default='desktop', choices=list(DEVICES),
                        help='Device (default: desktop)')
    parser.add_argument('--depth', type=int, default=10,
                        help='Results to fetch per keyword, 10 to 100, billed per 10 (default: 10)')
    parser.add_argument('--delay', type=float, default=2.0,
                        help='Seconds to wait between requests (default: 2.0)')
    parser.add_argument('--yes', action='store_true',
                        help='Skip the cost confirmation prompt')

    args = parser.parse_args()

    login = os.environ.get('DATAFORSEO_LOGIN')
    password = os.environ.get('DATAFORSEO_PASSWORD')
    if not login or not password:
        print("Error: DataForSEO credentials required.")
        print("Set the DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD environment variables.")
        sys.exit(1)

    if not normalise_domain(args.domain):
        print("Error: --domain is empty.")
        sys.exit(1)

    if not 10 <= args.depth <= 100:
        print("Error: --depth must be between 10 and 100.")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: Keyword file does not exist: {input_path}")
        sys.exit(1)

    try:
        keywords = read_keywords_file(input_path)
    except ValueError as e:
        print(f"Error reading keyword file: {e}")
        sys.exit(1)

    if not keywords:
        print("Error: No keywords found in the input file.")
        sys.exit(1)

    location_code, language_code = LOCATIONS[args.location]
    estimated_cost = estimate_cost(len(keywords), args.depth)

    print(f"Keywords: {len(keywords)}")
    print(f"  Domain: {normalise_domain(args.domain)}")
    print(f"  Location: {args.location} (code: {location_code}, language: {language_code})")
    print(f"  Device: {args.device}")
    print(f"  Results per keyword: {args.depth}")
    print(f"  Delay between requests: {args.delay}s")
    print(f"  Estimated cost: ${estimated_cost:.3f}")

    if not args.yes:
        if not sys.stdin.isatty():
            print("Run with --yes to confirm the cost when not running interactively.")
            sys.exit(1)
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer not in ('y', 'yes'):
            print("Cancelled.")
            sys.exit(0)

    def progress(done, total, keyword):
        if keyword:
            print(f"  [{done + 1}/{total}] {keyword}")

    rows, warnings, total_cost = run_report(
        keywords, args.domain, login, password, location_code, language_code,
        device=args.device, depth=args.depth, delay=args.delay, progress_callback=progress
    )

    for warning in warnings:
        print(f"Warning: {warning}")

    df = results_to_dataframe(rows)

    print(f"\nFound {len(df)} appearances of '{normalise_domain(args.domain)}' "
          f"across {len(keywords)} keyword(s).")
    print(f"Cost reported by DataForSEO: ${total_cost:.4f}")
    if not df.empty:
        print(df.to_string(index=False, max_colwidth=60))

    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\nCSV file saved to: {args.output}")


if __name__ == '__main__':
    main()
