#!/usr/bin/env python3
"""
SERP Appearance Report - CLI Version

Parse ValueSERP batch JSON output and report every organic appearance of a domain.

Usage:
    python serp_appearance_cli.py --input results.json --domain example.com
    python serp_appearance_cli.py --input ./batch_exports/ --domain example.com --output report.csv

Author: Lee Foot
Website: https://www.leefoot.com
"""

import argparse
import sys
from pathlib import Path

from serp_appearance_core import extract_appearances, read_json_file, results_to_dataframe


def collect_json_files(input_path):
    """Return a list of JSON files from a file or directory path."""
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob('*.json'))
        if not files:
            print(f"Error: No .json files found in directory: {path}")
            sys.exit(1)
        return files
    print(f"Error: Input path does not exist: {path}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Parse ValueSERP batch JSON output and report organic appearances of a domain'
    )
    parser.add_argument('--input', required=True,
                        help='Path to a ValueSERP batch JSON file, or a directory of JSON files')
    parser.add_argument('--domain', required=True,
                        help='Domain to filter organic results by, e.g. example.com')
    parser.add_argument('--output', default='serp_appearance_report.csv',
                        help='Output CSV path (default: serp_appearance_report.csv)')

    args = parser.parse_args()

    json_files = collect_json_files(args.input)

    all_results = []
    for json_file in json_files:
        print(f"Loading data from: {json_file}")
        try:
            data = read_json_file(json_file)
        except ValueError as e:
            print(f"Error reading file: {e}")
            sys.exit(1)

        results, warnings = extract_appearances(data, args.domain)
        for warning in warnings:
            print(f"Warning: {warning}")
        all_results.extend(results)

    df = results_to_dataframe(all_results)

    print(f"\nFound {len(df)} appearances of '{args.domain}' "
          f"across {len(json_files)} file(s).")
    if not df.empty:
        print(df.to_string(index=False, max_colwidth=60))

    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\nCSV file saved to: {args.output}")


if __name__ == '__main__':
    main()
