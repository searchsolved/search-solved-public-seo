#!/usr/bin/env python3
"""
Product Spec Extractor - CLI Version

Scrapes product specifications from e-commerce pages.

Usage:
    python product_spec_extractor_cli.py --urls urls.txt --dt-selector "dt" --dd-selector "dd"

Author: Lee Foot
Website: https://www.leefoot.com
"""

import argparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import sys


def extract_specs(url, dt_selector, dd_selector, parent_selector, headers, timeout):
    """Extract specifications from a single page."""
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        soup = BeautifulSoup(response.text, 'html.parser')

        if parent_selector:
            container = soup.select_one(parent_selector)
            if container:
                soup = container

        specs = {'URL': url}

        dt_tags = soup.select(dt_selector)
        dd_tags = soup.select(dd_selector)

        for j, dt in enumerate(dt_tags):
            key = dt.get_text(strip=True)
            if j < len(dd_tags):
                value = ' '.join(dd_tags[j].stripped_strings)
            else:
                value = ""
            specs[key] = value

        return specs, None

    except Exception as e:
        return {'URL': url, 'Error': str(e)}, str(e)


def main():
    parser = argparse.ArgumentParser(
        description='Scrape product specifications from e-commerce pages'
    )
    parser.add_argument('--urls', required=True,
                        help='Path to file with URLs (one per line) or CSV')
    parser.add_argument('--url-column', default=None,
                        help='Column name if using CSV (auto-detected if not specified)')
    parser.add_argument('--output', default='product_specs.csv',
                        help='Output CSV path (default: product_specs.csv)')
    parser.add_argument('--dt-selector', default='dt',
                        help='CSS selector for spec keys (default: dt)')
    parser.add_argument('--dd-selector', default='dd',
                        help='CSS selector for spec values (default: dd)')
    parser.add_argument('--parent-selector', default='',
                        help='CSS selector for parent container (optional)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between requests in seconds (default: 1.0)')
    parser.add_argument('--timeout', type=int, default=15,
                        help='Request timeout in seconds (default: 15)')
    parser.add_argument('--user-agent', default='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        help='User agent string')

    args = parser.parse_args()

    # Load URLs
    urls = []
    if args.urls.endswith('.csv'):
        df = pd.read_csv(args.urls)
        if args.url_column:
            url_col = args.url_column
        else:
            # Auto-detect URL column
            for col in df.columns:
                if 'url' in col.lower() or 'address' in col.lower():
                    url_col = col
                    break
            else:
                url_col = df.columns[0]
        urls = df[url_col].dropna().tolist()
        print(f"Loaded {len(urls)} URLs from CSV column '{url_col}'")
    else:
        with open(args.urls, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(urls)} URLs from text file")

    if not urls:
        print("Error: No URLs found")
        sys.exit(1)

    headers = {'User-Agent': args.user_agent}
    all_specs = []
    errors = []

    print(f"\nExtracting specs from {len(urls)} URLs...")
    print(f"  Key selector: {args.dt_selector}")
    print(f"  Value selector: {args.dd_selector}")
    if args.parent_selector:
        print(f"  Parent selector: {args.parent_selector}")

    for i, url in enumerate(urls):
        if i % 10 == 0:
            print(f"  Processing {i + 1}/{len(urls)}...")

        specs, error = extract_specs(
            url,
            args.dt_selector,
            args.dd_selector,
            args.parent_selector,
            headers,
            args.timeout
        )

        all_specs.append(specs)
        if error:
            errors.append({'URL': url, 'Error': error})

        time.sleep(args.delay)

    # Create DataFrame
    df_results = pd.DataFrame(all_specs)

    if not df_results.empty:
        # Sort columns by frequency
        col_counts = df_results.notna().sum().sort_values(ascending=False)
        sorted_cols = ['URL'] + [c for c in col_counts.index if c not in ['URL', 'Error']]
        if 'Error' in df_results.columns:
            sorted_cols.append('Error')
        df_results = df_results.reindex(columns=[c for c in sorted_cols if c in df_results.columns])

    # Save results
    df_results.to_csv(args.output, index=False, encoding='utf-8-sig')

    print(f"\nResults saved to: {args.output}")
    print(f"  URLs processed: {len(urls)}")
    print(f"  Successful: {len(urls) - len(errors)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Unique specs found: {len(df_results.columns) - 1}")

    if errors:
        print(f"\nErrors:")
        for err in errors[:5]:
            print(f"  {err['URL'][:50]}: {err['Error']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")


if __name__ == '__main__':
    main()
