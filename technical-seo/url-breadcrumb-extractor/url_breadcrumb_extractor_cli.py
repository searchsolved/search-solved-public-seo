#!/usr/bin/env python3
"""
URL Breadcrumb Extractor - CLI Version

Extract URLs from breadcrumb HTML extracted via Screaming Frog.

Usage:
    python url_breadcrumb_extractor_cli.py --input crawl.csv --output extracted.csv

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd
import re
import sys


def extract_urls(text):
    if pd.isna(text):
        return []
    text = str(text)
    url_pattern = r'https?://[^\s<>"\']+|(?:href=["\'])([^"\']+)["\']'
    urls = re.findall(url_pattern, text, re.IGNORECASE)
    clean_urls = []
    for url in urls:
        if isinstance(url, tuple):
            url = [u for u in url if u][0] if any(url) else ''
        if url and url.startswith(('http', '/')):
            clean_urls.append(url.strip())
    return clean_urls


def main():
    parser = argparse.ArgumentParser(description='Extract URLs from breadcrumb HTML')
    parser.add_argument('--input', required=True, help='Input crawl CSV')
    parser.add_argument('--output', default='breadcrumb_urls.csv', help='Output CSV path')
    parser.add_argument('--url-col', default='Address', help='URL column name')
    parser.add_argument('--breadcrumb-col', help='Breadcrumb column name (auto-detected if not specified)')
    parser.add_argument('--position', default='last', choices=['last', 'first', 'all'],
                        help='Which URL to extract: last (parent), first, or all')
    parser.add_argument('--exclude', nargs='*', default=[], help='URL patterns to exclude')

    args = parser.parse_args()

    print(f"Loading crawl data from: {args.input}")

    try:
        df = pd.read_csv(args.input, encoding='utf-8')
    except:
        df = pd.read_csv(args.input, encoding='latin-1')

    print(f"  Loaded {len(df):,} rows")

    cols = df.columns.tolist()

    # Find URL column
    url_col = args.url_col if args.url_col in cols else cols[0]

    # Find breadcrumb column
    if args.breadcrumb_col and args.breadcrumb_col in cols:
        bc_col = args.breadcrumb_col
    else:
        bc_options = [c for c in cols if 'breadcrumb' in c.lower() or 'extraction' in c.lower()]
        bc_col = bc_options[0] if bc_options else None

    if not bc_col:
        print(f"Error: Could not find breadcrumb column")
        print(f"Available columns: {cols}")
        sys.exit(1)

    print(f"  Using columns: URL='{url_col}', Breadcrumb='{bc_col}'")

    # Extract URLs
    df['extracted_urls'] = df[bc_col].apply(extract_urls)

    # Get desired URL position
    if args.position == 'last':
        df['breadcrumb_url'] = df['extracted_urls'].apply(lambda x: x[-1] if len(x) > 0 else None)
    elif args.position == 'first':
        df['breadcrumb_url'] = df['extracted_urls'].apply(lambda x: x[0] if len(x) > 0 else None)
    else:
        df['breadcrumb_url'] = df['extracted_urls'].apply(lambda x: ' | '.join(x) if x else None)

    # Apply exclusions
    for pattern in args.exclude:
        if pattern:
            df = df[~df[url_col].str.contains(pattern, na=False, case=False)]

    # Remove self-references
    df = df[df[url_col] != df['breadcrumb_url']]

    # Create result
    df_result = df[[url_col, bc_col, 'breadcrumb_url']].copy()
    df_result = df_result[df_result['breadcrumb_url'].notna()]
    df_result.columns = ['Page URL', 'Breadcrumb HTML', 'Extracted URL']

    df_result.to_csv(args.output, index=False, encoding='utf-8-sig')

    print(f"\nResults saved to: {args.output}")
    print(f"  URLs extracted: {len(df_result):,}")
    print(f"  Unique parent URLs: {df_result['Extracted URL'].nunique():,}")

    # Show top parents
    print(f"\nTop 5 Parent Categories:")
    top_parents = df_result['Extracted URL'].value_counts().head(5)
    for url, count in top_parents.items():
        print(f"  {count:,} children: {url[:60]}...")


if __name__ == '__main__':
    main()
