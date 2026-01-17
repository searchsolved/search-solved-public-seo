#!/usr/bin/env python3
"""
Striking Distance CSV - CLI Version

Find striking distance keywords from GSC data and check presence in page content.

Usage:
    python striking_distance_csv_cli.py --gsc gsc.csv --crawl crawl.csv --output results.csv

Author: Lee Foot
Website: https://www.leefoot.com
"""

import argparse
import pandas as pd
import re
import sys


def check_keyword_in_text(keyword, text):
    if pd.isna(text) or pd.isna(keyword):
        return False
    try:
        escaped = re.escape(str(keyword).lower())
        return bool(re.search(escaped, str(text).lower()))
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description='Find striking distance keyword opportunities')
    parser.add_argument('--gsc', required=True, help='GSC export CSV')
    parser.add_argument('--crawl', required=True, help='Crawl export CSV (Screaming Frog)')
    parser.add_argument('--output', default='striking_distance.csv', help='Output CSV path')
    parser.add_argument('--min-position', type=int, default=4, help='Minimum position')
    parser.add_argument('--max-position', type=int, default=20, help='Maximum position')
    parser.add_argument('--min-impressions', type=int, default=0, help='Minimum impressions')
    parser.add_argument('--max-keywords', type=int, default=10, help='Keywords per page')
    parser.add_argument('--sort-by', default='clicks', choices=['clicks', 'impressions'], help='Sort metric')
    parser.add_argument('--brand-exclude', nargs='*', default=[], help='Brand terms to exclude')

    args = parser.parse_args()

    print(f"Loading GSC data from: {args.gsc}")
    gsc_df = pd.read_csv(args.gsc, dtype=str)

    # Standardize GSC columns
    gsc_mapping = {
        'Top queries': 'query', 'Query': 'query', 'Queries': 'query',
        'Top pages': 'page', 'Page': 'page', 'URL': 'page',
        'Clicks': 'clicks', 'Impressions': 'impressions',
        'Position': 'position', 'Average position': 'position'
    }
    gsc_df.rename(columns=gsc_mapping, inplace=True)
    gsc_df.columns = gsc_df.columns.str.lower()

    gsc_df['clicks'] = pd.to_numeric(gsc_df['clicks'], errors='coerce').fillna(0).astype(int)
    gsc_df['impressions'] = pd.to_numeric(gsc_df['impressions'], errors='coerce').fillna(0).astype(int)
    gsc_df['position'] = pd.to_numeric(gsc_df['position'], errors='coerce').fillna(0)

    print(f"Loading crawl data from: {args.crawl}")
    crawl_df = pd.read_csv(args.crawl, dtype=str)

    # Find crawl columns
    cols = crawl_df.columns.tolist()
    url_col = 'Address' if 'Address' in cols else cols[0]
    title_col = next((c for c in cols if 'title' in c.lower()), None)
    h1_col = next((c for c in cols if 'h1' in c.lower()), None)

    print(f"  GSC: {len(gsc_df):,} rows, Crawl: {len(crawl_df):,} rows")

    # Filter brand terms
    if args.brand_exclude:
        original = len(gsc_df)
        for term in args.brand_exclude:
            gsc_df = gsc_df[~gsc_df['query'].str.lower().str.contains(term.lower(), na=False)]
        print(f"  Filtered {original - len(gsc_df):,} branded queries")

    # Apply position filter
    gsc_df = gsc_df[
        (gsc_df['position'] >= args.min_position) &
        (gsc_df['position'] <= args.max_position) &
        (gsc_df['impressions'] >= args.min_impressions)
    ]

    if len(gsc_df) == 0:
        print("No keywords found matching filters")
        sys.exit(0)

    print(f"  Found {len(gsc_df):,} striking distance keywords")

    # Get top keywords per page
    top_kw = (
        gsc_df.groupby('page')
        .apply(lambda x: x.nlargest(args.max_keywords, args.sort_by)[['query', args.sort_by, 'position']])
        .reset_index(level=0)
        .reset_index(drop=True)
    )

    # Create page lookup
    page_data = {}
    for _, row in crawl_df.iterrows():
        page = row.get(url_col, '')
        if page:
            page_data[page] = row

    # Check keyword presence
    results = []
    columns_to_check = []
    if title_col:
        columns_to_check.append(('Title', title_col))
    if h1_col:
        columns_to_check.append(('H1', h1_col))

    print(f"\nChecking keyword presence in: {[c[0] for c in columns_to_check]}")

    for idx, row in top_kw.iterrows():
        keyword = row['query']
        page = row['page']
        result = {
            'Page': page,
            'Keyword': keyword,
            f'Total {args.sort_by.capitalize()}': row[args.sort_by],
            'Position': row['position']
        }

        if page in page_data:
            pdata = page_data[page]
            for label, col in columns_to_check:
                result[f'In {label}'] = check_keyword_in_text(keyword, pdata.get(col, ''))
        else:
            for label, _ in columns_to_check:
                result[f'In {label}'] = False

        results.append(result)

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(top_kw)}...")

    df_results = pd.DataFrame(results)

    # Filter out keywords in all locations
    check_cols = [f'In {c[0]}' for c in columns_to_check if f'In {c[0]}' in df_results.columns]
    if check_cols:
        mask = df_results[check_cols].all(axis=1)
        df_results = df_results[~mask]

    df_results = df_results.sort_values('Position')
    df_results.to_csv(args.output, index=False, encoding='utf-8-sig')

    print(f"\nResults saved to: {args.output}")
    print(f"  Pages with opportunities: {df_results['Page'].nunique():,}")
    print(f"  Total keywords: {len(df_results):,}")

    if 'In Title' in df_results.columns:
        missing_title = (~df_results['In Title']).sum()
        print(f"  Missing from title: {missing_title:,}")
    if 'In H1' in df_results.columns:
        missing_h1 = (~df_results['In H1']).sum()
        print(f"  Missing from H1: {missing_h1:,}")


if __name__ == '__main__':
    main()
