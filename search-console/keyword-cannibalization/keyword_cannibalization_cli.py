#!/usr/bin/env python3
"""
Keyword Cannibalization Finder - CLI Version

Find keywords where multiple pages compete for the same query.

Usage:
    python keyword_cannibalization_cli.py --input gsc_data.csv --output cannibalization.csv

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd
import sys


def read_search_console_data(df):
    column_mapping = {
        'Top queries': 'query',
        'Query': 'query',
        'Queries': 'query',
        'Top pages': 'page',
        'Page': 'page',
        'Pages': 'page',
        'URL': 'page',
        'Landing Page': 'page',
        'Clicks': 'clicks',
        'Impressions': 'impressions',
        'CTR': 'ctr',
        'Click Through Rate': 'ctr',
        'Position': 'position',
        'Average position': 'position',
        'Avg. position': 'position'
    }

    df = df.rename(columns=column_mapping)
    df.columns = df.columns.str.lower()

    required = ['query', 'page', 'clicks', 'impressions', 'position']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df['clicks'] = pd.to_numeric(df['clicks'], errors='coerce').fillna(0).astype(int)
    df['impressions'] = pd.to_numeric(df['impressions'], errors='coerce').fillna(0).astype(int)
    df['position'] = pd.to_numeric(df['position'], errors='coerce').fillna(0)

    if 'ctr' not in df.columns:
        df['ctr'] = 0
    elif df['ctr'].dtype == 'object':
        df['ctr'] = df['ctr'].str.rstrip('%').astype(float) / 100
    df['ctr'] = pd.to_numeric(df['ctr'], errors='coerce').fillna(0)

    return df


def identify_cannibalization(df, position_range, min_impressions, min_clicks, min_pages):
    filtered = df[
        (df['position'] >= position_range[0]) &
        (df['position'] <= position_range[1]) &
        (df['impressions'] >= min_impressions) &
        (df['clicks'] >= min_clicks)
    ].copy()

    if len(filtered) == 0:
        return pd.DataFrame()

    cannibalization = filtered.groupby(['query', 'page']).agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'ctr': 'mean',
        'position': 'mean'
    }).reset_index()

    pages_per_query = cannibalization.groupby('query')['page'].transform('count')
    cannibalization['competing_pages'] = pages_per_query

    cannibalized = cannibalization[cannibalization['competing_pages'] >= min_pages].copy()
    cannibalized['position'] = cannibalized['position'].round(1)
    cannibalized['ctr'] = (cannibalized['ctr'] * 100).round(2)

    return cannibalized.sort_values(['competing_pages', 'impressions'], ascending=[False, False])


def main():
    parser = argparse.ArgumentParser(description='Find keyword cannibalization in GSC data')
    parser.add_argument('--input', required=True, help='Input GSC CSV file')
    parser.add_argument('--output', default='cannibalization.csv', help='Output CSV path')
    parser.add_argument('--min-position', type=int, default=1, help='Minimum position')
    parser.add_argument('--max-position', type=int, default=20, help='Maximum position')
    parser.add_argument('--min-impressions', type=int, default=0, help='Minimum impressions')
    parser.add_argument('--min-clicks', type=int, default=0, help='Minimum clicks')
    parser.add_argument('--min-pages', type=int, default=2, help='Minimum competing pages')

    args = parser.parse_args()

    print(f"Loading GSC data from: {args.input}")
    df = pd.read_csv(args.input, dtype=str)
    df = read_search_console_data(df)
    print(f"  Loaded {len(df):,} rows")

    print(f"\nFinding cannibalization (positions {args.min_position}-{args.max_position})...")
    cannibalized = identify_cannibalization(
        df,
        position_range=(args.min_position, args.max_position),
        min_impressions=args.min_impressions,
        min_clicks=args.min_clicks,
        min_pages=args.min_pages
    )

    if len(cannibalized) == 0:
        print("No cannibalization issues found with current filters.")
        sys.exit(0)

    # Save results
    cannibalized.to_csv(args.output, index=False, encoding='utf-8-sig')

    unique_queries = cannibalized['query'].nunique()
    total_pages = len(cannibalized)
    max_competing = cannibalized['competing_pages'].max()

    print(f"\nResults saved to: {args.output}")
    print(f"  Cannibalized queries: {unique_queries:,}")
    print(f"  Total competing pages: {total_pages:,}")
    print(f"  Max competing pages: {max_competing}")

    # Show top issues
    print(f"\nTop 5 Cannibalized Queries:")
    top_queries = cannibalized.groupby('query').agg({
        'competing_pages': 'first',
        'impressions': 'sum'
    }).sort_values('impressions', ascending=False).head(5)

    for query, row in top_queries.iterrows():
        print(f"  [{int(row['competing_pages'])} pages] {query} ({int(row['impressions']):,} impressions)")


if __name__ == '__main__':
    main()
