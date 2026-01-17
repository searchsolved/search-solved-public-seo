#!/usr/bin/env python3
"""
Category Page Title Suggester - CLI Version

Analyze category pages and suggest optimal title keywords.

Usage:
    python category_title_suggester_cli.py --crawl crawl.csv --gsc gsc_data.csv

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='Analyze category pages and suggest optimal title keywords'
    )
    parser.add_argument('--crawl', required=True,
                        help='Crawl CSV with URL and Title columns')
    parser.add_argument('--gsc', required=True,
                        help='GSC data CSV with page, query, clicks columns')
    parser.add_argument('--output', default='title_suggestions.csv',
                        help='Output CSV path (default: title_suggestions.csv)')
    parser.add_argument('--url-filter', default='/category/',
                        help='URL filter string (default: /category/)')
    parser.add_argument('--delimiter', default='|',
                        help='Title delimiter (default: |)')
    parser.add_argument('--brand', default='',
                        help='Brand name to exclude')
    parser.add_argument('--max-suggestions', type=int, default=10,
                        help='Max suggestions per page (default: 10)')
    parser.add_argument('--min-clicks', type=int, default=1,
                        help='Minimum clicks threshold (default: 1)')
    parser.add_argument('--url-col', default='Address',
                        help='URL column in crawl CSV')
    parser.add_argument('--title-col', default='Title 1',
                        help='Title column in crawl CSV')
    parser.add_argument('--gsc-page-col', default='page',
                        help='Page column in GSC CSV')
    parser.add_argument('--gsc-query-col', default='query',
                        help='Query column in GSC CSV')
    parser.add_argument('--gsc-clicks-col', default='clicks',
                        help='Clicks column in GSC CSV')

    args = parser.parse_args()

    # Load crawl data
    print(f"Loading crawl data from: {args.crawl}")
    crawl_df = pd.read_csv(args.crawl)

    # Find columns
    url_col = None
    title_col = None
    for col in crawl_df.columns:
        if col.lower() == args.url_col.lower() or 'address' in col.lower():
            url_col = col
        if col.lower() == args.title_col.lower() or 'title' in col.lower():
            title_col = col

    if not url_col:
        url_col = crawl_df.columns[0]
    if not title_col:
        title_col = crawl_df.columns[1]

    print(f"  Using columns: URL={url_col}, Title={title_col}")

    # Load GSC data
    print(f"\nLoading GSC data from: {args.gsc}")
    gsc_df = pd.read_csv(args.gsc)

    gsc_page_col = None
    gsc_query_col = None
    gsc_clicks_col = None
    for col in gsc_df.columns:
        if col.lower() == args.gsc_page_col.lower() or 'page' in col.lower():
            gsc_page_col = col
        if col.lower() == args.gsc_query_col.lower() or 'query' in col.lower():
            gsc_query_col = col
        if col.lower() == args.gsc_clicks_col.lower() or 'click' in col.lower():
            gsc_clicks_col = col

    print(f"  Using columns: page={gsc_page_col}, query={gsc_query_col}, clicks={gsc_clicks_col}")

    # Filter crawl to category pages
    df_pages = crawl_df[[url_col, title_col]].copy()
    df_pages = df_pages.rename(columns={url_col: 'page', title_col: 'title'})

    if args.url_filter:
        df_pages = df_pages[df_pages['page'].str.contains(args.url_filter, na=False)]

    print(f"\n  Found {len(df_pages)} pages matching URL filter")

    df_pages = df_pages[df_pages['title'].notna()]
    df_pages = df_pages[df_pages['page'].notna()]

    # Extract keywords from titles
    title_keywords = set()
    for title in df_pages['title']:
        parts = str(title).split(args.delimiter)
        for part in parts:
            kw = part.strip().lower()
            if kw and (not args.brand or args.brand.lower() not in kw):
                title_keywords.add(kw)

    print(f"  Extracted {len(title_keywords)} unique keywords from titles")

    # Prepare GSC data
    df_gsc = gsc_df[[gsc_page_col, gsc_query_col, gsc_clicks_col]].copy()
    df_gsc = df_gsc.rename(columns={
        gsc_page_col: 'page',
        gsc_query_col: 'query',
        gsc_clicks_col: 'clicks'
    })

    if args.url_filter:
        df_gsc = df_gsc[df_gsc['page'].str.contains(args.url_filter, na=False)]

    df_gsc['query'] = df_gsc['query'].str.lower()

    if args.brand:
        df_gsc = df_gsc[~df_gsc['query'].str.contains(args.brand.lower(), na=False)]

    df_gsc = df_gsc[df_gsc['clicks'] >= args.min_clicks]

    # Find suggestions: high clicks but not in title
    df_suggestions = df_gsc[~df_gsc['query'].isin(title_keywords)].copy()

    # Sort and limit
    df_suggestions = df_suggestions.sort_values(['page', 'clicks'], ascending=[True, False])
    df_suggestions = df_suggestions.groupby('page').head(args.max_suggestions)

    # Add current title
    page_to_title = dict(zip(df_pages['page'], df_pages['title']))
    df_suggestions['current_title'] = df_suggestions['page'].map(page_to_title)

    # Save
    df_suggestions.to_csv(args.output, index=False, encoding='utf-8-sig')

    print(f"\nResults saved to: {args.output}")
    print(f"  Pages with suggestions: {df_suggestions['page'].nunique()}")
    print(f"  Total suggestions: {len(df_suggestions)}")
    print(f"  Total potential clicks: {df_suggestions['clicks'].sum():,}")

    # Show top suggestions
    print(f"\nTop suggestions by clicks:")
    for _, row in df_suggestions.nlargest(10, 'clicks').iterrows():
        print(f"  [{row['clicks']:>5,}] {row['query'][:30]} -> {row['page'][:40]}...")


if __name__ == '__main__':
    main()
