#!/usr/bin/env python3
"""
Title Keyword Gap Finder - CLI Version

Compares GSC keywords against page titles to find missing opportunities.

Usage:
    python title_keyword_gap_cli.py --crawl crawl.csv --gsc gsc_data.csv

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd
import sys

try:
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


def load_csv(filepath):
    """Load CSV with encoding fallback."""
    try:
        return pd.read_csv(filepath, encoding='utf-8')
    except:
        return pd.read_csv(filepath, encoding='latin-1')


def find_column(df, possible_names):
    """Find a column by trying multiple possible names."""
    for name in possible_names:
        for col in df.columns:
            if name.lower() == col.lower() or name.lower() in col.lower():
                return col
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Compare GSC keywords vs page titles to find gaps'
    )
    parser.add_argument('--crawl', required=True, help='Path to Screaming Frog crawl CSV')
    parser.add_argument('--gsc', required=True, help='Path to GSC query data CSV')
    parser.add_argument('--output', default='title_keyword_gaps.csv',
                        help='Output file path (default: title_keyword_gaps.csv)')
    parser.add_argument('--delimiter', default='|',
                        help='Title delimiter for splitting (default: |)')
    parser.add_argument('--brand', default='',
                        help='Brand terms to exclude (comma-separated)')
    parser.add_argument('--url-filter', default='',
                        help='Only analyze URLs containing this text')
    parser.add_argument('--max-keywords', type=int, default=10,
                        help='Max keywords per page (default: 10)')
    parser.add_argument('--min-impressions', type=int, default=0,
                        help='Minimum impressions threshold (default: 0)')
    parser.add_argument('--excel', action='store_true',
                        help='Export as Excel with highlighting')

    args = parser.parse_args()

    # Load crawl data
    print(f"Loading crawl: {args.crawl}")
    df_crawl = load_csv(args.crawl)
    print(f"  Loaded {len(df_crawl):,} URLs")

    # Load GSC data
    print(f"Loading GSC data: {args.gsc}")
    df_gsc = load_csv(args.gsc)
    print(f"  Loaded {len(df_gsc):,} queries")

    # Find columns in crawl
    address_col = find_column(df_crawl, ['address', 'url'])
    title_col = find_column(df_crawl, ['title 1', 'title', 'page title'])

    # Find columns in GSC
    page_col = find_column(df_gsc, ['page', 'landing page', 'url'])
    query_col = find_column(df_gsc, ['query', 'keyword', 'top queries'])
    clicks_col = find_column(df_gsc, ['clicks', 'click'])
    impressions_col = find_column(df_gsc, ['impressions', 'impression'])

    if not all([address_col, title_col, page_col, query_col, clicks_col, impressions_col]):
        print("Error: Could not find all required columns")
        print(f"  Crawl: address={address_col}, title={title_col}")
        print(f"  GSC: page={page_col}, query={query_col}, clicks={clicks_col}, impressions={impressions_col}")
        sys.exit(1)

    print(f"  Using columns:")
    print(f"    Crawl: address={address_col}, title={title_col}")
    print(f"    GSC: page={page_col}, query={query_col}")

    # Prepare crawl data
    df_titles = df_crawl[[address_col, title_col]].copy()
    df_titles.columns = ['page', 'title']
    df_titles = df_titles.dropna(subset=['title'])

    # Apply URL filter
    if args.url_filter:
        df_titles = df_titles[df_titles['page'].str.contains(args.url_filter, na=False)]
        print(f"  Filtered to {len(df_titles):,} URLs matching '{args.url_filter}'")

    # Prepare GSC data
    df_queries = df_gsc[[page_col, query_col, clicks_col, impressions_col]].copy()
    df_queries.columns = ['page', 'query', 'clicks', 'impressions']

    # Filter by impressions
    if args.min_impressions > 0:
        df_queries = df_queries[df_queries['impressions'] >= args.min_impressions]
        print(f"  Filtered to {len(df_queries):,} queries with >= {args.min_impressions} impressions")

    # Filter by URL
    if args.url_filter:
        df_queries = df_queries[df_queries['page'].str.contains(args.url_filter, na=False)]

    # Filter out brand terms
    if args.brand:
        brand_terms = [b.strip().lower() for b in args.brand.split(',') if b.strip()]
        for term in brand_terms:
            df_queries = df_queries[~df_queries['query'].str.lower().str.contains(term, na=False)]
        print(f"  Filtered out brand terms: {brand_terms}")

    # Sort and limit keywords per page
    df_queries = df_queries.sort_values(['page', 'clicks'], ascending=[True, False])
    df_queries = df_queries.groupby('page').head(args.max_keywords)

    # Merge with titles
    df_merged = pd.merge(df_queries, df_titles, on='page', how='inner')

    if len(df_merged) == 0:
        print("No matching pages found between crawl and GSC data.")
        sys.exit(0)

    print(f"\nAnalyzing {len(df_merged):,} keyword-page combinations...")

    # Check if query is in title
    def check_query_in_title(row):
        query = str(row['query']).strip().lower()
        title = str(row['title']).strip().lower()

        if args.delimiter:
            title_parts = [p.strip() for p in title.split(args.delimiter)]
        else:
            title_parts = [title]

        for part in title_parts:
            if query in part:
                return True
        return False

    df_merged['in_title'] = df_merged.apply(check_query_in_title, axis=1)

    # Calculate totals per page
    df_merged['total_clicks'] = df_merged.groupby('page')['clicks'].transform('sum')
    df_merged['total_impressions'] = df_merged.groupby('page')['impressions'].transform('sum')

    # Sort by potential
    df_merged = df_merged.sort_values(
        by=['total_impressions', 'page', 'clicks'],
        ascending=[False, True, False]
    )

    # Summary stats
    in_title = df_merged['in_title'].sum()
    not_in_title = len(df_merged) - in_title

    print(f"\nResults:")
    print(f"  Pages analyzed: {df_merged['page'].nunique():,}")
    print(f"  Keywords analyzed: {len(df_merged):,}")
    print(f"  Already in title: {in_title:,}")
    print(f"  Missing from title: {not_in_title:,}")

    # Save results
    if args.excel and OPENPYXL_AVAILABLE:
        output_path = args.output.replace('.csv', '.xlsx') if args.output.endswith('.csv') else args.output + '.xlsx'

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_merged.to_excel(writer, index=False, sheet_name='Analysis')

            ws = writer.sheets['Analysis']
            green_fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")
            yellow_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")

            # Find in_title column
            in_title_col = None
            for idx, cell in enumerate(ws[1], 1):
                if cell.value == 'in_title':
                    in_title_col = idx
                    break

            if in_title_col:
                for row_idx in range(2, ws.max_row + 1):
                    cell_value = ws.cell(row=row_idx, column=in_title_col).value
                    fill = green_fill if cell_value else yellow_fill
                    for col_idx in range(1, ws.max_column + 1):
                        ws.cell(row=row_idx, column=col_idx).fill = fill

        print(f"\nExcel saved to: {output_path}")
    else:
        df_merged.to_csv(args.output, index=False, encoding='utf-8-sig')
        print(f"\nCSV saved to: {args.output}")

    # Show top gaps
    df_gaps = df_merged[df_merged['in_title'] == False].head(10)
    if len(df_gaps) > 0:
        print("\nTop keyword gaps (high impressions, missing from title):")
        for _, row in df_gaps.iterrows():
            print(f"  [{row['impressions']:,} impr] '{row['query']}' - {row['page'][:60]}")


if __name__ == '__main__':
    main()
