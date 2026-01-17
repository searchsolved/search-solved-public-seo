#!/usr/bin/env python3
"""
E-commerce Page Title Optimizer - CLI Version

Analyze page titles against GSC data to find missing keyword opportunities.

Usage:
    python ecom_page_title_optimizer_cli.py --crawl crawl.csv --gsc gsc.csv

Author: Lee Foot
Website: https://www.leefoot.com
"""

import argparse
import pandas as pd
import sys


def main():
    parser = argparse.ArgumentParser(description='Find page title optimization opportunities')
    parser.add_argument('--crawl', required=True, help='Crawl CSV with page titles')
    parser.add_argument('--gsc', required=True, help='GSC keyword CSV')
    parser.add_argument('--output', default='title_optimization.csv', help='Output CSV path')
    parser.add_argument('--delimiter', default='|', help='Title delimiter (e.g., |, -, :)')
    parser.add_argument('--brand', default='', help='Brand name to exclude')
    parser.add_argument('--url-filter', default='', help='URL path filter (e.g., /category/)')
    parser.add_argument('--max-suggestions', type=int, default=10, help='Max suggestions per page')

    args = parser.parse_args()

    print(f"Loading crawl data from: {args.crawl}")
    try:
        df_crawl = pd.read_csv(args.crawl, encoding='utf-8')
    except:
        df_crawl = pd.read_csv(args.crawl, encoding='latin-1')

    print(f"Loading GSC data from: {args.gsc}")
    try:
        df_gsc = pd.read_csv(args.gsc, encoding='utf-8')
    except:
        df_gsc = pd.read_csv(args.gsc, encoding='latin-1')

    # Find columns
    crawl_cols = df_crawl.columns.tolist()
    gsc_cols = df_gsc.columns.tolist()

    url_col = next((c for c in crawl_cols if c.lower() == 'address'), crawl_cols[0])
    title_col = next((c for c in crawl_cols if 'title' in c.lower()), crawl_cols[1] if len(crawl_cols) > 1 else crawl_cols[0])

    query_col = next((c for c in gsc_cols if 'query' in c.lower() or 'queries' in c.lower()), gsc_cols[0])
    page_col = next((c for c in gsc_cols if 'page' in c.lower()), gsc_cols[1] if len(gsc_cols) > 1 else gsc_cols[0])
    clicks_col = next((c for c in gsc_cols if 'click' in c.lower()), None)
    impressions_col = next((c for c in gsc_cols if 'impression' in c.lower()), None)

    print(f"  Crawl: {len(df_crawl):,} pages, GSC: {len(df_gsc):,} keywords")

    # Normalize columns
    df_crawl = df_crawl.rename(columns={url_col: "page", title_col: "title"})
    rename_map = {query_col: "query", page_col: "page"}
    if clicks_col:
        rename_map[clicks_col] = "clicks"
    if impressions_col:
        rename_map[impressions_col] = "impressions"
    df_gsc = df_gsc.rename(columns=rename_map)

    df_crawl = df_crawl[["page", "title"]].copy()
    gsc_columns = ["query", "page"]
    if "clicks" in df_gsc.columns:
        gsc_columns.append("clicks")
    if "impressions" in df_gsc.columns:
        gsc_columns.append("impressions")
    df_gsc = df_gsc[gsc_columns].copy()

    # Clean data
    df_crawl = df_crawl[df_crawl["title"].notna() & df_crawl["page"].notna()]
    df_gsc = df_gsc[df_gsc["query"].notna() & df_gsc["page"].notna()]

    if "clicks" in df_gsc.columns:
        df_gsc["clicks"] = pd.to_numeric(df_gsc["clicks"], errors='coerce').fillna(0)
    else:
        df_gsc["clicks"] = 0

    if "impressions" in df_gsc.columns:
        df_gsc["impressions"] = pd.to_numeric(df_gsc["impressions"], errors='coerce').fillna(0)
    else:
        df_gsc["impressions"] = 0

    # Apply filters
    if args.url_filter:
        df_crawl = df_crawl[df_crawl["page"].str.contains(args.url_filter, na=False)]
        df_gsc = df_gsc[df_gsc["page"].str.contains(args.url_filter, na=False)]

    if args.brand:
        df_gsc = df_gsc[~df_gsc["query"].str.lower().str.contains(args.brand.lower(), na=False)]

    # Extract title keywords
    df_titles = df_crawl.copy()
    df_titles["title_keywords"] = df_titles["title"].str.split(args.delimiter)
    df_titles = df_titles.explode("title_keywords")
    df_titles["title_keywords"] = df_titles["title_keywords"].str.strip().str.lower()
    df_titles = df_titles[df_titles["title_keywords"].notna() & (df_titles["title_keywords"] != "")]

    if args.brand:
        df_titles = df_titles[~df_titles["title_keywords"].str.contains(args.brand.lower(), na=False)]

    title_kw_sets = df_titles.groupby("page")["title_keywords"].apply(set).to_dict()

    # Find missing keywords
    df_gsc["query_lower"] = df_gsc["query"].str.lower()

    def check_in_title(row):
        page = row["page"]
        query = row["query_lower"]
        if page in title_kw_sets:
            query_words = set(query.split())
            title_words = title_kw_sets[page]
            return not bool(query_words & title_words)
        return True

    print("\nFinding missing keywords...")
    df_gsc["missing_from_title"] = df_gsc.apply(check_in_title, axis=1)

    df_missing = df_gsc[df_gsc["missing_from_title"] & (df_gsc["clicks"] > 0)].copy()
    df_missing = df_missing.sort_values("clicks", ascending=False)
    df_suggestions = df_missing.groupby("page").head(args.max_suggestions)

    # Merge with titles
    df_result = df_suggestions.merge(df_crawl[["page", "title"]], on="page", how="left")
    df_result = df_result[[
        "page", "title", "query", "clicks", "impressions"
    ]].sort_values("clicks", ascending=False)

    df_result.to_csv(args.output, index=False, encoding='utf-8-sig')

    print(f"\nResults saved to: {args.output}")
    print(f"  Pages analyzed: {df_crawl['page'].nunique():,}")
    print(f"  Pages with suggestions: {df_result['page'].nunique():,}")
    print(f"  Total suggestions: {len(df_result):,}")
    print(f"  Potential clicks: {int(df_result['clicks'].sum()):,}")

    # Top opportunities
    print(f"\nTop 5 Keyword Opportunities:")
    for _, row in df_result.head(5).iterrows():
        print(f"  [{int(row['clicks'])} clicks] '{row['query']}' -> {row['page'][:50]}...")


if __name__ == '__main__':
    main()
