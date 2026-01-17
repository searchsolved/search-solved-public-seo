#!/usr/bin/env python3
"""
Backlink Intersector - CLI Version

Find link building opportunities by intersecting competitor backlink profiles.

Usage:
    python backlink_intersector_cli.py --yours backlinks.csv --competitors comp1.csv comp2.csv

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd
import re
import sys


def extract_domain(url):
    if pd.isna(url):
        return None
    match = re.search(r'(?:https?://)?(?:www\.)?([^/]+)', str(url))
    return match.group(1) if match else url


def main():
    parser = argparse.ArgumentParser(description='Find backlink intersection opportunities')
    parser.add_argument('--yours', required=True, help='Your backlinks CSV')
    parser.add_argument('--competitors', nargs='+', required=True, help='Competitor backlink CSVs')
    parser.add_argument('--output', default='backlink_opportunities.csv', help='Output CSV path')
    parser.add_argument('--ref-col', default='Referring Page URL', help='Referring page column name')
    parser.add_argument('--traffic-col', default='Traffic', help='Traffic column name')
    parser.add_argument('--min-traffic', type=int, default=10, help='Minimum referring page traffic')
    parser.add_argument('--min-competitors', type=int, default=2, help='Minimum competitors linking')

    args = parser.parse_args()

    print(f"Loading your backlinks from: {args.yours}")
    try:
        df_yours = pd.read_csv(args.yours, encoding='utf-8')
    except:
        df_yours = pd.read_csv(args.yours, encoding='latin-1')

    print(f"  Loaded {len(df_yours):,} backlinks")

    # Load competitor backlinks
    competitor_dfs = []
    for f in args.competitors:
        print(f"Loading competitor backlinks from: {f}")
        try:
            df = pd.read_csv(f, encoding='utf-8')
        except:
            df = pd.read_csv(f, encoding='latin-1')
        df['_competitor'] = f
        competitor_dfs.append(df)
        print(f"  Loaded {len(df):,} backlinks")

    df_competitors = pd.concat(competitor_dfs, ignore_index=True)

    # Find column names
    your_cols = df_yours.columns.tolist()
    comp_cols = df_competitors.columns.tolist()

    # Find referring page column
    ref_col_yours = next((c for c in your_cols if 'referring' in c.lower() and 'page' in c.lower()), your_cols[0])
    ref_col_comp = next((c for c in comp_cols if 'referring' in c.lower() and 'page' in c.lower()), comp_cols[0])
    traffic_col = next((c for c in comp_cols if 'traffic' in c.lower()), None)

    # Normalize columns
    df_yours = df_yours.rename(columns={ref_col_yours: "referring_page"})
    rename_map = {ref_col_comp: "referring_page"}
    if traffic_col:
        rename_map[traffic_col] = "traffic"
    df_competitors = df_competitors.rename(columns=rename_map)

    # Ensure traffic column exists
    if "traffic" not in df_competitors.columns:
        df_competitors["traffic"] = 0
    df_competitors["traffic"] = pd.to_numeric(df_competitors["traffic"], errors='coerce').fillna(0)

    # Filter by traffic
    df_competitors = df_competitors[df_competitors["traffic"] >= args.min_traffic]

    # Get your referring pages
    your_links = set(df_yours["referring_page"].str.lower().unique())

    # Find opportunities (links you don't have)
    df_competitors["already_have"] = df_competitors["referring_page"].str.lower().isin(your_links)
    df_opportunities = df_competitors[~df_competitors["already_have"]].copy()

    # Count competitors per referring page
    df_opportunities["competitor_count"] = df_opportunities.groupby("referring_page")["referring_page"].transform("count")
    df_opportunities = df_opportunities[df_opportunities["competitor_count"] >= args.min_competitors]

    # Aggregate
    df_grouped = df_opportunities.groupby("referring_page").agg({
        "traffic": "mean",
        "competitor_count": "first",
        "_competitor": lambda x: ", ".join(x.unique())
    }).reset_index()

    df_grouped = df_grouped.rename(columns={
        "_competitor": "competitors_with_links",
        "traffic": "avg_traffic"
    })

    df_grouped = df_grouped.sort_values(["competitor_count", "avg_traffic"], ascending=[False, False])
    df_grouped.to_csv(args.output, index=False, encoding='utf-8-sig')

    print(f"\nResults saved to: {args.output}")
    print(f"  Total opportunities: {len(df_grouped):,}")
    print(f"  Links to 2+ competitors: {(df_grouped['competitor_count'] >= 2).sum():,}")
    print(f"  Links to 3+ competitors: {(df_grouped['competitor_count'] >= 3).sum():,}")

    # Top opportunities
    print(f"\nTop 5 Opportunities:")
    for _, row in df_grouped.head(5).iterrows():
        print(f"  [{int(row['competitor_count'])} competitors] {row['referring_page'][:60]}...")


if __name__ == '__main__':
    main()
