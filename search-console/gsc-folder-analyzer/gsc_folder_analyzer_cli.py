#!/usr/bin/env python3
"""
GSC Folder Analyzer - CLI Version

Aggregate GSC data by URL folder/path structure.

Usage:
    python gsc_folder_analyzer_cli.py --input gsc_data.csv --domain https://example.com/

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd
import sys


def main():
    parser = argparse.ArgumentParser(description='Analyze GSC data by folder structure')
    parser.add_argument('--input', required=True, help='Input GSC CSV file')
    parser.add_argument('--output', default='folder_analysis.csv', help='Output CSV path')
    parser.add_argument('--domain', required=True, help='Domain with trailing slash (e.g., https://example.com/)')
    parser.add_argument('--max-depth', type=int, default=5, help='Maximum folder depth to analyze')

    args = parser.parse_args()

    print(f"Loading GSC data from: {args.input}")
    df = pd.read_csv(args.input, dtype=str)

    # Standardize columns
    column_mapping = {
        'Top queries': 'query', 'Query': 'query', 'Queries': 'query',
        'Top pages': 'page', 'Page': 'page', 'Pages': 'page', 'URL': 'page',
        'Clicks': 'clicks', 'Impressions': 'impressions',
        'Position': 'position', 'Average position': 'position'
    }
    df.rename(columns=column_mapping, inplace=True)
    df.columns = df.columns.str.lower()

    required = ['query', 'page', 'clicks', 'impressions', 'position']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Error: Missing columns: {', '.join(missing)}")
        sys.exit(1)

    df['clicks'] = pd.to_numeric(df['clicks'], errors='coerce').fillna(0).astype(int)
    df['impressions'] = pd.to_numeric(df['impressions'], errors='coerce').fillna(0).astype(int)
    df['position'] = pd.to_numeric(df['position'], errors='coerce').fillna(0)

    print(f"  Loaded {len(df):,} rows")

    # Find top keyword per page
    df['clicks_max'] = df.groupby('page')['clicks'].transform('max')
    df.sort_values(['page', 'clicks_max'], ascending=[True, False], inplace=True)
    df['exact_clicks_match'] = df['clicks_max'] == df['clicks']

    df.loc[df['exact_clicks_match'], 'Top Keyword'] = df['query']
    df.loc[df['exact_clicks_match'], 'Volume'] = df['impressions']
    df.loc[df['exact_clicks_match'], 'Top Position'] = df['position']

    df = df.sort_values('page')
    df['Top Keyword'] = df.groupby('page')['Top Keyword'].ffill()
    df['Volume'] = df.groupby('page')['Volume'].ffill()
    df['Top Position'] = df.groupby('page')['Top Position'].ffill()

    # Clean page URLs
    domain = args.domain
    df['page'] = df['page'].str.replace(domain, "", regex=False)
    df['page'] = df['page'].str.split("?").str[0]
    df['page'] = df['page'].str.split("#").str[0]
    df.loc[df['page'] == "/", "page"] = domain
    df.loc[df['page'] == "", "page"] = domain
    df['page'] = df['page'].str.rstrip("/")

    # Calculate folder depth
    df["folder_depth"] = df["page"].str.count("/")
    actual_max_depth = min(df["folder_depth"].max() + 1, args.max_depth)
    cols = list(range(0, actual_max_depth))

    # Split path into columns
    df[cols] = df['page'].str.split('/', expand=True).iloc[:, :actual_max_depth]

    # Build cumulative paths
    for column in cols:
        n1 = column + 1
        if n1 in cols:
            try:
                df[n1] = df[column].astype(str) + "/" + df[n1].astype(str)
            except (ValueError, KeyError):
                pass

    # Aggregate by folder
    df_raw = df.drop_duplicates(subset=["page"]).copy()
    df_list = []
    df.sort_values(["clicks", "Volume"], ascending=[True, False], inplace=True)

    for i in cols:
        if i in df.columns:
            df_loop = df.groupby(i).agg({
                "clicks": "sum",
                "query": "count",
                "impressions": "sum",
                "Top Keyword": "first",
                "Volume": "first",
                "Top Position": "first"
            })
            df_list.append(df_loop)

    if not df_list:
        print("Error: No folder data could be extracted")
        sys.exit(1)

    df_final = pd.concat(df_list).reset_index()
    df_final.rename(columns={"index": "Path", "clicks": "Traffic", "query": "Keywords"}, inplace=True)

    # Add domain prefix
    df_final['Path'] = domain.rstrip('/') + "/" + df_final['Path'].astype(str)

    # Count pages
    count_list = []
    for path in df_final['Path']:
        search_path = path.replace(domain, "")
        temp = df_raw[df_raw["page"].str.contains(search_path, na=False, regex=False)]
        count_list.append(len(temp))
    df_final['Pages'] = count_list

    # Format output
    output_cols = ["Traffic", "Keywords", "Pages", "Path", "Top Keyword", "Volume", "Top Position"]
    df_final = df_final.reindex(columns=[c for c in output_cols if c in df_final.columns])
    if 'Top Position' in df_final.columns:
        df_final['Top Position'] = df_final['Top Position'].round(2)
    df_final = df_final.sort_values("Traffic", ascending=False)

    df_final.to_csv(args.output, index=False, encoding='utf-8-sig')

    print(f"\nResults saved to: {args.output}")
    print(f"  Total traffic: {df_final['Traffic'].sum():,}")
    print(f"  Unique paths: {len(df_final):,}")


if __name__ == '__main__':
    main()
