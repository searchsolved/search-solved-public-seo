#!/usr/bin/env python3
"""
Share of Voice Calculator - CLI Version

Calculate organic traffic share using ranking positions and CTR curves.

Usage:
    python share_of_voice_cli.py --input rankings.csv

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd
import sys

# CTR curve presets
CTR_CURVES = {
    "sistrix": {
        1: 0.2848, 2: 0.1548, 3: 0.1100, 4: 0.0765, 5: 0.0535,
        6: 0.0491, 7: 0.0306, 8: 0.0313, 9: 0.0279, 10: 0.0274
    },
    "awr": {
        1: 0.3100, 2: 0.1560, 3: 0.0990, 4: 0.0700, 5: 0.0512,
        6: 0.0390, 7: 0.0305, 8: 0.0248, 9: 0.0209, 10: 0.0182
    },
    "backlinko": {
        1: 0.2750, 2: 0.1510, 3: 0.1120, 4: 0.0810, 5: 0.0740,
        6: 0.0510, 7: 0.0410, 8: 0.0330, 9: 0.0290, 10: 0.0260
    },
    "conservative": {
        1: 0.2000, 2: 0.1000, 3: 0.0800, 4: 0.0600, 5: 0.0500,
        6: 0.0400, 7: 0.0350, 8: 0.0300, 9: 0.0250, 10: 0.0200
    }
}


def find_column(df, possible_names):
    """Find a column by trying multiple possible names."""
    for name in possible_names:
        for col in df.columns:
            if name.lower() == col.lower() or name.lower() in col.lower():
                return col
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Calculate share of voice from ranking data using CTR curves'
    )
    parser.add_argument('--input', required=True, help='Input CSV with ranking data')
    parser.add_argument('--output', default='share_of_voice.csv',
                        help='Output CSV path (default: share_of_voice.csv)')
    parser.add_argument('--ctr-curve', choices=list(CTR_CURVES.keys()), default='sistrix',
                        help='CTR curve preset (default: sistrix)')
    parser.add_argument('--keyword-col', help='Keyword column name')
    parser.add_argument('--volume-col', help='Search volume column name')
    parser.add_argument('--position-col', help='Position column name')
    parser.add_argument('--domain-col', help='Domain column name')
    parser.add_argument('--category-col', help='Category column name (optional)')
    parser.add_argument('--top-n', type=int, default=20,
                        help='Top N domains to show (default: 20)')

    args = parser.parse_args()

    # Load data
    print(f"Loading: {args.input}")
    try:
        df = pd.read_csv(args.input, encoding='utf-8')
    except:
        df = pd.read_csv(args.input, encoding='latin-1')
    print(f"  Loaded {len(df):,} rows")

    # Find columns
    keyword_col = args.keyword_col or find_column(df, ['keyword', 'query', 'search term'])
    volume_col = args.volume_col or find_column(df, ['volume', 'search volume', 'sv'])
    position_col = args.position_col or find_column(df, ['position', 'rank', 'pos'])
    domain_col = args.domain_col or find_column(df, ['domain', 'url', 'site'])
    category_col = args.category_col or find_column(df, ['category', 'group', 'vertical'])

    if not all([keyword_col, volume_col, position_col, domain_col]):
        print("Error: Could not find all required columns")
        print(f"  keyword={keyword_col}, volume={volume_col}, position={position_col}, domain={domain_col}")
        sys.exit(1)

    print(f"  Using columns: keyword={keyword_col}, volume={volume_col}, position={position_col}, domain={domain_col}")
    if category_col:
        print(f"  Category column: {category_col}")

    # Get CTR curve
    ctr_curve = CTR_CURVES[args.ctr_curve]
    print(f"  CTR curve: {args.ctr_curve}")

    # Prepare data
    df_work = df.copy()
    df_work[volume_col] = pd.to_numeric(df_work[volume_col], errors='coerce')
    df_work[position_col] = pd.to_numeric(df_work[position_col], errors='coerce')

    # Filter to top 10 positions
    df_work = df_work[(df_work[position_col] >= 1) & (df_work[position_col] <= 10)]
    df_work[position_col] = df_work[position_col].astype(int)

    print(f"  Rows in top 10: {len(df_work):,}")

    # Apply CTR curve
    df_work['ctr'] = df_work[position_col].map(ctr_curve)
    df_work['estimated_traffic'] = (df_work['ctr'] * df_work[volume_col]).round(0)

    # Group by domain
    if category_col and category_col in df_work.columns:
        grouped = df_work.groupby([category_col, domain_col]).agg({
            'estimated_traffic': 'sum',
            keyword_col: 'count'
        }).reset_index()
        grouped.columns = [category_col, 'Domain', 'Estimated Traffic', 'Keywords']

        # Calculate SOV within each category
        category_totals = grouped.groupby(category_col)['Estimated Traffic'].transform('sum')
        grouped['SOV (%)'] = (grouped['Estimated Traffic'] / category_totals * 100).round(2)

        grouped = grouped.sort_values([category_col, 'Estimated Traffic'], ascending=[True, False])
        grouped = grouped.groupby(category_col).head(args.top_n)
    else:
        grouped = df_work.groupby(domain_col).agg({
            'estimated_traffic': 'sum',
            keyword_col: 'count'
        }).reset_index()
        grouped.columns = ['Domain', 'Estimated Traffic', 'Keywords']

        total_traffic = grouped['Estimated Traffic'].sum()
        grouped['SOV (%)'] = (grouped['Estimated Traffic'] / total_traffic * 100).round(2)

        grouped = grouped.sort_values('Estimated Traffic', ascending=False)
        grouped = grouped.head(args.top_n)

    # Save results
    grouped.to_csv(args.output, index=False, encoding='utf-8-sig')

    print(f"\nResults saved to: {args.output}")
    print(f"  Domains: {len(grouped):,}")
    print(f"  Total estimated traffic: {grouped['Estimated Traffic'].sum():,.0f}")

    # Show top domains
    print(f"\nTop domains by SOV:")
    for _, row in grouped.head(10).iterrows():
        print(f"  [{row['SOV (%)']:5.1f}%] {row['Domain']}")


if __name__ == '__main__':
    main()
