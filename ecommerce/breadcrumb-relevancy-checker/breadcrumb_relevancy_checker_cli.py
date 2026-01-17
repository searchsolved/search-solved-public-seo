#!/usr/bin/env python3
"""
Breadcrumb Relevancy Checker - CLI Version

Check if products are in the most relevant categories using fuzzy matching.

Usage:
    python breadcrumb_relevancy_checker_cli.py --input crawl.csv --output results.csv

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd
from polyfuzz import PolyFuzz
import sys


def main():
    parser = argparse.ArgumentParser(description='Check breadcrumb relevancy using fuzzy matching')
    parser.add_argument('--input', required=True, help='Input crawl CSV (Screaming Frog export)')
    parser.add_argument('--output', default='breadcrumb_relevancy.csv', help='Output CSV path')
    parser.add_argument('--url-col', default='Address', help='URL column name')
    parser.add_argument('--h1-col', default='H1-1', help='H1 column name')
    parser.add_argument('--breadcrumb-col', default='Breadcrumb', help='Breadcrumb column name')
    parser.add_argument('--product-pattern', default='/product/', help='Product URL pattern')
    parser.add_argument('--category-pattern', default='/category/', help='Category URL pattern')
    parser.add_argument('--threshold', type=float, default=0.3, help='Similarity threshold for flagging')

    args = parser.parse_args()

    print(f"Loading crawl data from: {args.input}")
    df = pd.read_csv(args.input, dtype=str)
    print(f"  Loaded {len(df):,} rows")

    # Validate columns
    cols = df.columns.tolist()
    url_col = args.url_col if args.url_col in cols else cols[0]
    h1_col = args.h1_col if args.h1_col in cols else None
    bc_col = args.breadcrumb_col if args.breadcrumb_col in cols else None

    if not h1_col:
        h1_options = [c for c in cols if 'h1' in c.lower()]
        h1_col = h1_options[0] if h1_options else cols[1] if len(cols) > 1 else None

    if not bc_col:
        bc_options = [c for c in cols if 'breadcrumb' in c.lower()]
        bc_col = bc_options[0] if bc_options else None

    if not h1_col or not bc_col:
        print(f"Error: Could not find H1 or Breadcrumb columns")
        print(f"Available: {cols}")
        sys.exit(1)

    # Clean data
    df_clean = df[[url_col, h1_col, bc_col]].copy()
    df_clean.columns = ['Address', 'H1-1', 'Breadcrumb']
    df_clean = df_clean[df_clean["H1-1"].notna() & df_clean["Breadcrumb"].notna()]

    # Get categories and products
    df_cats = df_clean[df_clean["Address"].str.contains(args.category_pattern, na=False)].copy()
    df_products = df_clean[df_clean["Address"].str.contains(args.product_pattern, na=False)].copy()

    if len(df_products) == 0:
        print(f"Error: No products found matching pattern '{args.product_pattern}'")
        sys.exit(1)

    print(f"  Found {len(df_products):,} products and {len(df_cats):,} categories")

    # Calculate similarity between H1 and existing breadcrumb
    print("\nCalculating existing breadcrumb similarity...")
    h1_list = df_products["H1-1"].tolist()
    bread_list = df_products["Breadcrumb"].tolist()

    dfs = []
    for idx, (h1, bread) in enumerate(zip(h1_list, bread_list)):
        try:
            pf = PolyFuzz("TF-IDF").match([h1], [bread])
            dfs.append(pf.get_matches())
        except Exception:
            dfs.append(pd.DataFrame({'From': [h1], 'To': [bread], 'Similarity': [0.0]}))
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(h1_list)}...")

    df_concat = pd.concat(dfs)
    df_concat.rename(columns={"From": "H1-1"}, inplace=True)
    df_products = pd.merge(df_products, df_concat[['H1-1', 'Similarity']], on='H1-1', how='left')

    # Find best matching category
    print("\nFinding best category matches...")
    if len(df_cats) > 0:
        try:
            pf = PolyFuzz("TF-IDF").match(
                list(df_products["H1-1"].unique()),
                list(df_cats['Breadcrumb'].unique())
            )
            df_fuzzed = pf.get_matches()
            df_products = pd.merge(df_products, df_fuzzed, left_on="H1-1", right_on="From", how="left")
        except Exception as e:
            print(f"Warning: Category matching failed: {e}")
            df_products["To"] = None
            df_products["Similarity_y"] = 0

    # Clean up
    df_products = df_products.rename(columns={
        "Similarity_x": "Similarity (Existing)",
        "Similarity_y": "Similarity (Best Match)",
        "To": "Best Match Breadcrumb",
        "Breadcrumb": "Existing Breadcrumb"
    })

    if 'From' in df_products.columns:
        del df_products['From']

    df_products["Similarity (Best Match)"] = df_products["Similarity (Best Match)"].fillna(0)
    df_products["Similarity (Existing)"] = df_products["Similarity (Existing)"].fillna(0)
    df_products['Similarity Diff'] = df_products["Similarity (Best Match)"] - df_products["Similarity (Existing)"]

    # Round values
    for col in df_products.columns:
        if 'Similarity' in col and df_products[col].dtype in ['float64', 'float32']:
            df_products[col] = df_products[col].round(3)

    df_products = df_products.sort_values('Similarity Diff', ascending=False)
    df_products.to_csv(args.output, index=False, encoding='utf-8-sig')

    potential_issues = len(df_products[df_products['Similarity Diff'] >= args.threshold])

    print(f"\nResults saved to: {args.output}")
    print(f"  Products analyzed: {len(df_products):,}")
    print(f"  Potential miscategorizations: {potential_issues:,}")
    print(f"  Avg existing similarity: {df_products['Similarity (Existing)'].mean():.2%}")


if __name__ == '__main__':
    main()
