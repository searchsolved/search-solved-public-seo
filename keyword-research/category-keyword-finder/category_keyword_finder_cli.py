#!/usr/bin/env python3
"""
Category Keyword Finder - CLI Version

Extract n-gram keyword opportunities from product titles grouped by category.

Usage:
    python category_keyword_finder_cli.py --input crawl.csv --output keywords.csv

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd
import re
import string
import collections
from nltk.util import ngrams
import sys


def main():
    parser = argparse.ArgumentParser(description='Find category keyword opportunities from product titles')
    parser.add_argument('--input', required=True, help='Crawl CSV with product URLs and titles')
    parser.add_argument('--output', default='category_keywords.csv', help='Output CSV path')
    parser.add_argument('--keywords', help='Optional: keyword CSV with search volumes')
    parser.add_argument('--url-col', default='Address', help='URL column name')
    parser.add_argument('--h1-col', default='H1-1', help='H1/Title column name')
    parser.add_argument('--product-pattern', default='/product/', help='Product URL pattern')
    parser.add_argument('--category-pattern', default='/category/', help='Category URL pattern')
    parser.add_argument('--min-ngram', type=int, default=2, help='Minimum n-gram length')
    parser.add_argument('--max-ngram', type=int, default=5, help='Maximum n-gram length')
    parser.add_argument('--min-products', type=int, default=3, help='Minimum matching products')

    args = parser.parse_args()

    print(f"Loading crawl data from: {args.input}")
    try:
        df_crawl = pd.read_csv(args.input, encoding='utf-8')
    except:
        df_crawl = pd.read_csv(args.input, encoding='latin-1')

    print(f"  Loaded {len(df_crawl):,} URLs")

    # Find columns
    cols = df_crawl.columns.tolist()
    url_col = args.url_col if args.url_col in cols else cols[0]
    h1_col = args.h1_col if args.h1_col in cols else next((c for c in cols if 'h1' in c.lower() or 'title' in c.lower()), cols[1] if len(cols) > 1 else cols[0])

    # Filter to products
    df_products = df_crawl[df_crawl[url_col].str.contains(args.product_pattern, na=False)].copy()

    if len(df_products) == 0:
        print(f"Error: No products found matching pattern '{args.product_pattern}'")
        sys.exit(1)

    print(f"  Found {len(df_products):,} product pages")

    # Extract parent category
    def get_parent(url):
        parts = url.split('/')
        for i, part in enumerate(parts):
            if args.product_pattern.strip('/') in part.lower():
                return '/'.join(parts[:i]) + '/'
        return '/'.join(parts[:-1]) + '/'

    df_products["parent_category"] = df_products[url_col].apply(get_parent)
    df_products["h1_clean"] = df_products[h1_col].fillna("").str.lower()

    categories = df_products["parent_category"].unique()
    print(f"  Found {len(categories):,} parent categories")

    # Load optional keyword data
    df_keywords = None
    if args.keywords:
        print(f"Loading keyword data from: {args.keywords}")
        try:
            df_keywords = pd.read_csv(args.keywords, encoding='utf-8')
        except:
            df_keywords = pd.read_csv(args.keywords, encoding='latin-1')

        kw_cols = df_keywords.columns.tolist()
        kw_col = next((c for c in kw_cols if 'keyword' in c.lower()), kw_cols[0])
        vol_col = next((c for c in kw_cols if 'volume' in c.lower()), None)

    # Process n-grams
    print(f"\nExtracting n-grams (n={args.min_ngram}-{args.max_ngram})...")
    all_ngrams = []

    for category in categories:
        df_cat = df_products[df_products["parent_category"] == category]

        if len(df_cat) < args.min_products:
            continue

        text = " ".join(df_cat["h1_clean"].tolist())
        text = re.sub(r'\d+', '', text)
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        tokens = text.split()

        for n in range(args.min_ngram, args.max_ngram + 1):
            if len(tokens) >= n:
                n_grams = list(ngrams(tokens, n))
                counts = collections.Counter(n_grams)

                for gram, count in counts.most_common(50):
                    if count >= args.min_products:
                        phrase = ' '.join(gram)
                        all_ngrams.append({
                            "parent_category": category,
                            "keyword": phrase,
                            "frequency": count,
                            "n_gram_length": n
                        })

    if not all_ngrams:
        print("No n-grams found meeting threshold. Try lowering --min-products.")
        sys.exit(0)

    df_ngrams = pd.DataFrame(all_ngrams)

    # Count actual product matches
    def count_matches(row):
        kw = row["keyword"]
        cat = row["parent_category"]
        cat_products = df_products[df_products["parent_category"] == cat]
        return cat_products["h1_clean"].str.contains(kw, regex=False, na=False).sum()

    df_ngrams["matching_products"] = df_ngrams.apply(count_matches, axis=1)
    df_ngrams = df_ngrams[df_ngrams["matching_products"] >= args.min_products]

    # Remove duplicates
    df_ngrams = df_ngrams.sort_values("n_gram_length", ascending=False)
    df_ngrams = df_ngrams.drop_duplicates(subset=["parent_category", "keyword"], keep="first")

    # Match with keyword volumes
    if df_keywords is not None and vol_col:
        df_keywords["kw_lower"] = df_keywords[kw_col].str.lower().str.strip()
        df_ngrams = df_ngrams.merge(
            df_keywords[["kw_lower", vol_col]].rename(columns={"kw_lower": "keyword", vol_col: "search_volume"}),
            on="keyword",
            how="left"
        )
        df_ngrams["search_volume"] = df_ngrams["search_volume"].fillna(0)
        df_ngrams = df_ngrams.sort_values("search_volume", ascending=False)
    else:
        df_ngrams["search_volume"] = 0
        df_ngrams = df_ngrams.sort_values("matching_products", ascending=False)

    # Check existing categories
    existing_cats = df_crawl[df_crawl[url_col].str.contains(args.category_pattern, na=False)]
    if len(existing_cats) > 0:
        existing_h1s = existing_cats[h1_col].str.lower().unique()
        df_ngrams["exists_as_category"] = df_ngrams["keyword"].isin(existing_h1s)
    else:
        df_ngrams["exists_as_category"] = False

    df_ngrams.to_csv(args.output, index=False, encoding='utf-8-sig')

    new_opps = (~df_ngrams['exists_as_category']).sum()

    print(f"\nResults saved to: {args.output}")
    print(f"  Keywords found: {len(df_ngrams):,}")
    print(f"  New opportunities: {new_opps:,}")
    print(f"  Categories analyzed: {df_ngrams['parent_category'].nunique():,}")


if __name__ == '__main__':
    main()
