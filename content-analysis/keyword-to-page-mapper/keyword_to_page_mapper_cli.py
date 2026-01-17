#!/usr/bin/env python3
"""
Keyword-to-Page Mapper - CLI Version

Semantically match keywords to existing pages using ML embeddings.

Usage:
    python keyword_to_page_mapper_cli.py --pages crawl.csv --keywords keywords.csv

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Semantically match keywords to existing pages using ML embeddings'
    )
    parser.add_argument('--pages', required=True,
                        help='Pages CSV with URL and H1/Title columns')
    parser.add_argument('--keywords', required=True,
                        help='Keywords CSV')
    parser.add_argument('--output', default='keyword_mapping.csv',
                        help='Output CSV path (default: keyword_mapping.csv)')
    parser.add_argument('--url-col', default='Address',
                        help='URL column name in pages CSV')
    parser.add_argument('--h1-col', default='H1-1',
                        help='H1/Title column name in pages CSV')
    parser.add_argument('--keyword-col', default='keyword',
                        help='Keyword column name in keywords CSV')
    parser.add_argument('--volume-col', default='volume',
                        help='Volume column name in keywords CSV (optional)')
    parser.add_argument('--threshold', type=float, default=0.75,
                        help='Minimum similarity threshold (default: 0.75)')
    parser.add_argument('--model', default='all-MiniLM-L6-v2',
                        help='Sentence transformer model')
    parser.add_argument('--exclude-exact', action='store_true',
                        help='Exclude keywords found exactly in page content')

    args = parser.parse_args()

    # Import ML libraries
    print("Loading ML libraries...")
    try:
        from polyfuzz import PolyFuzz
        from polyfuzz.models import SentenceEmbeddings
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: Required packages not installed")
        print("Run: pip install polyfuzz sentence-transformers")
        sys.exit(1)

    # Load pages
    print(f"\nLoading pages from: {args.pages}")
    pages_df = pd.read_csv(args.pages)

    # Find columns
    url_col = None
    h1_col = None
    for col in pages_df.columns:
        if col.lower() == args.url_col.lower() or 'address' in col.lower() or 'url' in col.lower():
            url_col = col
        if col.lower() == args.h1_col.lower() or 'h1' in col.lower():
            h1_col = col

    if not url_col:
        url_col = pages_df.columns[0]
    if not h1_col:
        h1_col = pages_df.columns[1] if len(pages_df.columns) > 1 else pages_df.columns[0]

    print(f"  Using columns: URL={url_col}, H1={h1_col}")

    pages_df = pages_df[[url_col, h1_col]].copy()
    pages_df = pages_df[pages_df[h1_col].notna()]
    pages_df[h1_col] = pages_df[h1_col].str.lower()
    print(f"  Loaded {len(pages_df)} pages")

    # Load keywords
    print(f"\nLoading keywords from: {args.keywords}")
    keywords_df = pd.read_csv(args.keywords)

    # Find keyword column
    keyword_col = None
    volume_col = None
    for col in keywords_df.columns:
        if col.lower() == args.keyword_col.lower() or 'keyword' in col.lower():
            keyword_col = col
        if col.lower() == args.volume_col.lower() or 'volume' in col.lower():
            volume_col = col

    if not keyword_col:
        keyword_col = keywords_df.columns[0]

    print(f"  Using columns: keyword={keyword_col}, volume={volume_col}")

    keywords_df = keywords_df[keywords_df[keyword_col].notna()]
    print(f"  Loaded {len(keywords_df)} keywords")

    # Prepare data
    to_list = list(pages_df[h1_col])
    from_list = list(keywords_df[keyword_col].str.lower())

    # Load model
    print(f"\nLoading embedding model: {args.model}")
    embedding_model = SentenceTransformer(args.model)
    distance_model = SentenceEmbeddings(embedding_model)
    model = PolyFuzz(distance_model)

    # Match
    print(f"\nMatching {len(from_list)} keywords to {len(to_list)} pages...")
    model.match(from_list, to_list)

    df_matches = model.get_matches()

    # Filter by similarity
    df_matches = df_matches[df_matches['Similarity'] >= args.threshold]
    print(f"  Matches above threshold ({args.threshold}): {len(df_matches)}")

    # Remove exact matches if requested
    if args.exclude_exact:
        df_matches['From_lower'] = df_matches['From'].str.lower()
        df_matches['To_lower'] = df_matches['To'].str.lower()
        df_matches['is_exact'] = df_matches.apply(
            lambda row: row['From_lower'] in row['To_lower'], axis=1
        )
        df_matches = df_matches[~df_matches['is_exact']]
        df_matches = df_matches.drop(columns=['From_lower', 'To_lower', 'is_exact'])
        print(f"  After excluding exact matches: {len(df_matches)}")

    # Rename columns
    df_matches = df_matches.rename(columns={
        'From': 'Keyword',
        'To': 'Matched H1'
    })

    # Add URL
    h1_to_url = dict(zip(pages_df[h1_col], pages_df[url_col]))
    df_matches['URL'] = df_matches['Matched H1'].map(h1_to_url)

    # Add volume if available
    if volume_col:
        kw_to_vol = dict(zip(
            keywords_df[keyword_col].str.lower(),
            keywords_df[volume_col]
        ))
        df_matches['Volume'] = df_matches['Keyword'].map(kw_to_vol)

    # Sort
    if volume_col and 'Volume' in df_matches.columns:
        df_matches = df_matches.sort_values(['Volume', 'Similarity'], ascending=[False, False])
    else:
        df_matches = df_matches.sort_values('Similarity', ascending=False)

    # Save
    df_matches.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {args.output}")
    print(f"  Mapped keywords: {len(df_matches)}")

    # Find unmapped keywords
    unmapped = set(from_list) - set(df_matches['Keyword'].str.lower())
    if unmapped:
        unmapped_path = args.output.replace('.csv', '_gaps.csv')
        unmapped_df = pd.DataFrame({'Keyword': list(unmapped)})
        if volume_col:
            kw_to_vol = dict(zip(
                keywords_df[keyword_col].str.lower(),
                keywords_df[volume_col]
            ))
            unmapped_df['Volume'] = unmapped_df['Keyword'].map(kw_to_vol)
            unmapped_df = unmapped_df.sort_values('Volume', ascending=False)
        unmapped_df.to_csv(unmapped_path, index=False, encoding='utf-8-sig')
        print(f"  Unmapped keywords (content gaps): {len(unmapped)}")
        print(f"  Saved to: {unmapped_path}")

    # Show top matches
    print(f"\nTop keyword-page matches:")
    for _, row in df_matches.head(10).iterrows():
        vol_str = f" ({int(row['Volume']):,})" if 'Volume' in row and pd.notna(row['Volume']) else ""
        print(f"  [{row['Similarity']:.2f}] {row['Keyword']}{vol_str} -> {row['Matched H1'][:40]}...")


if __name__ == '__main__':
    main()
