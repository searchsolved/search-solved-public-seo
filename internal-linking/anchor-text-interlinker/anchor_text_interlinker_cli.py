#!/usr/bin/env python3
"""
Anchor Text Interlinker - CLI Version

Find internal linking opportunities based on keyword matches in content.

Usage:
    python anchor_text_interlinker_cli.py --crawl crawl.csv --keywords keywords.csv

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd
from rapidfuzz import fuzz
import re
import sys


def main():
    parser = argparse.ArgumentParser(description='Find internal linking opportunities')
    parser.add_argument('--crawl', required=True, help='Crawl CSV with content extraction')
    parser.add_argument('--keywords', required=True, help='Organic keywords CSV')
    parser.add_argument('--output', default='internal_linking.csv', help='Output CSV path')
    parser.add_argument('--min-words', type=int, default=2, help='Minimum words in keyword')
    parser.add_argument('--max-words', type=int, default=6, help='Maximum words in keyword')
    parser.add_argument('--max-position', type=int, default=50, help='Maximum ranking position')
    parser.add_argument('--min-similarity', type=int, default=50, help='Minimum keyword/H1 similarity')

    args = parser.parse_args()

    print(f"Loading crawl data from: {args.crawl}")
    try:
        df_crawl = pd.read_csv(args.crawl, encoding='utf-8')
    except:
        df_crawl = pd.read_csv(args.crawl, encoding='latin-1')

    print(f"Loading keywords from: {args.keywords}")
    try:
        df_keywords = pd.read_csv(args.keywords, encoding='utf-8')
    except:
        df_keywords = pd.read_csv(args.keywords, encoding='latin-1')

    print(f"  Crawl: {len(df_crawl):,} pages, Keywords: {len(df_keywords):,} keywords")

    # Find columns
    crawl_cols = df_crawl.columns.tolist()
    kw_cols = df_keywords.columns.tolist()

    url_col = next((c for c in crawl_cols if c.lower() == 'address'), crawl_cols[0])
    h1_col = next((c for c in crawl_cols if 'h1' in c.lower()), None)
    content_col = next((c for c in crawl_cols if 'extract' in c.lower() or 'content' in c.lower()), None)

    kw_url_col = next((c for c in kw_cols if c.lower() == 'url'), kw_cols[0])
    kw_col = next((c for c in kw_cols if c.lower() == 'keyword'), kw_cols[1] if len(kw_cols) > 1 else kw_cols[0])
    vol_col = next((c for c in kw_cols if 'volume' in c.lower()), None)
    pos_col = next((c for c in kw_cols if 'position' in c.lower()), None)

    if not content_col:
        print("Warning: No content column found. Using H1 for matching.")
        content_col = h1_col

    if not content_col:
        print("Error: Need content or H1 column")
        sys.exit(1)

    # Prepare data
    df_crawl = df_crawl[[url_col, h1_col, content_col]].copy() if h1_col else df_crawl[[url_col, content_col]].copy()
    df_crawl.columns = ["page", "h1", "content"] if h1_col else ["page", "content"]
    df_crawl = df_crawl[df_crawl["content"].notna()]
    df_crawl["content"] = df_crawl["content"].str.lower()
    if "h1" in df_crawl.columns:
        df_crawl["h1"] = df_crawl["h1"].fillna("").str.lower()

    # Prepare keywords
    kw_columns = [kw_url_col, kw_col]
    if vol_col:
        kw_columns.append(vol_col)
    if pos_col:
        kw_columns.append(pos_col)

    df_keywords = df_keywords[kw_columns].copy()
    df_keywords.columns = ["target_page", "keyword"] + (["volume"] if vol_col else []) + (["position"] if pos_col else [])

    if "volume" in df_keywords.columns:
        df_keywords["volume"] = pd.to_numeric(df_keywords["volume"], errors='coerce').fillna(0)
    else:
        df_keywords["volume"] = 0

    if "position" in df_keywords.columns:
        df_keywords["position"] = pd.to_numeric(df_keywords["position"], errors='coerce').fillna(100)
        df_keywords = df_keywords[df_keywords["position"] <= args.max_position]
    else:
        df_keywords["position"] = 0

    df_keywords["keyword"] = df_keywords["keyword"].str.lower().str.strip()
    df_keywords["word_count"] = df_keywords["keyword"].str.count(" ") + 1
    df_keywords = df_keywords[
        (df_keywords["word_count"] >= args.min_words) &
        (df_keywords["word_count"] <= args.max_words)
    ]

    df_keywords = df_keywords.sort_values("volume", ascending=False)
    df_keywords = df_keywords.drop_duplicates(subset=["keyword"], keep="first")

    print(f"  Processing {len(df_keywords):,} unique keywords across {len(df_crawl):,} pages")

    # Create keyword lookup
    keyword_targets = df_keywords.set_index("keyword")[["target_page", "volume", "position"]].to_dict('index')

    # Find matches
    opportunities = []
    total = len(df_crawl)

    for idx, row in df_crawl.iterrows():
        source_page = row["page"]
        content = row["content"]

        for keyword, target_info in keyword_targets.items():
            target_page = target_info["target_page"]

            if source_page == target_page:
                continue

            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, content):
                target_h1 = ""
                if "h1" in df_crawl.columns:
                    h1_matches = df_crawl[df_crawl["page"] == target_page]["h1"].values
                    target_h1 = h1_matches[0] if len(h1_matches) > 0 else ""

                similarity = fuzz.token_sort_ratio(keyword, target_h1)

                if similarity >= args.min_similarity:
                    opportunities.append({
                        "source_page": source_page,
                        "anchor_text": keyword,
                        "target_page": target_page,
                        "target_h1": target_h1,
                        "similarity": similarity,
                        "volume": target_info["volume"],
                        "position": target_info["position"]
                    })

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{total} pages...")

    if opportunities:
        df_opps = pd.DataFrame(opportunities)
        df_opps = df_opps.drop_duplicates(subset=["source_page", "target_page"], keep="first")
        df_opps = df_opps[df_opps["source_page"] != df_opps["target_page"]]
        df_opps = df_opps.sort_values("volume", ascending=False)

        df_opps.to_csv(args.output, index=False, encoding='utf-8-sig')

        print(f"\nResults saved to: {args.output}")
        print(f"  Total opportunities: {len(df_opps):,}")
        print(f"  Unique source pages: {df_opps['source_page'].nunique():,}")
        print(f"  Unique target pages: {df_opps['target_page'].nunique():,}")
    else:
        print("\nNo linking opportunities found. Try adjusting filters.")
        sys.exit(0)


if __name__ == '__main__':
    main()
