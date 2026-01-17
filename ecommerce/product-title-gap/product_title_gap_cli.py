#!/usr/bin/env python3
"""
Product Title Gap Analyzer - CLI Version

Compares your product titles with competitors using MPN matching.
Identifies missing words that competitors use in their titles.

Usage:
    python product_title_gap_cli.py --your-crawl your_crawl.csv --competitor-crawls comp1.csv comp2.csv

Author: Lee Foot
Website: https://www.leefoot.com
"""

import argparse
import pandas as pd
from collections import Counter
import string
import statistics
import sys

try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
except ImportError:
    print("Please install nltk: pip install nltk")
    sys.exit(1)


def preprocess_text(text, stop_words, min_length=2):
    """Remove punctuation and stop words from text."""
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) >= min_length]
    return ' '.join(words)


def find_column(df, possible_names):
    """Find a column by trying multiple possible names."""
    for name in possible_names:
        for col in df.columns:
            if name.lower() in col.lower():
                return col
    return None


def load_csv(filepath):
    """Load CSV with encoding fallback."""
    try:
        return pd.read_csv(filepath, encoding='utf-8')
    except:
        return pd.read_csv(filepath, encoding='latin-1')


def find_missing_words(row, competitor_dfs, stop_words, min_word_length):
    """Find words in competitor titles that are missing from source title."""
    words = preprocess_text(row['h1'], stop_words, min_word_length).split()
    missing_words = []
    matching_urls = []
    matching_h1s = []
    h1_lengths = []
    freq_source = Counter(words)

    for comp_df in competitor_dfs:
        matched_df = comp_df[comp_df['mpn'] == row['mpn_matching']]
        if not matched_df.empty:
            if 'url' in matched_df.columns:
                matching_urls.append(matched_df['url'].tolist()[0])
            matching_h1s.append(matched_df['h1'].tolist()[0])

            comparison_h1 = preprocess_text(
                matched_df['h1'].tolist()[0],
                stop_words,
                min_word_length
            )
            freq_comparison = Counter(comparison_h1.split())

            for word in freq_comparison:
                if word.lower() not in freq_source:
                    missing_words.append((word, freq_comparison[word]))

            h1_lengths.append(len(comparison_h1.split()))

    source_h1_length = len(words)
    median_comparison_h1_length = statistics.median(h1_lengths) if h1_lengths else 0
    median_length_difference = median_comparison_h1_length - source_h1_length

    return (missing_words, matching_urls, matching_h1s, source_h1_length,
            median_comparison_h1_length, median_length_difference)


def main():
    parser = argparse.ArgumentParser(
        description='Compare product titles with competitors using MPN matching'
    )
    parser.add_argument('--your-crawl', required=True, help='Path to your crawl CSV')
    parser.add_argument('--competitor-crawls', nargs='+', required=True,
                        help='Paths to competitor crawl CSVs')
    parser.add_argument('--output', default='product_title_gaps.csv',
                        help='Output CSV path (default: product_title_gaps.csv)')
    parser.add_argument('--language', default='english',
                        help='Stopwords language (default: english)')
    parser.add_argument('--min-word-length', type=int, default=2,
                        help='Minimum word length (default: 2)')
    parser.add_argument('--url-col', help='URL column name (auto-detected if not specified)')
    parser.add_argument('--title-col', help='Title/H1 column name (auto-detected if not specified)')
    parser.add_argument('--mpn-col', help='MPN column name (auto-detected if not specified)')

    args = parser.parse_args()

    # Load stopwords
    stop_words = set(stopwords.words(args.language))

    # Load your crawl
    print(f"Loading your crawl: {args.your_crawl}")
    df_source = load_csv(args.your_crawl)
    print(f"  Loaded {len(df_source):,} rows")

    # Find columns
    url_col = args.url_col or find_column(df_source, ['url', 'address'])
    title_col = args.title_col or find_column(df_source, ['h1', 'title 1', 'title'])
    mpn_col = args.mpn_col or find_column(df_source, ['mpn', 'sku', 'product_id'])

    if not all([url_col, title_col, mpn_col]):
        print("Error: Could not auto-detect all required columns.")
        print(f"  Found: url={url_col}, title={title_col}, mpn={mpn_col}")
        print("  Please specify columns with --url-col, --title-col, --mpn-col")
        sys.exit(1)

    print(f"  Using columns: url={url_col}, title={title_col}, mpn={mpn_col}")

    # Load competitor crawls
    competitor_dfs = []
    for comp_path in args.competitor_crawls:
        print(f"Loading competitor: {comp_path}")
        df_comp = load_csv(comp_path)
        print(f"  Loaded {len(df_comp):,} rows")

        comp_mpn_col = find_column(df_comp, ['mpn', 'sku', 'product_id'])
        comp_title_col = find_column(df_comp, ['h1', 'title 1', 'title'])
        comp_url_col = find_column(df_comp, ['url', 'address'])

        if comp_mpn_col and comp_title_col:
            df_comp = df_comp.rename(columns={
                comp_mpn_col: 'mpn',
                comp_title_col: 'h1',
            })
            if comp_url_col:
                df_comp = df_comp.rename(columns={comp_url_col: 'url'})
            df_comp['mpn'] = df_comp['mpn'].astype(str).str.lower().str.strip()
            competitor_dfs.append(df_comp)
        else:
            print(f"  Warning: Could not find required columns, skipping")

    if not competitor_dfs:
        print("Error: No valid competitor files loaded")
        sys.exit(1)

    # Prepare source data
    df_work = df_source.rename(columns={
        url_col: 'url',
        title_col: 'h1',
        mpn_col: 'mpn'
    })
    df_work['mpn_matching'] = df_work['mpn'].astype(str).str.lower().str.strip()
    df_work['h1_original'] = df_work['h1']
    df_work['h1'] = df_work['h1'].astype(str).str.lower().str.strip()
    df_work = df_work.dropna(subset=['h1'])
    df_work = df_work[df_work['h1'] != 'nan']

    # Process each row
    print(f"\nAnalyzing {len(df_work):,} products...")
    results = []
    for idx, (_, row) in enumerate(df_work.iterrows()):
        if idx % 100 == 0:
            print(f"  Processing {idx + 1}/{len(df_work)}...")
        result = find_missing_words(row, competitor_dfs, stop_words, args.min_word_length)
        results.append(result)

    df_work[['missing_words', 'matching_urls', 'matching_h1s',
             'source_h1_length', 'median_comparison_h1_length',
             'median_length_difference']] = pd.DataFrame(results, index=df_work.index)

    # Process results
    df_work['missing_words'] = df_work['missing_words'].apply(
        lambda x: Counter([word for word, freq in x]).most_common()
    )

    # Filter to only products with matches
    df_work = df_work[df_work['matching_urls'].apply(lambda x: len(x) > 0)]

    if len(df_work) == 0:
        print("No matching products found between your crawl and competitors.")
        sys.exit(0)

    # Find most verbose competitor title
    df_work['most_verbose_h1'] = df_work['matching_h1s'].apply(
        lambda x: sorted(x, key=lambda h1: len(str(h1).split()), reverse=True)[0] if x else ''
    )

    # Sort by gap size
    df_work = df_work.sort_values(by='median_length_difference', ascending=False)

    # Restore original title
    df_work['h1'] = df_work['h1_original']

    # Clean up output
    output_df = df_work[[
        'url', 'mpn', 'h1', 'missing_words', 'median_length_difference',
        'most_verbose_h1', 'matching_urls'
    ]].copy()
    output_df.columns = [
        'Your URL', 'MPN', 'Your Title', 'Missing Words',
        'Length Gap', 'Best Competitor Title', 'Competitor URLs'
    ]

    # Save results
    output_df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {args.output}")
    print(f"  Products analyzed: {len(output_df):,}")
    print(f"  Average length gap: {output_df['Length Gap'].mean():.1f} words")


if __name__ == '__main__':
    main()
