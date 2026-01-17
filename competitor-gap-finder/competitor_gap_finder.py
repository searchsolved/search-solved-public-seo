####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                                              #
####################################################################################

"""Competitor Content Gap Finder

Finds gaps in your product titles by comparing against competitors.
Uses a matching key (SKU, MPN, etc.) to pair products and identify missing words.

Usage:
    python competitor_gap_finder.py --source my_products.csv --compare competitors/ --output gaps.csv
    python competitor_gap_finder.py --source my_crawl.csv --compare competitor_crawl.csv --match-col sku --title-col h1
"""

import argparse
import pandas as pd
from collections import Counter
import glob
import statistics
import string
import nltk
from nltk.corpus import stopwords
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


def preprocess_text(text):
    """Remove punctuation and stop words from text."""
    if not isinstance(text, str):
        return ""
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(words)


def load_source_data(file_path, match_col, title_col, url_col=None):
    """Load source product data."""
    logger.info(f"Loading source data from: {file_path}")

    cols_to_load = [match_col, title_col]
    if url_col and url_col not in cols_to_load:
        cols_to_load.append(url_col)

    df = pd.read_csv(file_path)

    # Check required columns exist
    for col in [match_col, title_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in source file. Available: {list(df.columns)}")

    # Create matching key (lowercase, stripped)
    df['match_key'] = df[match_col].astype(str).str.lower().str.strip()

    # Keep original title and create lowercase version
    df['title_original'] = df[title_col]
    df['title'] = df[title_col].astype(str).str.lower().str.strip()

    logger.info(f"Loaded {len(df)} products from source")
    return df


def load_competitor_data(path_pattern, match_col, title_col, url_col=None):
    """Load competitor product data from one or more files."""
    files = glob.glob(path_pattern) if '*' in path_pattern else [path_pattern]

    if not files:
        raise ValueError(f"No files found matching: {path_pattern}")

    logger.info(f"Loading competitor data from {len(files)} file(s)")

    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file)

            if match_col not in df.columns or title_col not in df.columns:
                logger.warning(f"Skipping {file}: missing required columns")
                continue

            # Standardize match key
            df['match_key'] = df[match_col].astype(str).str.lower().str.strip()
            df['title'] = df[title_col].astype(str).str.lower().str.strip()

            if url_col and url_col in df.columns:
                df['url'] = df[url_col]
            else:
                df['url'] = file  # Use filename as identifier

            dfs.append(df[['match_key', 'title', 'url']])
            logger.info(f"  Loaded {len(df)} products from {os.path.basename(file)}")
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")

    if not dfs:
        raise ValueError("No valid competitor data loaded")

    return dfs


def find_missing_words(source_title, competitor_dfs, source_match_key):
    """Find words that competitors use but source doesn't."""
    source_words = set(preprocess_text(source_title).lower().split())
    source_word_freq = Counter(preprocess_text(source_title).lower().split())

    missing_words = []
    matching_urls = []
    matching_titles = []
    title_lengths = []

    for comp_df in competitor_dfs:
        matched = comp_df[comp_df['match_key'] == source_match_key]

        if not matched.empty:
            comp_title = matched['title'].iloc[0]
            comp_url = matched['url'].iloc[0]

            matching_urls.append(comp_url)
            matching_titles.append(comp_title)

            comp_processed = preprocess_text(comp_title).lower()
            comp_word_freq = Counter(comp_processed.split())

            for word in comp_word_freq:
                if word not in source_word_freq:
                    missing_words.append((word, comp_word_freq[word]))

            title_lengths.append(len(comp_processed.split()))

    source_length = len(preprocess_text(source_title).split())
    median_comp_length = statistics.median(title_lengths) if title_lengths else 0
    length_difference = median_comp_length - source_length

    return {
        'missing_words': missing_words,
        'matching_urls': matching_urls,
        'matching_titles': matching_titles,
        'source_length': source_length,
        'median_comp_length': median_comp_length,
        'length_difference': length_difference
    }


def analyze_gaps(source_df, competitor_dfs):
    """Analyze content gaps between source and competitors."""
    results = []

    for _, row in source_df.iterrows():
        gap_info = find_missing_words(
            row['title'],
            competitor_dfs,
            row['match_key']
        )

        # Only include if there are matches
        if gap_info['matching_urls']:
            # Aggregate missing words
            word_counts = Counter([word for word, freq in gap_info['missing_words']])

            results.append({
                'match_key': row['match_key'],
                'source_title': row['title_original'],
                'missing_words': word_counts.most_common(),
                'missing_word_count': len(word_counts),
                'competitor_matches': len(gap_info['matching_urls']),
                'matching_urls': gap_info['matching_urls'],
                'matching_titles': gap_info['matching_titles'],
                'most_verbose_title': max(gap_info['matching_titles'], key=lambda x: len(x.split())) if gap_info['matching_titles'] else '',
                'length_difference': gap_info['length_difference']
            })

    logger.info(f"Found {len(results)} products with competitor matches")
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Competitor Content Gap Finder - See what words competitors use that you're missing"
    )
    parser.add_argument(
        "--source", "-s",
        required=True,
        help="Path to your product CSV file"
    )
    parser.add_argument(
        "--compare", "-c",
        required=True,
        help="Path to competitor CSV file(s) - can use wildcards like 'competitors/*.csv'"
    )
    parser.add_argument(
        "--output", "-o",
        default="content_gaps.csv",
        help="Output CSV file path (default: content_gaps.csv)"
    )
    parser.add_argument(
        "--match-col",
        default="mpn",
        help="Column name for matching key (e.g., mpn, sku, product_id) (default: mpn)"
    )
    parser.add_argument(
        "--title-col",
        default="h1",
        help="Column name for product title (default: h1)"
    )
    parser.add_argument(
        "--url-col",
        default="url",
        help="Column name for URL (optional, default: url)"
    )

    args = parser.parse_args()

    try:
        # Load source data
        source_df = load_source_data(
            args.source,
            args.match_col,
            args.title_col,
            args.url_col
        )

        # Load competitor data
        competitor_dfs = load_competitor_data(
            args.compare,
            args.match_col,
            args.title_col,
            args.url_col
        )

        # Analyze gaps
        results_df = analyze_gaps(source_df, competitor_dfs)

        if results_df.empty:
            logger.warning("No matching products found between source and competitors")
            return

        # Sort by length difference (biggest gaps first)
        results_df = results_df.sort_values('length_difference', ascending=False)

        # Convert lists to strings for CSV
        results_df['missing_words'] = results_df['missing_words'].apply(str)
        results_df['matching_urls'] = results_df['matching_urls'].apply(lambda x: ' | '.join(x[:3]))
        results_df['matching_titles'] = results_df['matching_titles'].apply(lambda x: x[0] if x else '')

        # Save results
        results_df.to_csv(args.output, index=False, encoding='utf-8-sig')

        logger.info(f"Results saved to: {args.output}")
        logger.info(f"Found {len(results_df)} products with content gaps")

        # Show summary
        avg_gap = results_df['length_difference'].mean()
        logger.info(f"Average title length gap: {avg_gap:.1f} words")

        top_gaps = results_df.head(5)
        if not top_gaps.empty:
            logger.info("\nTop 5 products with biggest gaps:")
            for _, row in top_gaps.iterrows():
                logger.info(f"  {row['match_key']}: {row['length_difference']:.0f} words shorter")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
