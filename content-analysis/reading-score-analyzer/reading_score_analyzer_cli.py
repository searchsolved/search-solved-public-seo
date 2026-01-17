#!/usr/bin/env python3
"""
Reading Score Analyzer - CLI Version

Analyze content readability from URLs using Flesch scores.

Usage:
    python reading_score_analyzer_cli.py --sitemap https://example.com/sitemap.xml
    python reading_score_analyzer_cli.py --urls urls.txt

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import sys

try:
    import trafilatura
    import textstat
    from fake_useragent import UserAgent
except ImportError:
    print("Please install required packages:")
    print("  pip install trafilatura textstat fake-useragent")
    sys.exit(1)


def get_random_user_agent():
    """Get a random user agent string."""
    try:
        ua = UserAgent()
        return ua.random
    except:
        return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def fetch_urls_from_sitemap(sitemap_url, timeout=30):
    """Fetch URLs from an XML sitemap."""
    try:
        headers = {'User-Agent': get_random_user_agent()}
        response = requests.get(sitemap_url, headers=headers, timeout=timeout)
        soup = BeautifulSoup(response.content, 'xml')
        urls = [loc.text for loc in soup.find_all('loc')]

        # Filter out non-HTML URLs
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.svg', '.pdf', '.webp')
        urls = [url for url in urls if not url.lower().endswith(image_extensions)]

        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {str(e)}")
        return []


def extract_content(url, timeout=15):
    """Extract main content from a URL using Trafilatura."""
    try:
        headers = {'User-Agent': get_random_user_agent()}
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            content = trafilatura.extract(
                response.content,
                include_comments=False,
                include_tables=False
            )
            return content
        return None
    except Exception:
        return None


def calculate_reading_scores(text):
    """Calculate various readability scores."""
    if not text or len(text.split()) < 100:
        return None

    try:
        return {
            'flesch_reading_ease': round(textstat.flesch_reading_ease(text), 2),
            'flesch_kincaid_grade': round(textstat.flesch_kincaid_grade(text), 2),
            'gunning_fog': round(textstat.gunning_fog(text), 2),
            'smog_index': round(textstat.smog_index(text), 2),
            'automated_readability_index': round(textstat.automated_readability_index(text), 2),
            'coleman_liau_index': round(textstat.coleman_liau_index(text), 2),
            'word_count': textstat.lexicon_count(text, removepunct=True),
            'sentence_count': textstat.sentence_count(text),
            'avg_sentence_length': round(textstat.avg_sentence_length(text), 2),
            'difficult_words': textstat.difficult_words(text),
            'reading_time_mins': round(textstat.lexicon_count(text, removepunct=True) / 200, 1)
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Analyze content readability from URLs'
    )
    parser.add_argument('--sitemap', help='XML sitemap URL to fetch URLs from')
    parser.add_argument('--urls', help='File with URLs (one per line)')
    parser.add_argument('--output', default='reading_scores.csv',
                        help='Output CSV path (default: reading_scores.csv)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between requests in seconds (default: 1.0)')
    parser.add_argument('--timeout', type=int, default=15,
                        help='Request timeout in seconds (default: 15)')
    parser.add_argument('--max-urls', type=int, default=100,
                        help='Maximum URLs to process (default: 100)')
    parser.add_argument('--include-content', action='store_true',
                        help='Include extracted content in output')

    args = parser.parse_args()

    # Get URLs
    urls = []
    if args.sitemap:
        print(f"Fetching URLs from sitemap: {args.sitemap}")
        urls = fetch_urls_from_sitemap(args.sitemap, args.timeout)
        print(f"  Found {len(urls)} URLs")
    elif args.urls:
        with open(args.urls, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(urls)} URLs from file")
    else:
        print("Error: Provide URLs via --sitemap or --urls")
        sys.exit(1)

    if not urls:
        print("No URLs to process")
        sys.exit(1)

    # Limit URLs
    urls = urls[:args.max_urls]
    if len(urls) == args.max_urls:
        print(f"  Limited to first {args.max_urls} URLs")

    results = []
    errors = []

    print(f"\nAnalyzing {len(urls)} URLs...")

    for i, url in enumerate(urls):
        if i % 10 == 0:
            print(f"  Processing {i + 1}/{len(urls)}...")

        content = extract_content(url, args.timeout)

        if content:
            scores = calculate_reading_scores(content)
            if scores:
                result = {'url': url}
                result.update(scores)
                if args.include_content:
                    result['content'] = content
                results.append(result)
            else:
                errors.append({'url': url, 'error': 'Content too short for analysis'})
        else:
            errors.append({'url': url, 'error': 'Could not extract content'})

        time.sleep(args.delay)

    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(args.output, index=False, encoding='utf-8-sig')

        print(f"\nResults saved to: {args.output}")
        print(f"  Pages analyzed: {len(results)}")
        print(f"  Average Flesch score: {df_results['flesch_reading_ease'].mean():.1f}")
        print(f"  Average grade level: {df_results['flesch_kincaid_grade'].mean():.1f}")
        print(f"  Errors: {len(errors)}")

        # Show hardest pages
        hardest = df_results.nsmallest(5, 'flesch_reading_ease')
        print(f"\nHardest to read pages:")
        for _, row in hardest.iterrows():
            print(f"  [{row['flesch_reading_ease']:.0f}] {row['url'][:60]}")
    else:
        print("No pages could be analyzed")
        if errors:
            print("Errors:")
            for err in errors[:10]:
                print(f"  {err['url'][:50]}: {err['error']}")


if __name__ == '__main__':
    main()
