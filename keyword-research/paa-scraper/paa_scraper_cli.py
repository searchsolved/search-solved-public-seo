#!/usr/bin/env python3
"""
People Also Ask (PAA) Scraper - CLI Version
Recursively extracts PAA questions from search results using ValueSERP API.

Author: Lee Foot
Website: https://leefoot.co.uk

Usage:
    python paa_scraper_cli.py "what is SEO" --api-key YOUR_KEY
    python paa_scraper_cli.py -f keywords.txt --api-key YOUR_KEY --depth 3
"""

import argparse
import sys
import os
import time
import json
from datetime import datetime

import requests
import pandas as pd


COUNTRY_SETTINGS = {
    "us": ("United States", "google.com"),
    "uk": ("United Kingdom", "google.co.uk"),
    "ca": ("Canada", "google.ca"),
    "au": ("Australia", "google.com.au"),
    "de": ("Germany", "google.de"),
    "fr": ("France", "google.fr"),
    "es": ("Spain", "google.es"),
    "it": ("Italy", "google.it"),
    "nl": ("Netherlands", "google.nl"),
    "br": ("Brazil", "google.com.br"),
    "mx": ("Mexico", "google.com.mx"),
    "in": ("India", "google.co.in"),
    "jp": ("Japan", "google.co.jp")
}


def get_related_questions(query, api_key, location, country_code, google_domain,
                         language, device, max_depth, delay,
                         level=1, all_questions=None, parent=None,
                         original_query=None, verbose=False):
    """Recursively fetch related questions from ValueSERP API."""
    if all_questions is None:
        all_questions = []
        original_query = query

    if level > max_depth:
        return all_questions

    if verbose:
        print(f"  Level {level}: Querying '{query[:60]}...'")

    params = {
        'api_key': api_key,
        'q': query,
        'gl': country_code,
        'hl': language,
        'location': location,
        'google_domain': google_domain,
        'device': device.lower(),
        'output': 'json',
        'page': '1',
        'num': '10',
        'include_fields': 'related_questions'
    }

    try:
        response = requests.get('https://api.valueserp.com/search', params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        questions = data.get('related_questions', [])

        if not questions:
            return all_questions

        if verbose:
            print(f"    Found {len(questions)} PAA questions")

        for q in questions:
            question_text = q.get('question', '')

            if not question_text:
                continue

            question_data = {
                'original_query': original_query,
                'level': level,
                'parent_query': parent if parent else query,
                'question': question_text,
                'answer_snippet': q.get('answer', {}).get('text', '') if isinstance(q.get('answer'), dict) else '',
                'source_url': q.get('answer', {}).get('link', '') if isinstance(q.get('answer'), dict) else '',
                'source_title': q.get('answer', {}).get('title', '') if isinstance(q.get('answer'), dict) else ''
            }

            # Check for duplicates
            if not any(d.get('question') == question_text for d in all_questions):
                all_questions.append(question_data)

                # Recursively query next level
                if level < max_depth:
                    time.sleep(delay)
                    get_related_questions(
                        question_text,
                        api_key, location, country_code, google_domain,
                        language, device, max_depth, delay,
                        level=level + 1,
                        all_questions=all_questions,
                        parent=question_text,
                        original_query=original_query,
                        verbose=verbose
                    )

    except requests.exceptions.RequestException as e:
        print(f"Error querying '{query}': {str(e)}", file=sys.stderr)

    return all_questions


def main():
    parser = argparse.ArgumentParser(
        description="Extract 'People Also Ask' questions using ValueSERP API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s "what is SEO" --api-key YOUR_KEY
    %(prog)s "best keyword tools" --api-key YOUR_KEY --depth 3 --country uk
    %(prog)s -f keywords.txt --api-key YOUR_KEY -o paa_results.csv

Country codes: us, uk, ca, au, de, fr, es, it, nl, br, mx, in, jp

Author: Lee Foot (https://leefoot.co.uk)
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("keyword", nargs="?", help="Single keyword to analyze")
    input_group.add_argument("-f", "--file", help="File with keywords (one per line)")

    # API configuration
    parser.add_argument("--api-key", required=True, help="ValueSERP API key")

    # Search settings
    parser.add_argument("--country", default="us", choices=list(COUNTRY_SETTINGS.keys()),
                        help="Country code for search (default: us)")
    parser.add_argument("--language", default="en",
                        help="Language code (default: en)")
    parser.add_argument("--device", default="desktop", choices=["desktop", "mobile", "tablet"],
                        help="Device type (default: desktop)")

    # Scrape settings
    parser.add_argument("--depth", type=int, default=2, choices=range(1, 6),
                        help="Max depth to follow PAA questions (1-5, default: 2)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between API requests in seconds (default: 0.5)")

    # Output settings
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--format", choices=["csv", "json", "xlsx"], default="csv",
                        help="Output format (default: csv)")

    # Verbosity
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    # Get keywords
    if args.keyword:
        keywords = [args.keyword]
    else:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        with open(args.file, 'r') as f:
            keywords = [line.strip() for line in f if line.strip()]

    if not keywords:
        print("Error: No keywords to process.", file=sys.stderr)
        sys.exit(1)

    # Get country settings
    location, google_domain = COUNTRY_SETTINGS[args.country]

    if not args.quiet:
        print(f"Processing {len(keywords)} keyword(s)")
        print(f"Settings: {location}, {args.language}, {args.device}, depth={args.depth}")

    # Process keywords
    all_results = []
    start_time = time.time()

    for i, keyword in enumerate(keywords, 1):
        if not args.quiet:
            print(f"\n[{i}/{len(keywords)}] Processing: {keyword}")

        results = get_related_questions(
            keyword,
            args.api_key,
            location,
            args.country,
            google_domain,
            args.language,
            args.device,
            args.depth,
            args.delay,
            verbose=args.verbose
        )

        all_results.extend(results)

    elapsed = time.time() - start_time

    if not all_results:
        print("No PAA questions found.", file=sys.stderr)
        sys.exit(1)

    # Create DataFrame
    df = pd.DataFrame(all_results)
    columns = ['original_query', 'level', 'parent_query', 'question',
               'answer_snippet', 'source_url', 'source_title']
    columns = [c for c in columns if c in df.columns]
    df = df[columns]

    # Generate output filename
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"paa_questions_{timestamp}.{args.format}"

    # Save output
    if args.format == "csv":
        df.to_csv(output_file, index=False)
    elif args.format == "json":
        df.to_json(output_file, orient="records", indent=2)
    elif args.format == "xlsx":
        df.to_excel(output_file, index=False, sheet_name="PAA Questions")

    if not args.quiet:
        print(f"\nResults Summary:")
        print(f"  Total questions: {len(df):,}")
        print(f"  Keywords processed: {len(keywords)}")
        print(f"  Time elapsed: {elapsed:.1f}s")
        print(f"\nQuestions by level:")
        for level, count in df['level'].value_counts().sort_index().items():
            print(f"  Level {level}: {count}")
        print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()
