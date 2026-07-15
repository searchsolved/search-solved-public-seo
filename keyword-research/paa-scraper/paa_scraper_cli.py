# Author: Lee Foot
# Website: https://leefoot.com
"""
People Also Ask (PAA) Scraper - CLI Version
Extracts PAA questions from search results using the DataForSEO SERP API.
DataForSEO handles recursive PAA expansion server-side via people_also_ask_click_depth,
so only one API call is needed per seed keyword.

Usage:
    python paa_scraper_cli.py "what is SEO" --login YOUR_LOGIN --password YOUR_PASS
    python paa_scraper_cli.py -f keywords.txt --depth 3
    DATAFORSEO_LOGIN=x DATAFORSEO_PASSWORD=y python paa_scraper_cli.py "best tools"
"""

import argparse
import sys
import os
import time
import json
from base64 import b64encode
from datetime import datetime

import requests
import pandas as pd


LOCATION_CODES = {
    "uk": ("United Kingdom", 2826),
    "us": ("United States", 2840),
    "ca": ("Canada", 2124),
    "au": ("Australia", 2036),
    "de": ("Germany", 2276),
    "fr": ("France", 2250),
    "es": ("Spain", 2724),
    "it": ("Italy", 2380),
    "nl": ("Netherlands", 2528),
    "br": ("Brazil", 2076),
    "mx": ("Mexico", 2484),
    "in": ("India", 2356),
    "jp": ("Japan", 2392),
}


def build_auth_header(login, password):
    """Build the Basic Auth header for DataForSEO."""
    cred = b64encode(f"{login}:{password}".encode()).decode()
    return {
        "Authorization": f"Basic {cred}",
        "Content-Type": "application/json"
    }


def extract_paa_items(items, original_query, parent_query, level, all_questions, seen_questions):
    """
    Recursively extract PAA questions from the DataForSEO response items.
    Walks through top-level items and any nested expanded_element lists.
    """
    if items is None:
        return

    for item in items:
        if not isinstance(item, dict):
            continue

        # Check for PAA container at top level
        if item.get("type") == "people_also_ask":
            paa_questions = item.get("items", [])
            for q in paa_questions:
                if not isinstance(q, dict):
                    continue
                question_text = q.get("title", "")
                if not question_text or question_text in seen_questions:
                    continue
                seen_questions.add(question_text)

                question_data = {
                    "original_query": original_query,
                    "level": level,
                    "parent_query": parent_query,
                    "question": question_text,
                    "answer_snippet": q.get("snippet", ""),
                    "source_url": q.get("url", ""),
                    "source_title": q.get("domain", ""),
                }
                all_questions.append(question_data)

                # Recurse into expanded elements (deeper PAA levels)
                expanded = q.get("expanded_element", [])
                if expanded:
                    extract_paa_items(
                        expanded, original_query, question_text,
                        level + 1, all_questions, seen_questions
                    )


def fetch_paa_for_keyword(keyword, headers, location_code, language_code, device_type,
                          click_depth, verbose=False):
    """
    Fetch PAA questions for a single keyword using DataForSEO.
    One API call per keyword; DataForSEO handles recursive expansion.
    """
    if verbose:
        print(f"  Querying DataForSEO for '{keyword}'")

    payload = [{
        "keyword": keyword,
        "location_code": location_code,
        "language_code": language_code,
        "device": device_type,
        "depth": 10,
        "people_also_ask_click_depth": click_depth,
    }]

    try:
        response = requests.post(
            "https://api.dataforseo.com/v3/serp/google/organic/live/advanced",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        # Validate response
        if data.get("status_code") != 20000:
            msg = data.get("status_message", "Unknown API error")
            print(f"API error: {msg}", file=sys.stderr)
            return []

        tasks = data.get("tasks", [])
        if not tasks:
            return []

        task = tasks[0]
        if task.get("status_code") != 20000:
            msg = task.get("status_message", "Task error")
            print(f"Task error for '{keyword}': {msg}", file=sys.stderr)
            return []

        results = task.get("result", [])
        if not results:
            return []

        items = results[0].get("items", [])
        all_questions = []
        seen_questions = set()
        extract_paa_items(items, keyword, keyword, 1, all_questions, seen_questions)

        if verbose:
            print(f"    Found {len(all_questions)} PAA questions")

        return all_questions

    except requests.exceptions.RequestException as e:
        print(f"Error querying '{keyword}': {str(e)}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Extract 'People Also Ask' questions using the DataForSEO SERP API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s "what is SEO" --login YOUR_LOGIN --password YOUR_PASS
    %(prog)s "best keyword tools" --depth 3 --country uk
    %(prog)s -f keywords.txt -o paa_results.csv

Credentials can also be set via environment variables:
    DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD

Country codes: uk, us, ca, au, de, fr, es, it, nl, br, mx, in, jp

Author: Lee Foot (https://leefoot.com)
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("keyword", nargs="?", help="Single keyword to analyse")
    input_group.add_argument("-f", "--file", help="File with keywords (one per line)")

    # API configuration
    parser.add_argument("--login",
                        default=os.environ.get("DATAFORSEO_LOGIN", ""),
                        help="DataForSEO login (or set DATAFORSEO_LOGIN env var)")
    parser.add_argument("--password",
                        default=os.environ.get("DATAFORSEO_PASSWORD", ""),
                        help="DataForSEO password (or set DATAFORSEO_PASSWORD env var)")

    # Search settings
    parser.add_argument("--country", default="us", choices=list(LOCATION_CODES.keys()),
                        help="Country code for search (default: us)")
    parser.add_argument("--language", default="en",
                        help="Language code (default: en)")
    parser.add_argument("--device", default="desktop", choices=["desktop", "mobile"],
                        help="Device type (default: desktop)")

    # Scrape settings
    parser.add_argument("--depth", type=int, default=2, choices=range(1, 5),
                        metavar="{1,2,3,4}",
                        help="PAA click depth (1-4, default: 2). DataForSEO handles expansion server-side.")
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

    # Validate credentials
    if not args.login or not args.password:
        print("Error: DataForSEO credentials required. Provide --login and --password, "
              "or set DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD environment variables.",
              file=sys.stderr)
        sys.exit(1)

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

    # Get location settings
    location_name, location_code = LOCATION_CODES[args.country]
    headers = build_auth_header(args.login, args.password)

    if not args.quiet:
        estimated_cost = len(keywords) * 0.002
        print(f"Processing {len(keywords)} keyword(s)")
        print(f"Settings: {location_name}, lang={args.language}, device={args.device}, click_depth={args.depth}")
        print(f"Estimated cost: ${estimated_cost:.3f}")

    # Process keywords
    all_results = []
    start_time = time.time()

    for i, keyword in enumerate(keywords, 1):
        if not args.quiet:
            print(f"\n[{i}/{len(keywords)}] Processing: {keyword}")

        results = fetch_paa_for_keyword(
            keyword,
            headers,
            location_code,
            args.language,
            args.device,
            args.depth,
            verbose=args.verbose
        )

        all_results.extend(results)

        # Rate limit between keywords
        if i < len(keywords) and args.delay > 0:
            time.sleep(args.delay)

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
        actual_cost = len(keywords) * 0.002
        print(f"\nResults Summary:")
        print(f"  Total questions: {len(df):,}")
        print(f"  Keywords processed: {len(keywords)}")
        print(f"  Estimated cost: ${actual_cost:.3f}")
        print(f"  Time elapsed: {elapsed:.1f}s")
        print(f"\nQuestions by level:")
        for level, count in df['level'].value_counts().sort_index().items():
            print(f"  Level {level}: {count}")
        print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()
