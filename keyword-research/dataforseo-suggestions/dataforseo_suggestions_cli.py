#!/usr/bin/env python3
"""
DataForSEO Keyword Suggestions - CLI Version

Get keyword suggestions with search volumes from DataForSEO API.

Usage:
    python dataforseo_suggestions_cli.py --login your@email.com --password yourpassword --keywords "seo tools"

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import json
import sys


def get_keyword_suggestions(login, password, keyword, location, language, max_results, include_seed):
    """Get keyword suggestions from DataForSEO API."""
    base_url = "https://api.dataforseo.com/v3/dataforseo_labs/google/keyword_suggestions/live"

    post_data = [{
        "keyword": keyword,
        "location_name": location,
        "language_name": language,
        "include_serp_info": False,
        "include_seed_keyword": include_seed,
        "limit": max_results
    }]

    try:
        response = requests.post(
            base_url,
            auth=HTTPBasicAuth(login, password),
            headers={"Content-Type": "application/json"},
            data=json.dumps(post_data),
            timeout=60
        )

        response_data = response.json()

        if response_data.get("status_code") == 20000:
            results = []
            tasks = response_data.get('tasks', [])

            if tasks and tasks[0].get('result'):
                result = tasks[0]['result'][0]
                seed = result.get('seed_keyword', keyword)
                items = result.get('items', [])

                for item in items:
                    keyword_info = item.get('keyword_info', {})
                    results.append({
                        'seed_keyword': seed,
                        'suggested_keyword': item.get('keyword', '').replace('-', ' '),
                        'search_volume': keyword_info.get('search_volume', 0),
                        'cpc': keyword_info.get('cpc', 0),
                        'competition': keyword_info.get('competition', 0),
                        'competition_level': keyword_info.get('competition_level', '')
                    })

            return results, None
        else:
            error_msg = response_data.get('status_message', 'Unknown error')
            return None, f"API Error: {error_msg}"

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"
    except Exception as e:
        return None, f"Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description='Get keyword suggestions from DataForSEO API'
    )
    parser.add_argument('--login', required=True, help='DataForSEO login (email)')
    parser.add_argument('--password', required=True, help='DataForSEO API password')
    parser.add_argument('--keywords', nargs='+', help='Seed keywords (space-separated)')
    parser.add_argument('--keywords-file', help='File with seed keywords (one per line)')
    parser.add_argument('--output', default='keyword_suggestions.csv',
                        help='Output CSV path (default: keyword_suggestions.csv)')
    parser.add_argument('--location', default='United Kingdom',
                        help='Location for search volume (default: United Kingdom)')
    parser.add_argument('--language', default='English',
                        help='Language (default: English)')
    parser.add_argument('--limit', type=int, default=100,
                        help='Max suggestions per keyword (default: 100)')
    parser.add_argument('--include-seed', action='store_true',
                        help='Include seed keyword in results')

    args = parser.parse_args()

    # Get keywords
    seed_keywords = []
    if args.keywords:
        seed_keywords = args.keywords
    elif args.keywords_file:
        with open(args.keywords_file, 'r') as f:
            seed_keywords = [line.strip() for line in f if line.strip()]
    else:
        print("Error: Provide keywords via --keywords or --keywords-file")
        sys.exit(1)

    print(f"Processing {len(seed_keywords)} seed keywords...")
    print(f"  Location: {args.location}")
    print(f"  Language: {args.language}")
    print(f"  Max suggestions per keyword: {args.limit}")

    all_results = []
    errors = []

    for i, seed in enumerate(seed_keywords):
        print(f"  [{i + 1}/{len(seed_keywords)}] {seed}...")

        results, error = get_keyword_suggestions(
            args.login,
            args.password,
            seed,
            args.location,
            args.language,
            args.limit,
            args.include_seed
        )

        if results:
            all_results.extend(results)
            print(f"    Found {len(results)} suggestions")
        if error:
            errors.append({'seed_keyword': seed, 'error': error})
            print(f"    Error: {error}")

    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results = df_results.drop_duplicates(subset=['suggested_keyword'])
        df_results = df_results.sort_values('search_volume', ascending=False)

        df_results.to_csv(args.output, index=False, encoding='utf-8')

        print(f"\nResults saved to: {args.output}")
        print(f"  Total suggestions: {len(df_results):,}")
        print(f"  Total search volume: {df_results['search_volume'].sum():,}")

        if errors:
            print(f"  Errors: {len(errors)}")
    else:
        print("No results returned. Check your API credentials.")
        if errors:
            for err in errors:
                print(f"  {err['seed_keyword']}: {err['error']}")


if __name__ == '__main__':
    main()
