# Author: Lee Foot
# Website: https://leefoot.com
"""
DataForSEO Google Ads Search Volume Tool - CLI Version

Fetch search volume data from DataForSEO Google Ads Search Volume API.

Usage:
    python keywords_everywhere_api_cli.py --input keywords.csv --login YOUR_LOGIN --password YOUR_PASS --country uk
"""

import argparse
import pandas as pd
import requests
import sys
import os
import time
import math


# Country -> (location_code, language_code)
COUNTRIES = {
    "uk": (2826, "en"),
    "us": (2840, "en"),
    "au": (2036, "en"),
    "ca": (2124, "en"),
    "de": (2276, "de"),
    "fr": (2250, "fr"),
    "es": (2724, "es"),
    "it": (2380, "it"),
    "nl": (2528, "nl"),
    "br": (2076, "pt"),
    "in": (2356, "en"),
    "jp": (2392, "ja"),
}


def fetch_keyword_data(keywords, login, password, location_code, language_code):
    """Fetch keyword data from DataForSEO Google Ads Search Volume API in batches."""
    url = "https://api.dataforseo.com/v3/keywords_data/google_ads/search_volume/live"

    results = []
    batch_size = 700
    total_batches = math.ceil(len(keywords) / batch_size)

    for batch_num, i in enumerate(range(0, len(keywords), batch_size), start=1):
        chunk = keywords[i:i + batch_size]
        print(f"  Processing batch {batch_num}/{total_batches} ({len(chunk)} keywords)...")

        payload = [{
            "keywords": chunk,
            "location_code": location_code,
            "language_code": language_code,
        }]

        try:
            response = requests.post(
                url,
                json=payload,
                auth=(login, password),
            )

            if response.status_code == 200:
                resp_json = response.json()
                tasks = resp_json.get("tasks", [])
                if not tasks:
                    print("API returned no tasks.")
                    return None

                task = tasks[0]
                if task.get("status_code") != 20000:
                    print(f"API Error: {task.get('status_message', 'Unknown error')}")
                    return None

                task_results = task.get("result", [])
                if task_results is None:
                    task_results = []

                for item in task_results:
                    results.append({
                        "Keyword": item.get("keyword", ""),
                        "Search Volume": item.get("search_volume") or 0,
                        "CPC": item.get("cpc") or 0,
                        "Competition": item.get("competition") or "",
                        "Competition Index": item.get("competition_index") or 0,
                        "Low Top of Page Bid": item.get("low_top_of_page_bid") or 0,
                        "High Top of Page Bid": item.get("high_top_of_page_bid") or 0,
                    })
            else:
                try:
                    error_detail = response.json()
                    error_msg = error_detail.get("status_message", response.text)
                except Exception:
                    error_msg = response.text
                print(f"HTTP {response.status_code}: {error_msg}")
                return None

        except Exception as e:
            print(f"Error processing batch {batch_num}: {str(e)}")
            return None

        # Rate limit: 12 requests/min. Sleep 5s between batches if more than one.
        if total_batches > 1 and batch_num < total_batches:
            time.sleep(5)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Fetch search volume data from DataForSEO Google Ads Search Volume API'
    )
    parser.add_argument('--input', required=True,
                        help='Input CSV with keywords or text file (one per line)')
    parser.add_argument('--output', default='keyword_volumes.csv',
                        help='Output CSV path (default: keyword_volumes.csv)')
    parser.add_argument('--login',
                        help='DataForSEO login (or set DATAFORSEO_LOGIN env var)')
    parser.add_argument('--password',
                        help='DataForSEO password (or set DATAFORSEO_PASSWORD env var)')
    parser.add_argument('--country', choices=list(COUNTRIES.keys()), default='uk',
                        help='Country code (default: uk)')
    parser.add_argument('--keyword-col', default='keyword',
                        help='Column name for keywords in CSV (default: keyword)')

    args = parser.parse_args()

    # Get credentials
    login = args.login or os.environ.get('DATAFORSEO_LOGIN')
    password = args.password or os.environ.get('DATAFORSEO_PASSWORD')
    if not login or not password:
        print("Error: DataForSEO credentials required. "
              "Use --login and --password flags or set DATAFORSEO_LOGIN and "
              "DATAFORSEO_PASSWORD environment variables.")
        sys.exit(1)

    # Load keywords
    print(f"Loading keywords from: {args.input}")

    if args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
        # Find keyword column
        keyword_col = None
        for col in df.columns:
            if col.lower() == args.keyword_col.lower():
                keyword_col = col
                break
        if not keyword_col:
            keyword_col = df.columns[0]
        keywords = df[keyword_col].dropna().astype(str).tolist()
    else:
        # Text file, one keyword per line
        with open(args.input, 'r', encoding='utf-8') as f:
            keywords = [line.strip() for line in f if line.strip()]

    print(f"  Loaded {len(keywords)} keywords")

    # Get location and language codes
    location_code, language_code = COUNTRIES[args.country]
    print(f"  Country: {args.country}, Location code: {location_code}, Language: {language_code}")

    # Estimated cost
    num_requests = math.ceil(len(keywords) / 700)
    print(f"  This will use {num_requests} API request{'s' if num_requests != 1 else ''} "
          f"(up to 700 keywords each). DataForSEO charges per request, not per keyword.")

    # Fetch data
    print("Fetching keyword data...")
    results = fetch_keyword_data(keywords, login, password, location_code, language_code)

    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(args.output, index=False, encoding='utf-8-sig')

        print(f"\nResults saved to: {args.output}")
        print(f"  Keywords: {len(df_results)}")
        print(f"  Total volume: {df_results['Search Volume'].sum():,}")
        print(f"  Avg CPC: ${df_results['CPC'].mean():.2f} (USD)")

        # Show top keywords
        print(f"\nTop keywords by volume:")
        top_kws = df_results.nlargest(10, 'Search Volume')
        for _, row in top_kws.iterrows():
            print(f"  [{row['Search Volume']:>8,}] {row['Keyword']}")
    else:
        print("Failed to fetch keyword data")
        sys.exit(1)


if __name__ == '__main__':
    main()
