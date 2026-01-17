#!/usr/bin/env python3
"""
Keywords Everywhere API Tool - CLI Version

Fetch search volume data from Keywords Everywhere API.

Usage:
    python keywords_everywhere_api_cli.py --input keywords.csv --api-key YOUR_KEY --country uk

Author: Lee Foot
Website: https://www.leefoot.com
"""

import argparse
import pandas as pd
import requests
import sys
import os


COUNTRIES = {
    "uk": "GBP",
    "us": "USD",
    "au": "AUD",
    "ca": "CAD",
    "de": "EUR",
    "fr": "EUR",
    "es": "EUR",
    "it": "EUR",
    "nl": "EUR",
    "br": "BRL",
    "in": "INR",
    "jp": "JPY"
}


def fetch_keyword_data(keywords, api_key, country, currency, data_source):
    """Fetch keyword data from Keywords Everywhere API in batches."""
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    results = []
    batch_size = 100

    for i in range(0, len(keywords), batch_size):
        chunk = keywords[i:i + batch_size]
        print(f"  Processing batch {i//batch_size + 1} ({len(chunk)} keywords)...")

        data = {
            'country': country,
            'currency': currency,
            'dataSource': data_source,
            'kw[]': chunk
        }

        try:
            response = requests.post(
                'https://api.keywordseverywhere.com/v1/get_keyword_data',
                data=data,
                headers=headers
            )

            if response.status_code == 200:
                keywords_data = response.json().get('data', [])

                for idx, element in enumerate(keywords_data):
                    if idx < len(chunk):
                        results.append({
                            'Keyword': chunk[idx],
                            'Volume': element.get('vol', 0),
                            'CPC': element.get('cpc', {}).get('value', 0),
                            'Competition': element.get('competition', 0)
                        })
            else:
                error_msg = response.json().get('message', 'Unknown error')
                print(f"API Error: {error_msg}")
                return None

        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            return None

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Fetch search volume data from Keywords Everywhere API'
    )
    parser.add_argument('--input', required=True, help='Input CSV with keywords or text file (one per line)')
    parser.add_argument('--output', default='keyword_volumes.csv',
                        help='Output CSV path (default: keyword_volumes.csv)')
    parser.add_argument('--api-key', help='Keywords Everywhere API key (or set KWE_API_KEY env var)')
    parser.add_argument('--country', choices=list(COUNTRIES.keys()), default='uk',
                        help='Country code (default: uk)')
    parser.add_argument('--data-source', choices=['gkp', 'cli'], default='gkp',
                        help='Data source: gkp (Google Keyword Planner) or cli (Clickstream)')
    parser.add_argument('--keyword-col', default='keyword',
                        help='Column name for keywords in CSV (default: keyword)')

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get('KWE_API_KEY')
    if not api_key:
        print("Error: API key required. Use --api-key or set KWE_API_KEY environment variable")
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

    # Get currency
    currency = COUNTRIES[args.country]
    print(f"  Country: {args.country}, Currency: {currency}")

    # Fetch data
    print("Fetching keyword data...")
    results = fetch_keyword_data(keywords, api_key, args.country, currency, args.data_source)

    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(args.output, index=False, encoding='utf-8-sig')

        print(f"\nResults saved to: {args.output}")
        print(f"  Keywords: {len(df_results)}")
        print(f"  Total volume: {df_results['Volume'].sum():,}")
        print(f"  Avg CPC: {currency} {df_results['CPC'].mean():.2f}")

        # Show top keywords
        print(f"\nTop keywords by volume:")
        top_kws = df_results.nlargest(10, 'Volume')
        for _, row in top_kws.iterrows():
            print(f"  [{row['Volume']:>8,}] {row['Keyword']}")
    else:
        print("Failed to fetch keyword data")
        sys.exit(1)


if __name__ == '__main__':
    main()
