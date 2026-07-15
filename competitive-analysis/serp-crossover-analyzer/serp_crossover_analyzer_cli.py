# Author: Lee Foot
# Website: https://leefoot.com

"""
SERP Crossover Analyzer - CLI Version

Analyse URL overlap across multiple keyword SERPs using the DataForSEO API.

Usage:
    python serp_crossover_analyzer_cli.py --keywords "seo tools" "keyword research" --login YOUR_LOGIN --password YOUR_PASS
    python serp_crossover_analyzer_cli.py --keywords "seo tools" "keyword research"  # uses env vars

Environment variables:
    DATAFORSEO_LOGIN    - DataForSEO account login
    DATAFORSEO_PASSWORD - DataForSEO account password
"""

import argparse
import math
import pandas as pd
import requests
import os
import sys
from base64 import b64encode
from urllib.parse import urlparse

LOCATION_CODES = {
    "United Kingdom": 2826,
    "United States": 2840,
    "Australia": 2036,
    "Canada": 2124,
    "Germany": 2276,
    "France": 2250,
    "Spain": 2724,
    "Italy": 2380,
    "Netherlands": 2528,
    "India": 2356,
}


def extract_domain(url):
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc.replace('www.', '')
    except Exception:
        return url


def fetch_serp(keyword, login, password, location_code, device, num_results):
    """Fetch SERP results for a keyword using the DataForSEO API."""
    cred = b64encode(f"{login}:{password}".encode()).decode()
    headers = {
        'Authorization': f'Basic {cred}',
        'Content-Type': 'application/json'
    }

    payload = [{
        "keyword": keyword,
        "location_code": location_code,
        "language_code": "en",
        "device": device.lower(),
        "depth": num_results
    }]

    try:
        response = requests.post(
            'https://api.dataforseo.com/v3/serp/google/organic/live/advanced',
            headers=headers,
            json=payload,
            timeout=60
        )
        data = response.json()

        items = data["tasks"][0]["result"][0]["items"]
        organic = [item for item in items if item["type"] == "organic"]

        results = []
        for i, result in enumerate(organic[:num_results]):
            results.append({
                'position': result.get('rank_group', i + 1),
                'title': result.get('title', ''),
                'link': result.get('url', ''),
                'domain': extract_domain(result.get('url', ''))
            })

        return results

    except Exception as e:
        print(f"  Warning: Error fetching '{keyword}': {str(e)}")
        return []


def calculate_crossover(serp_data):
    """Calculate crossover matrix between keywords."""
    keywords = list(serp_data.keys())
    url_sets = {kw: set(r['link'] for r in results) for kw, results in serp_data.items()}

    matrix = pd.DataFrame(index=keywords, columns=keywords, dtype=float)

    for i, kw1 in enumerate(keywords):
        for j, kw2 in enumerate(keywords):
            if i == j:
                matrix.loc[kw1, kw2] = 100.0
            else:
                common = url_sets[kw1].intersection(url_sets[kw2])
                union = url_sets[kw1].union(url_sets[kw2])
                if union:
                    crossover = (len(common) / len(union)) * 100
                else:
                    crossover = 0.0
                matrix.loc[kw1, kw2] = round(crossover, 1)

    return matrix


def find_overlapping_urls(serp_data):
    """Find URLs that appear in multiple SERPs."""
    url_keywords = {}

    for keyword, results in serp_data.items():
        for result in results:
            url = result['link']
            if url not in url_keywords:
                url_keywords[url] = {
                    'title': result['title'],
                    'domain': result['domain'],
                    'keywords': [],
                    'positions': []
                }
            url_keywords[url]['keywords'].append(keyword)
            url_keywords[url]['positions'].append(result['position'])

    overlapping = []
    for url, data in url_keywords.items():
        if len(data['keywords']) > 1:
            overlapping.append({
                'URL': url,
                'Domain': data['domain'],
                'Title': data['title'],
                'Keywords Count': len(data['keywords']),
                'Keywords': ', '.join(data['keywords']),
                'Positions': ', '.join(map(str, data['positions']))
            })

    df = pd.DataFrame(overlapping)
    if not df.empty:
        df = df.sort_values('Keywords Count', ascending=False)

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Analyse URL overlap across multiple keyword SERPs using the DataForSEO API'
    )
    parser.add_argument('--keywords', required=True, nargs='+',
                        help='Keywords to compare')
    parser.add_argument('--login',
                        help='DataForSEO login (or set DATAFORSEO_LOGIN env var)')
    parser.add_argument('--password',
                        help='DataForSEO password (or set DATAFORSEO_PASSWORD env var)')
    parser.add_argument('--output', default='serp_crossover.csv',
                        help='Output CSV path (default: serp_crossover.csv)')
    parser.add_argument('--location', default='United Kingdom',
                        choices=list(LOCATION_CODES.keys()),
                        help='Search location name (default: United Kingdom)')
    parser.add_argument('--location-code', type=int, default=None,
                        help='DataForSEO location code (overrides --location, default: 2826)')
    parser.add_argument('--device', choices=['desktop', 'mobile', 'tablet'],
                        default='desktop', help='Device type (default: desktop)')
    parser.add_argument('--num-results', type=int, default=10,
                        help='Results per SERP (default: 10)')

    args = parser.parse_args()

    # Get credentials
    login = args.login or os.environ.get('DATAFORSEO_LOGIN')
    password = args.password or os.environ.get('DATAFORSEO_PASSWORD')

    if not login or not password:
        print("Error: DataForSEO credentials required.")
        print("Use --login and --password, or set DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD environment variables.")
        sys.exit(1)

    keywords = args.keywords
    if len(keywords) < 2:
        print("Error: Need at least 2 keywords to compare")
        sys.exit(1)

    # Resolve location code
    location_code = args.location_code if args.location_code else LOCATION_CODES[args.location]

    # Show estimated cost
    estimated_cost = len(keywords) * 0.002 * math.ceil(args.num_results / 10)

    print(f"Analysing {len(keywords)} keywords...")
    print(f"  Location: {args.location} (code: {location_code})")
    print(f"  Device: {args.device}")
    print(f"  Results per SERP: {args.num_results}")
    print(f"  Estimated cost: ${estimated_cost:.3f}")

    # Fetch SERPs
    serp_data = {}
    for keyword in keywords:
        print(f"  Fetching: {keyword}")
        results = fetch_serp(keyword, login, password, location_code, args.device, args.num_results)
        if results:
            serp_data[keyword] = results

    if len(serp_data) < 2:
        print("Error: Need at least 2 successful SERP fetches")
        sys.exit(1)

    # Calculate crossover
    matrix = calculate_crossover(serp_data)
    overlapping_df = find_overlapping_urls(serp_data)

    # Save matrix
    matrix.to_csv(args.output, encoding='utf-8-sig')
    print(f"\nCrossover matrix saved to: {args.output}")

    # Save overlapping URLs
    if not overlapping_df.empty:
        overlap_path = args.output.replace('.csv', '_overlaps.csv')
        overlapping_df.to_csv(overlap_path, index=False, encoding='utf-8-sig')
        print(f"Overlapping URLs saved to: {overlap_path}")

    # Print summary
    print(f"\nCrossover Matrix:")
    print(matrix.to_string())

    # Average crossover
    values = []
    for i, row in enumerate(matrix.values):
        for j, val in enumerate(row):
            if i != j:
                values.append(val)
    avg_crossover = sum(values) / len(values) if values else 0
    print(f"\nAverage crossover: {avg_crossover:.1f}%")

    if not overlapping_df.empty:
        print(f"\nOverlapping URLs: {len(overlapping_df)}")
        for _, row in overlapping_df.head(5).iterrows():
            print(f"  [{row['Keywords Count']} keywords] {row['Domain']}")


if __name__ == '__main__':
    main()
