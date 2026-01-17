#!/usr/bin/env python3
"""
Google Algorithm Tracker - CLI Version

Scrape Google's Search Status page for algorithm updates.

Usage:
    python google_algorithm_tracker_cli.py --output updates.csv

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta


def classify_update(summary):
    """Classify the type of algorithm update."""
    summary_lower = summary.lower()

    classifications = [
        ('core update', 'Core Update'),
        ('spam update', 'Spam Update'),
        ('helpful content update', 'Helpful Content Update'),
        ('helpful content system', 'Helpful Content Update'),
        ('product reviews update', 'Product Reviews Update'),
        ('reviews update', 'Reviews Update'),
        ('link spam update', 'Link Spam Update'),
        ('page experience update', 'Page Experience Update'),
        ('site reputation abuse', 'Site Reputation Abuse'),
        ('expired domain abuse', 'Expired Domain Abuse'),
        ('scaled content abuse', 'Scaled Content Abuse'),
    ]

    for keyword, update_type in classifications:
        if keyword in summary_lower:
            return update_type

    return 'Other'


def scrape_algorithm_updates():
    """Scrape Google Search Status page for algorithm updates."""
    url = 'https://status.search.google.com/products/rGHU1u87FJnkP6W2GwMi/history'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')
    tables = soup.find_all('table')

    all_data = []

    for table in tables:
        rows = table.find_all('tr')

        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 1:
                summary = cols[0].text.strip()
                date_text = cols[1].text.strip()

                try:
                    date_obj = datetime.strptime(date_text, "%d %b %Y")
                    date_formatted = date_obj.strftime("%Y-%m-%d")
                except ValueError:
                    date_formatted = None
                    date_obj = None

                update_type = classify_update(summary)

                all_data.append({
                    'Date': date_formatted,
                    'Date_obj': date_obj,
                    'Summary': summary,
                    'Update Type': update_type
                })

    df = pd.DataFrame(all_data)
    df = df.sort_values('Date', ascending=False)

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Scrape Google Search Status page for algorithm updates'
    )
    parser.add_argument('--output', default='algorithm_updates.csv',
                        help='Output file path (default: algorithm_updates.csv)')
    parser.add_argument('--days', type=int, default=0,
                        help='Filter to last N days (0 = all time)')
    parser.add_argument('--type', choices=['core', 'spam', 'helpful', 'reviews', 'all'],
                        default='all', help='Filter by update type')
    parser.add_argument('--format', choices=['csv', 'excel'], default='csv',
                        help='Output format (default: csv)')

    args = parser.parse_args()

    print("Fetching algorithm updates from Google...")
    df = scrape_algorithm_updates()

    print(f"  Found {len(df)} total updates")

    # Apply date filter
    if args.days > 0:
        cutoff = datetime.now() - timedelta(days=args.days)
        df = df[df['Date_obj'] >= cutoff]
        print(f"  After date filter: {len(df)} updates")

    # Apply type filter
    type_map = {
        'core': 'Core Update',
        'spam': 'Spam Update',
        'helpful': 'Helpful Content Update',
        'reviews': 'Reviews Update'
    }
    if args.type != 'all':
        df = df[df['Update Type'] == type_map.get(args.type, args.type)]
        print(f"  After type filter: {len(df)} updates")

    # Drop helper column
    df = df.drop(columns=['Date_obj'])

    # Save
    if args.format == 'excel' or args.output.endswith('.xlsx'):
        output_path = args.output if args.output.endswith('.xlsx') else args.output.replace('.csv', '.xlsx')
        df.to_excel(output_path, index=False)
    else:
        output_path = args.output
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\nResults saved to: {output_path}")

    # Summary
    print(f"\nUpdate type summary:")
    for update_type, count in df['Update Type'].value_counts().items():
        print(f"  {update_type}: {count}")

    # Show recent updates
    print(f"\nMost recent updates:")
    for _, row in df.head(5).iterrows():
        print(f"  [{row['Date']}] {row['Update Type']}: {row['Summary'][:50]}...")


if __name__ == '__main__':
    main()
