#!/usr/bin/env python3
"""
Content Extractor - CLI Version

Extract main content, H1, and title from URLs.

Usage:
    python content_extractor_cli.py --input urls.csv --output extracted.csv

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys


def fetch_page(url, headers, timeout_val):
    try:
        response = requests.get(url, headers=headers, timeout=timeout_val)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        return response.text, None
    except requests.RequestException as e:
        return None, str(e)


def html_to_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for element in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "div"]):
        element.append("\n\n")
    text = soup.get_text(separator=" ")
    return ' '.join(text.split())


def extract_h1(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    h1_tag = soup.find('h1')
    return h1_tag.get_text(strip=True) if h1_tag else None


def extract_title(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    title_tag = soup.find('title')
    return title_tag.get_text(strip=True) if title_tag else None


def process_url(url, headers, timeout_val, rate_limit):
    time.sleep(random.uniform(rate_limit * 0.5, rate_limit * 1.5))
    html_content, error = fetch_page(url, headers, timeout_val)

    if html_content:
        return {
            'URL': url,
            'Title': extract_title(html_content),
            'H1': extract_h1(html_content),
            'Content': html_to_text(html_content),
            'Content_Length': len(html_to_text(html_content)),
            'Status': 'Success',
            'Error': None
        }
    else:
        return {
            'URL': url,
            'Title': None,
            'H1': None,
            'Content': None,
            'Content_Length': 0,
            'Status': 'Failed',
            'Error': error
        }


def main():
    parser = argparse.ArgumentParser(description='Extract content from URLs')
    parser.add_argument('--input', required=True, help='Input CSV with URLs')
    parser.add_argument('--output', default='extracted_content.csv', help='Output CSV path')
    parser.add_argument('--url-col', default='url', help='URL column name')
    parser.add_argument('--rate-limit', type=float, default=1.0, help='Delay between requests (seconds)')
    parser.add_argument('--workers', type=int, default=3, help='Number of concurrent requests')
    parser.add_argument('--timeout', type=int, default=10, help='Request timeout (seconds)')
    parser.add_argument('--user-agent', default='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36', help='User agent string')

    args = parser.parse_args()

    print(f"Loading URLs from: {args.input}")
    df = pd.read_csv(args.input)

    # Find URL column
    url_col = None
    for col in df.columns:
        if col.lower() == args.url_col.lower():
            url_col = col
            break
    if not url_col:
        url_col = df.columns[0]

    urls = df[url_col].dropna().astype(str).tolist()
    print(f"Found {len(urls)} URLs to process")

    headers = {'User-Agent': args.user_agent}
    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_url, url, headers, args.timeout, args.rate_limit): url for url in urls}
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if completed % 10 == 0:
                print(f"  Processed {completed}/{len(urls)} URLs...")

    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output, index=False, encoding='utf-8-sig')

    success = len(df_results[df_results['Status'] == 'Success'])
    failed = len(df_results[df_results['Status'] != 'Success'])

    print(f"\nResults saved to: {args.output}")
    print(f"  Successful: {success}")
    print(f"  Failed: {failed}")


if __name__ == '__main__':
    main()
