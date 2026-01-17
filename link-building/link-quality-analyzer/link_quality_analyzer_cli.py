#!/usr/bin/env python3
"""
Link Quality Analyzer - CLI Version

Extract internal links from pages, check status codes, analyze anchor text.

Usage:
    python link_quality_analyzer_cli.py --urls urls.txt --selector "article"

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import time
import sys

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False


def get_domain(url):
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except:
        return ""


def extract_links_from_page(url, selector, user_agent, timeout):
    """Extract all links from a page within the specified selector."""
    try:
        headers = {'User-Agent': user_agent}
        response = requests.get(url, headers=headers, timeout=timeout)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Get page content for reading score
        page_text = ""
        content_container = soup.select_one(selector) if selector else soup
        if content_container:
            page_text = content_container.get_text(separator=' ', strip=True)
        else:
            page_text = soup.get_text(separator=' ', strip=True)

        # Get H1
        h1 = ""
        h1_tag = soup.find('h1')
        if h1_tag:
            h1 = h1_tag.get_text(strip=True)

        # Find links
        links = []
        container = soup.select_one(selector) if selector else soup

        if container:
            for a in container.find_all('a', href=True):
                href = a.get('href', '')
                anchor = a.get_text(strip=True)

                if not anchor:
                    anchor = "[IMAGE]" if a.find('img') else "[EMPTY]"

                full_url = urljoin(url, href)

                if any(skip in href for skip in ['mailto:', 'tel:', 'javascript:', '#']):
                    continue

                links.append({
                    'anchor_text': anchor,
                    'link_url': full_url,
                    'is_internal': get_domain(url) == get_domain(full_url)
                })

        return links, page_text, h1

    except Exception as e:
        return [], "", ""


def check_http_status(url, user_agent, timeout):
    """Check HTTP status code of a URL."""
    try:
        headers = {'User-Agent': user_agent}
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=False)
        return response.status_code
    except requests.exceptions.Timeout:
        return "Timeout"
    except requests.exceptions.ConnectionError:
        return "Connection Error"
    except:
        return "Error"


def main():
    parser = argparse.ArgumentParser(
        description='Extract and analyze internal links from pages'
    )
    parser.add_argument('--urls', required=True, help='File with URLs (one per line)')
    parser.add_argument('--output', default='link_analysis.csv',
                        help='Output CSV path (default: link_analysis.csv)')
    parser.add_argument('--page-stats-output', default='page_stats.csv',
                        help='Page stats output path (default: page_stats.csv)')
    parser.add_argument('--selector', default='body',
                        help='CSS selector for content area (default: body)')
    parser.add_argument('--check-status', action='store_true',
                        help='Check HTTP status codes for links')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between requests (default: 1.0)')
    parser.add_argument('--timeout', type=int, default=15,
                        help='Request timeout in seconds (default: 15)')
    parser.add_argument('--user-agent', default='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        help='User agent string')

    args = parser.parse_args()

    # Load URLs
    with open(args.urls, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(urls)} URLs")

    all_links = []
    page_stats = []

    print(f"\nExtracting links...")
    for i, url in enumerate(urls):
        if i % 10 == 0:
            print(f"  Processing {i + 1}/{len(urls)}...")

        links, page_text, h1 = extract_links_from_page(
            url, args.selector, args.user_agent, args.timeout
        )

        # Calculate reading score
        reading_score = None
        if TEXTSTAT_AVAILABLE and page_text and len(page_text.split()) > 50:
            try:
                reading_score = round(textstat.flesch_reading_ease(page_text), 2)
            except:
                pass

        page_stats.append({
            'source_url': url,
            'h1': h1,
            'links_count': len(links),
            'internal_links': sum(1 for l in links if l['is_internal']),
            'external_links': sum(1 for l in links if not l['is_internal']),
            'flesch_score': reading_score
        })

        for link in links:
            all_links.append({
                'source_url': url,
                'anchor_text': link['anchor_text'],
                'link_url': link['link_url'],
                'is_internal': link['is_internal']
            })

        time.sleep(args.delay)

    # Check status codes
    if args.check_status and all_links:
        print(f"\nChecking status codes...")
        unique_links = list(set(l['link_url'] for l in all_links))

        status_cache = {}
        for i, link_url in enumerate(unique_links):
            if i % 20 == 0:
                print(f"  Checking {i + 1}/{len(unique_links)}...")
            status_cache[link_url] = check_http_status(link_url, args.user_agent, args.timeout)
            time.sleep(args.delay / 2)

        for link in all_links:
            link['status_code'] = status_cache.get(link['link_url'], 'Unknown')

    # Create DataFrames
    df_links = pd.DataFrame(all_links)
    df_pages = pd.DataFrame(page_stats)

    # Save results
    df_links.to_csv(args.output, index=False, encoding='utf-8-sig')
    df_pages.to_csv(args.page_stats_output, index=False, encoding='utf-8-sig')

    print(f"\nResults saved:")
    print(f"  Links: {args.output} ({len(df_links):,} links)")
    print(f"  Page stats: {args.page_stats_output} ({len(df_pages):,} pages)")

    # Summary
    internal = sum(1 for l in all_links if l.get('is_internal'))
    print(f"\nSummary:")
    print(f"  Pages analyzed: {len(page_stats)}")
    print(f"  Total links: {len(all_links)}")
    print(f"  Internal: {internal}")
    print(f"  External: {len(all_links) - internal}")

    if args.check_status:
        broken = sum(1 for l in all_links if str(l.get('status_code', '')).startswith('4'))
        print(f"  Broken (4xx): {broken}")


if __name__ == '__main__':
    main()
