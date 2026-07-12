#!/usr/bin/env python3
"""
Categories Missing From Navigation - CLI Version

Fetches an XML sitemap and a page (usually the homepage), extracts links
found within a CSS selector for the navigation element, and reports which
sitemap URLs are missing from the navigation.

Usage:
    python categories_missing_from_navigation_cli.py \
        --sitemap https://www.example.com/sitemap.xml \
        --nav-selector "nav" \
        --output missing_from_navigation.csv

Author: Lee Foot
Website: https://www.leefoot.com
"""

import argparse
import sys

import pandas as pd

from categories_missing_from_navigation import (
    DEFAULT_DELAY,
    DEFAULT_NAV_SELECTOR,
    extract_navigation_links,
    extract_urls_from_sitemap,
    find_missing_urls,
    homepage_from_sitemap,
)


def main():
    parser = argparse.ArgumentParser(
        description="Report sitemap URLs missing from the site navigation"
    )
    parser.add_argument(
        "--sitemap", required=True,
        help="XML sitemap URL (sitemap index files are supported)"
    )
    parser.add_argument(
        "--nav-selector", default=DEFAULT_NAV_SELECTOR,
        help="CSS selector for the navigation element, for example nav, #main-nav or .header-menu "
             f"(default: {DEFAULT_NAV_SELECTOR})"
    )
    parser.add_argument(
        "--page", default=None,
        help="Page whose navigation will be checked (default: homepage of the sitemap domain)"
    )
    parser.add_argument(
        "--filter", dest="url_filter", default=None,
        help="Only report missing URLs containing this string, for example /category/"
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"Delay in seconds between fetches (default: {DEFAULT_DELAY})"
    )
    parser.add_argument(
        "--output", default="missing_from_navigation.csv",
        help="Output CSV path (default: missing_from_navigation.csv)"
    )

    args = parser.parse_args()

    page_url = args.page or homepage_from_sitemap(args.sitemap)

    print(f"Fetching sitemap: {args.sitemap}")

    def sitemap_progress(index, total, child_url):
        print(f"  Fetching child sitemap {index}/{total}: {child_url}")

    try:
        sitemap_urls = extract_urls_from_sitemap(args.sitemap, delay=args.delay,
                                                 progress_callback=sitemap_progress)
    except Exception as e:
        print(f"Error: failed to fetch sitemap: {e}")
        sys.exit(1)

    print(f"  Found {len(sitemap_urls):,} sitemap URLs")

    if not sitemap_urls:
        print("Error: no URLs found in the sitemap")
        sys.exit(1)

    print(f"Fetching page: {page_url}")
    print(f"  Extracting links from selector: {args.nav_selector}")

    try:
        navigation_urls = extract_navigation_links(page_url, args.nav_selector, delay=args.delay)
    except Exception as e:
        print(f"Error: failed to fetch page: {e}")
        sys.exit(1)

    print(f"  Found {len(set(navigation_urls)):,} unique navigation links")

    if not navigation_urls:
        print(f"Error: no links found for selector '{args.nav_selector}'.")
        print("Check the selector in your browser's inspector. Navigation rendered")
        print("with JavaScript will not be visible to this tool.")
        sys.exit(1)

    missing = find_missing_urls(sitemap_urls, navigation_urls, args.url_filter)

    if args.url_filter:
        print(f"Filtering results to URLs containing: {args.url_filter}")

    print(f"\n{len(missing):,} sitemap URLs missing from the navigation")

    df_missing = pd.DataFrame(missing, columns=["missing_url"])
    df_missing.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
