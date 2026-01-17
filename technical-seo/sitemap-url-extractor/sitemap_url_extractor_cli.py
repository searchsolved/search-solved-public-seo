#!/usr/bin/env python3
"""
Sitemap URL Extractor - CLI Version
Extracts all URLs from sitemap indexes and child sitemaps.

Author: Lee Foot
Website: https://leefoot.com

Usage:
    python sitemap_url_extractor_cli.py https://example.com/sitemap.xml -o urls.csv
    python sitemap_url_extractor_cli.py https://example.com/sitemap.xml --format txt
"""

import argparse
import sys
import time
import gzip
import re
import xml.etree.ElementTree as ET
from datetime import datetime

import requests
import pandas as pd


def extract_sitemap_urls(content, is_bytes=False):
    """Extract URLs and metadata from sitemap XML content."""
    urls_data = []

    if is_bytes:
        xml_content = content
        text_content = content.decode('utf-8', errors='ignore')
    else:
        xml_content = content.encode('utf-8')
        text_content = content

    # Try ElementTree parsing first
    try:
        root = ET.fromstring(xml_content)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        # Check if this is a sitemap index
        sitemaps = root.findall(".//sm:sitemap", ns)
        if not sitemaps:
            sitemaps = root.findall(".//sitemap")

        if sitemaps:
            # This is a sitemap index
            for sitemap in sitemaps:
                loc = sitemap.find("sm:loc", ns)
                if loc is None:
                    loc = sitemap.find("loc")
                if loc is not None and loc.text:
                    url_data = {"url": loc.text.strip(), "type": "sitemap"}

                    lastmod = sitemap.find("sm:lastmod", ns)
                    if lastmod is None:
                        lastmod = sitemap.find("lastmod")
                    if lastmod is not None and lastmod.text:
                        url_data["lastmod"] = lastmod.text.strip()

                    urls_data.append(url_data)
            return urls_data, True  # is_index = True

        # This is a regular sitemap
        url_elements = root.findall(".//sm:url", ns)
        if not url_elements:
            url_elements = root.findall(".//url")

        for url_elem in url_elements:
            loc = url_elem.find("sm:loc", ns)
            if loc is None:
                loc = url_elem.find("loc")
            if loc is not None and loc.text:
                url_data = {"url": loc.text.strip(), "type": "url"}

                lastmod = url_elem.find("sm:lastmod", ns)
                if lastmod is None:
                    lastmod = url_elem.find("lastmod")
                if lastmod is not None and lastmod.text:
                    url_data["lastmod"] = lastmod.text.strip()

                priority = url_elem.find("sm:priority", ns)
                if priority is None:
                    priority = url_elem.find("priority")
                if priority is not None and priority.text:
                    url_data["priority"] = priority.text.strip()

                changefreq = url_elem.find("sm:changefreq", ns)
                if changefreq is None:
                    changefreq = url_elem.find("changefreq")
                if changefreq is not None and changefreq.text:
                    url_data["changefreq"] = changefreq.text.strip()

                urls_data.append(url_data)

        return urls_data, False

    except ET.ParseError:
        pass

    # Fallback to regex
    loc_pattern = r"<loc>(.*?)</loc>"
    urls = re.findall(loc_pattern, text_content, re.DOTALL)
    is_index = "<sitemap>" in text_content.lower() or "<sitemap " in text_content.lower()

    for url in urls:
        url_data = {"url": url.strip(), "type": "sitemap" if is_index else "url"}
        urls_data.append(url_data)

    return urls_data, is_index


def fetch_sitemap(url, user_agent):
    """Fetch sitemap content, handling gzip compression."""
    headers = {"User-Agent": user_agent}

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    if url.endswith('.gz') or response.content.startswith(b'\x1f\x8b'):
        try:
            content = gzip.decompress(response.content)
            return content, True
        except Exception:
            pass

    return response.content, True


def process_sitemap(sitemap_url, user_agent, delay, verbose=False):
    """Process sitemap index and all child sitemaps."""
    all_urls = []
    sitemaps_to_process = [sitemap_url]
    processed_sitemaps = set()

    while sitemaps_to_process:
        current_url = sitemaps_to_process.pop(0)

        if current_url in processed_sitemaps:
            continue

        processed_sitemaps.add(current_url)

        if verbose:
            print(f"Processing: {current_url}")

        try:
            content, is_bytes = fetch_sitemap(current_url, user_agent)
            urls_data, is_index = extract_sitemap_urls(content, is_bytes)

            if is_index:
                sitemap_count = len(urls_data)
                if verbose:
                    print(f"  Found sitemap index with {sitemap_count} sitemaps")
                for item in urls_data:
                    if item["url"] not in processed_sitemaps:
                        sitemaps_to_process.append(item["url"])
            else:
                url_count = len(urls_data)
                if verbose:
                    print(f"  Found {url_count} URLs")
                for item in urls_data:
                    item["source_sitemap"] = current_url
                    all_urls.append(item)

            if delay > 0 and sitemaps_to_process:
                time.sleep(delay)

        except Exception as e:
            print(f"Error processing {current_url}: {str(e)}", file=sys.stderr)

    return all_urls


def main():
    parser = argparse.ArgumentParser(
        description="Extract all URLs from XML sitemap indexes and sitemaps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s https://example.com/sitemap.xml
    %(prog)s https://example.com/sitemap.xml -o urls.csv
    %(prog)s https://example.com/sitemap.xml --format txt -o urls.txt
    %(prog)s https://example.com/sitemap.xml --no-metadata

Author: Lee Foot (https://leefoot.com)
        """
    )

    parser.add_argument("url", help="URL of the sitemap or sitemap index")
    parser.add_argument("-o", "--output", help="Output file path (default: sitemap_urls_TIMESTAMP.csv)")
    parser.add_argument("--format", choices=["csv", "txt", "xlsx"], default="csv",
                        help="Output format (default: csv)")
    parser.add_argument("--no-metadata", action="store_true",
                        help="Only output URLs, no lastmod/priority/changefreq")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between requests in seconds (default: 1.0)")
    parser.add_argument("--user-agent", default="Mozilla/5.0 (compatible; SitemapExtractor/1.0)",
                        help="User agent string for requests")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress all output except errors")

    args = parser.parse_args()

    # Process sitemap
    if not args.quiet:
        print(f"Extracting URLs from: {args.url}")

    urls = process_sitemap(args.url, args.user_agent, args.delay, verbose=args.verbose)

    if not urls:
        print("No URLs found.", file=sys.stderr)
        sys.exit(1)

    # Create DataFrame
    df = pd.DataFrame(urls)

    # Handle columns
    if args.no_metadata:
        df = df[["url"]]
    else:
        columns = ["url"]
        for col in ["lastmod", "priority", "changefreq", "source_sitemap"]:
            if col in df.columns:
                columns.append(col)
        df = df[columns]

    # Generate output filename
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = args.format
        output_file = f"sitemap_urls_{timestamp}.{ext}"

    # Save output
    if args.format == "csv":
        df.to_csv(output_file, index=False)
    elif args.format == "txt":
        with open(output_file, "w") as f:
            for url in df["url"]:
                f.write(f"{url}\n")
    elif args.format == "xlsx":
        df.to_excel(output_file, index=False, sheet_name="URLs")

    if not args.quiet:
        print(f"Extracted {len(df):,} URLs to {output_file}")


if __name__ == "__main__":
    main()
