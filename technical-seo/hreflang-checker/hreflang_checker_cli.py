#!/usr/bin/env python3
"""
Hreflang Checker - CLI Version
Extracts and validates hreflang tags from web pages.

Author: Lee Foot
Website: https://leefoot.com

Usage:
    python hreflang_checker_cli.py https://example.com
    python hreflang_checker_cli.py -f urls.txt -o hreflang_report.csv
    python hreflang_checker_cli.py https://example.com --no-validate
"""

import argparse
import sys
import os
import time
import re
from datetime import datetime
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
import pandas as pd


def extract_hreflang_from_html(html_content, base_url):
    """Extract hreflang tags from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    hreflang_data = []

    hreflang_tags = soup.find_all('link', rel='alternate', hreflang=True)

    for tag in hreflang_tags:
        hreflang = tag.get('hreflang', '').strip()
        href = tag.get('href', '').strip()

        if hreflang and href:
            full_url = urljoin(base_url, href)
            hreflang_data.append({
                'hreflang': hreflang,
                'url': full_url,
                'source': 'HTML'
            })

    return hreflang_data


def extract_hreflang_from_headers(response):
    """Extract hreflang from HTTP Link headers."""
    hreflang_data = []

    link_header = response.headers.get('Link', '')
    if not link_header:
        return hreflang_data

    link_pattern = r'<([^>]+)>;\s*rel=["\']alternate["\'];\s*hreflang=["\']([^"\']+)["\']'
    matches = re.findall(link_pattern, link_header, re.IGNORECASE)

    for url, hreflang in matches:
        hreflang_data.append({
            'hreflang': hreflang.strip(),
            'url': url.strip(),
            'source': 'HTTP Header'
        })

    return hreflang_data


def validate_hreflang_data(hreflang_list, source_url):
    """Validate hreflang implementation and return issues."""
    issues = []

    hreflang_codes = [item['hreflang'] for item in hreflang_list]

    # Check self-referencing
    source_normalized = source_url.rstrip('/')
    has_self_reference = any(
        item['url'].rstrip('/') == source_normalized
        for item in hreflang_list
    )

    if not has_self_reference and hreflang_list:
        issues.append("Missing self-referencing hreflang tag")

    # Check x-default
    if 'x-default' not in hreflang_codes and len(hreflang_codes) > 1:
        issues.append("Missing x-default tag")

    # Check duplicates
    seen_codes = set()
    for code in hreflang_codes:
        if code in seen_codes:
            issues.append(f"Duplicate hreflang: {code}")
        seen_codes.add(code)

    # Check format
    valid_pattern = r'^[a-z]{2}(-[A-Z]{2})?$|^x-default$'
    for code in hreflang_codes:
        if not re.match(valid_pattern, code, re.IGNORECASE):
            issues.append(f"Invalid format: {code}")

    return issues


def fetch_url_hreflang(url, user_agent, timeout, check_headers=True, verbose=False):
    """Fetch a URL and extract hreflang data."""
    headers = {"User-Agent": user_agent}

    try:
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()

        hreflang_data = extract_hreflang_from_html(response.text, response.url)

        if check_headers:
            header_data = extract_hreflang_from_headers(response)
            hreflang_data.extend(header_data)

        if verbose:
            print(f"  Found {len(hreflang_data)} hreflang tags")

        return {
            'success': True,
            'final_url': response.url,
            'status_code': response.status_code,
            'hreflang': hreflang_data,
            'error': None
        }

    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"  Error: {str(e)}")
        return {
            'success': False,
            'final_url': url,
            'status_code': None,
            'hreflang': [],
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Extract and validate hreflang tags from web pages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s https://example.com
    %(prog)s https://example.com https://example.com/de
    %(prog)s -f urls.txt -o hreflang_report.csv
    %(prog)s https://example.com --no-headers --no-validate

Author: Lee Foot (https://leefoot.com)
        """
    )

    # Input options
    parser.add_argument("urls", nargs="*", help="URLs to check")
    parser.add_argument("-f", "--file", help="File with URLs (one per line)")

    # Output options
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--format", choices=["csv", "xlsx", "json"], default="csv",
                        help="Output format (default: csv)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only output summary, not individual tags")

    # Request options
    parser.add_argument("--user-agent",
                        default="Mozilla/5.0 (compatible; HreflangChecker/1.0)",
                        help="User agent string")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Request timeout in seconds (default: 30)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between requests in seconds (default: 1.0)")

    # Feature flags
    parser.add_argument("--no-headers", action="store_true",
                        help="Don't check HTTP headers for hreflang")
    parser.add_argument("--no-validate", action="store_true",
                        help="Don't validate hreflang implementation")

    # Verbosity
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    # Collect URLs
    urls = list(args.urls) if args.urls else []

    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        with open(args.file, 'r') as f:
            file_urls = [line.strip() for line in f if line.strip()]
            urls.extend(file_urls)

    if not urls:
        print("Error: No URLs provided. Use positional arguments or -f flag.", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Checking {len(urls)} URL(s) for hreflang tags...")

    # Process URLs
    results = []
    all_hreflang = []
    all_issues = []

    for i, url in enumerate(urls, 1):
        if not args.quiet:
            print(f"[{i}/{len(urls)}] {url}")

        result = fetch_url_hreflang(
            url,
            args.user_agent,
            args.timeout,
            check_headers=not args.no_headers,
            verbose=args.verbose
        )

        issues = []
        if result['success'] and not args.no_validate and result['hreflang']:
            issues = validate_hreflang_data(result['hreflang'], result['final_url'])

        results.append({
            'source_url': url,
            'final_url': result['final_url'],
            'status': 'OK' if result['success'] else 'ERROR',
            'hreflang_count': len(result['hreflang']),
            'issues': '; '.join(issues) if issues else ''
        })

        for item in result['hreflang']:
            all_hreflang.append({
                'source_url': url,
                'hreflang_code': item['hreflang'],
                'alternate_url': item['url'],
                'detection_source': item['source']
            })

        for issue in issues:
            all_issues.append({
                'source_url': url,
                'issue': issue
            })

        if args.delay > 0 and i < len(urls):
            time.sleep(args.delay)

    # Print summary
    if not args.quiet:
        print("\n" + "=" * 60)
        print("HREFLANG CHECK SUMMARY")
        print("=" * 60)
        print(f"URLs checked:     {len(results)}")
        print(f"Successful:       {sum(1 for r in results if r['status'] == 'OK')}")
        print(f"Total tags found: {len(all_hreflang)}")
        print(f"Issues found:     {len(all_issues)}")

        if all_hreflang:
            print("\nHreflang codes found:")
            codes = pd.DataFrame(all_hreflang)['hreflang_code'].value_counts()
            for code, count in codes.items():
                print(f"  {code}: {count}")

        if all_issues and not args.no_validate:
            print("\nValidation issues:")
            for issue in all_issues[:10]:
                print(f"  - {issue['source_url']}: {issue['issue']}")
            if len(all_issues) > 10:
                print(f"  ... and {len(all_issues) - 10} more")

    # Save output
    if args.output or args.summary_only:
        if args.summary_only:
            df = pd.DataFrame(results)
        else:
            df = pd.DataFrame(all_hreflang) if all_hreflang else pd.DataFrame(results)

        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"hreflang_report_{timestamp}.{args.format}"

        if args.format == "csv":
            df.to_csv(output_file, index=False)
        elif args.format == "xlsx":
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                pd.DataFrame(results).to_excel(writer, index=False, sheet_name='Summary')
                if all_hreflang:
                    pd.DataFrame(all_hreflang).to_excel(writer, index=False, sheet_name='Hreflang Tags')
                if all_issues:
                    pd.DataFrame(all_issues).to_excel(writer, index=False, sheet_name='Issues')
        elif args.format == "json":
            df.to_json(output_file, orient="records", indent=2)

        if not args.quiet:
            print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()
