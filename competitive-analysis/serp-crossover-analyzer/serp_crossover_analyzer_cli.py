# Author: Lee Foot
# Website: https://leefoot.com

"""
SERP Crossover Analyzer - CLI Version

Analyse URL overlap across multiple keyword SERPs using the DataForSEO API
(Standard queue).

Searches in the Standard queue can take 25 minutes or more to finish, so the
tool works in two steps. `send` queues the searches and writes a job file;
`collect` reads the job file, fetches the finished searches and writes the
crossover matrix. Add --wait to `collect` to keep checking until every search
is done.

Usage:
    python serp_crossover_analyzer_cli.py send --keywords "seo tools" "keyword research" --job job.csv
    python serp_crossover_analyzer_cli.py collect --job job.csv --output serp_crossover.csv
    python serp_crossover_analyzer_cli.py collect --job job.csv --wait

Credentials: pass --login and --password, or set the environment variables.

Environment variables:
    DATAFORSEO_LOGIN    - DataForSEO account login
    DATAFORSEO_PASSWORD - DataForSEO account password
"""

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from dataforseo_queue import (
    QueueError,
    collect_results,
    estimate_cost,
    is_unresolved,
    job_from_csv,
    job_to_csv,
    organic_items,
    send_searches,
)

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

LANGUAGE_CODE = "en"

WAIT_CHECK_SECONDS = 60
WAIT_LIMIT_MINUTES = 90


def extract_domain(url):
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc.replace('www.', '')
    except Exception:
        return url


def build_serp_data(ready, depths):
    """Turn collected searches into {keyword: [results]} for the crossover analysis.

    ready is {keyword: items} from collect_results(); depths is {keyword: results per SERP}.
    Organic results that came back as unresolved google.com/goto links are left out and
    counted, since their real URL and domain are unknown. Returns (serp_data, unresolved).
    """
    serp_data, unresolved = {}, 0
    for keyword, items in ready.items():
        organic = organic_items(items)[:depths.get(keyword, len(items))]
        results = []
        for i, result in enumerate(organic):
            url = result.get('url') or ''
            if is_unresolved(url):
                unresolved += 1
                continue
            results.append({
                'position': result.get('rank_group', i + 1),
                'title': result.get('title', ''),
                'link': url,
                'domain': extract_domain(url)
            })
        if results:
            serp_data[keyword] = results
    return serp_data, unresolved


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


def credentials(args):
    login = args.login or os.environ.get('DATAFORSEO_LOGIN')
    password = args.password or os.environ.get('DATAFORSEO_PASSWORD')
    if not login or not password:
        print("Error: DataForSEO credentials required.")
        print("Use --login and --password, or set DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD environment variables.")
        sys.exit(1)
    return login, password


def send(args):
    login, password = credentials(args)

    keywords = list(dict.fromkeys(kw.strip() for kw in args.keywords if kw.strip()))
    if len(keywords) < 2:
        print("Error: Need at least 2 keywords to compare")
        sys.exit(1)

    # Resolve location code
    location_code = args.location_code if args.location_code else LOCATION_CODES[args.location]

    print(f"Sending {len(keywords)} keywords...")
    print(f"  Location: {args.location} (code: {location_code})")
    print(f"  Device: {args.device}")
    print(f"  Results per SERP: {args.num_results}")
    print(f"  Estimated cost: ${estimate_cost(len(keywords), args.num_results):.4f}")

    try:
        job_rows, warnings, cost = send_searches(keywords, login, password, location_code, LANGUAGE_CODE,
                                                 device=args.device, depth=args.num_results)
    except QueueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    for warning in warnings:
        print(f"Warning: {warning}")
    if not job_rows:
        print("No searches were queued.")
        sys.exit(1)

    Path(args.job).write_text(job_to_csv(job_rows), encoding='utf-8')
    print(f"\nQueued {len(job_rows)} search(es). Cost reported by DataForSEO: ${cost:.4f}")
    print(f"Job file saved to: {args.job}")
    print("DataForSEO can take 25 minutes or more to finish. Then run:")
    print(f"  python {Path(__file__).name} collect --job {args.job}")


def collect(args):
    login, password = credentials(args)

    job_path = Path(args.job)
    if not job_path.is_file():
        print(f"Error: Job file does not exist: {job_path}")
        sys.exit(1)
    try:
        job_rows = job_from_csv(job_path.read_text(encoding='utf-8'))
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    started = time.time()
    while True:
        try:
            result = collect_results(job_rows, login, password)
        except QueueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        print(f"{len(result['ready'])} of {len(job_rows)} search(es) finished, "
              f"{len(result['pending'])} still running, {len(result['errors'])} failed.")
        if not result['pending'] or not args.wait:
            break
        if time.time() - started > WAIT_LIMIT_MINUTES * 60:
            print(f"Stopped waiting after {WAIT_LIMIT_MINUTES} minutes. Run collect again later.")
            break
        time.sleep(WAIT_CHECK_SECONDS)

    for error in result['errors']:
        print(f"Warning: {error}")
    if result['pending']:
        print(f"{len(result['pending'])} search(es) are still running, so the analysis below leaves them out. "
              "Run collect again later for the full analysis.")

    depths = {}
    for row in job_rows:
        try:
            depths[row['keyword']] = int(row['depth'])
        except (KeyError, TypeError, ValueError):
            pass
    serp_data, unresolved = build_serp_data(result['ready'], depths)
    if unresolved:
        print(f"Warning: {unresolved} result(s) came back as unresolved google.com/goto links "
              "and are left out of the crossover, as their real URL is unknown.")

    if len(serp_data) < 2:
        print("Error: Need at least 2 finished searches with results to analyse")
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
    print("\nCrossover Matrix:")
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


def add_credential_args(parser):
    parser.add_argument('--login',
                        help='DataForSEO login (or set DATAFORSEO_LOGIN env var)')
    parser.add_argument('--password',
                        help='DataForSEO password (or set DATAFORSEO_PASSWORD env var)')


def main():
    parser = argparse.ArgumentParser(
        description='Analyse URL overlap across multiple keyword SERPs using the DataForSEO API'
    )
    sub = parser.add_subparsers(dest='command', required=True)

    p_send = sub.add_parser('send', help='Queue the searches and write a job file')
    p_send.add_argument('--keywords', required=True, nargs='+',
                        help='Keywords to compare')
    p_send.add_argument('--job', default='serp_crossover_job.csv',
                        help='Job file to write (default: serp_crossover_job.csv)')
    p_send.add_argument('--location', default='United Kingdom',
                        choices=list(LOCATION_CODES.keys()),
                        help='Search location name (default: United Kingdom)')
    p_send.add_argument('--location-code', type=int, default=None,
                        help='DataForSEO location code (overrides --location, default: 2826)')
    p_send.add_argument('--device', choices=['desktop', 'mobile'],
                        default='desktop', help='Device type (default: desktop)')
    p_send.add_argument('--num-results', type=int, default=10,
                        help='Results per SERP, billed per 10 (default: 10)')
    add_credential_args(p_send)
    p_send.set_defaults(func=send)

    p_collect = sub.add_parser('collect', help='Collect finished searches and write the crossover matrix')
    p_collect.add_argument('--job', required=True, help='Job file written by send')
    p_collect.add_argument('--output', default='serp_crossover.csv',
                           help='Output CSV path (default: serp_crossover.csv)')
    p_collect.add_argument('--wait', action='store_true',
                           help=f'Check every {WAIT_CHECK_SECONDS}s until all searches finish '
                                f'(up to {WAIT_LIMIT_MINUTES} minutes)')
    add_credential_args(p_collect)
    p_collect.set_defaults(func=collect)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
