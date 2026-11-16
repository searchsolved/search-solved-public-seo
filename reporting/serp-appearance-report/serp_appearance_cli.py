#!/usr/bin/env python3
"""
SERP Appearance Report - CLI Version

Send a keyword list to the DataForSEO SERP API (Standard queue), then collect
the Google results and report every organic appearance of a domain.

Searches in the Standard queue can take 25 minutes or more to finish, so the
tool works in two steps. `send` queues the searches and writes a job file;
`collect` reads the job file, fetches the finished searches and writes the
report. Add --wait to `collect` to keep checking until every search is done.

Usage:
    python serp_appearance_cli.py send --input keywords.txt --job job.csv
    python serp_appearance_cli.py collect --job job.csv --domain example.com --output report.csv
    python serp_appearance_cli.py collect --job job.csv --domain example.com --wait

The keyword file is either a text file with one keyword per line, or a CSV
with a keyword column (keyword, keywords, query, queries, search term or
top queries; otherwise the first column is used).

Environment variables:
    DATAFORSEO_LOGIN    - DataForSEO account login
    DATAFORSEO_PASSWORD - DataForSEO account password

Author: Lee Foot
Website: https://www.leefoot.com
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dataforseo_queue import QueueError, collect_results, estimate_cost, job_from_csv, job_to_csv, send_searches
from serp_appearance_core import (
    DEVICES,
    LOCATIONS,
    build_report,
    normalise_domain,
    read_keywords_file,
    results_to_dataframe,
)

WAIT_CHECK_SECONDS = 60
WAIT_LIMIT_MINUTES = 90


def credentials():
    login = os.environ.get('DATAFORSEO_LOGIN')
    password = os.environ.get('DATAFORSEO_PASSWORD')
    if not login or not password:
        print("Error: DataForSEO credentials required.")
        print("Set the DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD environment variables.")
        sys.exit(1)
    return login, password


def send(args):
    login, password = credentials()

    if not 10 <= args.depth <= 100:
        print("Error: --depth must be between 10 and 100.")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: Keyword file does not exist: {input_path}")
        sys.exit(1)
    try:
        keywords = read_keywords_file(input_path)
    except ValueError as e:
        print(f"Error reading keyword file: {e}")
        sys.exit(1)
    if not keywords:
        print("Error: No keywords found in the input file.")
        sys.exit(1)

    location_code, language_code = LOCATIONS[args.location]
    print(f"Keywords: {len(keywords)}")
    print(f"  Location: {args.location} (code: {location_code}, language: {language_code})")
    print(f"  Device: {args.device}")
    print(f"  Results per keyword: {args.depth}")
    print(f"  Estimated cost: ${estimate_cost(len(keywords), args.depth):.4f}")

    if not args.yes:
        if not sys.stdin.isatty():
            print("Run with --yes to confirm the cost when not running interactively.")
            sys.exit(1)
        if input("Send the searches? [y/N] ").strip().lower() not in ('y', 'yes'):
            print("Cancelled.")
            sys.exit(0)

    try:
        job_rows, warnings, cost = send_searches(keywords, login, password, location_code, language_code,
                                                 device=args.device, depth=args.depth)
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
    print(f"  python {Path(__file__).name} collect --job {args.job} --domain example.com")


def collect(args):
    login, password = credentials()

    if not normalise_domain(args.domain):
        print("Error: --domain is empty.")
        sys.exit(1)
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
        done = len(result['ready']) + len(result['errors'])
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
        print(f"{len(result['pending'])} search(es) are still running, so the report below leaves them out. "
              "Run collect again later for the full report.")
    if done == 0:
        sys.exit(0)

    rows, unresolved = build_report(result['ready'], args.domain)
    df = results_to_dataframe(rows)
    if unresolved:
        print(f"Warning: {unresolved} result(s) came back as unresolved google.com/goto links "
              "and could not be checked against the domain.")

    print(f"\nFound {len(df)} appearances of '{normalise_domain(args.domain)}' "
          f"across {len(result['ready'])} keyword(s).")
    if not df.empty:
        print(df.to_string(index=False, max_colwidth=60))

    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\nCSV file saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description='Report organic appearances of a domain in Google results from DataForSEO'
    )
    sub = parser.add_subparsers(dest='command', required=True)

    p_send = sub.add_parser('send', help='Queue the searches and write a job file')
    p_send.add_argument('--input', required=True,
                        help='Keyword file: .txt with one keyword per line, or .csv with a keyword column')
    p_send.add_argument('--job', default='serp_appearance_job.csv',
                        help='Job file to write (default: serp_appearance_job.csv)')
    p_send.add_argument('--location', default='United Kingdom', choices=list(LOCATIONS.keys()),
                        help='Search location (default: United Kingdom)')
    p_send.add_argument('--device', default='desktop', choices=list(DEVICES),
                        help='Device (default: desktop)')
    p_send.add_argument('--depth', type=int, default=10,
                        help='Results to fetch per keyword, 10 to 100, billed per 10 (default: 10)')
    p_send.add_argument('--yes', action='store_true', help='Skip the cost confirmation prompt')
    p_send.set_defaults(func=send)

    p_collect = sub.add_parser('collect', help='Collect finished searches and write the report')
    p_collect.add_argument('--job', required=True, help='Job file written by send')
    p_collect.add_argument('--domain', required=True,
                           help='Domain to report, e.g. example.com (subdomains are included)')
    p_collect.add_argument('--output', default='serp_appearance_report.csv',
                           help='Output CSV path (default: serp_appearance_report.csv)')
    p_collect.add_argument('--wait', action='store_true',
                           help=f'Check every {WAIT_CHECK_SECONDS}s until all searches finish '
                                f'(up to {WAIT_LIMIT_MINUTES} minutes)')
    p_collect.set_defaults(func=collect)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
