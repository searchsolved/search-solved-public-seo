#!/usr/bin/env python3
"""
Redirect/URL Mapping Validator - CLI Version
Validates that implemented redirects match your redirect mapping specification.

Author: Lee Foot
Website: https://leefoot.co.uk

Usage:
    python redirect_validator_cli.py crawled.csv mapping.xlsx -o report.csv
    python redirect_validator_cli.py crawled.csv mapping.csv --crawled-cols "Address,Redirect URL"
"""

import argparse
import sys
import os
from datetime import datetime

import pandas as pd


def clean_url(url):
    """Clean and standardize a URL for comparison."""
    if pd.isna(url) or url is None:
        return ""
    url = str(url).strip().lower()
    # Remove URL parameters
    url = url.split('?')[0]
    # Remove trailing slashes
    url = url.rstrip('/')
    return url


def load_file(filepath):
    """Load CSV or Excel file."""
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    else:
        return pd.read_excel(filepath)


def find_url_columns(df):
    """Auto-detect URL columns in a dataframe."""
    # Common column names for source/destination URLs
    source_patterns = ['address', 'source', 'from', 'old', 'original']
    dest_patterns = ['redirect', 'destination', 'to', 'new', 'target']

    source_col = None
    dest_col = None

    columns_lower = {col.lower(): col for col in df.columns}

    # Find source column
    for pattern in source_patterns:
        for col_lower, col_orig in columns_lower.items():
            if pattern in col_lower:
                source_col = col_orig
                break
        if source_col:
            break

    # Find destination column
    for pattern in dest_patterns:
        for col_lower, col_orig in columns_lower.items():
            if pattern in col_lower and col_orig != source_col:
                dest_col = col_orig
                break
        if dest_col:
            break

    # Fallback to first two columns
    if not source_col and len(df.columns) >= 1:
        source_col = df.columns[0]
    if not dest_col and len(df.columns) >= 2:
        dest_col = df.columns[1]

    return source_col, dest_col


def verify_mappings(crawled_df, source_df, crawled_source_col, crawled_dest_col,
                   source_source_col, source_dest_col):
    """Verify that source mappings match crawled mappings."""

    # Clean URLs
    crawled_clean = crawled_df.copy()
    source_clean = source_df.copy()

    crawled_clean['_clean_source'] = crawled_clean[crawled_source_col].apply(clean_url)
    crawled_clean['_clean_dest'] = crawled_clean[crawled_dest_col].apply(clean_url)
    source_clean['_clean_source'] = source_clean[source_source_col].apply(clean_url)
    source_clean['_clean_dest'] = source_clean[source_dest_col].apply(clean_url)

    # Create dictionaries
    crawled_mapping = dict(zip(crawled_clean['_clean_source'], crawled_clean['_clean_dest']))
    source_mapping = dict(zip(source_clean['_clean_source'], source_clean['_clean_dest']))

    matches = []
    mismatches = []
    missing = []
    extra = []

    # Check source mappings against crawled
    for source_url, expected_dest in source_mapping.items():
        if not source_url:
            continue
        if source_url in crawled_mapping:
            actual_dest = crawled_mapping[source_url]
            if actual_dest == expected_dest:
                matches.append({
                    'source_url': source_url,
                    'expected_destination': expected_dest,
                    'actual_destination': expected_dest,
                    'status': 'MATCH'
                })
            else:
                mismatches.append({
                    'source_url': source_url,
                    'expected_destination': expected_dest,
                    'actual_destination': actual_dest,
                    'status': 'MISMATCH'
                })
        else:
            missing.append({
                'source_url': source_url,
                'expected_destination': expected_dest,
                'actual_destination': 'NOT_FOUND',
                'status': 'MISSING'
            })

    # Check for extra redirects
    for crawled_url, dest in crawled_mapping.items():
        if not crawled_url:
            continue
        if crawled_url not in source_mapping:
            extra.append({
                'source_url': crawled_url,
                'expected_destination': 'NOT_IN_SOURCE',
                'actual_destination': dest,
                'status': 'EXTRA'
            })

    return {
        'matches': matches,
        'mismatches': mismatches,
        'missing': missing,
        'extra': extra
    }


def print_report(results, quiet=False):
    """Print validation report to console."""
    total = len(results['matches']) + len(results['mismatches']) + len(results['missing'])
    accuracy = (len(results['matches']) / total * 100) if total > 0 else 0

    print("\n" + "=" * 70)
    print("REDIRECT VALIDATION REPORT")
    print("=" * 70)

    print(f"\nMatches:    {len(results['matches']):,}")
    print(f"Mismatches: {len(results['mismatches']):,}")
    print(f"Missing:    {len(results['missing']):,}")
    print(f"Extra:      {len(results['extra']):,}")
    print(f"\nAccuracy:   {accuracy:.1f}% ({len(results['matches']):,}/{total:,})")

    if not quiet:
        if results['mismatches']:
            print(f"\n--- MISMATCHES ({len(results['mismatches'])}) ---")
            for i, item in enumerate(results['mismatches'][:10], 1):
                print(f"{i}. {item['source_url']}")
                print(f"   Expected: {item['expected_destination']}")
                print(f"   Actual:   {item['actual_destination']}")
            if len(results['mismatches']) > 10:
                print(f"... and {len(results['mismatches']) - 10} more")

        if results['missing']:
            print(f"\n--- MISSING ({len(results['missing'])}) ---")
            for i, item in enumerate(results['missing'][:10], 1):
                print(f"{i}. {item['source_url']} -> {item['expected_destination']}")
            if len(results['missing']) > 10:
                print(f"... and {len(results['missing']) - 10} more")

    # Status message
    if accuracy == 100 and not results['extra']:
        print("\nPerfect match! All redirects are correctly implemented.")
    elif accuracy >= 95:
        print("\nVery good match. Minor issues to address.")
    elif accuracy >= 80:
        print("\nGood match with some issues to review.")
    else:
        print("\nSignificant discrepancies found. Review needed.")


def main():
    parser = argparse.ArgumentParser(
        description="Validate redirect implementations against mapping specifications.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s crawled.csv mapping.xlsx
    %(prog)s crawled.csv mapping.xlsx -o validation_report.csv
    %(prog)s crawled.csv mapping.csv --crawled-cols "Address,Redirect URL"
    %(prog)s crawled.xlsx mapping.xlsx --source-cols "old_url,new_url"

Author: Lee Foot (https://leefoot.co.uk)
        """
    )

    parser.add_argument("crawled_file", help="Crawled redirects file (CSV or Excel)")
    parser.add_argument("mapping_file", help="Redirect mapping specification file (CSV or Excel)")

    parser.add_argument("-o", "--output", help="Output file path for full report")
    parser.add_argument("--format", choices=["csv", "xlsx"], default="csv",
                        help="Output format (default: csv)")

    parser.add_argument("--crawled-cols",
                        help="Crawled file columns as 'source,dest' (auto-detected if not specified)")
    parser.add_argument("--source-cols",
                        help="Mapping file columns as 'source,dest' (auto-detected if not specified)")

    parser.add_argument("--mismatches-only", action="store_true",
                        help="Only output mismatches")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Only output summary statistics")

    args = parser.parse_args()

    # Validate files exist
    for filepath in [args.crawled_file, args.mapping_file]:
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}", file=sys.stderr)
            sys.exit(1)

    # Load files
    print(f"Loading crawled file: {args.crawled_file}")
    crawled_df = load_file(args.crawled_file)
    print(f"  Loaded {len(crawled_df):,} rows")

    print(f"Loading mapping file: {args.mapping_file}")
    source_df = load_file(args.mapping_file)
    print(f"  Loaded {len(source_df):,} rows")

    # Determine columns
    if args.crawled_cols:
        crawled_cols = args.crawled_cols.split(',')
        crawled_source_col, crawled_dest_col = crawled_cols[0].strip(), crawled_cols[1].strip()
    else:
        crawled_source_col, crawled_dest_col = find_url_columns(crawled_df)

    if args.source_cols:
        source_cols = args.source_cols.split(',')
        source_source_col, source_dest_col = source_cols[0].strip(), source_cols[1].strip()
    else:
        source_source_col, source_dest_col = find_url_columns(source_df)

    print(f"\nUsing columns:")
    print(f"  Crawled: '{crawled_source_col}' -> '{crawled_dest_col}'")
    print(f"  Mapping: '{source_source_col}' -> '{source_dest_col}'")

    # Validate columns exist
    for col in [crawled_source_col, crawled_dest_col]:
        if col not in crawled_df.columns:
            print(f"Error: Column '{col}' not found in crawled file", file=sys.stderr)
            sys.exit(1)
    for col in [source_source_col, source_dest_col]:
        if col not in source_df.columns:
            print(f"Error: Column '{col}' not found in mapping file", file=sys.stderr)
            sys.exit(1)

    # Run validation
    print("\nValidating redirects...")
    results = verify_mappings(
        crawled_df, source_df,
        crawled_source_col, crawled_dest_col,
        source_source_col, source_dest_col
    )

    # Print report
    print_report(results, quiet=args.quiet)

    # Save output
    if args.output or args.mismatches_only:
        if args.mismatches_only:
            output_data = results['mismatches']
        else:
            output_data = (
                results['matches'] +
                results['mismatches'] +
                results['missing'] +
                results['extra']
            )

        if not output_data:
            print("\nNo data to save.")
            return

        df = pd.DataFrame(output_data)

        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "_mismatches" if args.mismatches_only else ""
            output_file = f"redirect_validation{suffix}_{timestamp}.{args.format}"

        if args.format == "csv":
            df.to_csv(output_file, index=False)
        else:
            df.to_excel(output_file, index=False, sheet_name="Validation Report")

        print(f"\nReport saved to: {output_file}")


if __name__ == "__main__":
    main()
