#!/usr/bin/env python3
"""
Navigation Label Mismatch Detector - CLI Version

Find navigation links whose anchor text does not match the destination page's
H1 or primary title keyword, using Screaming Frog exports.

Usage:
    python nav_label_mismatch_detector_cli.py --inlinks all_inlinks.csv --internal internal_html.csv --output results.csv

Author: Lee Foot
Website: https://www.leefoot.com
"""

import argparse
import sys

import pandas as pd

TITLE_SEPARATOR = "|"


def normalise(series):
    return series.fillna("").str.strip().str.casefold()


def main():
    parser = argparse.ArgumentParser(
        description="Compare navigation anchor text against destination page H1s and title keywords"
    )
    parser.add_argument('--inlinks', required=True, help='Screaming Frog All Inlinks export (CSV)')
    parser.add_argument('--internal', required=True, help='Screaming Frog Internal HTML export (CSV)')
    parser.add_argument('--output', default='nav_label_mismatches.csv', help='Output CSV path')

    args = parser.parse_args()

    print(f"Loading inlinks from: {args.inlinks}")
    try:
        df_inlinks = pd.read_csv(
            args.inlinks,
            usecols=["Source", "Destination", "Alt Text", "Anchor"],
            dtype=str
        )
    except ValueError as e:
        print(f"Error: inlinks export is missing required columns (Source, Destination, Alt Text, Anchor): {e}")
        sys.exit(1)
    print(f"  Loaded {len(df_inlinks):,} rows")

    print(f"Loading internal HTML from: {args.internal}")
    try:
        df_internal = pd.read_csv(
            args.internal,
            usecols=["Address", "H1-1", "Title 1"],
            dtype=str
        )
    except ValueError as e:
        print(f"Error: internal HTML export is missing required columns (Address, H1-1, Title 1): {e}")
        sys.exit(1)
    print(f"  Loaded {len(df_internal):,} rows")

    # Identify navigation links: alt text matches anchor text
    df_nav = df_inlinks[df_inlinks["Alt Text"] == df_inlinks["Anchor"]]
    df_nav = df_nav[["Source", "Destination", "Anchor"]]

    if len(df_nav) == 0:
        print("Error: No navigation links found (no rows where alt text matches anchor text).")
        sys.exit(1)

    print(f"  Found {len(df_nav):,} navigation links (alt text matches anchor text)")

    # Merge with internal HTML data
    df_nav = pd.merge(df_nav, df_internal, left_on="Destination", right_on="Address", how="left")
    df_nav = df_nav[df_nav["Address"].notna()]
    del df_nav["Address"]

    if len(df_nav) == 0:
        print("Error: No navigation links matched a page in the internal HTML export. Check both exports come from the same crawl.")
        sys.exit(1)

    # Deduplicate on anchor, H1 and title combination
    df_nav.drop_duplicates(subset=["Anchor", "H1-1", "Title 1"], keep="first", inplace=True)

    # Extract the primary keyword from the page title
    df_nav["Page Title Primary KW"] = df_nav["Title 1"].str.split(TITLE_SEPARATOR, regex=False).str[0]
    del df_nav["Title 1"]
    df_nav["Page Title Primary KW"] = df_nav["Page Title Primary KW"].str.rstrip()

    # Compare navigation label against H1 and title primary keyword
    anchor_norm = normalise(df_nav["Anchor"])
    df_nav["Anchor Matches H1"] = anchor_norm == normalise(df_nav["H1-1"])
    df_nav["Anchor Matches Title KW"] = anchor_norm == normalise(df_nav["Page Title Primary KW"])
    df_nav["Mismatch"] = ~(df_nav["Anchor Matches H1"] & df_nav["Anchor Matches Title KW"])

    # Sort mismatches to the top
    df_nav = df_nav.sort_values(["Mismatch", "Anchor"], ascending=[False, True])

    df_nav.to_csv(args.output, index=False, encoding="utf-8-sig")

    mismatch_count = int(df_nav["Mismatch"].sum())

    print(f"\nResults saved to: {args.output}")
    print(f"  Navigation labels analysed: {len(df_nav):,}")
    print(f"  Mismatched labels: {mismatch_count:,}")


if __name__ == '__main__':
    main()
