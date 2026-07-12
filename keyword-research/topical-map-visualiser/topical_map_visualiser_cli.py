#!/usr/bin/env python3
"""
Topical Map Visualiser - CLI Version

Render a tagged keyword CSV (parent topic > subtopic > keyword) as an
interactive, zoomable D3.js circle packing chart. Pairs with the
Topical Map Generator tool in this repository.

Usage:
    python topical_map_visualiser_cli.py --input tagged_keywords.csv --output topical_map.html

Author: Lee Foot
Website: https://www.leefoot.com
"""

import argparse
import sys

import pandas as pd

from topical_map_chart import METRIC_CHOICES, render_chart


def main():
    parser = argparse.ArgumentParser(description='Render a tagged keyword CSV as an interactive D3.js circle packing chart')
    parser.add_argument('--input', required=True, help='Input CSV with tagged keywords')
    parser.add_argument('--output', default='topical_map.html', help='Output HTML path')
    parser.add_argument('--metric', default='count', choices=METRIC_CHOICES,
                        help='Metric used to size circles (default: count)')
    parser.add_argument('--title', help='Optional chart title (defaults to a title based on the metric)')
    parser.add_argument('--parent-col', default='Parent', help='Column holding the parent topic (default: Parent)')
    parser.add_argument('--child-col', default='Child', help='Column holding the subtopic (default: Child)')
    parser.add_argument('--keyword-col', default='query', help='Column holding the keyword (default: query)')
    parser.add_argument('--position-col', default='position',
                        help='Column holding average position, used by first_page_count and top_3_count (default: position)')
    parser.add_argument('--impressions-col', default='impressions',
                        help='Column holding impressions, used by the impressions metric (default: impressions)')
    parser.add_argument('--clicks-col', default='clicks',
                        help='Column holding clicks, used by the clicks metric (default: clicks)')

    args = parser.parse_args()

    print(f"Loading keywords from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  Found {len(df):,} rows")

    # Validate required columns before building the hierarchy
    required = [args.parent_col, args.child_col, args.keyword_col]
    if args.metric == 'impressions':
        required.append(args.impressions_col)
    elif args.metric == 'clicks':
        required.append(args.clicks_col)
    elif args.metric in ('first_page_count', 'top_3_count'):
        required.append(args.position_col)

    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Error: missing column(s) in input CSV: {', '.join(missing)}")
        print(f"Available columns: {', '.join(df.columns)}")
        print("Use the column mapping flags (--parent-col, --child-col, --keyword-col, etc.) to match your CSV.")
        sys.exit(1)

    df = df.dropna(subset=required)

    html = render_chart(
        df,
        metric=args.metric,
        chart_title=args.title,
        parent_col=args.parent_col,
        child_col=args.child_col,
        keyword_col=args.keyword_col,
        position_col=args.position_col,
        impressions_col=args.impressions_col,
        clicks_col=args.clicks_col,
    )

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nChart generated using the '{args.metric}' metric and saved to: {args.output}")
    print("Open the file in a browser and click a circle to zoom in. Click the background to zoom back out.")


if __name__ == '__main__':
    main()
