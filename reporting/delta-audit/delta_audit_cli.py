#!/usr/bin/env python3
"""
Delta Audit Tool - CLI Version

Detect significant traffic changes in GSC data.

Usage:
    python delta_audit_cli.py --input gsc_data.csv --output report.csv

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd
import numpy as np
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Detect significant traffic changes in GSC data'
    )
    parser.add_argument('--input', required=True, help='Input GSC CSV file')
    parser.add_argument('--output', default='delta_audit_report.csv', help='Output CSV path')
    parser.add_argument('--date-col', default='date', help='Date column name')
    parser.add_argument('--clicks-col', default='clicks', help='Clicks column name')
    parser.add_argument('--impressions-col', default='impressions', help='Impressions column name')
    parser.add_argument('--window', type=int, default=7, help='Rolling window size in days')

    args = parser.parse_args()

    print(f"Loading data from: {args.input}")
    data = pd.read_csv(args.input, low_memory=False)

    # Validate columns
    required_cols = [args.date_col, args.clicks_col, args.impressions_col]
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        print(f"Error: Missing columns: {', '.join(missing)}")
        print(f"Available columns: {', '.join(data.columns.tolist())}")
        sys.exit(1)

    # Process data
    data[args.date_col] = pd.to_datetime(data[args.date_col], errors='coerce')
    data = data.dropna(subset=[args.date_col])
    data[args.clicks_col] = pd.to_numeric(data[args.clicks_col], errors='coerce')
    data[args.impressions_col] = pd.to_numeric(data[args.impressions_col], errors='coerce')
    data = data.dropna(subset=[args.clicks_col, args.impressions_col])

    data.set_index(args.date_col, inplace=True)
    data = data.groupby(data.index).sum(numeric_only=True)

    # Calculate daily clicks
    daily_clicks = data[args.clicks_col].resample('D').sum()
    rolling_clicks = daily_clicks.rolling(window=args.window).mean()
    rolling_clicks_diff = rolling_clicks.diff().abs()
    significant_change_date = rolling_clicks_diff.idxmax()

    print(f"\nMost Significant Traffic Change: {significant_change_date.strftime('%Y-%m-%d')}")

    # Calculate weekly data
    weekly_data = data.resample('W-MON').sum()

    # Snap to week
    significant_week_start = significant_change_date - pd.Timedelta(days=significant_change_date.weekday())
    significant_week_end = significant_week_start + pd.Timedelta(days=6)

    print(f"Significant Week: {significant_week_start.strftime('%Y-%m-%d')} to {significant_week_end.strftime('%Y-%m-%d')}")

    # Week over week comparison
    pre_week_start = significant_week_start - pd.Timedelta(days=7)
    post_week_start = significant_week_start + pd.Timedelta(days=7)

    comparison = []
    if pre_week_start in weekly_data.index and post_week_start in weekly_data.index:
        pre_week_data = weekly_data.loc[pre_week_start]
        post_week_data = weekly_data.loc[post_week_start]

        abs_change_clicks = post_week_data[args.clicks_col] - pre_week_data[args.clicks_col]
        rel_change_clicks = (abs_change_clicks / pre_week_data[args.clicks_col]) * 100 if pre_week_data[args.clicks_col] > 0 else 0

        abs_change_impressions = post_week_data[args.impressions_col] - pre_week_data[args.impressions_col]
        rel_change_impressions = (abs_change_impressions / pre_week_data[args.impressions_col]) * 100 if pre_week_data[args.impressions_col] > 0 else 0

        comparison = [
            {'metric': 'Clicks', 'pre_week': pre_week_data[args.clicks_col], 'post_week': post_week_data[args.clicks_col], 'absolute_change': abs_change_clicks, 'relative_change_pct': rel_change_clicks},
            {'metric': 'Impressions', 'pre_week': pre_week_data[args.impressions_col], 'post_week': post_week_data[args.impressions_col], 'absolute_change': abs_change_impressions, 'relative_change_pct': rel_change_impressions}
        ]

        print(f"\nWeek-over-Week Changes:")
        print(f"  Clicks: {int(abs_change_clicks):+,} ({rel_change_clicks:+.1f}%)")
        print(f"  Impressions: {int(abs_change_impressions):+,} ({rel_change_impressions:+.1f}%)")

    # Save results
    df_comparison = pd.DataFrame(comparison)
    df_comparison.to_csv(args.output, index=False)
    print(f"\nReport saved to: {args.output}")


if __name__ == '__main__':
    main()
