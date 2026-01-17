#!/usr/bin/env python3
"""
Keyword Trends Analyzer - CLI Version

Analyze Google Trends data for keywords and calculate trend slopes.

Usage:
    python keyword_trends_analyzer_cli.py --input keywords.csv --output trends.csv

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd
from datetime import datetime
import time
from random import randint
import sys

try:
    from pytrends.request import TrendReq
except ImportError:
    print("Error: pytrends not installed. Install with: pip install pytrends")
    sys.exit(1)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    parser = argparse.ArgumentParser(description='Analyze Google Trends for keywords')
    parser.add_argument('--input', required=True, help='Input CSV/TXT with keywords')
    parser.add_argument('--output', default='keyword_trends.csv', help='Output CSV path')
    parser.add_argument('--keyword-col', default='keyword', help='Keyword column name')
    parser.add_argument('--geo', default='GB', help='Geographic region (e.g., US, GB, DE)')
    parser.add_argument('--timeframe', default='today 5-y', help='Time range')
    parser.add_argument('--delay', type=int, default=3, help='Delay between requests (seconds)')
    parser.add_argument('--max-keywords', type=int, default=100, help='Maximum keywords to analyze')

    args = parser.parse_args()

    print(f"Loading keywords from: {args.input}")

    if args.input.endswith('.txt'):
        with open(args.input, 'r') as f:
            keywords = [line.strip() for line in f if line.strip()]
    else:
        df = pd.read_csv(args.input)
        cols = df.columns.tolist()
        kw_col = next((c for c in cols if c.lower() == args.keyword_col.lower()), cols[0])
        keywords = df[kw_col].dropna().unique().tolist()

    keywords = keywords[:args.max_keywords]
    print(f"  Analyzing {len(keywords)} keywords")

    pytrend = TrendReq(hl='en-US', tz=0)
    all_results = []
    keyword_chunks = list(chunks(keywords, 5))

    print(f"\nFetching trends data (geo={args.geo}, timeframe={args.timeframe})...")

    for i, chunk in enumerate(keyword_chunks):
        print(f"  Batch {i+1}/{len(keyword_chunks)}: {', '.join(chunk[:3])}...")

        try:
            pytrend.build_payload(kw_list=chunk, timeframe=args.timeframe, geo=args.geo)
            interest_df = pytrend.interest_over_time()

            if not interest_df.empty:
                if 'isPartial' in interest_df.columns:
                    interest_df = interest_df.drop('isPartial', axis=1)

                year_today = datetime.now().year
                last_year = year_today - 1
                prev_year = year_today - 2

                interest_df = interest_df.reset_index()

                for kw in chunk:
                    if kw in interest_df.columns:
                        interest_df['year'] = interest_df['date'].dt.year

                        last_year_data = interest_df[interest_df['year'] == last_year][kw].mean()
                        prev_year_data = interest_df[interest_df['year'] == prev_year][kw].mean()

                        last_year_data = last_year_data if pd.notna(last_year_data) else 0
                        prev_year_data = prev_year_data if pd.notna(prev_year_data) else 0

                        if prev_year_data > 0:
                            slope = ((last_year_data - prev_year_data) / prev_year_data) * 100
                        else:
                            slope = 0

                        avg_interest = interest_df[kw].mean()

                        all_results.append({
                            "keyword": kw,
                            "avg_interest": round(avg_interest, 1),
                            "last_year_avg": round(last_year_data, 1),
                            "prev_year_avg": round(prev_year_data, 1),
                            "slope_pct": round(slope, 1),
                            "trend": "Rising" if slope > 10 else ("Declining" if slope < -10 else "Stable")
                        })

        except Exception as e:
            print(f"    Error: {str(e)[:50]}")

        if i < len(keyword_chunks) - 1:
            time.sleep(randint(1, args.delay))

    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results = df_results.sort_values("slope_pct", ascending=False)
        df_results.to_csv(args.output, index=False, encoding='utf-8-sig')

        rising = (df_results["slope_pct"] > 10).sum()
        declining = (df_results["slope_pct"] < -10).sum()
        stable = len(df_results) - rising - declining

        print(f"\nResults saved to: {args.output}")
        print(f"  Keywords analyzed: {len(df_results)}")
        print(f"  Rising trends: {rising}")
        print(f"  Declining trends: {declining}")
        print(f"  Stable: {stable}")

        print(f"\nTop 5 Rising:")
        for _, row in df_results[df_results["slope_pct"] > 0].head(5).iterrows():
            print(f"  +{row['slope_pct']:.1f}% {row['keyword']}")

        print(f"\nTop 5 Declining:")
        for _, row in df_results[df_results["slope_pct"] < 0].sort_values("slope_pct").head(5).iterrows():
            print(f"  {row['slope_pct']:.1f}% {row['keyword']}")
    else:
        print("No trend data retrieved. Check your connection or try different keywords.")
        sys.exit(1)


if __name__ == '__main__':
    main()
