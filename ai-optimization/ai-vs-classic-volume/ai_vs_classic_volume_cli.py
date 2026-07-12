#!/usr/bin/env python3
"""
AI vs Classic Search Volume - CLI Version

Compare AI search volume (AI Overviews/ChatGPT) against traditional Google
search volume per keyword.

Usage:
    python ai_vs_classic_volume_cli.py --login your@email.com --password yourpassword --keywords "best crm" "how to code"
    python ai_vs_classic_volume_cli.py --login your@email.com --password yourpassword --keywords-file keywords.txt

Author: Lee Foot
Website: https://www.leefoot.com
"""

import argparse
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import json
import sys
import math


def estimate_cost(num_keywords):
    """Estimate the API cost for a given number of keywords."""
    ai_batches = math.ceil(num_keywords / 1000)
    classic_batches = math.ceil(num_keywords / 1000)
    ai_cost = ai_batches * 0.01
    classic_cost = classic_batches * 0.05
    return ai_cost + classic_cost


def fetch_ai_volume(login, password, kw_list, loc_code, lang_code):
    """Fetch AI search volume from DataForSEO AI Keyword Data endpoint."""
    url = "https://api.dataforseo.com/v3/ai_optimization/ai_keyword_data/keywords_search_volume/live"

    post_data = [{
        "keywords": kw_list,
        "location_code": loc_code,
        "language_code": lang_code,
    }]

    try:
        response = requests.post(
            url,
            auth=HTTPBasicAuth(login, password),
            headers={"Content-Type": "application/json"},
            data=json.dumps(post_data),
            timeout=120
        )
        response_data = response.json()

        if response_data.get("status_code") == 20000:
            results = {}
            tasks = response_data.get("tasks", [])

            for task in tasks:
                task_result = task.get("result")
                if not task_result:
                    continue

                # Handle both response shapes defensively
                for item in task_result:
                    if isinstance(item, dict):
                        if "keyword" in item:
                            kw = item["keyword"]
                            vol = item.get("search_volume") or 0
                            results[kw] = vol
                        elif "items" in item:
                            for sub_item in item["items"]:
                                if isinstance(sub_item, dict) and "keyword" in sub_item:
                                    kw = sub_item["keyword"]
                                    vol = sub_item.get("search_volume") or 0
                                    results[kw] = vol

            return results, None
        else:
            error_msg = response_data.get("status_message", "Unknown error")
            return None, f"AI Volume API Error: {error_msg}"

    except requests.exceptions.RequestException as e:
        return None, f"AI Volume request failed: {str(e)}"
    except Exception as e:
        return None, f"AI Volume error: {str(e)}"


def fetch_classic_volume(login, password, kw_list, loc_code, lang_code):
    """Fetch traditional Google Ads search volume from DataForSEO."""
    url = "https://api.dataforseo.com/v3/keywords_data/google_ads/search_volume/live"

    post_data = [{
        "keywords": kw_list,
        "location_code": loc_code,
        "language_code": lang_code,
    }]

    try:
        response = requests.post(
            url,
            auth=HTTPBasicAuth(login, password),
            headers={"Content-Type": "application/json"},
            data=json.dumps(post_data),
            timeout=120
        )
        response_data = response.json()

        if response_data.get("status_code") == 20000:
            results = {}
            tasks = response_data.get("tasks", [])

            for task in tasks:
                task_result = task.get("result")
                if not task_result:
                    continue

                for item in task_result:
                    if isinstance(item, dict) and "keyword" in item:
                        kw = item["keyword"]
                        vol = item.get("search_volume") or 0
                        results[kw] = {
                            "classic_search_volume": vol,
                            "competition": item.get("competition", 0),
                            "cpc": item.get("cpc", 0),
                        }

            return results, None
        else:
            error_msg = response_data.get("status_message", "Unknown error")
            return None, f"Classic Volume API Error: {error_msg}"

    except requests.exceptions.RequestException as e:
        return None, f"Classic Volume request failed: {str(e)}"
    except Exception as e:
        return None, f"Classic Volume error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Compare AI search volume against traditional Google search volume"
    )
    parser.add_argument("--login", required=True, help="DataForSEO login (email)")
    parser.add_argument("--password", required=True, help="DataForSEO API password")
    parser.add_argument("--keywords", nargs="+", help="Keywords (space-separated, quote multi-word)")
    parser.add_argument("--keywords-file", help="File with keywords (one per line)")
    parser.add_argument("--output", default="ai_vs_classic_volume.csv",
                        help="Output CSV path (default: ai_vs_classic_volume.csv)")
    parser.add_argument("--location-code", type=int, default=2826,
                        help="Location code (default: 2826 for UK)")
    parser.add_argument("--language-code", default="en",
                        help="Language code (default: en)")

    args = parser.parse_args()

    # Get keywords
    keywords = []
    if args.keywords:
        keywords = args.keywords
    elif args.keywords_file:
        with open(args.keywords_file, "r") as f:
            keywords = [line.strip() for line in f if line.strip()]
    else:
        print("Error: Provide keywords via --keywords or --keywords-file")
        sys.exit(1)

    if len(keywords) > 1000:
        print(f"Warning: Truncating to 1000 keywords (received {len(keywords)})")
        keywords = keywords[:1000]

    cost = estimate_cost(len(keywords))
    print(f"Processing {len(keywords)} keywords...")
    print(f"  Location code: {args.location_code}")
    print(f"  Language code: {args.language_code}")
    print(f"  Estimated cost: ${cost:.2f}")
    print()

    # Batch keywords (max 1000 per request)
    batches = [keywords[i:i + 1000] for i in range(0, len(keywords), 1000)]

    # Fetch AI volume
    print("Fetching AI search volume...")
    ai_results = {}
    for idx, batch in enumerate(batches):
        result, error = fetch_ai_volume(args.login, args.password, batch, args.location_code, args.language_code)
        if result:
            ai_results.update(result)
            print(f"  Batch {idx + 1}: got data for {len(result)} keywords")
        if error:
            print(f"  Batch {idx + 1} error: {error}")

    # Fetch classic volume
    print("Fetching classic Google search volume...")
    classic_results = {}
    for idx, batch in enumerate(batches):
        result, error = fetch_classic_volume(args.login, args.password, batch, args.location_code, args.language_code)
        if result:
            classic_results.update(result)
            print(f"  Batch {idx + 1}: got data for {len(result)} keywords")
        if error:
            print(f"  Batch {idx + 1} error: {error}")

    # Combine results
    print("\nCombining results...")
    rows = []
    for kw in keywords:
        ai_vol = ai_results.get(kw, 0) or 0
        classic_data = classic_results.get(kw, {})
        classic_vol = classic_data.get("classic_search_volume", 0) if isinstance(classic_data, dict) else 0
        classic_vol = classic_vol or 0

        total = ai_vol + classic_vol
        ai_share = round((ai_vol / total) * 100, 1) if total > 0 else 0.0
        delta = ai_vol - classic_vol

        rows.append({
            "keyword": kw,
            "ai_search_volume": int(ai_vol),
            "classic_search_volume": int(classic_vol),
            "ai_share_pct": ai_share,
            "delta": int(delta),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("ai_share_pct", ascending=False).reset_index(drop=True)

    # Save output
    df.to_csv(args.output, index=False, encoding="utf-8")

    # Print summary
    print(f"\nResults saved to: {args.output}")
    print(f"  Total keywords: {len(df):,}")
    print(f"  Average AI share: {df['ai_share_pct'].mean():.1f}%")
    print(f"  Keywords where AI > Classic: {int((df['ai_search_volume'] > df['classic_search_volume']).sum()):,}")
    print(f"  Keywords where Classic > AI: {int((df['classic_search_volume'] > df['ai_search_volume']).sum()):,}")

    # Show top 10
    print("\nTop 10 keywords by AI share:")
    print("-" * 70)
    for _, row in df.head(10).iterrows():
        print(
            f"  {row['keyword']:<30} "
            f"AI: {row['ai_search_volume']:>8,}  "
            f"Classic: {row['classic_search_volume']:>8,}  "
            f"Share: {row['ai_share_pct']:>5.1f}%"
        )


if __name__ == "__main__":
    main()
