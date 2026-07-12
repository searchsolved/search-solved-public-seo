#!/usr/bin/env python3
# Author   : Lee Foot
# Website  : https://www.leefoot.com
"""
Fan-Out Query Explorer - CLI Version

Surfaces the sub-questions (fan-out queries) that AI generates when answering
queries in your topic space.

Usage:
    python fan_out_query_explorer_cli.py --login you@email.com --password yourpass --keyword "welding helmets"
    python fan_out_query_explorer_cli.py --login you@email.com --password yourpass --domain example.com

Environment variables DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD are used as fallbacks.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth


def fetch_llm_mentions(login, password, target, target_is_domain, platform,
                       location_code, language_code, limit, include_subdomains=True):
    """
    Call the DataForSEO LLM Mentions Search endpoint and return parsed results.

    Returns:
        tuple: (mentions_list, error_message)
    """
    url = "https://api.dataforseo.com/v3/ai_optimization/llm_mentions/search/live"

    payload = {
        "search_scope": ["any"],
        "ai_platform": platform,
        "location_code": location_code,
        "language_code": language_code,
        "limit": limit,
    }

    if target_is_domain:
        payload["target"] = target
        payload["include_subdomains"] = include_subdomains
    else:
        payload["keyword"] = target

    try:
        response = requests.post(
            url,
            auth=HTTPBasicAuth(login, password),
            headers={"Content-Type": "application/json"},
            data=json.dumps([payload]),
            timeout=120,
        )

        data = response.json()

        if data.get("status_code") != 20000:
            error_msg = data.get("status_message", "Unknown API error")
            return None, f"API Error ({data.get('status_code')}): {error_msg}"

        tasks = data.get("tasks", [])
        if not tasks or not tasks[0].get("result"):
            return [], None

        result = tasks[0]["result"][0]
        items = result.get("items", [])

        mentions = []
        for item in items:
            question = item.get("question", "")
            fan_outs = item.get("fan_out_queries", []) or []
            ai_sv = item.get("ai_search_volume", 0)
            sources = item.get("sources", []) or []
            brands = item.get("brand_entities", []) or []

            mentions.append({
                "question": question,
                "fan_out_queries": fan_outs,
                "ai_search_volume": ai_sv,
                "source_count": len(sources),
                "sources": sources,
                "brand_entities": brands,
                "fan_out_count": len(fan_outs),
            })

        return mentions, None

    except requests.exceptions.Timeout:
        return None, "Request timed out. Try reducing the limit."
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"
    except (KeyError, IndexError, TypeError) as e:
        return None, f"Error parsing response: {str(e)}"


def build_fan_out_table(mentions):
    """
    Build a deduplicated, frequency-counted fan-out query table from mentions.

    Returns:
        pd.DataFrame with columns: fan_out_query, frequency, parent_questions
    """
    records = []
    for mention in mentions:
        parent_q = mention["question"]
        for fq in mention["fan_out_queries"]:
            records.append({
                "fan_out_query": fq,
                "parent_question": parent_q,
            })

    if not records:
        return pd.DataFrame(columns=["fan_out_query", "frequency", "parent_questions"])

    df_raw = pd.DataFrame(records)

    # Count frequency
    counter = Counter(df_raw["fan_out_query"].tolist())

    # Group parent questions
    parents = (
        df_raw.groupby("fan_out_query")["parent_question"]
        .apply(lambda x: "; ".join(sorted(set(x))))
        .reset_index()
    )
    parents.columns = ["fan_out_query", "parent_questions"]

    # Build final table
    df = pd.DataFrame(
        [(q, count) for q, count in counter.most_common()],
        columns=["fan_out_query", "frequency"]
    )
    df = df.merge(parents, on="fan_out_query", how="left")

    return df


def build_parent_table(mentions):
    """Build the parent questions summary table."""
    rows = []
    for m in mentions:
        rows.append({
            "parent_question": m["question"],
            "fan_out_count": m["fan_out_count"],
            "ai_search_volume": m["ai_search_volume"],
            "source_count": m["source_count"],
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("fan_out_count", ascending=False)
    return df


LOCATION_CODES = {
    "uk": 2826,
    "us": 2840,
    "au": 2036,
    "ca": 2124,
    "de": 2276,
    "fr": 2250,
    "es": 2724,
    "it": 2380,
    "nl": 2528,
    "in": 2356,
    "br": 2076,
    "jp": 2392,
}

LANGUAGE_CODES = {
    "en": "en",
    "de": "de",
    "fr": "fr",
    "es": "es",
    "it": "it",
    "nl": "nl",
    "pt": "pt",
    "ja": "ja",
}


def main():
    parser = argparse.ArgumentParser(
        description="Fan-Out Query Explorer: discover the sub-questions AI generates for your topic."
    )

    # Auth
    parser.add_argument("--login", default=os.environ.get("DATAFORSEO_LOGIN", ""),
                        help="DataForSEO login (email). Falls back to DATAFORSEO_LOGIN env var.")
    parser.add_argument("--password", default=os.environ.get("DATAFORSEO_PASSWORD", ""),
                        help="DataForSEO API password. Falls back to DATAFORSEO_PASSWORD env var.")

    # Target (mutually exclusive)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--keyword", help="Keyword or topic to explore")
    target_group.add_argument("--domain", help="Domain to explore mentions for")

    # Options
    parser.add_argument("--platform", choices=["google", "chat_gpt"], default="google",
                        help="AI platform (default: google)")
    parser.add_argument("--location", choices=list(LOCATION_CODES.keys()), default="uk",
                        help="Location code (default: uk)")
    parser.add_argument("--language", choices=list(LANGUAGE_CODES.keys()), default="en",
                        help="Language code (default: en)")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max mention items to retrieve (default: 100)")
    parser.add_argument("--include-subdomains", action="store_true", default=True,
                        help="Include subdomains when using --domain (default: True)")
    parser.add_argument("--output", default="fan_out_queries.csv",
                        help="Output CSV path for fan-out queries (default: fan_out_queries.csv)")
    parser.add_argument("--output-parents", default="parent_questions.csv",
                        help="Output CSV path for parent questions (default: parent_questions.csv)")

    args = parser.parse_args()

    # Validate credentials
    if not args.login or not args.password:
        print("Error: DataForSEO credentials required.")
        print("Provide via --login/--password or DATAFORSEO_LOGIN/DATAFORSEO_PASSWORD env vars.")
        sys.exit(1)

    target = args.keyword if args.keyword else args.domain
    target_is_domain = args.domain is not None

    print(f"Fan-Out Query Explorer")
    print(f"  Target: {target} ({'domain' if target_is_domain else 'keyword'})")
    print(f"  Platform: {args.platform}")
    print(f"  Location: {args.location.upper()}")
    print(f"  Language: {args.language}")
    print(f"  Limit: {args.limit}")
    print(f"  Estimated cost: ~$0.10")
    print()

    # Fetch data
    print("Querying DataForSEO LLM Mentions endpoint...")
    mentions, error = fetch_llm_mentions(
        login=args.login,
        password=args.password,
        target=target,
        target_is_domain=target_is_domain,
        platform=args.platform,
        location_code=LOCATION_CODES[args.location],
        language_code=LANGUAGE_CODES[args.language],
        limit=args.limit,
        include_subdomains=args.include_subdomains,
    )

    if error:
        print(f"Error: {error}")
        sys.exit(1)

    if not mentions:
        print("No mentions returned for this target.")
        sys.exit(0)

    print(f"  Retrieved {len(mentions)} mention items.")

    # Build tables
    df_fan_outs = build_fan_out_table(mentions)
    df_parents = build_parent_table(mentions)

    if df_fan_outs.empty:
        print("No fan-out queries found in the returned mentions.")
        sys.exit(0)

    # Summary
    total_fan_outs = sum(m["fan_out_count"] for m in mentions)
    print()
    print(f"Results:")
    print(f"  Parent questions:       {len(mentions):,}")
    print(f"  Total fan-out queries:  {total_fan_outs:,}")
    print(f"  Unique fan-outs:        {len(df_fan_outs):,}")
    print(f"  Avg fan-outs/question:  {total_fan_outs / len(mentions):.1f}")
    print()

    # Show top 20
    print("Top 20 fan-out queries by frequency:")
    print("-" * 70)
    for _, row in df_fan_outs.head(20).iterrows():
        print(f"  [{row['frequency']}x] {row['fan_out_query']}")
    print()

    # Save
    df_fan_outs.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Fan-out queries saved to: {args.output}")

    df_parents.to_csv(args.output_parents, index=False, encoding="utf-8")
    print(f"Parent questions saved to: {args.output_parents}")


if __name__ == "__main__":
    main()
