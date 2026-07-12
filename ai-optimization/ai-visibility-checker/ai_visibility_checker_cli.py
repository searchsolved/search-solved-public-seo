#!/usr/bin/env python3
####################################################################################
#                                                                                  #
#  AI Visibility Checker - CLI Version                                             #
#                                                                                  #
#  See who gets cited in Google AI Overviews and ChatGPT answers                   #
#  for a given domain's topic space.                                               #
#                                                                                  #
####################################################################################
# Author: Lee Foot                                                                 #
# Website: https://leefoot.com                                                     #
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                       #
####################################################################################

"""
AI Visibility Checker - CLI

Shows who gets cited in Google AI Overviews and ChatGPT answers
for a given domain's topic space.

Usage:
    export DATAFORSEO_LOGIN=your@email.com
    export DATAFORSEO_PASSWORD=yourpassword
    python ai_visibility_checker_cli.py --entities example.com --platform google

Or pass credentials directly:
    python ai_visibility_checker_cli.py --login your@email.com --password pass --entities example.com

Author: Lee Foot
Website: https://www.leefoot.com
"""

import argparse
import os
import sys
import json
from collections import Counter
from datetime import datetime

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth


def fetch_llm_mentions(login, password, entities, platform_value, location, language, max_results):
    """
    Call the DataForSEO AI Optimization LLM Mentions Search endpoint.

    Returns (list_of_mentions, error_string_or_none).
    """
    url = "https://api.dataforseo.com/v3/ai_optimization/llm_mentions/search/live"

    post_data = [{
        "entities": entities,
        "search_scope": ["any"],
        "platform": platform_value,
        "location_name": location,
        "language_name": language,
        "limit": max_results
    }]

    try:
        response = requests.post(
            url,
            auth=HTTPBasicAuth(login, password),
            headers={"Content-Type": "application/json"},
            data=json.dumps(post_data),
            timeout=120
        )

        data = response.json()

        if data.get("status_code") == 20000:
            mentions = []
            tasks = data.get("tasks", [])

            if tasks and tasks[0].get("result"):
                for result_block in tasks[0]["result"]:
                    items = result_block.get("items", [])
                    if items:
                        mentions.extend(items)
            return mentions, None
        else:
            error_msg = data.get("status_message", "Unknown error")
            task_errors = ""
            tasks = data.get("tasks", [])
            if tasks and tasks[0].get("status_message"):
                task_errors = f" - {tasks[0]['status_message']}"
            return None, f"API Error ({data.get('status_code')}): {error_msg}{task_errors}"

    except requests.exceptions.Timeout:
        return None, "Request timed out. Try reducing the results limit."
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"
    except Exception as e:
        return None, f"Error: {str(e)}"


def aggregate_cited_domains(mentions):
    """Aggregate source domains across all mentions."""
    domain_counter = Counter()
    for mention in mentions:
        sources = mention.get("sources") or []
        for source in sources:
            domain = source.get("domain", "")
            if domain:
                domain_counter[domain] += 1
    return domain_counter.most_common(50)


def build_export_dataframe(mentions):
    """Build a flat DataFrame for CSV export."""
    rows = []
    for mention in mentions:
        sources = mention.get("sources") or []
        fan_out = mention.get("fan_out_queries") or []
        brand_entities = mention.get("brand_entities") or []

        source_domains = "; ".join(
            s.get("domain", "") for s in sources if s.get("domain")
        )
        source_urls = "; ".join(
            s.get("url", "") for s in sources if s.get("url")
        )
        fan_out_list = "; ".join(fan_out)
        brand_list = "; ".join(brand_entities)

        rows.append({
            "question": mention.get("question", ""),
            "ai_search_volume": mention.get("ai_search_volume", 0),
            "platform": mention.get("platform", ""),
            "model_name": mention.get("model_name", ""),
            "sources_count": len(sources),
            "source_domains": source_domains,
            "source_urls": source_urls,
            "fan_out_queries_count": len(fan_out),
            "fan_out_queries": fan_out_list,
            "brand_entities": brand_list,
            "first_response_at": mention.get("first_response_at", ""),
            "last_response_at": mention.get("last_response_at", ""),
            "answer": mention.get("answer", ""),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="AI Visibility Checker - Find LLM citations for your domain"
    )
    parser.add_argument(
        "--login",
        default=os.environ.get("DATAFORSEO_LOGIN", ""),
        help="DataForSEO login email (or set DATAFORSEO_LOGIN env var)"
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("DATAFORSEO_PASSWORD", ""),
        help="DataForSEO API password (or set DATAFORSEO_PASSWORD env var)"
    )
    parser.add_argument(
        "--entities", nargs="+", required=True,
        help="Domains or brand entities to check (space-separated, max 10)"
    )
    parser.add_argument(
        "--platform",
        choices=["google", "chat_gpt", "both"],
        default="google",
        help="Platform to query (default: google)"
    )
    parser.add_argument(
        "--location",
        default="United Kingdom",
        help="Location name (default: United Kingdom)"
    )
    parser.add_argument(
        "--language",
        default="English",
        help="Language name (default: English)"
    )
    parser.add_argument(
        "--limit", type=int, default=50,
        help="Max results to retrieve (default: 50, max: 1000)"
    )
    parser.add_argument(
        "--output",
        default="ai_visibility_results.csv",
        help="Output CSV path (default: ai_visibility_results.csv)"
    )
    parser.add_argument(
        "--show-domains", action="store_true",
        help="Print top cited domains summary to stdout"
    )

    args = parser.parse_args()

    # Validate credentials
    if not args.login or not args.password:
        print("Error: DataForSEO credentials required.")
        print("  Set DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD environment variables,")
        print("  or pass --login and --password arguments.")
        sys.exit(1)

    # Validate entities
    entities = args.entities[:10]
    if len(args.entities) > 10:
        print(f"Warning: Only the first 10 entities will be used (got {len(args.entities)}).")

    # Validate limit
    result_limit = min(max(args.limit, 1), 1000)

    # Determine platforms
    platforms_to_query = []
    if args.platform == "both":
        platforms_to_query = ["google", "chat_gpt"]
    else:
        platforms_to_query = [args.platform]

    call_count = len(platforms_to_query)
    estimated_cost = call_count * 0.10

    print(f"AI Visibility Checker")
    print(f"  Entities: {', '.join(entities)}")
    print(f"  Platform: {args.platform}")
    print(f"  Location: {args.location}")
    print(f"  Language: {args.language}")
    print(f"  Limit: {result_limit}")
    print(f"  Estimated cost: ~${estimated_cost:.2f}")
    print()

    all_mentions = []
    errors = []

    for pf in platforms_to_query:
        pf_label = "Google AI Overviews" if pf == "google" else "ChatGPT"
        print(f"  Querying {pf_label}...", end=" ", flush=True)

        mentions, error = fetch_llm_mentions(
            args.login,
            args.password,
            entities,
            pf,
            args.location,
            args.language,
            result_limit
        )

        if mentions:
            all_mentions.extend(mentions)
            print(f"Found {len(mentions)} mentions")
        else:
            print("0 mentions")

        if error:
            errors.append(f"{pf_label}: {error}")
            print(f"    Error: {error}")

    print()

    if errors:
        print("Errors:")
        for err in errors:
            print(f"  - {err}")
        print()

    if all_mentions:
        # Summary stats
        total_volume = sum(m.get("ai_search_volume", 0) for m in all_mentions)
        unique_questions = len(set(m.get("question", "") for m in all_mentions))

        print(f"Results Summary:")
        print(f"  Total mentions: {len(all_mentions):,}")
        print(f"  Total AI search volume: {int(total_volume):,}")
        print(f"  Unique questions: {unique_questions:,}")

        # Top cited domains
        if args.show_domains:
            top_domains = aggregate_cited_domains(all_mentions)
            if top_domains:
                print(f"\nTop Cited Domains:")
                for domain, count in top_domains[:20]:
                    print(f"  {domain}: {count}")

        # Export
        df_export = build_export_dataframe(all_mentions)
        df_export = df_export.sort_values("ai_search_volume", ascending=False)
        df_export.to_csv(args.output, index=False, encoding="utf-8")
        print(f"\nResults saved to: {args.output}")
        print(f"  Rows: {len(df_export):,}")
    else:
        print("No mentions found for the given entities and configuration.")
        sys.exit(0)


if __name__ == "__main__":
    main()
