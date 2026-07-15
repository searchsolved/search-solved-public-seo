# Author: Lee Foot
# Website: https://leefoot.com

"""
Related Searches Tree Builder - CLI Version

Build hierarchical trees of related searches using the DataForSEO SERP API.

Usage:
    python related_searches_tree_cli.py --keyword "seo tools" --login YOUR_LOGIN --password YOUR_PASSWORD

Environment variables:
    DATAFORSEO_LOGIN     - DataForSEO account login
    DATAFORSEO_PASSWORD  - DataForSEO account password
"""

import argparse
import pandas as pd
import requests
import os
import sys
from base64 import b64encode


# Location code mapping
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
    "Ireland": 2372,
}


def _build_auth_headers(login, password):
    """Build DataForSEO Basic auth headers."""
    cred = b64encode(f"{login}:{password}".encode()).decode()
    return {
        'Authorization': f'Basic {cred}',
        'Content-Type': 'application/json'
    }


def get_related_searches(keyword, login, password, location_code):
    """Fetch related searches for a keyword from the DataForSEO SERP API."""
    headers = _build_auth_headers(login, password)
    payload = [{
        "keyword": keyword,
        "location_code": location_code,
        "language_code": "en",
        "device": "desktop",
        "depth": 10
    }]

    try:
        response = requests.post(
            'https://api.dataforseo.com/v3/serp/google/organic/live/advanced',
            headers=headers,
            json=payload,
            timeout=60
        )
        data = response.json()

        if data.get("status_code") != 20000:
            msg = data.get("status_message", "Unknown error")
            print(f"  Warning: API error for '{keyword}': {msg}")
            return []

        items = data["tasks"][0]["result"][0]["items"]
        related = []
        for item in items:
            if item["type"] == "related_searches":
                for rs in item.get("items", []):
                    related.append(rs["title"])
        return related

    except Exception as e:
        print(f"  Warning: Error fetching '{keyword}': {str(e)}")
        return []


def build_tree(seed_keywords, login, password, location_code, max_depth, max_results):
    """Build a tree of related searches."""
    relationships = {'Parent': [], 'Child': []}
    visited = set()
    queue = [(kw, 0) for kw in seed_keywords]
    total_queries = 0

    while queue:
        keyword, depth = queue.pop(0)

        if keyword in visited or depth >= max_depth:
            continue

        visited.add(keyword)
        total_queries += 1

        print(f"  Depth {depth}: Exploring '{keyword}'")

        related = get_related_searches(keyword, login, password, location_code)

        for r in related[:max_results]:
            r_lower = r.lower()
            if r_lower not in visited:
                relationships['Parent'].append(keyword.lower())
                relationships['Child'].append(r_lower)

                if depth + 1 < max_depth:
                    queue.append((r, depth + 1))

    print(f"  Total API queries: {total_queries}")
    est_cost = total_queries * 0.002
    print(f"  Estimated cost: ${est_cost:.3f}")
    return relationships


def generate_dot(relationships, root_keywords, output_path):
    """Generate DOT format for Graphviz."""
    df = pd.DataFrame(relationships)
    if df.empty:
        return

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('digraph RelatedSearches {\n')
        f.write('    rankdir=TB;\n')
        f.write('    node [shape=box, style=filled, fillcolor=lightblue];\n')

        for root in root_keywords:
            f.write(f'    "{root.lower()}" [fillcolor=lightgreen];\n')

        for _, row in df.iterrows():
            parent = row['Parent'].replace('"', '\\"')
            child = row['Child'].replace('"', '\\"')
            f.write(f'    "{parent}" -> "{child}";\n')

        f.write('}\n')


def print_tree(relationships, root_keywords):
    """Print ASCII tree representation."""
    df = pd.DataFrame(relationships)
    if df.empty:
        return

    def render_node(node, prefix="", is_last=True, rendered=None):
        if rendered is None:
            rendered = set()

        if node in rendered:
            print(f"{prefix}{'└── ' if is_last else '├── '}{node} (circular)")
            return

        rendered.add(node)
        print(f"{prefix}{'└── ' if is_last else '├── '}{node}")

        children = df[df['Parent'] == node]['Child'].tolist()
        for i, child in enumerate(children):
            child_is_last = (i == len(children) - 1)
            child_prefix = prefix + ("    " if is_last else "│   ")
            render_node(child, child_prefix, child_is_last, rendered.copy())

    for i, root in enumerate(root_keywords):
        is_last = (i == len(root_keywords) - 1)
        render_node(root.lower(), "", is_last)


def main():
    parser = argparse.ArgumentParser(
        description='Build hierarchical trees of related searches using the DataForSEO SERP API'
    )
    parser.add_argument('--keyword', required=True, nargs='+',
                        help='Seed keyword(s) to explore')
    parser.add_argument('--login',
                        help='DataForSEO login (or set DATAFORSEO_LOGIN env var)')
    parser.add_argument('--password',
                        help='DataForSEO password (or set DATAFORSEO_PASSWORD env var)')
    parser.add_argument('--output', default='related_searches.csv',
                        help='Output CSV path (default: related_searches.csv)')
    parser.add_argument('--dot-output', help='Output DOT file path for Graphviz')
    parser.add_argument('--depth', type=int, default=2,
                        help='Crawl depth (default: 2)')
    parser.add_argument('--max-results', type=int, default=10,
                        help='Max related searches per keyword (default: 10)')
    parser.add_argument('--location-code', type=int, default=2826,
                        help='DataForSEO location code (default: 2826 for United Kingdom)')

    args = parser.parse_args()

    # Get credentials
    login = args.login or os.environ.get('DATAFORSEO_LOGIN')
    password = args.password or os.environ.get('DATAFORSEO_PASSWORD')
    if not login or not password:
        print("Error: DataForSEO credentials required.")
        print("  Use --login and --password, or set DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD environment variables.")
        sys.exit(1)

    seed_keywords = args.keyword
    print(f"Building tree for: {', '.join(seed_keywords)}")
    print(f"  Depth: {args.depth}")
    print(f"  Location code: {args.location_code}")
    print(f"  Max results per keyword: {args.max_results}")

    # Estimate cost
    est_calls = sum(args.max_results ** i for i in range(args.depth + 1)) * len(seed_keywords)
    est_cost = est_calls * 0.002
    print(f"  Estimated max API calls: ~{est_calls} (approx. ${est_cost:.2f})")

    # Build tree
    relationships = build_tree(
        seed_keywords,
        login,
        password,
        args.location_code,
        args.depth,
        args.max_results
    )

    if relationships['Parent']:
        df = pd.DataFrame(relationships)
        df = df.drop_duplicates()
        df = df[df['Parent'] != df['Child']]

        # Save CSV
        df.to_csv(args.output, index=False, encoding='utf-8-sig')
        print(f"\nResults saved to: {args.output}")
        print(f"  Relationships: {len(df)}")

        unique_keywords = set(df['Parent'].tolist() + df['Child'].tolist())
        print(f"  Unique keywords: {len(unique_keywords)}")

        # Save DOT file if requested
        if args.dot_output:
            generate_dot(relationships, seed_keywords, args.dot_output)
            print(f"  DOT file: {args.dot_output}")

        # Print tree
        print("\nTree structure:")
        print_tree(relationships, seed_keywords)

    else:
        print("No related searches found")


if __name__ == '__main__':
    main()
