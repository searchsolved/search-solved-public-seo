#!/usr/bin/env python3
"""
Related Searches Tree Builder - CLI Version

Build hierarchical trees of related searches using ValueSERP API.

Usage:
    python related_searches_tree_cli.py --keyword "seo tools" --api-key YOUR_KEY

Author: Lee Foot
Website: https://www.leefoot.com
"""

import argparse
import pandas as pd
import requests
import os
import sys


def get_related_searches(keyword, api_key, location):
    """Fetch related searches for a keyword from ValueSERP."""
    params = {
        'api_key': api_key,
        'q': keyword,
        'location': location,
        'num': '10'
    }

    try:
        response = requests.get('https://api.valueserp.com/search', params=params)
        data = response.json()

        related = data.get('related_searches', [])
        return [r['query'] for r in related] if related else []

    except Exception as e:
        print(f"  Warning: Error fetching '{keyword}': {str(e)}")
        return []


def build_tree(seed_keywords, api_key, location, max_depth, max_results):
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

        related = get_related_searches(keyword, api_key, location)

        for r in related[:max_results]:
            r_lower = r.lower()
            if r_lower not in visited:
                relationships['Parent'].append(keyword.lower())
                relationships['Child'].append(r_lower)

                if depth + 1 < max_depth:
                    queue.append((r, depth + 1))

    print(f"  Total API queries: {total_queries}")
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
        description='Build hierarchical trees of related searches using ValueSERP API'
    )
    parser.add_argument('--keyword', required=True, nargs='+',
                        help='Seed keyword(s) to explore')
    parser.add_argument('--api-key', help='ValueSERP API key (or set VALUESERP_API_KEY env var)')
    parser.add_argument('--output', default='related_searches.csv',
                        help='Output CSV path (default: related_searches.csv)')
    parser.add_argument('--dot-output', help='Output DOT file path for Graphviz')
    parser.add_argument('--depth', type=int, default=2,
                        help='Crawl depth (default: 2)')
    parser.add_argument('--max-results', type=int, default=10,
                        help='Max related searches per keyword (default: 10)')
    parser.add_argument('--location', default='United Kingdom',
                        help='Search location (default: United Kingdom)')

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get('VALUESERP_API_KEY')
    if not api_key:
        print("Error: API key required. Use --api-key or set VALUESERP_API_KEY environment variable")
        sys.exit(1)

    seed_keywords = args.keyword
    print(f"Building tree for: {', '.join(seed_keywords)}")
    print(f"  Depth: {args.depth}")
    print(f"  Location: {args.location}")
    print(f"  Max results per keyword: {args.max_results}")

    # Build tree
    relationships = build_tree(
        seed_keywords,
        api_key,
        args.location,
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
