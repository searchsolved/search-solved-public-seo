#!/usr/bin/env python3
"""
LLM Sitemap Creator - CLI Version

Use GPT to generate hierarchical sitemap structures from keywords.

Usage:
    python llm_sitemap_creator_cli.py --input keywords.csv --api-key YOUR_KEY

Author: Lee Foot
Website: https://www.leefoot.com
"""

import argparse
import pandas as pd
import json
import os
import sys
from openai import OpenAI


def flatten_sitemap(sitemap):
    """Extract all keywords from sitemap structure."""
    keywords = []
    for key, value in sitemap.items():
        keywords.append(key)
        if isinstance(value, dict):
            keywords.extend(flatten_sitemap(value))
    return keywords


def generate_sitemap_with_llm(keyword_volumes, api_key, max_categories, max_depth, model):
    """Generate sitemap structure using OpenAI."""
    client = OpenAI(api_key=api_key)

    keywords_str = json.dumps(keyword_volumes, indent=2)

    messages = [
        {
            "role": "system",
            "content": "You are an SEO expert assistant. Create a sitemap structure using the provided keywords and their search volumes. Return ONLY valid JSON."
        },
        {
            "role": "user",
            "content": f"""Given the following keywords and their search volumes:

{keywords_str}

Create a sitemap structure for an SEO strategy. Requirements:

1. Create a maximum of {max_categories} top-level categories.
2. The sitemap should have a maximum depth of {max_depth} levels.
3. Group related keywords together under appropriate categories.
4. Higher volume keywords should generally be higher in the structure.
5. IMPORTANT: Every single provided keyword MUST be included in the sitemap.

Return the sitemap as JSON where each key is a keyword and its value is either:
- An empty object {{}} for leaf nodes
- Another object for nodes with children

Return ONLY the JSON, no explanations."""
        }
    ]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=4000
        )

        response_content = completion.choices[0].message.content
        sitemap = json.loads(response_content)

        return sitemap, None

    except Exception as e:
        return None, str(e)


def print_sitemap(sitemap, keyword_to_volume, level=0, prefix=""):
    """Print sitemap tree to console."""
    items = list(sitemap.items())
    for i, (key, value) in enumerate(items):
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        volume = keyword_to_volume.get(key, "")
        volume_str = f" ({volume:,})" if isinstance(volume, (int, float)) else ""
        print(f"{prefix}{connector}{key}{volume_str}")

        if isinstance(value, dict) and value:
            extension = "    " if is_last else "│   "
            print_sitemap(value, keyword_to_volume, level + 1, prefix + extension)


def sitemap_to_df(sitemap, parent="", level=0, rows=None):
    """Convert sitemap to DataFrame rows."""
    if rows is None:
        rows = []

    for key, value in sitemap.items():
        rows.append({
            'Level': level,
            'Parent': parent,
            'Keyword': key
        })
        if isinstance(value, dict):
            sitemap_to_df(value, key, level + 1, rows)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description='Use GPT to generate hierarchical sitemap structures from keywords'
    )
    parser.add_argument('--input', required=True,
                        help='Input CSV with keywords and volumes')
    parser.add_argument('--output', default='sitemap_structure.csv',
                        help='Output CSV path (default: sitemap_structure.csv)')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--keyword-col', default='keyword',
                        help='Keyword column name (default: keyword)')
    parser.add_argument('--volume-col', default='volume',
                        help='Volume column name (default: volume)')
    parser.add_argument('--max-categories', type=int, default=8,
                        help='Max top-level categories (default: 8)')
    parser.add_argument('--max-depth', type=int, default=3,
                        help='Max nesting depth (default: 3)')
    parser.add_argument('--model', default='gpt-4o-mini',
                        choices=['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo'],
                        help='OpenAI model (default: gpt-4o-mini)')
    parser.add_argument('--json-output', help='Also save JSON structure to this path')

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Error: API key required. Use --api-key or set OPENAI_API_KEY environment variable")
        sys.exit(1)

    # Load keywords
    print(f"Loading keywords from: {args.input}")
    df = pd.read_csv(args.input)

    # Find columns
    keyword_col = None
    volume_col = None
    for col in df.columns:
        if col.lower() == args.keyword_col.lower():
            keyword_col = col
        if col.lower() == args.volume_col.lower():
            volume_col = col

    if not keyword_col:
        keyword_col = df.columns[0]
    if not volume_col and len(df.columns) > 1:
        volume_col = df.columns[1]

    print(f"  Using columns: keyword={keyword_col}, volume={volume_col}")

    # Build keyword dict
    keywords_data = {}
    for _, row in df.iterrows():
        kw = str(row[keyword_col]).strip()
        vol = row[volume_col] if volume_col else 0
        if kw and pd.notna(vol):
            keywords_data[kw] = int(vol)

    print(f"  Prepared {len(keywords_data)} keywords")

    # Generate sitemap
    print(f"\nGenerating sitemap with {args.model}...")
    sitemap, error = generate_sitemap_with_llm(
        keywords_data,
        api_key,
        args.max_categories,
        args.max_depth,
        args.model
    )

    if error:
        print(f"Error: {error}")
        sys.exit(1)

    # Validate
    generated_kws = set(kw.lower() for kw in flatten_sitemap(sitemap))
    input_kws = set(kw.lower() for kw in keywords_data.keys())

    missing = input_kws - generated_kws
    if missing:
        print(f"Warning: {len(missing)} keywords missing from sitemap")

    # Convert to DataFrame
    rows = sitemap_to_df(sitemap)
    df_sitemap = pd.DataFrame(rows)
    df_sitemap['Volume'] = df_sitemap['Keyword'].apply(
        lambda x: keywords_data.get(x, '')
    )

    # Save CSV
    df_sitemap.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\nCSV saved to: {args.output}")

    # Save JSON if requested
    if args.json_output:
        with open(args.json_output, 'w', encoding='utf-8') as f:
            json.dump(sitemap, f, indent=2)
        print(f"JSON saved to: {args.json_output}")

    # Print tree
    print("\nSitemap Structure:")
    print("Home")
    print_sitemap(sitemap, keywords_data)

    # Summary
    all_kws = flatten_sitemap(sitemap)
    print(f"\nSummary:")
    print(f"  Total keywords: {len(all_kws)}")
    print(f"  Top-level categories: {len(sitemap)}")
    total_vol = sum(keywords_data.get(kw, 0) for kw in all_kws)
    print(f"  Total volume: {total_vol:,}")


if __name__ == '__main__':
    main()
