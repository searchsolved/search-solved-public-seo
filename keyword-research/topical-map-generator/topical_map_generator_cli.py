#!/usr/bin/env python3
"""
Topical Map Generator - CLI Version

Use AI to organize keywords into hierarchical topical maps.

Usage:
    python topical_map_generator_cli.py --input keywords.csv --api-key YOUR_KEY

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd
import json
import sys
import os

try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI library not installed. Install with: pip install openai")
    sys.exit(1)


def create_topical_map(keywords, api_key, model, depth, levels):
    client = OpenAI(api_key=api_key)

    level_desc = "\n".join([f"{i+1}. {levels[i]}" for i in range(depth)])

    prompt_content = f"""Create a detailed topical map from the following keywords: {keywords}.
The topical map should organize keywords into a hierarchical structure with {depth} levels:
{level_desc}

Group related keywords together logically. Each keyword should appear in only one place.
Return JSON format with:
{{
    "topical_map": [
        {{
            "{levels[0]}": "Topic Name",
            "subtopics": [
                {{
                    "{levels[1]}": "Subtopic Name",
                    "keywords": ["keyword1", "keyword2"]
                }}
            ]
        }}
    ]
}}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an SEO expert that organizes keywords into topical maps. Output JSON only."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.7
        )
        content = response.choices[0].message.content
        return json.loads(content), None
    except Exception as e:
        return None, str(e)


def flatten_topical_map(data, levels, parent_path=None):
    if parent_path is None:
        parent_path = {}
    rows = []

    if isinstance(data, dict):
        if 'topical_map' in data:
            for item in data['topical_map']:
                rows.extend(flatten_topical_map(item, levels, parent_path.copy()))
        else:
            current_path = parent_path.copy()
            for level in levels:
                if level in data:
                    current_path[level] = data[level]
            if 'keywords' in data:
                for kw in data['keywords']:
                    row = current_path.copy()
                    row['Keyword'] = kw
                    rows.append(row)
            if 'subtopics' in data:
                for subtopic in data['subtopics']:
                    rows.extend(flatten_topical_map(subtopic, levels, current_path.copy()))
    elif isinstance(data, list):
        for item in data:
            rows.extend(flatten_topical_map(item, levels, parent_path.copy()))

    return rows


def main():
    parser = argparse.ArgumentParser(description='Generate topical maps using AI')
    parser.add_argument('--input', required=True, help='Input CSV with keywords')
    parser.add_argument('--output', default='topical_map.csv', help='Output CSV path')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--keyword-col', default='keyword', help='Keyword column name')
    parser.add_argument('--model', default='gpt-4o-mini', choices=['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo'], help='OpenAI model')
    parser.add_argument('--depth', type=int, default=4, help='Hierarchy depth (2-5)')
    parser.add_argument('--json-output', help='Optional: save raw JSON output')

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Error: API key required. Use --api-key or set OPENAI_API_KEY")
        sys.exit(1)

    print(f"Loading keywords from: {args.input}")

    if args.input.endswith('.txt'):
        with open(args.input, 'r') as f:
            keywords = [line.strip() for line in f if line.strip()]
    else:
        df = pd.read_csv(args.input)
        keyword_col = None
        for col in df.columns:
            if col.lower() == args.keyword_col.lower():
                keyword_col = col
                break
        if not keyword_col:
            keyword_col = df.columns[0]
        keywords = df[keyword_col].dropna().astype(str).tolist()

    print(f"  Found {len(keywords)} keywords")

    if len(keywords) > 200:
        print(f"  Warning: Large keyword list. Limiting to 200 keywords.")
        keywords = keywords[:200]

    levels = ['Parent Topic', 'Niche Topic 1', 'Niche Topic 2', 'Niche Topic 3', 'Niche Topic 4'][:args.depth]

    print(f"\nGenerating topical map with {args.model}...")
    result, error = create_topical_map(keywords, api_key, args.model, args.depth, levels)

    if error:
        print(f"Error: {error}")
        sys.exit(1)

    # Save JSON if requested
    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"JSON saved to: {args.json_output}")

    # Flatten and save CSV
    rows = flatten_topical_map(result, levels)
    if rows:
        df_result = pd.DataFrame(rows)
        cols = levels + ['Keyword']
        cols = [c for c in cols if c in df_result.columns]
        df_result = df_result[cols]
        df_result.to_csv(args.output, index=False, encoding='utf-8-sig')

        print(f"\nResults saved to: {args.output}")
        print(f"  Total keywords mapped: {len(df_result)}")
        print(f"  Unique {levels[0]}s: {df_result[levels[0]].nunique() if levels[0] in df_result.columns else 'N/A'}")
    else:
        print("Could not parse topical map. Check JSON output for raw results.")
        sys.exit(1)


if __name__ == '__main__':
    main()
