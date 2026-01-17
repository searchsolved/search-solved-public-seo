#!/usr/bin/env python3
"""
Micro-Moments Classifier - CLI Version

Classify keywords into Google's 4 micro-moments using OpenAI.

Usage:
    python micro_moments_classifier_cli.py --input keywords.csv --api-key YOUR_KEY

Author: Lee Foot
Website: https://www.leefoot.com
"""

import argparse
import pandas as pd
import json
import os
import sys
from openai import OpenAI


def classify_keywords_batch(keywords, client, model, include_confidence):
    """Classify a batch of keywords using OpenAI."""
    keyword_list = "\n".join([f"- {kw}" for kw in keywords])

    confidence_instruction = ""
    if include_confidence:
        confidence_instruction = "Also provide a confidence score from 1-5 for each classification."

    messages = [
        {
            "role": "system",
            "content": """You are an SEO expert that classifies search queries into Google's 4 micro-moments:

1. I-want-to-BUY - Transactional intent, user wants to purchase something
2. I-want-to-KNOW - Informational intent, user wants to learn something
3. I-want-to-DO - Instructional intent, user wants to accomplish a task
4. I-want-to-GO - Navigational/local intent, user wants to find a specific place or website

Return ONLY valid JSON, no explanations."""
        },
        {
            "role": "user",
            "content": f"""Classify each of these keywords into one of the 4 micro-moments:

{keyword_list}

{confidence_instruction}

Return JSON in this exact format:
{{
  "classifications": [
    {{"keyword": "keyword1", "micro_moment": "I-want-to-BUY", "confidence": 5}},
    {{"keyword": "keyword2", "micro_moment": "I-want-to-KNOW", "confidence": 4}}
  ]
}}

Include ALL keywords in your response."""
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
        result = json.loads(response_content)

        return result.get('classifications', []), None

    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(
        description='Classify keywords into Google\'s 4 micro-moments using OpenAI'
    )
    parser.add_argument('--input', required=True,
                        help='Input CSV with keywords')
    parser.add_argument('--output', default='micro_moments_classified.csv',
                        help='Output CSV path (default: micro_moments_classified.csv)')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--keyword-col', default='keyword',
                        help='Keyword column name (default: keyword)')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Keywords per API call (default: 50)')
    parser.add_argument('--model', default='gpt-4o-mini',
                        choices=['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo'],
                        help='OpenAI model (default: gpt-4o-mini)')
    parser.add_argument('--no-confidence', action='store_true',
                        help='Skip confidence scores')

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Error: API key required. Use --api-key or set OPENAI_API_KEY environment variable")
        sys.exit(1)

    # Load keywords
    print(f"Loading keywords from: {args.input}")
    df = pd.read_csv(args.input)

    # Find keyword column
    keyword_col = None
    for col in df.columns:
        if col.lower() == args.keyword_col.lower():
            keyword_col = col
            break
    if not keyword_col:
        keyword_col = df.columns[0]

    print(f"  Using column: {keyword_col}")

    keywords = df[keyword_col].dropna().astype(str).tolist()
    print(f"  Found {len(keywords)} keywords")

    # Initialize client
    client = OpenAI(api_key=api_key)

    # Process in batches
    all_results = []
    num_batches = (len(keywords) + args.batch_size - 1) // args.batch_size

    print(f"\nClassifying keywords with {args.model}...")

    for i in range(0, len(keywords), args.batch_size):
        batch = keywords[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1

        print(f"  Batch {batch_num}/{num_batches} ({len(batch)} keywords)...")

        results, error = classify_keywords_batch(
            batch, client, args.model, not args.no_confidence
        )

        if error:
            print(f"    Error: {error}")
        elif results:
            all_results.extend(results)

    if all_results:
        df_results = pd.DataFrame(all_results)

        # Check for missing keywords
        classified_kws = set(df_results['keyword'].str.lower())
        missing_kws = [kw for kw in keywords if kw.lower() not in classified_kws]

        if missing_kws:
            print(f"\nWarning: {len(missing_kws)} keywords couldn't be classified")
            for kw in missing_kws:
                df_results = pd.concat([df_results, pd.DataFrame([{
                    'keyword': kw,
                    'micro_moment': 'Unclassified',
                    'confidence': 0
                }])], ignore_index=True)

        # Save
        df_results.to_csv(args.output, index=False, encoding='utf-8-sig')

        print(f"\nResults saved to: {args.output}")
        print(f"  Total classified: {len(df_results)}")

        # Summary
        print(f"\nMicro-Moment Distribution:")
        for moment, count in df_results['micro_moment'].value_counts().items():
            pct = count / len(df_results) * 100
            print(f"  {moment}: {count} ({pct:.1f}%)")

        # Examples
        print(f"\nExample classifications:")
        for moment in df_results['micro_moment'].unique():
            example = df_results[df_results['micro_moment'] == moment].iloc[0]
            print(f"  [{moment}] {example['keyword']}")

    else:
        print("No results to save")
        sys.exit(1)


if __name__ == '__main__':
    main()
