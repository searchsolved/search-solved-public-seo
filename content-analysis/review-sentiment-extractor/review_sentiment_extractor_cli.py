#!/usr/bin/env python3
"""
Review Sentiment Extractor - CLI Version

Use OpenAI to extract positive and negative sentiments from reviews.

Usage:
    python review_sentiment_extractor_cli.py --api-key YOUR_KEY --input reviews.csv

Author: Lee Foot
Website: https://www.leefoot.com
"""

import argparse
import pandas as pd
import time
import json
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)


def get_system_prompt(positive=True, negative=True, context=""):
    """Generate system prompt based on extraction options."""
    context_hint = f"\n\nContext: These reviews are for {context}." if context else ""

    if positive and negative:
        return f"""You are a review analyst. For each review, extract:
1. POSITIVE aspects (praise, what customers liked)
2. NEGATIVE aspects (complaints, pain points)

If a review has no positive aspects, use "N/A" for positive.
If a review has no negative aspects, use "N/A" for negative.
Keep summaries concise (1-2 sentences each).{context_hint}

Respond ONLY with valid JSON in this format:
{{
  "reviews": [
    {{"id": "1", "positive": "summary", "negative": "summary", "sentiment": "positive|negative|mixed|neutral"}}
  ]
}}"""
    elif positive:
        return f"""You are a review analyst focused on POSITIVE aspects only.
For each review, extract what customers liked, praised, or found valuable.
If a review has no positive aspects, use "N/A".{context_hint}

Respond ONLY with valid JSON:
{{"reviews": [{{"id": "1", "positive": "summary"}}]}}"""
    else:
        return f"""You are a review analyst focused on NEGATIVE aspects only.
For each review, extract complaints, pain points, and issues.
If a review has no negative aspects, use "N/A".{context_hint}

Respond ONLY with valid JSON:
{{"reviews": [{{"id": "1", "negative": "summary"}}]}}"""


def main():
    parser = argparse.ArgumentParser(
        description='Use OpenAI to extract sentiments from reviews'
    )
    parser.add_argument('--api-key', required=True, help='OpenAI API key')
    parser.add_argument('--input', required=True, help='Input CSV with reviews')
    parser.add_argument('--output', default='review_sentiments.csv',
                        help='Output CSV path (default: review_sentiments.csv)')
    parser.add_argument('--review-column', required=True, help='Name of review text column')
    parser.add_argument('--id-column', default=None,
                        help='Name of ID column (optional, auto-generates if not specified)')
    parser.add_argument('--model', default='gpt-4o-mini',
                        help='OpenAI model (default: gpt-4o-mini)')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Reviews per API call (default: 5)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between batches (default: 1.0)')
    parser.add_argument('--positive-only', action='store_true',
                        help='Extract only positive sentiments')
    parser.add_argument('--negative-only', action='store_true',
                        help='Extract only negative sentiments')
    parser.add_argument('--context', default='',
                        help='Product/service context (e.g., "curtains")')
    parser.add_argument('--max-reviews', type=int, default=None,
                        help='Maximum reviews to process')

    args = parser.parse_args()

    # Determine extraction mode
    extract_positive = not args.negative_only
    extract_negative = not args.positive_only

    # Initialize client
    client = OpenAI(api_key=args.api_key)

    # Load data
    print(f"Loading: {args.input}")
    try:
        df = pd.read_csv(args.input, encoding='utf-8')
    except:
        df = pd.read_csv(args.input, encoding='latin-1')

    if args.max_reviews:
        df = df.head(args.max_reviews)

    print(f"  Loaded {len(df):,} reviews")
    print(f"  Model: {args.model}")
    print(f"  Extracting: {'positive ' if extract_positive else ''}{'negative' if extract_negative else ''}")

    # Prepare ID column
    if args.id_column and args.id_column in df.columns:
        id_col = args.id_column
    else:
        df['_id'] = range(1, len(df) + 1)
        id_col = '_id'

    system_prompt = get_system_prompt(extract_positive, extract_negative, args.context)
    results = []
    total_batches = (len(df) + args.batch_size - 1) // args.batch_size

    print(f"\nProcessing {total_batches} batches...")

    for batch_idx in range(total_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(df))

        print(f"  Batch {batch_idx + 1}/{total_batches}...")

        batch_df = df.iloc[start_idx:end_idx]
        reviews_data = []
        for _, row in batch_df.iterrows():
            reviews_data.append({
                "id": str(row[id_col]),
                "review": str(row[args.review_column])[:1000]
            })

        user_content = json.dumps({"reviews": reviews_data})

        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            batch_results = json.loads(response.choices[0].message.content)

            for item in batch_results.get("reviews", []):
                result = {"id": item.get("id")}
                if extract_positive:
                    result["positive"] = item.get("positive", "N/A")
                if extract_negative:
                    result["negative"] = item.get("negative", "N/A")
                if extract_positive and extract_negative:
                    result["sentiment"] = item.get("sentiment", "unknown")
                results.append(result)

        except Exception as e:
            print(f"    Error: {str(e)}")
            for _, row in batch_df.iterrows():
                results.append({"id": str(row[id_col]), "error": str(e)})

        time.sleep(args.delay)

    # Merge results with original data
    df_results = pd.DataFrame(results)
    df[id_col] = df[id_col].astype(str)
    df_results['id'] = df_results['id'].astype(str)
    df_final = pd.merge(df, df_results, left_on=id_col, right_on='id', how='left')

    # Save results
    df_final.to_csv(args.output, index=False, encoding='utf-8-sig')

    print(f"\nResults saved to: {args.output}")
    print(f"  Reviews processed: {len(df_final)}")

    if extract_positive and extract_negative and 'sentiment' in df_final.columns:
        print(f"\nSentiment distribution:")
        for sent, count in df_final['sentiment'].value_counts().items():
            print(f"  {sent}: {count}")


if __name__ == '__main__':
    main()
