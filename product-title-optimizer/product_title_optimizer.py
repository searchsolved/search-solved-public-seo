####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://leefoot.com                                               #
# Contact  : https://leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                                              #
####################################################################################

"""Product Title Optimizer

LLM-powered title restructuring with data integrity checks.
Optimizes product titles to follow consistent word order while preserving all information.

Usage:
    python product_title_optimizer.py --input products.csv --output optimized.csv
    python product_title_optimizer.py --input products.csv --api-key YOUR_KEY --model gpt-4o
    python product_title_optimizer.py --input products.csv --base-url http://localhost:1234/v1  # Local LLM
"""

import argparse
import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm
import re
import string
import tiktoken
import time
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('title_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_client(api_key, base_url):
    """Initialize OpenAI client."""
    if base_url:
        return OpenAI(base_url=base_url, api_key=api_key or "local")
    elif api_key:
        return OpenAI(api_key=api_key)
    else:
        # Try environment variable
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            return OpenAI(api_key=api_key)
        raise ValueError("No API key provided. Use --api-key or set OPENAI_API_KEY environment variable")


def num_tokens_from_string(text, model="gpt-4"):
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        return len(text.split()) * 1.3  # Rough estimate


def clean_json_string(json_str):
    """Clean JSON string of invalid characters."""
    json_str = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_str)
    json_str = ''.join(filter(lambda x: x in string.printable, json_str))
    return json_str


def extract_json(text):
    """Extract JSON object from text."""
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return clean_json_string(json_match.group())
    return None


def create_title_template(client, model, titles, category, max_tokens=4000):
    """Create a consistent template for optimizing titles in a category."""
    prompt = f"""Analyze the following product titles in the {category} category and create a consistent template for optimizing them. The template MUST:
1. Maintain consistency across similar products while allowing for variation in specific details.
2. Ensure that ALL important information from the original title is retained, especially:
   - All numbers and measurements
   - All product codes or model numbers
   - All specific features or materials mentioned
3. Improve readability and clarity.
4. Use UK English spelling.
5. Follow this adapted Basic English Order of Words rule for product titles:
   Brand + Product Type + Model/Series + Key Features + Specifications (including ALL numbers) + Additional Info

Example template:
{{
  "template": "{{Brand}} {{Product Type}} - {{Model/Series}} - {{Key Features}} - {{Specifications}} - {{Additional Info}}",
  "instructions": "Fill in each field with the appropriate information from the original title. ALL numbers, measurements, and specific details MUST be included. If a field is not applicable, omit it and remove the extra dash. Ensure no important information is lost in the optimization process."
}}

Product titles to analyze:
{json.dumps(titles[:50])}
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a product title template creator that always responds with a valid JSON object."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )

        json_str = extract_json(response.choices[0].message.content)
        if json_str:
            return json.loads(json_str)
        else:
            raise ValueError("No JSON object found in the response")
    except Exception as e:
        logger.error(f"Error creating title template for category {category}: {e}")
        return {"template": "", "instructions": ""}


def optimize_titles_batch(client, model, titles, category, template, max_tokens=4000):
    """Optimize a batch of titles using the template."""
    prompt = f"""Optimize the following product titles using the provided template and instructions. You MUST ensure consistency across all titles within the {category} category while retaining ALL important information from the original titles. Output the result in JSON format.

Template: {template['template']}
Instructions: {template['instructions']}

Additional MANDATORY requirements:
1. DO NOT omit any important information from the original title, especially:
   - All numbers and measurements
   - All product codes or model numbers
   - All specific features or materials mentioned
2. Improve readability and clarity without losing any details.
3. Use UK English spelling.
4. Maintain consistent structure across similar products.
5. Follow this adapted Basic English Order of Words rule for product titles:
   Brand (if present) + Product Type + Model/Series (if present) + Key Features + Specifications (including ALL numbers) + Additional Info
6. IMPORTANT: DO NOT add any information that is not present in the original title. If a brand, model number, or any other detail is missing, do not invent or add it.

Original titles:
{json.dumps(titles)}

Output format:
{{
  "optimised_titles": ["optimized title 1", "optimized title 2", ...]
}}

IMPORTANT: Ensure that ALL numbers, measurements, and specific details from the original title are present in the optimized title. Do not add any information that is not in the original title.
    """

    backoff_time = 1
    max_retries = 5

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a product title optimizer that always responds with a valid JSON object."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )

            json_str = extract_json(response.choices[0].message.content)
            if json_str:
                result = json.loads(json_str)
                return result.get('optimised_titles', titles)
            else:
                raise ValueError("No JSON object found in the response")
        except Exception as e:
            if "rate limit" in str(e).lower():
                logger.warning(f"Rate limit exceeded. Backing off for {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 2
            else:
                logger.error(f"Error processing titles: {e}")
                if attempt == max_retries - 1:
                    return titles  # Return originals on final failure

    return titles


def verify_numerical_data(original, optimized):
    """Verify all numbers from original appear in optimized."""
    if not isinstance(original, str) or not isinstance(optimized, str):
        return str(original)

    original_numbers = set(re.findall(r'\d+(?:\.\d+)?', original))
    optimized_numbers = set(re.findall(r'\d+(?:\.\d+)?', optimized))

    if original_numbers != optimized_numbers:
        logger.warning(f"Numerical data mismatch.\nOriginal: {original}\nOptimized: {optimized}")
        return original
    return optimized


def find_missing_words(original, optimized):
    """Find words in original that are missing from optimized."""
    if not isinstance(original, str) or not isinstance(optimized, str):
        return []

    original_words = set(original.lower().split())
    optimized_words = set(optimized.lower().split())

    return list(original_words - optimized_words)


def title_corresponds(original, optimized, threshold=0.8):
    """Check if optimized title corresponds to original (80%+ word overlap)."""
    if not isinstance(original, str) or not isinstance(optimized, str):
        return False

    original_words = set(original.lower().split())
    optimized_words = set(optimized.lower().split())

    common_words = original_words.intersection(optimized_words)
    correspondence_ratio = len(common_words) / len(original_words) if original_words else 0

    return correspondence_ratio >= threshold


def process_category(client, model, category, titles, template_cache, batch_size=20):
    """Process all titles in a category."""
    if category not in template_cache:
        template_cache[category] = create_title_template(client, model, titles, category)

    template = template_cache[category]
    all_results = []

    # Process in batches
    for i in range(0, len(titles), batch_size):
        batch = titles[i:i + batch_size]
        optimized_batch = optimize_titles_batch(client, model, batch, category, template)

        # Ensure we have the right number of results
        if len(optimized_batch) < len(batch):
            optimized_batch.extend(batch[len(optimized_batch):])

        for original, optimized in zip(batch, optimized_batch):
            optimized = verify_numerical_data(original, optimized)

            if not title_corresponds(original, optimized):
                logger.warning(f"Title correspondence failed, keeping original")
                optimized = original

            missing = find_missing_words(original, optimized)
            is_same = original.lower() == optimized.lower()

            all_results.append({
                'original': original,
                'optimized': optimized,
                'is_same': is_same,
                'missing_words': ', '.join(missing) if missing else ''
            })

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Product Title Optimizer - LLM-powered title restructuring"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input CSV file with product titles"
    )
    parser.add_argument(
        "--output", "-o",
        default="optimized_titles.csv",
        help="Output CSV file (default: optimized_titles.csv)"
    )
    parser.add_argument(
        "--title-col",
        default="Name",
        help="Column name for product titles (default: Name)"
    )
    parser.add_argument(
        "--category-col",
        default="Categories",
        help="Column name for categories (default: Categories)"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--base-url",
        help="Custom API base URL (for local LLMs like LM Studio)"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of titles to process per API call (default: 20)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of products to process (0 = no limit)"
    )

    args = parser.parse_args()

    try:
        # Initialize client
        client = get_client(args.api_key, args.base_url)
        logger.info(f"Using model: {args.model}")
        if args.base_url:
            logger.info(f"Using custom endpoint: {args.base_url}")

        # Load data
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} products from {args.input}")

        # Check required columns
        if args.title_col not in df.columns:
            raise ValueError(f"Title column '{args.title_col}' not found. Available: {list(df.columns)}")
        if args.category_col not in df.columns:
            raise ValueError(f"Category column '{args.category_col}' not found. Available: {list(df.columns)}")

        # Apply limit if specified
        if args.limit > 0:
            df = df.head(args.limit)
            logger.info(f"Limited to {len(df)} products")

        # Process by category
        template_cache = {}
        all_results = []

        for category, group in tqdm(df.groupby(args.category_col), desc="Processing categories"):
            titles = group[args.title_col].tolist()
            results = process_category(client, args.model, category, titles, template_cache, args.batch_size)

            for idx, result in zip(group.index, results):
                all_results.append({
                    'index': idx,
                    **result
                })

        # Merge results back
        results_df = pd.DataFrame(all_results).set_index('index')
        df['Optimized Title'] = results_df['optimized']
        df['Is Same'] = results_df['is_same']
        df['Missing Words'] = results_df['missing_words']

        # Save
        df.to_csv(args.output, index=False, encoding='utf-8-sig')
        logger.info(f"Saved results to: {args.output}")

        # Statistics
        changed = len(df[~df['Is Same']])
        logger.info(f"Titles changed: {changed} / {len(df)} ({100*changed/len(df):.1f}%)")

        with_missing = len(df[df['Missing Words'] != ''])
        if with_missing > 0:
            logger.warning(f"Titles with potentially missing words: {with_missing}")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
