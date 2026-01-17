#!/usr/bin/env python3
"""
Page Intent Classifier - CLI Version

Use OpenAI to classify page intent and expected user actions.

Usage:
    python page_intent_classifier_cli.py --api-key YOUR_KEY --urls urls.txt

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import json
import sys

try:
    import html2text
    from openai import OpenAI
except ImportError:
    print("Please install required packages:")
    print("  pip install openai html2text")
    sys.exit(1)


SYSTEM_PROMPT = """You are a web page intent analyzer. Analyze the provided web page content to determine:
1. The primary PURPOSE of the page (what the page is designed to achieve)
2. The expected USER ACTION (what the user is supposed to do)

Respond ONLY with valid JSON in this exact format:
{
  "intent": "brief description of page purpose (6 words or fewer)",
  "action": "expected user action (3 words or fewer)",
  "category": "one of: signup, purchase, browse, learn, contact, compare, download, other"
}"""


def fetch_page_content(url, selector, max_chars, timeout):
    """Fetch and extract content from a URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        if selector:
            content_el = soup.select_one(selector)
            if content_el:
                html_content = str(content_el)
            else:
                html_content = str(soup.body) if soup.body else str(soup)
        else:
            html_content = str(soup.body) if soup.body else str(soup)

        text_maker = html2text.HTML2Text()
        text_maker.ignore_links = True
        text_maker.ignore_images = True
        text_maker.bypass_tables = True

        text = text_maker.handle(html_content)

        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        return text, None

    except Exception as e:
        return None, str(e)


def classify_intent(client, model, content):
    """Use OpenAI to classify page intent."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ],
            temperature=0.3,
            max_tokens=150,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return result, None

    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(
        description='Use OpenAI to classify page intent'
    )
    parser.add_argument('--api-key', required=True, help='OpenAI API key')
    parser.add_argument('--urls', required=True, help='File with URLs (one per line)')
    parser.add_argument('--output', default='page_intent.csv',
                        help='Output CSV path (default: page_intent.csv)')
    parser.add_argument('--model', default='gpt-4o-mini',
                        help='OpenAI model (default: gpt-4o-mini)')
    parser.add_argument('--selector', default='',
                        help='CSS selector for content (optional)')
    parser.add_argument('--max-chars', type=int, default=3000,
                        help='Max characters per page (default: 3000)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between requests (default: 1.0)')
    parser.add_argument('--timeout', type=int, default=15,
                        help='Request timeout (default: 15)')

    args = parser.parse_args()

    # Initialize client
    client = OpenAI(api_key=args.api_key)

    # Load URLs
    with open(args.urls, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(urls)} URLs")
    print(f"Using model: {args.model}")

    results = []

    print(f"\nClassifying pages...")
    for i, url in enumerate(urls):
        print(f"  [{i + 1}/{len(urls)}] {url[:50]}...")

        # Fetch content
        content, fetch_error = fetch_page_content(url, args.selector, args.max_chars, args.timeout)

        if fetch_error:
            results.append({
                'url': url,
                'intent': None,
                'action': None,
                'category': None,
                'error': fetch_error
            })
            continue

        # Classify
        classification, classify_error = classify_intent(client, args.model, content)

        if classify_error:
            results.append({
                'url': url,
                'intent': None,
                'action': None,
                'category': None,
                'error': classify_error
            })
        else:
            results.append({
                'url': url,
                'intent': classification.get('intent', ''),
                'action': classification.get('action', ''),
                'category': classification.get('category', ''),
                'error': None
            })
            print(f"    -> {classification.get('category')}: {classification.get('intent')}")

        time.sleep(args.delay)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False, encoding='utf-8-sig')

    print(f"\nResults saved to: {args.output}")
    print(f"  Pages processed: {len(df)}")
    print(f"  Successfully classified: {df['intent'].notna().sum()}")
    print(f"  Errors: {df['error'].notna().sum()}")

    # Show category distribution
    if df['category'].notna().any():
        print(f"\nCategory distribution:")
        for cat, count in df['category'].value_counts().items():
            print(f"  {cat}: {count}")


if __name__ == '__main__':
    main()
