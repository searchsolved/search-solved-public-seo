#!/usr/bin/env python3
"""
Product Q&A Extractor - CLI Version

Extract product reviews and Q&A from e-commerce pages.

Usage:
    python product_qa_extractor_cli.py --urls urls.txt

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
import sys


def extract_number(text):
    """Extract first number from text."""
    if not text:
        return None
    match = re.search(r'[\d.]+', text.replace(',', ''))
    if match:
        try:
            return float(match.group())
        except:
            return None
    return None


def extract_product_data(url, rating_selector, review_count_selector,
                         review_selector, review_star_selector, review_text_selector,
                         qa_selector, question_selector, answer_selector,
                         user_agent, timeout):
    """Extract reviews and Q&A from a product page."""
    try:
        headers = {'User-Agent': user_agent}
        response = requests.get(url, headers=headers, timeout=timeout)
        soup = BeautifulSoup(response.text, 'html.parser')

        data = {
            'url': url,
            'rating': None,
            'review_count': None,
            'reviews': [],
            'qa_items': []
        }

        # Extract overall rating
        rating_el = soup.select_one(rating_selector)
        if rating_el:
            data['rating'] = extract_number(rating_el.get_text(strip=True))
            if data['rating'] is None:
                for attr in ['content', 'value', 'data-rating']:
                    if rating_el.get(attr):
                        data['rating'] = extract_number(rating_el.get(attr))
                        break

        # Extract review count
        count_el = soup.select_one(review_count_selector)
        if count_el:
            data['review_count'] = extract_number(count_el.get_text(strip=True))

        # Extract individual reviews
        reviews = soup.select(review_selector)
        for rev in reviews:
            review_data = {}
            text_el = rev.select_one(review_text_selector)
            if text_el:
                review_data['text'] = text_el.get_text(strip=True)
            star_el = rev.select_one(review_star_selector)
            if star_el:
                review_data['rating'] = extract_number(star_el.get_text(strip=True))
            if review_data:
                data['reviews'].append(review_data)

        # Extract Q&A
        qa_items = soup.select(qa_selector)
        for qa in qa_items:
            q_el = qa.select_one(question_selector)
            a_el = qa.select_one(answer_selector)
            if q_el or a_el:
                data['qa_items'].append({
                    'question': q_el.get_text(strip=True) if q_el else None,
                    'answer': a_el.get_text(strip=True) if a_el else None
                })

        return data, None

    except Exception as e:
        return {'url': url, 'error': str(e)}, str(e)


def main():
    parser = argparse.ArgumentParser(
        description='Extract product reviews and Q&A from e-commerce pages'
    )
    parser.add_argument('--urls', required=True, help='File with URLs (one per line)')
    parser.add_argument('--output', default='product_qa_summary.csv',
                        help='Summary output path (default: product_qa_summary.csv)')
    parser.add_argument('--reviews-output', default='product_reviews.csv',
                        help='Reviews output path (default: product_reviews.csv)')
    parser.add_argument('--qa-output', default='product_qa.csv',
                        help='Q&A output path (default: product_qa.csv)')

    # Selectors
    parser.add_argument('--rating-selector', default='.rating, [itemprop="ratingValue"]',
                        help='CSS selector for overall rating')
    parser.add_argument('--review-count-selector', default='.review-count, [itemprop="reviewCount"]',
                        help='CSS selector for review count')
    parser.add_argument('--review-selector', default='.review, [itemprop="review"]',
                        help='CSS selector for individual reviews')
    parser.add_argument('--review-star-selector', default='.review-stars, .star-rating',
                        help='CSS selector for review stars')
    parser.add_argument('--review-text-selector', default='.review-text, [itemprop="reviewBody"]',
                        help='CSS selector for review text')
    parser.add_argument('--qa-selector', default='.qa-item, .question-answer',
                        help='CSS selector for Q&A items')
    parser.add_argument('--question-selector', default='.question, dt',
                        help='CSS selector for questions')
    parser.add_argument('--answer-selector', default='.answer, dd',
                        help='CSS selector for answers')

    # Request settings
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between requests (default: 1.0)')
    parser.add_argument('--timeout', type=int, default=15,
                        help='Request timeout (default: 15)')
    parser.add_argument('--user-agent', default='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        help='User agent string')

    args = parser.parse_args()

    # Load URLs
    with open(args.urls, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(urls)} URLs")

    all_data = []
    all_reviews = []
    all_qa = []

    print(f"\nExtracting data...")
    for i, url in enumerate(urls):
        if i % 10 == 0:
            print(f"  Processing {i + 1}/{len(urls)}...")

        data, error = extract_product_data(
            url,
            args.rating_selector,
            args.review_count_selector,
            args.review_selector,
            args.review_star_selector,
            args.review_text_selector,
            args.qa_selector,
            args.question_selector,
            args.answer_selector,
            args.user_agent,
            args.timeout
        )

        if 'error' not in data:
            all_data.append({
                'url': data['url'],
                'rating': data['rating'],
                'review_count': data['review_count'],
                'reviews_extracted': len(data.get('reviews', [])),
                'qa_extracted': len(data.get('qa_items', []))
            })

            for rev in data.get('reviews', []):
                all_reviews.append({
                    'url': url,
                    'review_rating': rev.get('rating'),
                    'review_text': rev.get('text')
                })

            for qa in data.get('qa_items', []):
                all_qa.append({
                    'url': url,
                    'question': qa.get('question'),
                    'answer': qa.get('answer')
                })
        else:
            all_data.append({
                'url': url,
                'rating': None,
                'review_count': None,
                'reviews_extracted': 0,
                'qa_extracted': 0,
                'error': data.get('error')
            })

        time.sleep(args.delay)

    # Save results
    df_summary = pd.DataFrame(all_data)
    df_reviews = pd.DataFrame(all_reviews)
    df_qa = pd.DataFrame(all_qa)

    df_summary.to_csv(args.output, index=False, encoding='utf-8-sig')
    if len(df_reviews) > 0:
        df_reviews.to_csv(args.reviews_output, index=False, encoding='utf-8-sig')
    if len(df_qa) > 0:
        df_qa.to_csv(args.qa_output, index=False, encoding='utf-8-sig')

    print(f"\nResults saved:")
    print(f"  Summary: {args.output}")
    print(f"  Reviews: {args.reviews_output} ({len(df_reviews)} reviews)")
    print(f"  Q&A: {args.qa_output} ({len(df_qa)} items)")

    print(f"\nSummary:")
    print(f"  Products processed: {len(all_data)}")
    print(f"  With ratings: {df_summary['rating'].notna().sum()}")
    print(f"  Reviews extracted: {len(df_reviews)}")
    print(f"  Q&A extracted: {len(df_qa)}")


if __name__ == '__main__':
    main()
