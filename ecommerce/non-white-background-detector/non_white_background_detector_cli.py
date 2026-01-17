#!/usr/bin/env python3
"""
Non-White Background Detector - CLI Version

Detect product images that don't have white backgrounds.

Usage:
    python non_white_background_detector_cli.py --input images.csv --output results.csv

Author: Lee Foot
Website: https://leefoot.com
"""

import argparse
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys


def get_corner_pixels(img, margin=5):
    width, height = img.size
    if img.mode != 'RGB':
        img = img.convert('RGB')

    corners = {
        'Top-Left': img.getpixel((margin, margin)),
        'Top-Right': img.getpixel((width - margin - 1, margin)),
        'Bottom-Left': img.getpixel((margin, height - margin - 1)),
        'Bottom-Right': img.getpixel((width - margin - 1, height - margin - 1))
    }
    return corners


def is_pixel_white(pixel, threshold):
    if isinstance(pixel, int):
        return pixel >= threshold
    return all(channel >= threshold for channel in pixel[:3])


def analyze_image(url, threshold, corners_to_check, margin, require_all, timeout):
    result = {
        'url': url,
        'has_nonwhite_bg': False,
        'nonwhite_corners': [],
        'image_size': None,
        'error': None
    }

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        result['image_size'] = f"{img.width}x{img.height}"

        corners = get_corner_pixels(img, margin)
        nonwhite_corners = []

        for corner_name in corners_to_check:
            pixel = corners.get(corner_name)
            if pixel and not is_pixel_white(pixel, threshold):
                nonwhite_corners.append(corner_name)

        result['nonwhite_corners'] = nonwhite_corners

        if require_all:
            result['has_nonwhite_bg'] = len(nonwhite_corners) == len(corners_to_check)
        else:
            result['has_nonwhite_bg'] = len(nonwhite_corners) > 0

    except requests.exceptions.Timeout:
        result['error'] = "Timeout"
    except requests.exceptions.RequestException as e:
        result['error'] = f"Request error: {str(e)[:50]}"
    except Exception as e:
        result['error'] = f"Error: {str(e)[:50]}"

    return result


def main():
    parser = argparse.ArgumentParser(description='Detect non-white backgrounds in images')
    parser.add_argument('--input', required=True, help='Input CSV with image URLs')
    parser.add_argument('--output', default='background_detection.csv', help='Output CSV path')
    parser.add_argument('--url-col', default='image_url', help='Image URL column name')
    parser.add_argument('--threshold', type=int, default=245, help='Whiteness threshold (0-255)')
    parser.add_argument('--margin', type=int, default=5, help='Corner margin in pixels')
    parser.add_argument('--require-all', action='store_true', help='Require ALL corners non-white')
    parser.add_argument('--workers', type=int, default=5, help='Parallel downloads')
    parser.add_argument('--timeout', type=int, default=10, help='Request timeout (seconds)')

    args = parser.parse_args()

    print(f"Loading image URLs from: {args.input}")
    df = pd.read_csv(args.input, dtype=str)

    # Find URL column
    url_col = None
    for col in df.columns:
        if col.lower() == args.url_col.lower() or 'image' in col.lower() or 'url' in col.lower():
            url_col = col
            break
    if not url_col:
        url_col = df.columns[0]

    urls = df[url_col].dropna().tolist()
    urls = [u for u in urls if isinstance(u, str) and u.startswith('http')]
    print(f"  Found {len(urls)} valid image URLs")

    if not urls:
        print("Error: No valid URLs found")
        sys.exit(1)

    corners_to_check = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
    results = []

    print(f"\nAnalyzing images (threshold={args.threshold})...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(analyze_image, url, args.threshold, corners_to_check, args.margin, args.require_all, args.timeout): url
            for url in urls
        }
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if completed % 20 == 0:
                print(f"  Processed {completed}/{len(urls)} images...")

    # Create results DataFrame
    df_results = pd.DataFrame([
        {
            'Image URL': r['url'],
            'Has Non-White BG': r['has_nonwhite_bg'],
            'Non-White Corners': ', '.join(r['nonwhite_corners']) if r['nonwhite_corners'] else '',
            'Image Size': r['image_size'] or '',
            'Error': r['error'] or ''
        }
        for r in results
    ])

    df_results.to_csv(args.output, index=False, encoding='utf-8-sig')

    nonwhite_count = sum(1 for r in results if r['has_nonwhite_bg'])
    white_count = sum(1 for r in results if not r['has_nonwhite_bg'] and not r['error'])
    error_count = sum(1 for r in results if r['error'])

    print(f"\nResults saved to: {args.output}")
    print(f"  Total images: {len(results)}")
    print(f"  Non-white background: {nonwhite_count}")
    print(f"  White background: {white_count}")
    print(f"  Errors: {error_count}")


if __name__ == '__main__':
    main()
