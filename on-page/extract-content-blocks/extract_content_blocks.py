####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://leefoot.co.uk                                               #
# Contact  : https://leefoot.co.uk/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                                              #
####################################################################################

import requests
from bs4 import BeautifulSoup
import time
import csv
import json
import os
import uuid
from datetime import datetime
from anthropic import Anthropic
import pandas as pd
import glob

# DEBUG MODE: Set to True to process only 2 URLs
DEBUG_MODE = False

# INCREMENTAL SAVE: Save every N rows
SAVE_EVERY_N_ROWS = 50

# Paths
INPUT_FILE = r"C:\python_scripts\zb_extract_content_blocks\input\urls.txt"
OUTPUT_DIR = r"C:\python_scripts\zb_extract_content_blocks\output"

# Your API key
api_key = "YOUR CLAUDE KEY HERE"
client = Anthropic(api_key=api_key)


def fetch_webpage(url):
    """Fetch webpage with 1 second delay"""
    print(f"Fetching {url}...")
    time.sleep(1)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def filter_html(html_content):
    """Remove scripts, styles, header, footer, nav to reduce tokens"""
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove unnecessary tags
    for tag in soup(['script', 'style', 'noscript', 'meta', 'link', 'header', 'footer', 'nav']):
        tag.decompose()

    cleaned = str(soup)
    print(f"Reduced HTML from {len(html_content)} to {len(cleaned)} chars")
    return cleaned


def extract_blocks(html_content):
    """Call Claude Haiku to extract content blocks with XPath"""

    prompt = f"""Analyze this HTML and identify major content blocks/sections.

For each block provide:
- name: descriptive name
- xpath: robust XPath expression to select this element
- notes: brief description

Focus on main content areas (hero sections, feature blocks, carousels). Skip small utility elements.

Return ONLY a JSON array with this exact format:
[
  {{"name": "Hero Section", "xpath": "//div[@class='hero']", "notes": "Main hero banner"}},
  {{"name": "Features", "xpath": "//section[@class='features']", "notes": "Feature grid"}}
]

HTML:
{html_content}
"""

    print("Calling Claude Haiku API...")

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4000,
        system=[
            {
                "type": "text",
                "text": "You are an expert web scraper. Extract content blocks and provide XPath selectors. Return only valid JSON.",
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            }
        ]
    )

    # Show usage stats
    usage = response.usage
    print(
        f"\nTokens - Input: {usage.input_tokens}, Cache creation: {getattr(usage, 'cache_creation_input_tokens', 0)}, Cache read: {getattr(usage, 'cache_read_input_tokens', 0)}, Output: {usage.output_tokens}")

    content = response.content[0].text

    # Extract JSON
    try:
        start = content.find('[')
        end = content.rfind(']') + 1
        if start != -1 and end > start:
            return json.loads(content[start:end])
        else:
            print("No JSON found in response")
            print("Response:", content)
            return []
    except json.JSONDecodeError as e:
        print(f"JSON error: {e}")
        print("Response:", content)
        return []


def save_csv_batch(blocks, output_dir):
    """Save batch of blocks to CSV with unique filename"""
    if not blocks:
        return None

    # Generate unique filename with timestamp and UUID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"content_blocks_{timestamp}_{unique_id}.csv"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['url', 'name', 'xpath', 'notes'])
        writer.writeheader()
        writer.writerows(blocks)

    print(f"  → Saved {len(blocks)} blocks to {filename}")
    return filepath


def read_urls(filepath):
    """Read URLs from file, one per line"""
    with open(filepath, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    return urls


def combine_and_process_csvs(output_dir):
    """
    Combine all content_blocks_*.csv files, add frequency count,
    standardize names by XPath, and save final processed file
    """
    print("\n" + "=" * 80)
    print("POST-PROCESSING: Combining and analyzing results...")
    print("=" * 80)

    # Find all content_blocks CSV files
    pattern = os.path.join(output_dir, "content_blocks_*.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        print("No CSV files found to combine.")
        return

    print(f"Found {len(csv_files)} CSV file(s) to combine:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")

    # Read and combine all CSV files
    print("\nCombining CSV files...")
    dfs = []
    for filepath in csv_files:
        df = pd.read_csv(filepath)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined {len(combined_df)} total rows")

    # Count the frequency of each xpath
    print("\nCounting XPath frequencies...")
    xpath_counts = combined_df['xpath'].value_counts().to_dict()
    combined_df['frequency'] = combined_df['xpath'].map(xpath_counts)

    # Standardize names: for each unique xpath, use the first name that appears
    print("Standardizing names for each XPath group...")
    xpath_to_first_name = combined_df.groupby('xpath')['name'].first().to_dict()
    combined_df['name'] = combined_df['xpath'].map(xpath_to_first_name)

    # Sort by frequency (highest first)
    print("Sorting by frequency...")
    combined_df = combined_df.sort_values('frequency', ascending=False)

    # Save the final processed file
    output_file = os.path.join(output_dir, "combined_output_with_frequency.csv")
    combined_df.to_csv(output_file, index=False)

    print(f"\n✓ Saved processed file: {output_file}")
    print(f"  Total rows: {len(combined_df)}")
    print(f"  Columns: {list(combined_df.columns)}")

    # Show top XPaths by frequency
    print(f"\nTop 10 XPaths by frequency:")
    print("=" * 80)
    for xpath, count in combined_df['xpath'].value_counts().head(10).items():
        # Get the standardized name for this xpath
        name = xpath_to_first_name[xpath]
        print(f"  {count:3d}x - {name}")
        print(f"        {xpath[:100]}{'...' if len(xpath) > 100 else ''}")

    print("\n" + "=" * 80)
    print("POST-PROCESSING COMPLETE")
    print("=" * 80)


def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read URLs
    print(f"Reading URLs from {INPUT_FILE}")
    urls = read_urls(INPUT_FILE)

    # Limit URLs in debug mode
    if DEBUG_MODE:
        urls = urls[:2]
        print(f"\n*** DEBUG MODE: Processing only {len(urls)} URLs ***\n")

    print(f"Processing {len(urls)} URLs...")
    print(f"Incremental save: every {SAVE_EVERY_N_ROWS} rows\n")

    all_blocks = []
    unsaved_blocks = []
    saved_files = []

    for i, url in enumerate(urls, 1):
        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(urls)}] Processing: {url}")
        print('=' * 80)

        try:
            # Fetch and process
            html = fetch_webpage(url)
            filtered = filter_html(html)

            # Extract with Claude
            blocks = extract_blocks(filtered)

            # Add URL to each block
            for block in blocks:
                block['url'] = url

            all_blocks.extend(blocks)
            unsaved_blocks.extend(blocks)
            print(f"Extracted {len(blocks)} blocks from {url}")

            # Incremental save if threshold reached
            if len(unsaved_blocks) >= SAVE_EVERY_N_ROWS:
                print(f"\nIncremental save ({len(unsaved_blocks)} rows)...")
                filepath = save_csv_batch(unsaved_blocks, OUTPUT_DIR)
                if filepath:
                    saved_files.append(filepath)
                unsaved_blocks = []

        except Exception as e:
            print(f"ERROR processing {url}: {e}")
            continue

    # Save any remaining unsaved blocks
    if unsaved_blocks:
        print(f"\nFinal save ({len(unsaved_blocks)} rows)...")
        filepath = save_csv_batch(unsaved_blocks, OUTPUT_DIR)
        if filepath:
            saved_files.append(filepath)

    # Print summary
    print("\n" + "=" * 80)
    print(f"EXTRACTION COMPLETE: Processed {len(urls)} URLs, extracted {len(all_blocks)} total blocks")
    print(f"Saved to {len(saved_files)} file(s) in {OUTPUT_DIR}")
    print("=" * 80)

    # NEW: Combine all CSVs and add frequency analysis
    combine_and_process_csvs(OUTPUT_DIR)


if __name__ == "__main__":
    main()
