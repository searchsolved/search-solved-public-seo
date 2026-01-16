####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://leefoot.com                                               #
# Contact  : https://leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                                              #
####################################################################################

"""Brand Stockists SERP Scraper using ValueSERP API"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


# Constants
VALUESERP_API_KEY = "your-key-here"  # Replace with your actual ValueSERP API key

# Google Domain Examples:
# UK: google.co.uk with gl=gb, hl=en
# USA: google.com with gl=us, hl=en
# Canada: google.ca with gl=ca, hl=en
# Australia: google.com.au with gl=au, hl=en

GOOGLE_DOMAIN = "google.co.uk"  # Change to google.com for USA results
COUNTRY_CODE = "gb"             # Change to "us" for USA results
LANGUAGE = "en"
STOCKISTS_SUFFIX = " Stockists"
TARGET_POSITION = 1
HOMEPAGE_DEPTH = 2
OUTPUT_FILENAME = "brand_links_output.csv"
BRANDS_FILENAME = "brands.txt"


def read_file(filepath: Path) -> str:
    """Read content from a file."""
    try:
        return filepath.read_text(encoding='utf-8').strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Required file not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error reading file {filepath}: {e}")


def get_api_key() -> str:
    """Return the ValueSERP API key."""
    return VALUESERP_API_KEY


def load_brands(base_path: Path) -> List[str]:
    """Load brand names from file and return as list."""
    content = read_file(base_path / BRANDS_FILENAME)
    return [line.strip() for line in content.splitlines() if line.strip()]


def create_search_queries(brands: List[str]) -> List[str]:
    """Create search queries by appending 'Stockists' to each brand."""
    return [f"{brand}{STOCKISTS_SUFFIX}" for brand in brands]


def make_serp_request(query: str, api_key: str) -> Dict:
    """Make a single SERP API request using ValueSERP."""
    params = {
        'api_key': api_key,
        'q': query,
        'google_domain': GOOGLE_DOMAIN,
        'gl': COUNTRY_CODE,
        'hl': LANGUAGE
    }
    
    try:
        api_result = requests.get('https://api.valueserp.com/search', params)
        api_result.raise_for_status()
        return api_result.json()
    except requests.RequestException as e:
        print(f"API request failed for query '{query}': {e}")
        return {}


def extract_organic_result(result: Dict, query: str, position: int) -> Optional[Dict[str, str]]:
    """Extract data from a single organic search result."""
    try:
        return {
            'query': query,
            'url': result.get('link', 'MISSING'),
            'title': result.get('title', 'MISSING'),
            'description': result.get('snippet', 'MISSING'),
            'position': position
        }
    except Exception as e:
        print(f"Error extracting result data: {e}")
        return None


def process_search_results(search_data: Dict, query: str) -> List[Dict[str, str]]:
    """Process search results and extract organic data."""
    results = []
    organic_results = search_data.get('organic_results', [])
    
    for position, result in enumerate(organic_results, 1):
        extracted_result = extract_organic_result(result, query, position)
        if extracted_result:
            results.append(extracted_result)
    
    return results


def calculate_url_depth(url: str) -> int:
    """Calculate the depth of a URL by counting forward slashes."""
    if url == 'MISSING':
        return 0
    
    # Normalize URL by removing trailing slashes
    normalized_url = url.rstrip('/')
    return normalized_url.count('/')


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and filter the dataframe according to business rules."""
    # Filter to position 1 results only
    df = df[df['position'] == TARGET_POSITION].copy()
    
    # Remove rows with missing data
    df = df[~df['url'].isin(['MISSING'])]
    df = df[~df['description'].isin(['MISSING'])]
    df = df[~df['title'].isin(['MISSING'])]
    
    # Calculate URL depth and remove homepages
    df['url_depth'] = df['url'].apply(calculate_url_depth)
    df = df[df['url_depth'] != HOMEPAGE_DEPTH]
    
    # Select final columns and remove duplicates
    final_columns = ['query', 'url', 'title', 'description']
    df = df[final_columns].copy()
    df.drop_duplicates(subset=['url'], keep='first', inplace=True)
    
    return df


def scrape_brand_stockists(base_path: Optional[Path] = None) -> pd.DataFrame:
    """Main function to scrape brand stockists data."""
    if base_path is None:
        base_path = Path.cwd()
    
    print(f"Working directory: {base_path}")
    print(f"Using Google domain: {GOOGLE_DOMAIN} (Country: {COUNTRY_CODE.upper()})")
    
    # Load configuration
    api_key = get_api_key()
    brands = load_brands(base_path)
    search_queries = create_search_queries(brands)
    
    print(f"Processing {len(search_queries)} brand queries...")
    
    # Collect all results
    all_results = []
    
    for i, query in enumerate(search_queries, 1):
        print(f"Searching: {query.strip()} ({i} of {len(search_queries)})")
        
        search_data = make_serp_request(query, api_key)
        if search_data:
            results = process_search_results(search_data, query)
            all_results.extend(results)
        
        # Be respectful to the API
        time.sleep(1)
    
    # Create and clean dataframe
    if not all_results:
        print("No results found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    df_cleaned = clean_dataframe(df)
    
    return df_cleaned


def save_results(df: pd.DataFrame, output_path: Path) -> None:
    """Save results to CSV file with UTF-8-BOM encoding."""
    try:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Results saved to: {output_path}")
        print(f"Total records: {len(df)}")
        print(f"Encoding: UTF-8 with BOM (utf-8-sig)")
    except Exception as e:
        print(f"Error saving results: {e}")


def main() -> None:
    """Main execution function."""
    start_time = time.time()
    
    try:
        # Run the scraping process
        results_df = scrape_brand_stockists()
        
        if not results_df.empty:
            output_path = Path.cwd() / OUTPUT_FILENAME
            save_results(results_df, output_path)
        else:
            print("No valid results to save.")
            
    except Exception as e:
        print(f"Script failed: {e}")
        return
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
