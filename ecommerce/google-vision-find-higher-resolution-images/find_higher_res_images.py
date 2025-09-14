####################################################################################
# Website  : https://leefoot.co.uk/                                                #
# Contact  : https://leefoot.co.uk/hire-me/                                        #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Twitter  : https://twitter.com/LeeFootSEO                                        #
####################################################################################

# Standard libraries
import io
import json
import logging
import os
import hashlib
import signal
import sys
import atexit
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from urllib.parse import urlparse

# External libraries
import pandas as pd
import requests
from requests.exceptions import RequestException, Timeout
from PIL import Image, UnidentifiedImageError
from fake_useragent import UserAgent
from google.cloud import vision
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
GOOGLE_APPLICATION_CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_PATH',
                                                "/python_scripts/cloud_vision_api.json")
INPUT_FILE_PATH = '/python_scripts/google_vision/images_all.csv'
OUTPUT_FILE_PATH = "/python_scripts/google_vision/higher_resolution_images_all.xlsx"
REQUEST_TIMEOUT = 20


# Configuration for match types
@dataclass
class Config:
    MIN_RESOLUTION: Tuple[int, int] = (1000, 1000)
    MIN_IMPROVEMENT_RATIO: float = 1.2
    MAX_IMAGES: int = 5
    REQUEST_TIMEOUT: int = 20
    MAX_WORKERS: int = 10
    INCLUDE_PARTIAL_MATCHES: bool = True
    INCLUDE_VISUALLY_SIMILAR: bool = True
    INCLUDE_PAGE_MATCHES: bool = True
    SKIPPED_FILE_TYPES: List[str] = None
    SAME_DOMAIN_DELAY: float = 0.5  # Delay in seconds between requests to the same domain

    def __post_init__(self):
        if self.SKIPPED_FILE_TYPES is None:
            self.SKIPPED_FILE_TYPES = ['.svg']


config = Config()


class MatchType(Enum):
    FULL = "full_match"
    PARTIAL = "partial_match"
    SIMILAR = "visually_similar"
    PAGE = "page_match"


# Initialize a session for persistent requests
session = requests.Session()

# Global variables
processed_hashes = set()
processed_results = {}
all_results_global = {match_type: [] for match_type in MatchType}
interrupted = False

# Domain tracking for rate limiting
domain_last_access = {}
domain_access_lock = None

stats = {
    'skipped_format': 0,
    'skipped_high_res': 0,
    'skipped_duplicate': 0,
    'processed': 0,
    'full_matches': 0,
    'partial_matches': 0,
    'similar_matches': 0,
    'page_matches': 0,
    'total_urls': 0,
    'completed_urls': 0
}


# ====== Signal Handling for Clean Exit ====== #

def signal_handler(sig, frame):
    """Handle interrupt signals (Ctrl+C) and save results"""
    global interrupted
    interrupted = True
    logging.info("\n" + "=" * 60)
    logging.info("INTERRUPT RECEIVED - Saving results...")
    logging.info("=" * 60)
    save_and_exit()


def save_and_exit():
    """Save current results and exit"""
    global all_results_global
    try:
        if any(len(results) > 0 for results in all_results_global.values()):
            create_excel_output(all_results_global, OUTPUT_FILE_PATH)
            logging.info(f"Results saved to: {OUTPUT_FILE_PATH}")
        else:
            logging.info("No results to save yet")

        log_final_statistics()
    except Exception as e:
        logging.error(f"Error saving results: {e}")
    finally:
        sys.exit(0)


def log_final_statistics():
    """Log final statistics"""
    global all_results_global
    logging.info("=" * 60)
    logging.info("STATISTICS")
    logging.info("=" * 60)
    logging.info(f"Total URLs: {stats.get('total_urls', 0)}")
    logging.info(f"Processed: {stats.get('completed_urls', 0)}")
    logging.info(f"Remaining: {stats.get('total_urls', 0) - stats.get('completed_urls', 0)}")
    logging.info(f"Matches found:")
    logging.info(f"  - Full: {len(all_results_global[MatchType.FULL])}")
    logging.info(f"  - Partial: {len(all_results_global[MatchType.PARTIAL])}")
    logging.info(f"  - Similar: {len(all_results_global[MatchType.SIMILAR])}")
    logging.info(f"  - From pages: {len(all_results_global[MatchType.PAGE])}")
    logging.info("=" * 60)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
if sys.platform == "win32":
    signal.signal(signal.SIGBREAK, signal_handler)


# ====== Helper Functions ====== #

def initialize_client():
    """Initializes the Google Vision client"""
    logging.info("Initializing Google Vision client...")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS_PATH

    # Initialize domain access lock for thread safety
    global domain_access_lock
    import threading
    domain_access_lock = threading.Lock()

    return vision.ImageAnnotatorClient(), vision.Image()


def apply_domain_rate_limit(url):
    """Apply rate limiting for same-domain requests"""
    global domain_last_access, domain_access_lock

    domain = urlparse(url).netloc
    if not domain:
        return

    with domain_access_lock:
        current_time = time.time()

        if domain in domain_last_access:
            time_since_last = current_time - domain_last_access[domain]
            if time_since_last < config.SAME_DOMAIN_DELAY:
                sleep_time = config.SAME_DOMAIN_DELAY - time_since_last
                logging.debug(f"Rate limiting: Waiting {sleep_time:.2f}s before accessing {domain}")
                time.sleep(sleep_time)

        domain_last_access[domain] = time.time()


def get_image_hash(image_data):
    """Generate a hash for the given image data"""
    if image_data:
        return hashlib.md5(image_data).hexdigest()
    return None


def is_image_processed(image_hash):
    """Check if an image hash is already processed"""
    return image_hash in processed_hashes


def mark_image_processed(image_hash, data, is_duplicate=False):
    """Mark an image hash as processed"""
    if not is_duplicate:
        processed_hashes.add(image_hash)
    processed_results[image_hash] = data


def check_file_type(url):
    """Check and skip unwanted file types"""
    url_lower = url.lower()
    if any(url_lower.endswith(ext) for ext in config.SKIPPED_FILE_TYPES):
        logging.info(f"Skipping URL due to file type: {url}")
        stats['skipped_format'] += 1
        return True
    return False


def fetch_image_with_requests(url, user_agent):
    """Fetch the image using requests library with rate limiting"""
    # Apply rate limiting for the domain
    apply_domain_rate_limit(url)

    logging.debug(f"Fetching image with requests for URL {url}")
    headers = {'User-Agent': user_agent.random}
    response = session.get(url, timeout=REQUEST_TIMEOUT, headers=headers, stream=True)
    return response


def fetch_image_dimensions(url, user_agent):
    """Fetch the dimensions of the image with rate limiting"""
    try:
        response = fetch_image_with_requests(url, user_agent)

        if response.status_code == 403:
            logging.warning(f"Received 403 error for URL {url}")
            return (0, 0), None

        logging.info(f"Fetching image with requests for URL {url}")

        try:
            im = Image.open(response.raw)
        except Exception as e:
            logging.debug(f"Direct open failed, trying with full content read: {e}")
            content = response.content
            im = Image.open(io.BytesIO(content))

        if im.format == 'GIF':
            im.seek(0)
            logging.debug(f"Processing GIF image, using first frame")

        dimensions = im.size

        img_byte_arr = io.BytesIO()
        save_format = im.format if im.format else 'PNG'

        if save_format == 'GIF':
            if im.mode == 'P':
                im = im.convert('RGB')
            im.save(img_byte_arr, format='PNG')
        elif save_format in ['JPEG', 'JPG'] and im.mode in ('RGBA', 'LA', 'P'):
            rgb_im = Image.new('RGB', im.size, (255, 255, 255))
            rgb_im.paste(im, mask=im.split()[-1] if len(im.split()) > 3 else None)
            rgb_im.save(img_byte_arr, format='JPEG')
        else:
            im.save(img_byte_arr, format=save_format if save_format != 'JPG' else 'JPEG')

        image_bytes = img_byte_arr.getvalue()

        logging.debug(f"Fetched image dimensions for URL {url}: {dimensions}")
        return dimensions, image_bytes

    except (RequestException, UnidentifiedImageError, Timeout) as e:
        logging.error(f"Exception in fetch_image_dimensions for URL {url}: {e}")
        return (0, 0), None


# ====== Enhanced Matching Functions ====== #

def fetch_matching_images_enhanced(url: str, client, image) -> Dict[MatchType, List[Dict]]:
    """Fetch ALL types of matching images from Google Vision API"""
    try:
        logging.info(f"Fetching all match types for URL: {url}")

        # Apply rate limiting even for Google Vision API calls to the same domain
        apply_domain_rate_limit(url)

        image.source.image_uri = url

        web_response = client.web_detection(image=image)
        web_content = web_response.web_detection
        json_string = type(web_content).to_json(web_content)
        data = json.loads(json_string)

        matches = {
            MatchType.FULL: data.get('fullMatchingImages', []),
            MatchType.PARTIAL: data.get('partialMatchingImages', []) if config.INCLUDE_PARTIAL_MATCHES else [],
            MatchType.SIMILAR: data.get('visuallySimilarImages', []) if config.INCLUDE_VISUALLY_SIMILAR else [],
            MatchType.PAGE: data.get('pagesWithMatchingImages', []) if config.INCLUDE_PAGE_MATCHES else []
        }

        for match_type, items in matches.items():
            if items:
                logging.info(f"Found {len(items)} {match_type.value} matches for {url}")

        return matches
    except Exception as e:
        logging.error(f"Error fetching matching images for URL {url}: {e}")
        return {match_type: [] for match_type in MatchType}


def calculate_resolution_score(orig_dims: Tuple[int, int],
                               new_dims: Tuple[int, int]) -> float:
    """Calculate improvement score for resolution"""
    if not orig_dims or not new_dims or 0 in orig_dims or 0 in new_dims:
        return 0

    orig_pixels = orig_dims[0] * orig_dims[1]
    new_pixels = new_dims[0] * new_dims[1]

    if new_pixels <= orig_pixels:
        return 0

    pixel_ratio = new_pixels / orig_pixels

    orig_aspect = orig_dims[0] / orig_dims[1]
    new_aspect = new_dims[0] / new_dims[1]
    aspect_similarity = 1 - abs(orig_aspect - new_aspect) / max(orig_aspect, new_aspect)

    return pixel_ratio * (0.7 + 0.3 * aspect_similarity)


def extract_images_from_page(page_url: str, user_agent) -> List[str]:
    """Extract image URLs from a web page with rate limiting"""
    image_urls = []
    try:
        # Apply rate limiting for page fetching
        apply_domain_rate_limit(page_url)

        headers = {'User-Agent': user_agent.random}
        response = session.get(page_url, timeout=10, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            for img in soup.find_all('img'):
                img_url = img.get('src', '')
                if img_url:
                    img_url = urljoin(page_url, img_url)
                    if img_url.startswith('http'):
                        image_urls.append(img_url)

            for link in soup.find_all('a', href=True):
                href = link['href']
                if any(href.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                    img_url = urljoin(page_url, href)
                    if img_url.startswith('http'):
                        image_urls.append(img_url)

            image_urls = list(dict.fromkeys(image_urls))[:20]

    except Exception as e:
        logging.debug(f"Could not extract images from page {page_url}: {e}")

    return image_urls


def process_single_match(match_url: str, original_url: str,
                         original_dims: Tuple[int, int],
                         match_type: MatchType,
                         user_agent,
                         source_page_url: str = None) -> Optional[Tuple]:
    """Process a single matching image URL"""
    try:
        dims, img_bytes = fetch_image_dimensions(match_url, user_agent)

        if dims == (0, 0) or not img_bytes:
            return None

        score = calculate_resolution_score(original_dims, dims)

        if score > config.MIN_IMPROVEMENT_RATIO:
            result = (
                original_url,
                match_url,
                dims[0],
                dims[1],
                original_dims[0],
                original_dims[1],
                dims[0] - original_dims[0],
                dims[1] - original_dims[1],
                round(score, 2),
                source_page_url if source_page_url else '',
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )

            logging.info(f"Found improved {match_type.value}: {dims} vs original {original_dims}, "
                         f"score: {score:.2f}")
            return result
    except Exception as e:
        logging.error(f"Error processing match URL {match_url}: {e}")

    return None


def process_all_match_types(url: str, all_matches: Dict[MatchType, List],
                            original_dims: Tuple[int, int],
                            user_agent) -> Dict[MatchType, List[Tuple]]:
    """Process all types of matches"""
    local_results = {match_type: [] for match_type in MatchType}
    seen_urls = set()

    for match_type, matches in all_matches.items():
        for match in matches:
            if match_type == MatchType.PAGE:
                page_url = match.get('url', '')
                page_images = extract_images_from_page(page_url, user_agent)

                for img_url in page_images:
                    if img_url not in seen_urls:
                        result = process_single_match(
                            img_url, url, original_dims,
                            match_type, user_agent, page_url
                        )
                        if result:
                            local_results[match_type].append(result)
                            seen_urls.add(img_url)
            else:
                match_url = match.get('url', '')
                if match_url and match_url not in seen_urls:
                    result = process_single_match(
                        match_url, url, original_dims,
                        match_type, user_agent
                    )
                    if result:
                        local_results[match_type].append(result)
                        seen_urls.add(match_url)

    for match_type in local_results:
        local_results[match_type].sort(key=lambda x: x[8], reverse=True)
        local_results[match_type] = local_results[match_type][:config.MAX_IMAGES]

    return local_results


def fetch_and_process_image_enhanced(url: str, client, image, user_agent):
    """Enhanced processing with all match types"""
    try:
        if check_file_type(url):
            return {match_type: [] for match_type in MatchType}

        original_dims, original_bytes = fetch_image_dimensions(url, user_agent)

        if original_dims == (0, 0):
            return {match_type: [] for match_type in MatchType}

        if original_dims[0] >= config.MIN_RESOLUTION[0] and original_dims[1] >= config.MIN_RESOLUTION[1]:
            logging.info(f"Image {url} is already high resolution {original_dims}, checking for even higher...")

        original_hash = get_image_hash(original_bytes)
        if is_image_processed(original_hash):
            logging.info(f"Image {url} already processed, skipping...")
            stats['skipped_duplicate'] += 1
            return {match_type: [] for match_type in MatchType}

        all_matches = fetch_matching_images_enhanced(url, client, image)
        local_results = process_all_match_types(url, all_matches, original_dims, user_agent)

        mark_image_processed(original_hash, local_results)

        total_results = sum(len(results) for results in local_results.values())
        if total_results > 0:
            stats['processed'] += 1
            logging.info(f"Found {total_results} higher resolution versions for {url}")
        else:
            logging.info(f"No higher resolution versions found for {url}")

        return local_results

    except Exception as e:
        logging.error(f"Unexpected error processing URL {url}: {e}")
        return {match_type: [] for match_type in MatchType}


def process_images_enhanced(df, client, image, user_agent):
    """Process images concurrently - FIXED VERSION"""
    global interrupted, all_results_global

    logging.info(f"Processing {len(df['Address'])} images...")
    logging.info(f"Same-domain delay: {config.SAME_DOMAIN_DELAY}s")
    stats['total_urls'] = len(df['Address'])

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        future_to_url = {
            executor.submit(fetch_and_process_image_enhanced, url, client, image, user_agent): url
            for url in df['Address']
        }

        completed_count = 0
        for future in as_completed(future_to_url):
            if interrupted:
                logging.info("Interrupt detected, canceling remaining tasks...")
                for f in future_to_url:
                    if not f.done():
                        f.cancel()
                break

            url = future_to_url[future]
            try:
                # Get results from this URL
                local_results = future.result()

                # Add to global results
                for match_type, results in local_results.items():
                    all_results_global[match_type].extend(results)

                completed_count += 1
                stats['completed_urls'] = completed_count

                # Log progress
                total_found = sum(len(r) for r in all_results_global.values())
                logging.info(
                    f"Processed {completed_count}/{stats['total_urls']} images - Found {total_found} matches total")

            except Exception as e:
                logging.error(f"Error processing URL {url}: {e}")
                completed_count += 1

    return all_results_global


def create_excel_output(results_by_type: Dict[MatchType, List], output_path: str):
    """Create Excel file with multiple worksheets"""
    logging.info("Creating Excel output...")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = []
        for match_type in MatchType:
            results = results_by_type.get(match_type, [])
            summary_data.append({
                'Match Type': match_type.value,
                'Total Matches': len(results),
                'Unique Sources': len(set(r[0] for r in results)) if results else 0,
                'Avg Score': round(sum(r[8] for r in results) / len(results), 2) if results else 0
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Individual match type sheets
        columns = [
            'original_url', 'matching_img_url',
            'width_matching', 'height_matching',
            'width_source', 'height_source',
            'width_diff', 'height_diff',
            'resolution_score', 'source_page_url', 'timestamp'
        ]

        for match_type in MatchType:
            results = results_by_type.get(match_type, [])

            if results:
                df = pd.DataFrame(results, columns=columns)
                df['pixel_increase_%'] = round(
                    ((df['width_matching'] * df['height_matching']) -
                     (df['width_source'] * df['height_source'])) /
                    (df['width_source'] * df['height_source']) * 100, 1
                )
                df = df.sort_values('resolution_score', ascending=False)
                sheet_name = match_type.value.replace('_', ' ').title()
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                df = pd.DataFrame(columns=columns)
                sheet_name = match_type.value.replace('_', ' ').title()
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    logging.info(f"Excel file saved to {output_path}")


# ====== Main Function ====== #

def main():
    """Main function"""
    global interrupted, all_results_global

    logging.info("=" * 60)
    logging.info("STARTING IMAGE RESOLUTION FINDER")
    logging.info("Press Ctrl+C to stop and save results")
    logging.info(f"Rate limiting: {config.SAME_DOMAIN_DELAY}s delay between same-domain requests")
    logging.info("=" * 60)

    try:
        client, image = initialize_client()
        user_agent = UserAgent()

        df = pd.read_csv(INPUT_FILE_PATH)
        logging.info(f"Loaded {len(df)} URLs from {INPUT_FILE_PATH}")

        # Process images - results are now stored in all_results_global
        process_images_enhanced(df, client, image, user_agent)

        # Save results
        if not interrupted and any(len(results) > 0 for results in all_results_global.values()):
            create_excel_output(all_results_global, OUTPUT_FILE_PATH)
            logging.info("=" * 60)
            logging.info("PROCESSING COMPLETE")
            log_final_statistics()
            logging.info(f"Output saved to: {OUTPUT_FILE_PATH}")
        elif not interrupted:
            logging.info("No higher resolution images found")

    except KeyboardInterrupt:
        signal_handler(None, None)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        save_and_exit()


if __name__ == "__main__":
    main()
