####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                                              #
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
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from urllib.parse import urlparse

# Progress bar and colored output
from tqdm import tqdm
from colorama import init, Fore, Style

init(autoreset=True)  # Initialize colorama

# External libraries
import pandas as pd
import requests
from requests.exceptions import RequestException, Timeout
from PIL import Image, UnidentifiedImageError
from fake_useragent import UserAgent
from google.cloud import vision
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Playwright imports (optional fallback)
try:
    from playwright.sync_api import sync_playwright

    PLAYWRIGHT_AVAILABLE = True
    logging.info("Playwright available for fallback")
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logging.warning("Playwright not installed - install with: pip install playwright && playwright install chromium")

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
GOOGLE_APPLICATION_CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_PATH',
                                                "/python_scripts/cloud_vision_api.json")
INPUT_FILE_PATH = '/python_scripts/google_vision/images_all.csv'
OUTPUT_FILE_PATH = "/python_scripts/google_vision/higher_resolution_images_all.xlsx"
OUTPUT_FOLDER = "/python_scripts/google_vision/output"  # Incremental save folder
INCREMENTAL_SAVE_INTERVAL = 100  # Save every N processed images
REQUEST_TIMEOUT = 20


# Configuration for match types
@dataclass
class Config:
    MIN_RESOLUTION: Tuple[int, int] = (1000, 1000)
    MIN_IMPROVEMENT_RATIO: float = 1.2
    MAX_IMAGES: int = 5
    REQUEST_TIMEOUT: int = 30  # Increased timeout for slower, more patient requests
    MAX_WORKERS: int = 2  # Further reduced for more human-like behavior
    INCLUDE_PARTIAL_MATCHES: bool = True
    INCLUDE_VISUALLY_SIMILAR: bool = True
    INCLUDE_PAGE_MATCHES: bool = True
    SKIPPED_FILE_TYPES: List[str] = None
    SAME_DOMAIN_DELAY: float = 3.0  # Base delay between same-domain requests
    SAME_DOMAIN_DELAY_JITTER: float = 2.0  # Random jitter (0 to 2 seconds added)
    GLOBAL_DELAY_MIN: float = 0.5  # Minimum global delay
    GLOBAL_DELAY_MAX: float = 1.5  # Maximum global delay
    USER_AGENT_ROTATION: bool = True  # Rotate user agents
    SIMULATE_HUMAN_BREAKS: bool = True  # Take breaks after batches
    BREAK_AFTER_REQUESTS: int = 20  # Take a break after this many requests
    BREAK_DURATION_MIN: float = 10.0  # Minimum break duration
    BREAK_DURATION_MAX: float = 30.0  # Maximum break duration
    USE_PLAYWRIGHT_FALLBACK: bool = True  # Use Playwright when requests fail
    PLAYWRIGHT_HEADLESS: bool = True  # Run Playwright in headless mode
    PLAYWRIGHT_TIMEOUT: int = 30000  # Playwright timeout in milliseconds
    PLAYWRIGHT_VIEWPORT: Tuple[int, int] = (1920, 1080)  # Browser viewport size
    VERBOSE: bool = True  # Enable verbose output
    SHOW_DELAYS: bool = True  # Show delay countdown

    def __post_init__(self):
        if self.SKIPPED_FILE_TYPES is None:
            self.SKIPPED_FILE_TYPES = ['.svg']


config = Config()


class MatchType(Enum):
    FULL = "full_match"
    PARTIAL = "partial_match"
    SIMILAR = "visually_similar"
    PAGE = "page_match"


# Initialize a session for persistent requests with browser-like configuration
session = requests.Session()
# Browser-like session configuration
session.max_redirects = 10
adapter = requests.adapters.HTTPAdapter(
    pool_connections=2,  # Keep low for more human-like behavior
    pool_maxsize=2,
    max_retries=0,  # We handle retries manually
    pool_block=False
)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Global variables
processed_hashes = set()
processed_results = {}
all_results_global = {match_type: [] for match_type in MatchType}
interrupted = False

# Domain tracking for rate limiting
domain_last_access = {}
domain_access_lock = None
global_last_request_time = 0  # Track last request globally
total_requests_made = 0  # Track total requests for break simulation

# Playwright browser instance (lazy initialization)
playwright_instance = None
browser_instance = None
browser_context = None

# Track which domains need Playwright (learning from failures)
domains_requiring_playwright = set()
domains_requests_success = set()  # Domains where requests work fine

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
    'completed_urls': 0,
    'last_incremental_save': 0  # Track last save point
}


# ====== Signal Handling for Clean Exit ====== #

def print_verbose(message, color=Fore.WHITE, force=False):
    """Print verbose messages with color"""
    if config.VERBOSE or force:
        print(f"{color}{message}{Style.RESET_ALL}")


def print_success(message):
    """Print success messages in green"""
    print_verbose(f"✓ {message}", Fore.GREEN, force=True)


def print_error(message):
    """Print error messages in red"""
    print_verbose(f"✗ {message}", Fore.RED, force=True)


def print_warning(message):
    """Print warning messages in yellow"""
    print_verbose(f"⚠ {message}", Fore.YELLOW)


def print_info(message):
    """Print info messages in cyan"""
    print_verbose(f"ℹ {message}", Fore.CYAN)


def print_action(message):
    """Print action messages in magenta"""
    print_verbose(f"➤ {message}", Fore.MAGENTA)


def show_delay_countdown(seconds, message="Waiting"):
    """Show countdown timer for delays"""
    if config.SHOW_DELAYS and seconds > 1:
        for remaining in tqdm(range(int(seconds)), desc=message, bar_format='{desc}: {bar} {remaining}s'):
            time.sleep(1)
        if seconds % 1 > 0:  # Handle fractional seconds
            time.sleep(seconds % 1)
    else:
        time.sleep(seconds)


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
            # Save final complete file
            create_excel_output(all_results_global, OUTPUT_FILE_PATH)
            logging.info(f"Final results saved to: {OUTPUT_FILE_PATH}")

            # Also save as final incremental
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            final_incremental = os.path.join(OUTPUT_FOLDER, f"FINAL_save_{timestamp}.xlsx")
            create_excel_output(all_results_global, final_incremental)
            logging.info(f"Final incremental save: {final_incremental}")
        else:
            logging.info("No results to save yet")

        log_final_statistics()

        # Log domain learning
        if domains_requiring_playwright:
            logging.info(f"Domains that required Playwright: {', '.join(domains_requiring_playwright)}")
        if domains_requests_success:
            logging.info(f"Domains that worked with requests: {len(domains_requests_success)}")

    except Exception as e:
        logging.error(f"Error saving results: {e}")
    finally:
        cleanup_playwright()
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
    logging.info(f"Last incremental save at: {stats.get('last_incremental_save', 0)} rows")
    logging.info(f"Matches found:")
    logging.info(f"  - Full: {len(all_results_global[MatchType.FULL])}")
    logging.info(f"  - Partial: {len(all_results_global[MatchType.PARTIAL])}")
    logging.info(f"  - Similar: {len(all_results_global[MatchType.SIMILAR])}")
    logging.info(f"  - From pages: {len(all_results_global[MatchType.PAGE])}")

    # List saved files
    if os.path.exists(OUTPUT_FOLDER):
        saved_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith('.xlsx')]
        if saved_files:
            logging.info(f"Incremental saves in {OUTPUT_FOLDER}:")
            for file in sorted(saved_files)[-5:]:  # Show last 5 files
                filepath = os.path.join(OUTPUT_FOLDER, file)
                size_kb = os.path.getsize(filepath) / 1024
                logging.info(f"  - {file} ({size_kb:.1f} KB)")

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

    # Create output folder if it doesn't exist
    create_output_folder()

    return vision.ImageAnnotatorClient(), vision.Image()


def create_output_folder():
    """Create output folder for incremental saves"""
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        logging.info(f"Created output folder: {OUTPUT_FOLDER}")
    else:
        logging.info(f"Output folder exists: {OUTPUT_FOLDER}")


def save_incremental_results(results_by_type: Dict[MatchType, List], save_number: int):
    """Save incremental results with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"incremental_save_{save_number:04d}_rows_{timestamp}.xlsx"
    filepath = os.path.join(OUTPUT_FOLDER, filename)

    try:
        create_excel_output(results_by_type, filepath)
        logging.info(f"Incremental save #{save_number}: {filepath}")
        return True
    except Exception as e:
        logging.error(f"Failed to save incremental results: {e}")
        return False


def initialize_playwright():
    """Initialize Playwright browser instance (lazy initialization)"""
    global playwright_instance, browser_instance, browser_context

    if not PLAYWRIGHT_AVAILABLE or not config.USE_PLAYWRIGHT_FALLBACK:
        return False

    if browser_instance is None:
        try:
            logging.info("Initializing Playwright browser...")
            playwright_instance = sync_playwright().start()

            # Launch Chromium with anti-detection settings
            browser_instance = playwright_instance.chromium.launch(
                headless=config.PLAYWRIGHT_HEADLESS,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-web-security',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--disable-site-isolation-trials',
                    '--disable-setuid-sandbox',
                    '--disable-accelerated-2d-canvas',
                    '--no-zygote',
                    '--single-process',
                    '--deterministic-fetch',
                ]
            )

            # Create context with realistic settings
            browser_context = browser_instance.new_context(
                viewport={'width': config.PLAYWRIGHT_VIEWPORT[0], 'height': config.PLAYWRIGHT_VIEWPORT[1]},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                locale='en-US',
                timezone_id='America/New_York',
                permissions=['geolocation'],
                color_scheme='light',
                reduced_motion='no-preference',
                device_scale_factor=1,
            )

            logging.info("Playwright browser initialized successfully")
            return True

        except Exception as e:
            logging.error(f"Failed to initialize Playwright: {e}")
            return False

    return True


def cleanup_playwright():
    """Clean up Playwright browser instance"""
    global playwright_instance, browser_instance, browser_context

    try:
        if browser_context:
            browser_context.close()
        if browser_instance:
            browser_instance.close()
        if playwright_instance:
            playwright_instance.stop()
    except:
        pass

    browser_context = None
    browser_instance = None
    playwright_instance = None


def apply_domain_rate_limit(url):
    """Apply human-like rate limiting with randomization"""
    global domain_last_access, domain_access_lock, global_last_request_time, total_requests_made

    domain = urlparse(url).netloc
    if not domain:
        return

    with domain_access_lock:
        current_time = time.time()

        # Check if we need a human break
        if config.SIMULATE_HUMAN_BREAKS and total_requests_made > 0 and total_requests_made % config.BREAK_AFTER_REQUESTS == 0:
            break_duration = random.uniform(config.BREAK_DURATION_MIN, config.BREAK_DURATION_MAX)
            print_info(f"Taking a human-like break after {total_requests_made} requests...")
            show_delay_countdown(break_duration, f"Break time ({break_duration:.1f}s)")
            current_time = time.time()

        # Apply global rate limit with randomization
        global_delay = random.uniform(config.GLOBAL_DELAY_MIN, config.GLOBAL_DELAY_MAX)
        time_since_last_global = current_time - global_last_request_time
        if time_since_last_global < global_delay:
            sleep_time = global_delay - time_since_last_global
            print_verbose(f"Global rate limit: {sleep_time:.2f}s delay", Fore.YELLOW)
            show_delay_countdown(sleep_time, "Global cooldown")
            current_time = time.time()

        # Apply domain-specific rate limit with jitter
        if domain in domain_last_access:
            time_since_last = current_time - domain_last_access[domain]
            # Add random jitter to the delay
            domain_delay = config.SAME_DOMAIN_DELAY + random.uniform(0, config.SAME_DOMAIN_DELAY_JITTER)
            if time_since_last < domain_delay:
                sleep_time = domain_delay - time_since_last
                print_verbose(f"Domain rate limit for {domain}: {sleep_time:.2f}s", Fore.YELLOW)
                show_delay_countdown(sleep_time, f"Domain cooldown [{domain}]")
                current_time = time.time()

        # Add small random "thinking time" before each request (100-500ms)
        thinking_time = random.uniform(0.1, 0.5)
        if config.VERBOSE:
            print_verbose(f"Simulating human thinking time: {thinking_time:.2f}s", Fore.BLUE)
        time.sleep(thinking_time)

        domain_last_access[domain] = time.time()
        global_last_request_time = time.time()
        total_requests_made += 1


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


def fetch_image_with_playwright(url):
    """Fetch image using Playwright browser (fallback for blocked requests)"""
    global browser_context

    if not initialize_playwright():
        return (0, 0), None

    domain = urlparse(url).netloc
    page = None

    try:
        print_action(f"Using Playwright browser for {domain}")

        # Apply rate limiting even for Playwright
        apply_domain_rate_limit(url)

        # Create new page with random viewport variation
        width = config.PLAYWRIGHT_VIEWPORT[0] + random.randint(-50, 50)
        height = config.PLAYWRIGHT_VIEWPORT[1] + random.randint(-50, 50)
        page = browser_context.new_page()
        page.set_viewport_size({"width": width, "height": height})

        print_verbose(f"Browser viewport: {width}x{height}", Fore.BLUE)

        # Random mouse movement to appear human
        mouse_x, mouse_y = random.randint(100, 500), random.randint(100, 500)
        page.mouse.move(mouse_x, mouse_y)
        print_verbose(f"Mouse moved to ({mouse_x}, {mouse_y})", Fore.BLUE)

        # Navigate to the image URL
        print_verbose(f"Navigating to {url[:50]}...", Fore.CYAN)
        response = page.goto(url, wait_until='networkidle', timeout=config.PLAYWRIGHT_TIMEOUT)

        if response and response.status in [200, 304]:
            # Wait a bit for image to fully load
            wait_time = random.randint(500, 1500)
            print_verbose(f"Waiting {wait_time}ms for image to load", Fore.BLUE)
            page.wait_for_timeout(wait_time)

            # Try to get image dimensions from the page
            dimensions = page.evaluate('''() => {
                const img = document.querySelector('img');
                if (img && img.complete) {
                    return {width: img.naturalWidth, height: img.naturalHeight};
                }
                return null;
            }''')

            if not dimensions:
                # Fallback: check if the body contains an image
                dimensions = page.evaluate('''() => {
                    if (document.images.length > 0) {
                        const img = document.images[0];
                        return {width: img.naturalWidth, height: img.naturalHeight};
                    }
                    return null;
                }''')

            # Get the image bytes
            img_bytes = page.screenshot(full_page=False)

            if dimensions:
                print_success(f"Playwright successfully fetched {domain}: {dimensions['width']}x{dimensions['height']}")
                return (dimensions['width'], dimensions['height']), img_bytes
            else:
                # Parse dimensions from screenshot
                img = Image.open(io.BytesIO(img_bytes))
                print_success(f"Playwright fetched {domain} (from screenshot): {img.size}")
                return img.size, img_bytes
        else:
            status = response.status if response else 'None'
            print_warning(f"Playwright got status {status} for {domain}")
            return (0, 0), None

    except Exception as e:
        print_error(f"Playwright failed for {domain}: {str(e)[:100]}")
        return (0, 0), None
    finally:
        if page:
            try:
                page.close()
            except:
                pass


def fetch_image_with_requests(url, user_agent, max_retries=3):
    """Fetch the image using requests library with human-like behavior"""
    # Apply rate limiting for the domain
    apply_domain_rate_limit(url)

    domain = urlparse(url).netloc

    for attempt in range(max_retries):
        try:
            print_action(f"Fetching {domain} - Attempt {attempt + 1}/{max_retries}")

            # Build human-like headers
            headers = {
                'User-Agent': user_agent.random if config.USER_AGENT_ROTATION else user_agent.chrome,
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'image',
                'Sec-Fetch-Mode': 'no-cors',
                'Sec-Fetch-Site': 'cross-site',
                'Cache-Control': 'max-age=0',
                'Referer': 'https://www.google.com/'  # Simulate coming from Google
            }

            # Randomly decide whether to include some headers (more human-like)
            if random.random() > 0.5:
                headers['Sec-CH-UA'] = '"Chromium";v="118", "Google Chrome";v="118", "Not=A?Brand";v="99"'
                headers['Sec-CH-UA-Mobile'] = '?0'
                headers['Sec-CH-UA-Platform'] = '"Windows"'

            print_verbose(f"Using User-Agent: {headers['User-Agent'][:50]}...", Fore.BLUE)

            response = session.get(url, timeout=REQUEST_TIMEOUT, headers=headers, stream=True, allow_redirects=True)

            # If successful, return the response
            if response.status_code == 200:
                print_success(f"Successfully fetched from {domain} (200 OK)")
                return response

            # For 403/443 errors, wait longer before retry with different user agent
            if response.status_code in [403, 443] and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5 + random.uniform(0, 3)  # 5-8s, 10-13s, etc.
                print_warning(f"Got {response.status_code} error from {domain}")
                show_delay_countdown(wait_time, f"Retry wait ({response.status_code} error)")
                # Force new user agent on retry
                user_agent = UserAgent()
                continue

            # For other errors, still retry but with shorter wait
            if response.status_code >= 400 and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2 + random.uniform(0, 2)
                print_warning(f"Got {response.status_code} error from {domain}")
                show_delay_countdown(wait_time, f"Retry wait (HTTP {response.status_code})")
                continue

            return response

        except (RequestException, Timeout) as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 3 + random.uniform(0, 2)
                print_error(f"Request failed: {str(e)[:100]}")
                show_delay_countdown(wait_time, "Retry after error")
            else:
                raise

    return response


def fetch_image_dimensions(url, user_agent):
    """Fetch the dimensions of the image with Playwright fallback"""
    global domains_requiring_playwright, domains_requests_success

    domain = urlparse(url).netloc

    # Check if we know this domain needs Playwright
    if domain in domains_requiring_playwright and config.USE_PLAYWRIGHT_FALLBACK:
        logging.debug(f"Domain {domain} known to require Playwright, using it directly")
        return fetch_image_with_playwright(url)

    # First try with requests (unless we know it won't work)
    if domain not in domains_requiring_playwright:
        try:
            response = fetch_image_with_requests(url, user_agent)

            if response.status_code == 403 or response.status_code == 443:
                logging.warning(f"Received {response.status_code} error for URL {url}")

                # Try Playwright as fallback
                if config.USE_PLAYWRIGHT_FALLBACK and PLAYWRIGHT_AVAILABLE:
                    logging.info(f"Attempting Playwright fallback for {url}")
                    dims, img_bytes = fetch_image_with_playwright(url)

                    if dims != (0, 0) and img_bytes:
                        # Mark this domain as requiring Playwright
                        domains_requiring_playwright.add(domain)
                        logging.info(f"Domain {domain} added to Playwright-required list")
                        return dims, img_bytes

                return (0, 0), None

            if response.status_code != 200:
                logging.warning(f"Got status {response.status_code} for {url}")
                # Try Playwright for other error codes too
                if config.USE_PLAYWRIGHT_FALLBACK and PLAYWRIGHT_AVAILABLE:
                    dims, img_bytes = fetch_image_with_playwright(url)
                    if dims != (0, 0):
                        domains_requiring_playwright.add(domain)
                        return dims, img_bytes
                return (0, 0), None

            logging.info(f"Successfully fetched {url} with requests")

            # Process the image
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

            # Mark domain as working with requests
            domains_requests_success.add(domain)

            logging.debug(f"Fetched image dimensions for URL {url}: {dimensions}")
            return dimensions, image_bytes

        except (RequestException, UnidentifiedImageError, Timeout) as e:
            logging.error(f"Exception in fetch_image_dimensions for URL {url}: {e}")

            # Try Playwright as fallback
            if config.USE_PLAYWRIGHT_FALLBACK and PLAYWRIGHT_AVAILABLE:
                logging.info(f"Attempting Playwright fallback after exception for {url}")
                dims, img_bytes = fetch_image_with_playwright(url)

                if dims != (0, 0) and img_bytes:
                    domains_requiring_playwright.add(domain)
                    return dims, img_bytes

            return (0, 0), None

    # If we get here, domain is known to need Playwright but it's not available
    return (0, 0), None


# ====== Enhanced Matching Functions ====== #

def fetch_matching_images_enhanced(url: str, client, image) -> Dict[MatchType, List[Dict]]:
    """Fetch ALL types of matching images from Google Vision API"""
    try:
        domain = urlparse(url).netloc
        print_action(f"Calling Google Vision API for {domain}")

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

        total_matches = sum(len(items) for items in matches.values())
        if total_matches > 0:
            print_success(f"Google Vision found {total_matches} potential matches:")
            for match_type, items in matches.items():
                if items:
                    print_verbose(f"  - {match_type.value}: {len(items)} matches", Fore.GREEN)
        else:
            print_verbose(f"No matches found by Google Vision for {domain}", Fore.YELLOW)

        return matches
    except Exception as e:
        print_error(f"Google Vision API error for {url}: {str(e)[:100]}")
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
    """Process images concurrently with human-like behavior and incremental saves"""
    global interrupted, all_results_global

    print_info(f"Starting to process {len(df['Address'])} images")
    print_info(f"Incremental saves every {INCREMENTAL_SAVE_INTERVAL} images to {OUTPUT_FOLDER}")
    print_info(f"Using {config.MAX_WORKERS} concurrent workers")
    stats['total_urls'] = len(df['Address'])

    # Shuffle URLs to simulate non-sequential browsing (more human-like)
    urls = list(df['Address'])
    random.shuffle(urls)
    print_info("URLs shuffled for random access pattern")

    # Initialize progress bars
    pbar = None
    match_pbar = None

    try:
        # Create main progress bar
        pbar = tqdm(total=len(urls), desc="Processing images", unit="img",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        # Create secondary progress bar for matches found - simpler format
        match_counts = {'full': 0, 'partial': 0, 'similar': 0, 'pages': 0}
        match_pbar = tqdm(total=0, desc="Matches", unit="match", position=1,
                          bar_format='{desc}: {n_fmt} matches found')

        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            future_to_url = {
                executor.submit(fetch_and_process_image_enhanced, url, client, image, user_agent): url
                for url in urls
            }

            completed_count = 0
            incremental_save_counter = 1

            for future in as_completed(future_to_url):
                if interrupted:
                    print_warning("Interrupt detected, canceling remaining tasks...")
                    for f in future_to_url:
                        if not f.done():
                            f.cancel()
                    break

                url = future_to_url[future]
                domain = urlparse(url).netloc if url else "unknown"

                try:
                    # Get results from this URL
                    local_results = future.result()

                    # Add to global results
                    new_matches = 0
                    for match_type, results in local_results.items():
                        all_results_global[match_type].extend(results)
                        new_matches += len(results)

                    completed_count += 1
                    stats['completed_urls'] = completed_count

                    # Update progress bars
                    pbar.update(1)

                    # Update match counts
                    total_found = sum(len(r) for r in all_results_global.values())
                    match_counts = {
                        'full': len(all_results_global[MatchType.FULL]),
                        'partial': len(all_results_global[MatchType.PARTIAL]),
                        'similar': len(all_results_global[MatchType.SIMILAR]),
                        'pages': len(all_results_global[MatchType.PAGE])
                    }

                    # Update match progress bar with custom description
                    match_pbar.total = total_found
                    match_pbar.n = total_found
                    match_desc = f"Matches [F:{match_counts['full']} P:{match_counts['partial']} S:{match_counts['similar']} Pg:{match_counts['pages']}]"
                    match_pbar.set_description(match_desc)
                    match_pbar.refresh()

                    # Update main progress bar description with current domain
                    if new_matches > 0:
                        pbar.set_description(f"Processing [{domain[:20]}] +{new_matches}")
                    else:
                        pbar.set_description(f"Processing [{domain[:20]}]")

                    if new_matches > 0:
                        # Don't print during active progress bar updates, it messes up the display
                        # The progress bars show all the info needed
                        pass

                    # Incremental save check
                    if completed_count % INCREMENTAL_SAVE_INTERVAL == 0:
                        # Temporarily clear progress bars for save message
                        pbar.clear()
                        match_pbar.clear()
                        print_info(
                            f"Reached {completed_count} images - Performing incremental save #{incremental_save_counter}")
                        if save_incremental_results(all_results_global, incremental_save_counter):
                            stats['last_incremental_save'] = completed_count
                            incremental_save_counter += 1
                            print_success(
                                f"Incremental save #{incremental_save_counter - 1} completed. Total matches: {total_found}")
                        else:
                            print_warning("Incremental save failed, continuing...")
                        # Refresh progress bars
                        pbar.refresh()
                        match_pbar.refresh()

                except Exception as e:
                    # Update progress even on error
                    completed_count += 1
                    stats['completed_urls'] = completed_count
                    pbar.update(1)
                    pbar.set_description(f"Error on [{domain[:20]}]")

                    # Log error without interrupting progress bars too much
                    if config.VERBOSE:
                        # Clear and restore progress bars around error message
                        pbar.clear()
                        match_pbar.clear()
                        print_error(f"Error processing {domain}: {str(e)[:100]}")
                        pbar.refresh()
                        match_pbar.refresh()

                    # Still check for incremental save even on errors
                    if completed_count % INCREMENTAL_SAVE_INTERVAL == 0:
                        save_incremental_results(all_results_global, incremental_save_counter)
                        incremental_save_counter += 1

    except Exception as e:
        print_error(f"Critical error in process_images_enhanced: {str(e)}")
        if config.VERBOSE:
            import traceback
            traceback.print_exc()

    finally:
        # Always close progress bars properly
        if pbar:
            pbar.close()
        if match_pbar:
            match_pbar.close()

        # Print final summary after progress bars are closed
        total_found = sum(len(r) for r in all_results_global.values())
        if total_found > 0:
            print("")  # New line after progress bars
            print_success(
                f"Processing complete: {stats['completed_urls']} images processed, {total_found} matches found")
            print_info(f"  Full matches: {len(all_results_global[MatchType.FULL])}")
            print_info(f"  Partial matches: {len(all_results_global[MatchType.PARTIAL])}")
            print_info(f"  Similar matches: {len(all_results_global[MatchType.SIMILAR])}")
            print_info(f"  Page matches: {len(all_results_global[MatchType.PAGE])}")

        # Final save if there are unsaved results
        if stats['completed_urls'] > stats.get('last_incremental_save', 0):
            print_info(
                f"Saving final incremental results (rows {stats.get('last_incremental_save', 0) + 1} to {stats['completed_urls']})")
            save_incremental_results(all_results_global, 9999)  # Use 9999 as final save number

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

    # Print banner
    print(Fore.CYAN + "=" * 60)
    print(Fore.CYAN + "     IMAGE RESOLUTION FINDER - ENHANCED VERSION")
    print(Fore.CYAN + "=" * 60)
    print(Fore.GREEN + "✓ Human-like browsing behavior")
    print(Fore.GREEN + "✓ Playwright fallback for blocked sites")
    print(Fore.GREEN + "✓ Incremental saves every 100 images")
    print(Fore.GREEN + "✓ Smart rate limiting with visual feedback")
    print(Fore.CYAN + "=" * 60)
    print(Fore.YELLOW + "Press Ctrl+C at any time to stop and save results")
    print(Fore.CYAN + "=" * 60 + Style.RESET_ALL)

    print_info(f"Configuration:")
    print(f"  • Workers: {Fore.YELLOW}{config.MAX_WORKERS}{Style.RESET_ALL}")
    print(
        f"  • Domain delay: {Fore.YELLOW}{config.SAME_DOMAIN_DELAY}-{config.SAME_DOMAIN_DELAY + config.SAME_DOMAIN_DELAY_JITTER}s{Style.RESET_ALL}")
    print(f"  • Global delay: {Fore.YELLOW}{config.GLOBAL_DELAY_MIN}-{config.GLOBAL_DELAY_MAX}s{Style.RESET_ALL}")
    print(f"  • Timeout: {Fore.YELLOW}{config.REQUEST_TIMEOUT}s{Style.RESET_ALL}")
    print(
        f"  • Breaks: Every {Fore.YELLOW}{config.BREAK_AFTER_REQUESTS}{Style.RESET_ALL} requests for {Fore.YELLOW}{config.BREAK_DURATION_MIN}-{config.BREAK_DURATION_MAX}s{Style.RESET_ALL}")
    print(
        f"  • Playwright: {Fore.GREEN if PLAYWRIGHT_AVAILABLE else Fore.RED}{'Available' if PLAYWRIGHT_AVAILABLE else 'Not installed'}{Style.RESET_ALL}")
    print(
        f"  • Saves: Every {Fore.YELLOW}{INCREMENTAL_SAVE_INTERVAL}{Style.RESET_ALL} images to {Fore.YELLOW}{OUTPUT_FOLDER}{Style.RESET_ALL}")
    print(Fore.CYAN + "=" * 60 + Style.RESET_ALL)

    # Check for existing incremental saves
    if os.path.exists(OUTPUT_FOLDER):
        saved_files = sorted(
            [f for f in os.listdir(OUTPUT_FOLDER) if f.startswith('incremental_save_') and f.endswith('.xlsx')])
        if saved_files:
            print_warning(f"Found {len(saved_files)} existing incremental save(s)")
            latest_file = saved_files[-1]
            print_info(f"Most recent: {latest_file}")
            print_info("Note: Script will process ALL images from scratch")

    try:
        print_action("Initializing Google Vision client...")
        client, image = initialize_client()
        print_success("Google Vision client ready")

        print_action("Initializing User Agent generator...")
        user_agent = UserAgent()
        print_success("User Agent generator ready")

        print_action(f"Loading images from {INPUT_FILE_PATH}...")
        df = pd.read_csv(INPUT_FILE_PATH)

        # Debug: Show CSV structure
        print_info(f"CSV columns found: {list(df.columns)}")
        print_info(f"CSV shape: {df.shape[0]} rows, {df.shape[1]} columns")

        # Try to find the URL column
        url_column = None
        possible_names = ['Address', 'address', 'URL', 'url', 'Url', 'Image URL', 'Image_URL', 'image_url', 'Image',
                          'image']

        for col_name in possible_names:
            if col_name in df.columns:
                url_column = col_name
                print_success(f"Found URL column: '{url_column}'")
                break

        if url_column is None:
            # If no standard name found, check if it's the first column or if there's only one column
            if len(df.columns) == 1:
                url_column = df.columns[0]
                print_warning(f"Using the only column as URL column: '{url_column}'")
            else:
                print_error("Could not identify URL column!")
                print_info("Available columns:")
                for i, col in enumerate(df.columns):
                    print(f"  {i + 1}. {col}")
                print_error("Please ensure your CSV has a column named 'Address' or 'URL'")
                print_info("Or modify the script's INPUT_FILE_PATH to use the correct column")
                return

        # Create a standardized dataframe with 'Address' column
        urls_df = pd.DataFrame()
        urls_df['Address'] = df[url_column]

        # Remove any NaN or empty URLs
        urls_df = urls_df.dropna(subset=['Address'])
        urls_df = urls_df[urls_df['Address'].str.strip() != '']

        # Validate URLs
        valid_urls = urls_df['Address'].str.startswith(('http://', 'https://'))
        if not valid_urls.all():
            invalid_count = (~valid_urls).sum()
            print_warning(f"Found {invalid_count} invalid URLs (not starting with http:// or https://)")
            print_info("Filtering out invalid URLs...")
            urls_df = urls_df[valid_urls]

        print_success(f"Loaded {len(urls_df)} valid URLs to process")

        if len(urls_df) == 0:
            print_error("No valid URLs found to process!")
            return

        # Show sample URLs
        print_info("Sample URLs to process:")
        for i, url in enumerate(urls_df['Address'].head(3)):
            print(f"  {i + 1}. {url[:100]}...")

        print(Fore.CYAN + "=" * 60 + Style.RESET_ALL)
        print_info("Starting image processing...")
        print(Fore.CYAN + "=" * 60 + Style.RESET_ALL)

        # Process images - results are now stored in all_results_global
        process_images_enhanced(urls_df, client, image, user_agent)

        # Save results
        if not interrupted and any(len(results) > 0 for results in all_results_global.values()):
            create_excel_output(all_results_global, OUTPUT_FILE_PATH)

            # Also create final timestamped version in output folder
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            final_output = os.path.join(OUTPUT_FOLDER, f"COMPLETE_results_{timestamp}.xlsx")
            create_excel_output(all_results_global, final_output)

            print(Fore.GREEN + "=" * 60)
            print(Fore.GREEN + "PROCESSING COMPLETE!")
            print(Fore.GREEN + "=" * 60 + Style.RESET_ALL)
            log_final_statistics()
            print_success(f"Final output: {OUTPUT_FILE_PATH}")
            print_success(f"Complete results: {final_output}")
        elif not interrupted:
            print_warning("No higher resolution images found")

    except KeyboardInterrupt:
        signal_handler(None, None)
    except Exception as e:
        print_error(f"Unexpected error in main: {str(e)}")
        import traceback
        traceback.print_exc()
        save_and_exit()
    finally:
        cleanup_playwright()


if __name__ == "__main__":
    main()
