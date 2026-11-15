"""
SERP Appearance Report - Core Logic

Fetches live Google organic results for a list of keywords from the
DataForSEO SERP API (Google Organic, Live Advanced) and keeps the results
that belong to a user-supplied domain.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import io
import math
import time
from base64 import b64encode
from urllib.parse import urlparse

import pandas as pd
import requests

ENCODINGS = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1']

DATAFORSEO_ENDPOINT = 'https://api.dataforseo.com/v3/serp/google/organic/live/advanced'

# DataForSEO bills live SERP requests per 10 results
COST_PER_10_RESULTS = 0.002

# Region name -> (DataForSEO location_code, language_code)
LOCATIONS = {
    "United Kingdom": (2826, "en"),
    "United States": (2840, "en"),
    "Australia": (2036, "en"),
    "Canada": (2124, "en"),
    "Germany": (2276, "de"),
    "France": (2250, "fr"),
    "Spain": (2724, "es"),
    "Italy": (2380, "it"),
    "Netherlands": (2528, "nl"),
    "India": (2356, "en"),
}

# DataForSEO SERP API supports desktop and mobile only
DEVICES = ("desktop", "mobile")

KEYWORD_COLUMN_NAMES = ('keyword', 'keywords', 'query', 'queries', 'search term', 'top queries')

COLUMNS = ['query', 'position', 'link', 'title', 'snippet']


# ----------------------------------------------------------------------
# Keyword input
# ----------------------------------------------------------------------

def decode_bytes(raw_bytes):
    """Decode raw bytes (e.g. an uploaded file), trying several encodings."""
    for encoding in ENCODINGS:
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode the file with any of the encodings: {ENCODINGS}")


def dedupe_keywords(keywords):
    """Strip blanks and duplicates (case-insensitive), keeping the first spelling and order."""
    seen = set()
    cleaned = []
    for keyword in keywords:
        if keyword is None or (isinstance(keyword, float) and math.isnan(keyword)):
            continue
        keyword = str(keyword).strip()
        if not keyword or keyword.lower() in seen:
            continue
        seen.add(keyword.lower())
        cleaned.append(keyword)
    return cleaned


def parse_keyword_lines(text):
    """Return keywords from pasted text, one per line."""
    return dedupe_keywords(text.splitlines())


def keywords_from_csv_text(text):
    """Return keywords from CSV text.

    Uses the first column named keyword, keywords, query, queries, search term
    or top queries (any case); otherwise the first column.
    """
    df = pd.read_csv(io.StringIO(text), dtype=str)
    if df.empty or len(df.columns) == 0:
        return []
    column = df.columns[0]
    for col in df.columns:
        if str(col).strip().lower() in KEYWORD_COLUMN_NAMES:
            column = col
            break
    return dedupe_keywords(df[column].tolist())


def keywords_from_bytes(raw_bytes, filename):
    """Return keywords from an uploaded .csv or .txt file."""
    text = decode_bytes(raw_bytes)
    if filename.lower().endswith('.csv'):
        return keywords_from_csv_text(text)
    return parse_keyword_lines(text)


def read_keywords_file(file_path):
    """Read keywords from a .csv file (keyword column) or a text file (one per line)."""
    with open(file_path, 'rb') as file:
        raw_bytes = file.read()
    return keywords_from_bytes(raw_bytes, str(file_path))


# ----------------------------------------------------------------------
# Domain matching
# ----------------------------------------------------------------------

def normalise_domain(domain):
    """Strip scheme, www prefix and trailing slashes from a domain string."""
    domain = domain.strip().lower()
    for prefix in ('https://', 'http://'):
        if domain.startswith(prefix):
            domain = domain[len(prefix):]
    if domain.startswith('www.'):
        domain = domain[len('www.'):]
    return domain.rstrip('/')


def url_matches_domain(url, domain):
    """True when the URL is on the domain or one of its subdomains.

    If the domain includes a path (e.g. example.com/blog), the URL path must
    also start with that path.
    """
    if not url:
        return False
    domain_host, _, domain_path = domain.partition('/')
    parsed = urlparse(url if '://' in url else f'https://{url}')
    host = parsed.netloc.lower().split(':')[0]
    if host.startswith('www.'):
        host = host[len('www.'):]
    if not (host == domain_host or host.endswith('.' + domain_host)):
        return False
    if domain_path:
        return parsed.path.lower().lstrip('/').startswith(domain_path)
    return True


# ----------------------------------------------------------------------
# DataForSEO
# ----------------------------------------------------------------------

class DataForSEOError(ValueError):
    """An error reported by the DataForSEO API, with its status code when known."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code

    @property
    def is_account_error(self):
        """True for authentication (401xx) and payment or balance (402xx) errors."""
        return self.status_code is not None and 40100 <= self.status_code < 40300


def estimate_cost(num_keywords, depth):
    """Estimated DataForSEO cost in USD for a run."""
    return num_keywords * COST_PER_10_RESULTS * math.ceil(depth / 10)


def build_headers(login, password):
    """Basic auth headers for the DataForSEO API."""
    cred = b64encode(f"{login}:{password}".encode()).decode()
    return {
        'Authorization': f'Basic {cred}',
        'Content-Type': 'application/json'
    }


def build_payload(keyword, location_code, language_code, device, depth):
    """Request body for one keyword."""
    return [{
        "keyword": keyword,
        "location_code": location_code,
        "language_code": language_code,
        "device": device.lower(),
        "depth": depth
    }]


def parse_serp_response(response_data):
    """Return (organic_items, cost) from a DataForSEO SERP response.

    Raises ValueError with a readable message when the API or the task
    reports an error.
    """
    if not isinstance(response_data, dict):
        raise DataForSEOError("Unexpected response from DataForSEO")
    if response_data.get('status_code') != 20000:
        raise DataForSEOError(f"API error: {response_data.get('status_message', 'Unknown error')}",
                              response_data.get('status_code'))
    cost = response_data.get('cost') or 0.0
    try:
        task = response_data["tasks"][0]
    except (KeyError, TypeError, IndexError):
        raise DataForSEOError("No task returned by DataForSEO")
    if task.get('status_code') != 20000:
        raise DataForSEOError(f"Task error: {task.get('status_message', 'Unknown error')}",
                              task.get('status_code'))
    try:
        items = task["result"][0]["items"] or []
    except (KeyError, TypeError, IndexError):
        items = []
    organic = [item for item in items if item.get("type") == "organic"]
    return organic, cost


def fetch_serp(keyword, headers, location_code, language_code, device, depth, timeout=60):
    """Fetch one keyword's live SERP. Returns (organic_items, cost); raises DataForSEOError on failure."""
    try:
        response = requests.post(
            DATAFORSEO_ENDPOINT,
            headers=headers,
            json=build_payload(keyword, location_code, language_code, device, depth),
            timeout=timeout
        )
    except requests.exceptions.RequestException as e:
        raise DataForSEOError(f"Request failed: {e}")
    try:
        response_data = response.json()
    except ValueError:
        status = 40100 if response.status_code == 401 else None
        raise DataForSEOError(f"Could not parse the API response (HTTP {response.status_code})", status)
    return parse_serp_response(response_data)


def extract_appearances(organic_items, query, domain):
    """Return one row per organic result on the domain, with query, position, link, title and snippet."""
    domain = normalise_domain(domain)
    if not domain:
        raise ValueError("A domain must be supplied.")

    rows = []
    for item in organic_items:
        link = item.get('url') or ''
        if url_matches_domain(link, domain):
            rows.append({
                'query': query,
                'position': item.get('rank_group', ''),
                'link': link,
                'title': item.get('title') or '',
                'snippet': item.get('description') or '',
            })
    return rows


def run_report(keywords, domain, login, password, location_code, language_code,
               device='desktop', depth=10, delay=2.0, progress_callback=None):
    """Fetch every keyword and collect the domain's organic appearances.

    Requests are sent one at a time with `delay` seconds between them.
    Returns (rows, warnings, total_cost).
    """
    if not normalise_domain(domain):
        raise ValueError("A domain must be supplied.")

    headers = build_headers(login, password)
    all_rows = []
    warnings = []
    total_cost = 0.0

    for i, keyword in enumerate(keywords):
        if progress_callback:
            progress_callback(i, len(keywords), keyword)
        try:
            organic, cost = fetch_serp(keyword, headers, location_code, language_code, device, depth)
            total_cost += cost
            all_rows.extend(extract_appearances(organic, keyword, domain))
        except DataForSEOError as e:
            warnings.append(f"{keyword}: {e}")
            # Stop on credential or balance errors rather than repeating them for every keyword
            if e.is_account_error:
                warnings.append("Stopped: fix the DataForSEO account error above and run again.")
                break

        if i < len(keywords) - 1 and delay > 0:
            time.sleep(delay)

    if progress_callback:
        progress_callback(len(keywords), len(keywords), None)

    return all_rows, warnings, total_cost


def results_to_dataframe(all_results):
    """Convert extracted results into a DataFrame with fixed columns."""
    return pd.DataFrame(all_results, columns=COLUMNS)
