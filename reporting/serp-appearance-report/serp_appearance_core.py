"""
SERP Appearance Report - Core Logic

Matches Google organic results collected from the DataForSEO SERP API
(Standard queue, see dataforseo_queue.py) against a user-supplied domain.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import io
import math
from urllib.parse import urlparse

import pandas as pd

from dataforseo_queue import is_unresolved, organic_items

ENCODINGS = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1']

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
# Report
# ----------------------------------------------------------------------

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


def build_report(ready, domain):
    """Rows for every organic appearance of the domain in the collected searches.

    ready is {keyword: items} from dataforseo_queue.collect_results(). Returns
    (rows, unresolved) where unresolved counts organic results that came back as
    google.com/goto links, which cannot be matched to a domain.
    """
    rows, unresolved = [], 0
    for keyword, items in ready.items():
        organic = organic_items(items)
        unresolved += sum(is_unresolved(item.get('url')) for item in organic)
        rows.extend(extract_appearances(organic, keyword, domain))
    return rows, unresolved


def results_to_dataframe(all_results):
    """Convert extracted results into a DataFrame with fixed columns."""
    return pd.DataFrame(all_results, columns=COLUMNS)
