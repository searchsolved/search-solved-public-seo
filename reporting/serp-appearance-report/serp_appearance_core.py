"""
SERP Appearance Report - Core Logic

Parses ValueSERP batch JSON output and filters organic results to those
containing a user-supplied domain.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import json

import pandas as pd

ENCODINGS = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1']


def read_json_file(file_path):
    """Read a JSON file, trying several encodings."""
    for encoding in ENCODINGS:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return json.load(file)
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Unable to read the file with any of the encodings: {ENCODINGS}")


def load_json_bytes(raw_bytes):
    """Decode raw bytes (e.g. an uploaded file) to JSON, trying several encodings."""
    for encoding in ENCODINGS:
        try:
            return json.loads(raw_bytes.decode(encoding))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Unable to decode the file with any of the encodings: {ENCODINGS}")


def normalise_domain(domain):
    """Strip scheme, www prefix and trailing slashes from a domain string."""
    domain = domain.strip().lower()
    for prefix in ('https://', 'http://'):
        if domain.startswith(prefix):
            domain = domain[len(prefix):]
    if domain.startswith('www.'):
        domain = domain[len('www.'):]
    return domain.rstrip('/')


def extract_appearances(data, domain):
    """Extract organic results matching the given domain from ValueSERP batch JSON.

    Returns a list of dicts with query, position, link, title and snippet,
    plus a list of warning strings for any skipped items.
    """
    domain = normalise_domain(domain)
    if not domain:
        raise ValueError("A domain must be supplied.")

    all_results = []
    warnings = []

    # Ensure data is a list
    if not isinstance(data, list):
        data = [data]

    for item in data:
        if not isinstance(item, dict) or 'result' not in item:
            warnings.append("'result' key not found in an item, skipping.")
            continue

        result = item['result']

        # Extract the search query
        search_params = result.get('search_parameters', {})
        query = search_params.get('q', '')

        # Extract organic results
        organic_results = result.get('organic_results', [])

        for organic_result in organic_results:
            position = organic_result.get('position', '')
            link = organic_result.get('link', '')
            title = organic_result.get('title', '')
            snippet = organic_result.get('snippet', '')

            # Only include results where the link contains the target domain
            if domain in link.lower():
                all_results.append({
                    'query': query,
                    'position': position,
                    'link': link,
                    'title': title,
                    'snippet': snippet,
                })

    return all_results, warnings


def results_to_dataframe(all_results):
    """Convert extracted results into a DataFrame with fixed columns."""
    columns = ['query', 'position', 'link', 'title', 'snippet']
    return pd.DataFrame(all_results, columns=columns)
