#!/usr/bin/env python3
"""
OnCrawl API Suite - Comprehensive CLI Version
Full-featured OnCrawl API client for data extraction, crawl management, and analysis.

Author: Lee Foot
Website: https://leefoot.co.uk

Usage:
    # List resources
    python oncrawl_extractor_cli.py --token TOKEN --list-workspaces
    python oncrawl_extractor_cli.py --token TOKEN --workspace WS_ID --list-projects
    python oncrawl_extractor_cli.py --token TOKEN --workspace WS_ID --project PROJ_ID --list-crawls
    python oncrawl_extractor_cli.py --token TOKEN --list-presets

    # Data extraction
    python oncrawl_extractor_cli.py --token TOKEN --crawl CRAWL_ID --preset "404-pages"
    python oncrawl_extractor_cli.py --token TOKEN --crawl CRAWL_ID --preset "301-redirects" -o redirects.csv
    python oncrawl_extractor_cli.py --token TOKEN --crawl CRAWL_ID --fields url,status_code,title --oql '{"and":[{"field":["status_code","equals",200]}]}'

    # Crawl management
    python oncrawl_extractor_cli.py --token TOKEN --project PROJ_ID --launch-crawl
    python oncrawl_extractor_cli.py --token TOKEN --crawl CRAWL_ID --pause-crawl
    python oncrawl_extractor_cli.py --token TOKEN --crawl CRAWL_ID --resume-crawl
    python oncrawl_extractor_cli.py --token TOKEN --crawl CRAWL_ID --cancel-crawl

    # Crawl comparison
    python oncrawl_extractor_cli.py --token TOKEN --crawl CRAWL_ID --compare-to REF_CRAWL_ID --change-type new

    # Aggregations
    python oncrawl_extractor_cli.py --token TOKEN --crawl CRAWL_ID --aggregate status_code

    # Link export
    python oncrawl_extractor_cli.py --token TOKEN --crawl CRAWL_ID --export-links internal

    # Project management
    python oncrawl_extractor_cli.py --token TOKEN --workspace WS_ID --create-project --name "My Project" --start-url "https://example.com"
    python oncrawl_extractor_cli.py --token TOKEN --project PROJ_ID --delete-project
"""

import argparse
import sys
import json
from datetime import datetime
from io import StringIO

import requests
import pandas as pd

# Constants
BASE_URL = "https://app.oncrawl.com/api/v2"

# Comprehensive preset queries
PRESET_QUERIES = {
    # Status Code Issues
    "404-pages": {
        "name": "404 Pages",
        "description": "All pages returning 404 status code",
        "fields": ["url", "status_code", "depth", "inrank", "nb_inlinks"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 404]}]}
    },
    "5xx-errors": {
        "name": "5xx Server Errors",
        "description": "All pages with server errors (500-599)",
        "fields": ["url", "status_code", "depth", "nb_inlinks"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "between", [500, 599]]}]}
    },
    "4xx-errors": {
        "name": "4xx Client Errors",
        "description": "All pages with client errors (400-499)",
        "fields": ["url", "status_code", "depth", "nb_inlinks"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "between", [400, 499]]}]}
    },
    # Redirects
    "301-redirects": {
        "name": "301 Redirects",
        "description": "All 301 permanent redirects",
        "fields": ["url", "status_code", "redirect_location", "final_redirect_location", "redirect_count"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 301]}]}
    },
    "302-redirects": {
        "name": "302 Redirects",
        "description": "All 302 temporary redirects",
        "fields": ["url", "status_code", "redirect_location", "final_redirect_location", "redirect_count"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 302]}]}
    },
    "redirect-chains": {
        "name": "Redirect Chains",
        "description": "Pages with multiple redirects (>1)",
        "fields": ["url", "status_code", "redirect_count", "redirect_location", "final_redirect_location"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["redirect_count", "gt", 1]}]}
    },
    "redirect-loops": {
        "name": "Redirect Loops",
        "description": "Pages caught in redirect loops",
        "fields": ["url", "status_code", "redirect_count", "redirect_location", "final_redirect_location"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["is_redirect_loop", "equals", True]}]}
    },
    "stale-links": {
        "name": "Stale Links",
        "description": "Redirecting pages still receiving internal links",
        "fields": ["url", "status_code", "redirect_location", "nb_inlinks", "inrank"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "between", [300, 399]]}, {"field": ["nb_inlinks", "gt", 0]}]}
    },
    # Indexability
    "indexable": {
        "name": "Indexable Pages",
        "description": "All indexable pages",
        "fields": ["url", "status_code", "meta_robots_index", "canonical_evaluation", "title", "h1"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["meta_robots_index", "equals", True]}, {"field": ["robots_txt_denied", "equals", False]}, {"field": ["canonical_evaluation", "equals", "matching"]}]}
    },
    "non-indexable": {
        "name": "Non-Indexable Pages",
        "description": "Pages blocked from indexing",
        "fields": ["url", "status_code", "meta_robots_index", "robots_txt_denied", "canonical_evaluation"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"or": [{"field": ["meta_robots_index", "equals", False]}, {"field": ["robots_txt_denied", "equals", True]}, {"field": ["canonical_evaluation", "equals", "not_matching"]}]}]}
    },
    "noindex": {
        "name": "Noindex Pages",
        "description": "Pages with noindex directive",
        "fields": ["url", "status_code", "meta_robots_index", "x_robots_tag_index", "title"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["meta_robots_index", "equals", False]}]}
    },
    "robots-blocked": {
        "name": "Robots.txt Blocked",
        "description": "Pages blocked by robots.txt",
        "fields": ["url", "status_code", "robots_txt_denied", "nb_inlinks"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["robots_txt_denied", "equals", True]}]}
    },
    # Canonical Issues
    "non-matching-canonicals": {
        "name": "Non-Matching Canonicals",
        "description": "Pages where canonical doesn't match URL",
        "fields": ["url", "canonical", "canonical_evaluation", "status_code"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["canonical_evaluation", "equals", "not_matching"]}]}
    },
    "missing-canonicals": {
        "name": "Missing Canonicals",
        "description": "Pages without canonical tags",
        "fields": ["url", "canonical", "status_code", "title"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["canonical", "is_empty", True]}]}
    },
    # Content Issues
    "no-title": {
        "name": "Pages Without Title",
        "description": "Pages missing title tags",
        "fields": ["url", "status_code", "title", "h1"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["title", "is_empty", True]}]}
    },
    "no-h1": {
        "name": "Pages Without H1",
        "description": "Pages missing H1 tags",
        "fields": ["url", "status_code", "title", "h1"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["h1", "is_empty", True]}]}
    },
    "no-description": {
        "name": "Pages Without Description",
        "description": "Pages missing meta descriptions",
        "fields": ["url", "status_code", "title", "meta_description"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["meta_description", "is_empty", True]}]}
    },
    "duplicate-titles": {
        "name": "Duplicate Titles",
        "description": "Pages with duplicate titles",
        "fields": ["url", "status_code", "title", "title_duplicates_count"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["title_duplicates_count", "gt", 1]}]}
    },
    "duplicate-descriptions": {
        "name": "Duplicate Descriptions",
        "description": "Pages with duplicate meta descriptions",
        "fields": ["url", "status_code", "meta_description", "meta_description_duplicates_count"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["meta_description_duplicates_count", "gt", 1]}]}
    },
    "short-titles": {
        "name": "Short Titles (<30 chars)",
        "description": "Pages with very short titles",
        "fields": ["url", "title", "title_length", "status_code"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["title_length", "lt", 30]}, {"field": ["title_length", "gt", 0]}]}
    },
    "long-titles": {
        "name": "Long Titles (>60 chars)",
        "description": "Pages with overly long titles",
        "fields": ["url", "title", "title_length", "status_code"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["title_length", "gt", 60]}]}
    },
    "thin-content": {
        "name": "Thin Content (<300 words)",
        "description": "Pages with thin content",
        "fields": ["url", "word_count", "title", "status_code"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["word_count", "lt", 300]}]}
    },
    # Link Structure
    "orphan-pages": {
        "name": "Orphan Pages",
        "description": "Pages with no internal links",
        "fields": ["url", "status_code", "nb_inlinks", "depth"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["nb_inlinks", "equals", 0]}]}
    },
    "deep-pages": {
        "name": "Deep Pages (Depth > 5)",
        "description": "Pages more than 5 clicks deep",
        "fields": ["url", "status_code", "depth", "nb_inlinks"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["depth", "gt", 5]}]}
    },
    "low-inrank": {
        "name": "Low Inrank Pages",
        "description": "Important pages with low internal PageRank",
        "fields": ["url", "inrank", "nb_inlinks", "depth", "status_code"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["inrank", "lt", 0.1]}]}
    },
    "high-outlinks": {
        "name": "High Outlink Pages (>100)",
        "description": "Pages with many outgoing links",
        "fields": ["url", "nb_outlinks", "nb_internal_outlinks", "nb_external_outlinks"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["nb_outlinks", "gt", 100]}]}
    },
    # Performance
    "slow-pages": {
        "name": "Slow Pages (>3s)",
        "description": "Pages loading over 3 seconds",
        "fields": ["url", "status_code", "delay_total", "delay_first_byte", "size_total"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["delay_total", "gt", 3000]}]}
    },
    "very-slow-pages": {
        "name": "Very Slow Pages (>5s)",
        "description": "Pages loading over 5 seconds",
        "fields": ["url", "status_code", "delay_total", "delay_first_byte", "size_total"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["delay_total", "gt", 5000]}]}
    },
    "large-pages": {
        "name": "Large Pages (>1MB)",
        "description": "Pages over 1MB in size",
        "fields": ["url", "size_total", "size_html", "delay_total"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["size_total", "gt", 1000000]}]}
    },
    "slow-ttfb": {
        "name": "Slow TTFB (>1s)",
        "description": "Pages with slow Time to First Byte",
        "fields": ["url", "delay_first_byte", "delay_total", "status_code"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["delay_first_byte", "gt", 1000]}]}
    },
    # Images
    "images-without-alt": {
        "name": "Images Without Alt",
        "description": "Pages with images missing alt text",
        "fields": ["url", "nb_images", "nb_images_without_alt", "status_code"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["nb_images_without_alt", "gt", 0]}]}
    },
    # Sitemap
    "not-in-sitemap": {
        "name": "Not in Sitemap",
        "description": "Indexable pages not in sitemap",
        "fields": ["url", "in_sitemap", "meta_robots_index", "canonical_evaluation"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["meta_robots_index", "equals", True]}, {"field": ["in_sitemap", "equals", False]}]}
    },
    "sitemap-non-200": {
        "name": "In Sitemap (Non-200)",
        "description": "Non-200 pages found in sitemap",
        "fields": ["url", "status_code", "in_sitemap"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["in_sitemap", "equals", True]}, {"field": ["status_code", "not_equals", 200]}]}
    },
    # HTTPS/Security
    "http-pages": {
        "name": "HTTP Pages",
        "description": "Non-HTTPS pages",
        "fields": ["url", "is_https", "status_code"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["is_https", "equals", False]}]}
    },
    "mixed-content": {
        "name": "Mixed Content",
        "description": "HTTPS pages with mixed content",
        "fields": ["url", "is_https", "has_mixed_content", "status_code"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["has_mixed_content", "equals", True]}]}
    },
    # All Pages
    "all-pages": {
        "name": "All Fetched Pages",
        "description": "Complete list of all crawled pages",
        "fields": ["url", "status_code", "depth", "nb_inlinks", "inrank"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}]}
    },
    "all-200": {
        "name": "All 200 Pages",
        "description": "All pages returning 200 OK",
        "fields": ["url", "title", "h1", "depth", "nb_inlinks", "word_count"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}]}
    }
}

# Aggregate field options
AGGREGATE_FIELDS = [
    "status_code", "depth", "urlsegment1", "urlsegment2", "urlsegment3",
    "content_type", "canonical_evaluation", "meta_robots_index", "in_sitemap",
    "lang", "charset"
]


def get_headers(api_token):
    """Return headers for API authentication."""
    return {
        'Authorization': f'Bearer {api_token}',
        'Content-Type': 'application/json'
    }


# =============================================================================
# API Functions - Account & Workspaces
# =============================================================================

def get_account(api_token):
    """Fetch account information."""
    try:
        response = requests.get(
            f"{BASE_URL}/account",
            headers=get_headers(api_token),
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get('account', {}), None
        return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)


def get_workspaces(api_token):
    """Fetch all workspaces."""
    try:
        account, error = get_account(api_token)
        if error:
            return [], error

        workspace_ids = account.get('workspace_ids', [])
        workspaces = []

        for ws_id in workspace_ids:
            ws_response = requests.get(
                f"{BASE_URL}/workspaces/{ws_id}",
                headers=get_headers(api_token),
                timeout=30
            )
            if ws_response.status_code == 200:
                ws_data = ws_response.json().get('workspace', {})
                workspaces.append({
                    'id': ws_id,
                    'name': ws_data.get('name', ws_id),
                    'crawl_pages_limit': ws_data.get('crawl_pages_limit'),
                    'crawl_pages_used': ws_data.get('crawl_pages_used')
                })
        return workspaces, None
    except Exception as e:
        return [], str(e)


# =============================================================================
# API Functions - Projects
# =============================================================================

def get_projects(api_token, workspace_id):
    """Fetch all projects in a workspace."""
    try:
        all_projects = []
        offset = 0
        limit = 100

        while True:
            response = requests.get(
                f"{BASE_URL}/workspaces/{workspace_id}/projects",
                headers=get_headers(api_token),
                params={'offset': offset, 'limit': limit},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                projects = data.get('projects', [])
                all_projects.extend(projects)

                total = data.get('meta', {}).get('total', 0)
                if offset + limit >= total or len(projects) == 0:
                    break
                offset += limit
            else:
                return [], f"Error {response.status_code}: {response.text}"

        return all_projects, None
    except Exception as e:
        return [], str(e)


def create_project(api_token, workspace_id, name, start_url, user_agent="oncrawl"):
    """Create a new project."""
    try:
        data = {
            "name": name,
            "start_url": start_url,
            "user_agent": user_agent
        }
        response = requests.post(
            f"{BASE_URL}/workspaces/{workspace_id}/projects",
            headers=get_headers(api_token),
            json=data,
            timeout=30
        )
        if response.status_code in [200, 201]:
            return response.json().get('project', {}), None
        return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)


def delete_project(api_token, project_id):
    """Delete a project."""
    try:
        response = requests.delete(
            f"{BASE_URL}/projects/{project_id}",
            headers=get_headers(api_token),
            timeout=30
        )
        if response.status_code in [200, 204]:
            return True, None
        return False, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return False, str(e)


# =============================================================================
# API Functions - Crawls
# =============================================================================

def get_crawls(api_token, workspace_id, project_id=None, status=None):
    """Fetch crawls for a workspace."""
    try:
        params = {}
        if project_id:
            params['filters[project_id]'] = project_id
        if status:
            params['filters[status]'] = status

        response = requests.get(
            f"{BASE_URL}/workspaces/{workspace_id}/crawls",
            headers=get_headers(api_token),
            params=params,
            timeout=30
        )

        if response.status_code == 200:
            crawls = response.json().get('crawls', [])
            crawls = sorted(crawls, key=lambda x: x.get('created_at', 0), reverse=True)
            return crawls, None
        return [], f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return [], str(e)


def get_crawl(api_token, crawl_id):
    """Get details of a specific crawl."""
    try:
        response = requests.get(
            f"{BASE_URL}/crawls/{crawl_id}",
            headers=get_headers(api_token),
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get('crawl', {}), None
        return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)


def launch_crawl(api_token, project_id, crawl_config_id=None):
    """Launch a new crawl."""
    try:
        data = {}
        if crawl_config_id:
            data['crawl_config_id'] = crawl_config_id

        response = requests.post(
            f"{BASE_URL}/projects/{project_id}/crawls",
            headers=get_headers(api_token),
            json=data,
            timeout=30
        )
        if response.status_code in [200, 201, 202]:
            return response.json().get('crawl', {}), None
        return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)


def update_crawl_state(api_token, crawl_id, action):
    """Update crawl state (pause, resume, cancel)."""
    try:
        response = requests.put(
            f"{BASE_URL}/crawls/{crawl_id}",
            headers=get_headers(api_token),
            json={"action": action},
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get('crawl', {}), None
        return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)


def delete_crawl(api_token, crawl_id):
    """Delete a crawl."""
    try:
        response = requests.delete(
            f"{BASE_URL}/crawls/{crawl_id}",
            headers=get_headers(api_token),
            timeout=30
        )
        if response.status_code in [200, 204]:
            return True, None
        return False, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return False, str(e)


# =============================================================================
# API Functions - Crawl Configurations
# =============================================================================

def get_crawl_configs(api_token, project_id):
    """Fetch crawl configurations for a project."""
    try:
        response = requests.get(
            f"{BASE_URL}/projects/{project_id}/crawl_configs",
            headers=get_headers(api_token),
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get('crawl_configs', []), None
        return [], f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return [], str(e)


# =============================================================================
# API Functions - Data Queries
# =============================================================================

def export_crawl_data(api_token, crawl_id, fields, oql_query, url_filter=None):
    """Export crawl data to CSV."""
    try:
        query = {
            "fields": fields,
            "oql": oql_query
        }

        if url_filter:
            query["oql"] = {
                "and": [
                    oql_query,
                    {"field": ["url", "contains", url_filter]}
                ]
            }

        response = requests.post(
            f"{BASE_URL}/data/crawl/{crawl_id}/pages?export=true",
            headers=get_headers(api_token),
            json=query,
            timeout=300
        )

        if response.status_code == 200:
            csv_content = response.content.decode('utf-8')
            df = pd.read_csv(StringIO(csv_content), sep=';', quotechar='"')
            return df, None
        return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)


def aggregate_crawl_data(api_token, crawl_id, group_by, oql_query=None):
    """Run aggregate query on crawl data."""
    try:
        query = {
            "agg": [
                {
                    "groupBy": [{"field": group_by}],
                    "metric": {"count": {"field": "url"}}
                }
            ]
        }

        if oql_query:
            query["oql"] = oql_query

        response = requests.post(
            f"{BASE_URL}/data/crawl/{crawl_id}/pages",
            headers=get_headers(api_token),
            json=query,
            timeout=120
        )

        if response.status_code == 200:
            data = response.json()
            return data.get('aggs', []), None
        return [], f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return [], str(e)


def export_links(api_token, crawl_id, link_type="internal"):
    """Export links from crawl."""
    try:
        endpoint = "links" if link_type == "internal" else "external_links"
        response = requests.post(
            f"{BASE_URL}/data/crawl/{crawl_id}/{endpoint}?export=true",
            headers=get_headers(api_token),
            json={"fields": ["source", "target", "anchor", "type", "follow"]},
            timeout=300
        )

        if response.status_code == 200:
            csv_content = response.content.decode('utf-8')
            df = pd.read_csv(StringIO(csv_content), sep=';', quotechar='"')
            return df, None
        return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)


# =============================================================================
# API Functions - Crawl Comparison
# =============================================================================

def export_crawl_comparison(api_token, crawl_id, reference_crawl_id, fields, oql_query=None, change_type=None):
    """Export crawl comparison data."""
    try:
        query = {
            "reference": reference_crawl_id,
            "fields": fields
        }

        if oql_query:
            query["oql"] = oql_query
        if change_type:
            query["change"] = change_type

        response = requests.post(
            f"{BASE_URL}/data/crawl/{crawl_id}/pages/coc?export=true",
            headers=get_headers(api_token),
            json=query,
            timeout=300
        )

        if response.status_code == 200:
            csv_content = response.content.decode('utf-8')
            df = pd.read_csv(StringIO(csv_content), sep=';', quotechar='"')
            return df, None
        return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)


# =============================================================================
# API Functions - Scheduling
# =============================================================================

def get_schedules(api_token, project_id):
    """Get schedules for a project."""
    try:
        response = requests.get(
            f"{BASE_URL}/projects/{project_id}/schedules",
            headers=get_headers(api_token),
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get('schedules', []), None
        return [], f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return [], str(e)


def create_schedule(api_token, project_id, crawl_config_id, frequency, day_of_week=None, day_of_month=None, hour=0):
    """Create a crawl schedule."""
    try:
        data = {
            "crawl_config_id": crawl_config_id,
            "frequency": frequency,
            "hour": hour
        }
        if day_of_week is not None:
            data["day_of_week"] = day_of_week
        if day_of_month is not None:
            data["day_of_month"] = day_of_month

        response = requests.post(
            f"{BASE_URL}/projects/{project_id}/schedules",
            headers=get_headers(api_token),
            json=data,
            timeout=30
        )
        if response.status_code in [200, 201]:
            return response.json().get('schedule', {}), None
        return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)


def delete_schedule(api_token, schedule_id):
    """Delete a schedule."""
    try:
        response = requests.delete(
            f"{BASE_URL}/schedules/{schedule_id}",
            headers=get_headers(api_token),
            timeout=30
        )
        if response.status_code in [200, 204]:
            return True, None
        return False, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return False, str(e)


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OnCrawl API Suite - Comprehensive CLI for data extraction, crawl management, and analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available presets (use with --preset):
{chr(10).join(f'  {k}: {v["description"]}' for k, v in PRESET_QUERIES.items())}

Aggregate fields (use with --aggregate):
  {', '.join(AGGREGATE_FIELDS)}

Examples:
    # List resources
    %(prog)s --token TOKEN --list-workspaces
    %(prog)s --token TOKEN --workspace WS_ID --list-projects
    %(prog)s --token TOKEN --workspace WS_ID --project PROJ_ID --list-crawls

    # Data extraction
    %(prog)s --token TOKEN --crawl CRAWL_ID --preset 404-pages
    %(prog)s --token TOKEN --crawl CRAWL_ID --preset 301-redirects -o redirects.csv
    %(prog)s --token TOKEN --crawl CRAWL_ID --fields url,status_code --oql '{{"and":[{{"field":["status_code","equals",200]}}]}}'

    # Crawl management
    %(prog)s --token TOKEN --project PROJ_ID --launch-crawl
    %(prog)s --token TOKEN --crawl CRAWL_ID --pause-crawl
    %(prog)s --token TOKEN --crawl CRAWL_ID --resume-crawl
    %(prog)s --token TOKEN --crawl CRAWL_ID --cancel-crawl
    %(prog)s --token TOKEN --crawl CRAWL_ID --delete-crawl

    # Crawl comparison
    %(prog)s --token TOKEN --crawl CRAWL_ID --compare-to REF_CRAWL_ID
    %(prog)s --token TOKEN --crawl CRAWL_ID --compare-to REF_CRAWL_ID --change-type new

    # Aggregations
    %(prog)s --token TOKEN --crawl CRAWL_ID --aggregate status_code

    # Link export
    %(prog)s --token TOKEN --crawl CRAWL_ID --export-links internal

    # Project management
    %(prog)s --token TOKEN --workspace WS_ID --create-project --name "My Project" --start-url "https://example.com"
    %(prog)s --token TOKEN --project PROJ_ID --delete-project

    # Scheduling
    %(prog)s --token TOKEN --project PROJ_ID --list-schedules
    %(prog)s --token TOKEN --project PROJ_ID --create-schedule --config-id CFG_ID --frequency daily --hour 2
    %(prog)s --token TOKEN --schedule SCHED_ID --delete-schedule

Author: Lee Foot (https://leefoot.co.uk)
        """
    )

    # Authentication
    parser.add_argument("--token", required=True, help="OnCrawl API access token")

    # Resource selection
    parser.add_argument("--workspace", help="Workspace ID")
    parser.add_argument("--project", help="Project ID")
    parser.add_argument("--crawl", help="Crawl ID")
    parser.add_argument("--schedule", help="Schedule ID")
    parser.add_argument("--config-id", help="Crawl configuration ID")

    # Listing commands
    parser.add_argument("--list-workspaces", action="store_true", help="List all workspaces")
    parser.add_argument("--list-projects", action="store_true", help="List projects in workspace")
    parser.add_argument("--list-crawls", action="store_true", help="List crawls for project")
    parser.add_argument("--list-configs", action="store_true", help="List crawl configurations")
    parser.add_argument("--list-schedules", action="store_true", help="List schedules for project")
    parser.add_argument("--list-presets", action="store_true", help="List available preset queries")
    parser.add_argument("--crawl-status", choices=["all", "pending", "running", "paused", "done", "canceled", "failed"],
                        default="all", help="Filter crawls by status")

    # Data extraction
    parser.add_argument("--preset", help="Preset query name (e.g., 404-pages, 301-redirects)")
    parser.add_argument("--fields", help="Comma-separated list of fields to extract")
    parser.add_argument("--oql", help="OQL query as JSON string")
    parser.add_argument("--url-filter", help="Filter results to URLs containing this text")

    # Crawl management
    parser.add_argument("--launch-crawl", action="store_true", help="Launch a new crawl")
    parser.add_argument("--pause-crawl", action="store_true", help="Pause a running crawl")
    parser.add_argument("--resume-crawl", action="store_true", help="Resume a paused crawl")
    parser.add_argument("--cancel-crawl", action="store_true", help="Cancel a crawl")
    parser.add_argument("--delete-crawl", action="store_true", help="Delete a crawl")
    parser.add_argument("--crawl-info", action="store_true", help="Get crawl details")

    # Crawl comparison
    parser.add_argument("--compare-to", help="Reference crawl ID for comparison")
    parser.add_argument("--change-type", choices=["new", "lost", "changed", "unchanged"],
                        help="Filter comparison by change type")

    # Aggregations
    parser.add_argument("--aggregate", choices=AGGREGATE_FIELDS, help="Run aggregation by field")

    # Link export
    parser.add_argument("--export-links", choices=["internal", "external"], help="Export links")

    # Project management
    parser.add_argument("--create-project", action="store_true", help="Create a new project")
    parser.add_argument("--delete-project", action="store_true", help="Delete a project")
    parser.add_argument("--name", help="Project name (for --create-project)")
    parser.add_argument("--start-url", help="Start URL (for --create-project)")
    parser.add_argument("--user-agent", default="oncrawl", help="User agent (for --create-project)")

    # Scheduling
    parser.add_argument("--create-schedule", action="store_true", help="Create a schedule")
    parser.add_argument("--delete-schedule", action="store_true", help="Delete a schedule")
    parser.add_argument("--frequency", choices=["daily", "weekly", "monthly"], help="Schedule frequency")
    parser.add_argument("--hour", type=int, default=2, help="Hour (UTC) for scheduled crawl")
    parser.add_argument("--day-of-week", type=int, choices=range(7), help="Day of week (0=Sun, 1=Mon, etc.)")
    parser.add_argument("--day-of-month", type=int, choices=range(1, 29), help="Day of month (1-28)")

    # Output options
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--format", choices=["csv", "xlsx", "json"], default="csv", help="Output format")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    # ==========================================================================
    # List Presets
    # ==========================================================================
    if args.list_presets:
        print("Available preset queries:\n")
        for key, config in PRESET_QUERIES.items():
            print(f"  {key}")
            print(f"    {config['description']}")
            print(f"    Fields: {', '.join(config['fields'])}")
            print()
        return

    # ==========================================================================
    # List Workspaces
    # ==========================================================================
    if args.list_workspaces:
        workspaces, error = get_workspaces(args.token)
        if error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)
        if workspaces:
            print("Workspaces:\n")
            for ws in workspaces:
                print(f"  ID: {ws['id']}")
                print(f"  Name: {ws['name']}")
                if ws.get('crawl_pages_limit'):
                    print(f"  Pages: {ws.get('crawl_pages_used', 0):,} / {ws['crawl_pages_limit']:,}")
                print()
        else:
            print("No workspaces found.")
        return

    # ==========================================================================
    # List Projects
    # ==========================================================================
    if args.list_projects:
        if not args.workspace:
            print("Error: --workspace required for --list-projects", file=sys.stderr)
            sys.exit(1)

        projects, error = get_projects(args.token, args.workspace)
        if error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)
        if projects:
            print(f"Projects in workspace {args.workspace}:\n")
            for p in projects:
                print(f"  ID: {p['id']}")
                print(f"  Name: {p['name']}")
                print(f"  URL: {p.get('start_url', 'N/A')}")
                print()
        else:
            print("No projects found.")
        return

    # ==========================================================================
    # List Crawls
    # ==========================================================================
    if args.list_crawls:
        if not args.workspace or not args.project:
            print("Error: --workspace and --project required for --list-crawls", file=sys.stderr)
            sys.exit(1)

        status_filter = None if args.crawl_status == "all" else args.crawl_status
        crawls, error = get_crawls(args.token, args.workspace, args.project, status_filter)
        if error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)
        if crawls:
            print(f"Crawls for project {args.project}:\n")
            for c in crawls[:20]:
                crawl_date = datetime.fromtimestamp(c.get('created_at', 0) / 1000).strftime('%Y-%m-%d %H:%M')
                print(f"  ID: {c['id']}")
                print(f"  Date: {crawl_date}")
                print(f"  Status: {c.get('status', 'unknown')}")
                print(f"  URLs: {c.get('fetched_urls', 'N/A')}")
                print()
        else:
            print("No crawls found.")
        return

    # ==========================================================================
    # List Configs
    # ==========================================================================
    if args.list_configs:
        if not args.project:
            print("Error: --project required for --list-configs", file=sys.stderr)
            sys.exit(1)

        configs, error = get_crawl_configs(args.token, args.project)
        if error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)
        if configs:
            print(f"Crawl configurations:\n")
            for c in configs:
                print(f"  ID: {c['id']}")
                print(f"  Name: {c.get('name', 'Unnamed')}")
                print()
        else:
            print("No configurations found.")
        return

    # ==========================================================================
    # List Schedules
    # ==========================================================================
    if args.list_schedules:
        if not args.project:
            print("Error: --project required for --list-schedules", file=sys.stderr)
            sys.exit(1)

        schedules, error = get_schedules(args.token, args.project)
        if error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)
        if schedules:
            print(f"Schedules:\n")
            for s in schedules:
                print(f"  ID: {s['id']}")
                print(f"  Frequency: {s.get('frequency', 'unknown')}")
                print(f"  Hour: {s.get('hour', 0)}:00 UTC")
                print()
        else:
            print("No schedules found.")
        return

    # ==========================================================================
    # Crawl Info
    # ==========================================================================
    if args.crawl_info:
        if not args.crawl:
            print("Error: --crawl required for --crawl-info", file=sys.stderr)
            sys.exit(1)

        crawl, error = get_crawl(args.token, args.crawl)
        if error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(crawl, indent=2))
        return

    # ==========================================================================
    # Crawl Management
    # ==========================================================================
    if args.launch_crawl:
        if not args.project:
            print("Error: --project required for --launch-crawl", file=sys.stderr)
            sys.exit(1)

        crawl, error = launch_crawl(args.token, args.project, args.config_id)
        if error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)
        print(f"Crawl launched! ID: {crawl.get('id')}")
        return

    if args.pause_crawl:
        if not args.crawl:
            print("Error: --crawl required for --pause-crawl", file=sys.stderr)
            sys.exit(1)

        crawl, error = update_crawl_state(args.token, args.crawl, "pause")
        if error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)
        print(f"Crawl paused. Status: {crawl.get('status')}")
        return

    if args.resume_crawl:
        if not args.crawl:
            print("Error: --crawl required for --resume-crawl", file=sys.stderr)
            sys.exit(1)

        crawl, error = update_crawl_state(args.token, args.crawl, "resume")
        if error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)
        print(f"Crawl resumed. Status: {crawl.get('status')}")
        return

    if args.cancel_crawl:
        if not args.crawl:
            print("Error: --crawl required for --cancel-crawl", file=sys.stderr)
            sys.exit(1)

        crawl, error = update_crawl_state(args.token, args.crawl, "cancel")
        if error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)
        print(f"Crawl canceled. Status: {crawl.get('status')}")
        return

    if args.delete_crawl:
        if not args.crawl:
            print("Error: --crawl required for --delete-crawl", file=sys.stderr)
            sys.exit(1)

        success, error = delete_crawl(args.token, args.crawl)
        if error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)
        print("Crawl deleted.")
        return

    # ==========================================================================
    # Project Management
    # ==========================================================================
    if args.create_project:
        if not args.workspace or not args.name or not args.start_url:
            print("Error: --workspace, --name, and --start-url required for --create-project", file=sys.stderr)
            sys.exit(1)

        project, error = create_project(args.token, args.workspace, args.name, args.start_url, args.user_agent)
        if error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)
        print(f"Project created! ID: {project.get('id')}")
        return

    if args.delete_project:
        if not args.project:
            print("Error: --project required for --delete-project", file=sys.stderr)
            sys.exit(1)

        success, error = delete_project(args.token, args.project)
        if error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)
        print("Project deleted.")
        return

    # ==========================================================================
    # Scheduling
    # ==========================================================================
    if args.create_schedule:
        if not args.project or not args.config_id or not args.frequency:
            print("Error: --project, --config-id, and --frequency required for --create-schedule", file=sys.stderr)
            sys.exit(1)

        schedule, error = create_schedule(
            args.token, args.project, args.config_id,
            args.frequency, args.day_of_week, args.day_of_month, args.hour
        )
        if error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)
        print(f"Schedule created! ID: {schedule.get('id')}")
        return

    if args.delete_schedule:
        if not args.schedule:
            print("Error: --schedule required for --delete-schedule", file=sys.stderr)
            sys.exit(1)

        success, error = delete_schedule(args.token, args.schedule)
        if error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)
        print("Schedule deleted.")
        return

    # ==========================================================================
    # Aggregations
    # ==========================================================================
    if args.aggregate:
        if not args.crawl:
            print("Error: --crawl required for --aggregate", file=sys.stderr)
            sys.exit(1)

        if not args.quiet:
            print(f"Running aggregation by {args.aggregate}...")

        aggs, error = aggregate_crawl_data(args.token, args.crawl, args.aggregate)
        if error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)

        results = []
        for agg in aggs:
            for bucket in agg.get('buckets', []):
                results.append({
                    args.aggregate: bucket.get('key', 'N/A'),
                    'count': bucket.get('metrics', {}).get('count', 0)
                })

        if results:
            df = pd.DataFrame(results)
            df = df.sort_values('count', ascending=False)

            if args.output:
                if args.format == "csv":
                    df.to_csv(args.output, index=False)
                elif args.format == "xlsx":
                    df.to_excel(args.output, index=False)
                elif args.format == "json":
                    df.to_json(args.output, orient="records", indent=2)
                if not args.quiet:
                    print(f"Output saved to: {args.output}")
            else:
                print(df.to_string(index=False))
        else:
            print("No aggregation data.")
        return

    # ==========================================================================
    # Link Export
    # ==========================================================================
    if args.export_links:
        if not args.crawl:
            print("Error: --crawl required for --export-links", file=sys.stderr)
            sys.exit(1)

        if not args.quiet:
            print(f"Exporting {args.export_links} links...")

        df, error = export_links(args.token, args.crawl, args.export_links)
        if error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)

        if df is not None and not df.empty:
            if not args.quiet:
                print(f"Exported {len(df):,} links")

            if args.output:
                output_file = args.output
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"oncrawl_{args.export_links}_links_{timestamp}.{args.format}"

            if args.format == "csv":
                df.to_csv(output_file, index=False)
            elif args.format == "xlsx":
                df.to_excel(output_file, index=False)
            elif args.format == "json":
                df.to_json(output_file, orient="records", indent=2)

            if not args.quiet:
                print(f"Output saved to: {output_file}")
        else:
            print("No links found.")
        return

    # ==========================================================================
    # Crawl Comparison
    # ==========================================================================
    if args.compare_to:
        if not args.crawl:
            print("Error: --crawl required for --compare-to", file=sys.stderr)
            sys.exit(1)

        fields = ["url", "status_code", "title", "depth"]
        if args.fields:
            fields = [f.strip() for f in args.fields.split(",")]

        if not args.quiet:
            print(f"Comparing crawl {args.crawl} to {args.compare_to}...")

        df, error = export_crawl_comparison(
            args.token, args.crawl, args.compare_to,
            fields, change_type=args.change_type
        )

        if error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)

        if df is not None and not df.empty:
            if not args.quiet:
                print(f"Found {len(df):,} pages")

            if args.output:
                output_file = args.output
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"oncrawl_comparison_{timestamp}.{args.format}"

            if args.format == "csv":
                df.to_csv(output_file, index=False)
            elif args.format == "xlsx":
                df.to_excel(output_file, index=False)
            elif args.format == "json":
                df.to_json(output_file, orient="records", indent=2)

            if not args.quiet:
                print(f"Output saved to: {output_file}")
        else:
            print("No comparison data found.")
        return

    # ==========================================================================
    # Data Extraction (Preset or Custom)
    # ==========================================================================
    if args.preset or args.fields:
        if not args.crawl:
            print("Error: --crawl required for data extraction", file=sys.stderr)
            sys.exit(1)

        if args.preset:
            preset_key = args.preset.lower().replace(" ", "-")

            if preset_key not in PRESET_QUERIES:
                print(f"Error: Unknown preset '{args.preset}'", file=sys.stderr)
                print(f"Available presets: {', '.join(PRESET_QUERIES.keys())}", file=sys.stderr)
                sys.exit(1)

            preset_config = PRESET_QUERIES[preset_key]
            fields = preset_config['fields']
            oql_query = preset_config['oql']

            if not args.quiet:
                print(f"Extracting: {preset_config['name']}")
        else:
            if not args.fields:
                print("Error: --fields required for custom extraction", file=sys.stderr)
                sys.exit(1)

            fields = [f.strip() for f in args.fields.split(",")]

            if args.oql:
                try:
                    oql_query = json.loads(args.oql)
                except json.JSONDecodeError as e:
                    print(f"Error: Invalid OQL JSON - {e}", file=sys.stderr)
                    sys.exit(1)
            else:
                oql_query = {"and": [{"field": ["fetched", "equals", True]}]}

            if not args.quiet:
                print("Extracting custom query...")

        if not args.quiet:
            print(f"Crawl ID: {args.crawl}")
            if args.url_filter:
                print(f"URL filter: {args.url_filter}")

        df, error = export_crawl_data(args.token, args.crawl, fields, oql_query, args.url_filter)

        if error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)

        if df is None or df.empty:
            print("No data found matching the query.")
            return

        if not args.quiet:
            print(f"Extracted {len(df):,} rows")

        # Generate output filename
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if args.preset:
                output_file = f"oncrawl_{preset_key}_{timestamp}.{args.format}"
            else:
                output_file = f"oncrawl_custom_{timestamp}.{args.format}"

        # Save output
        if args.format == "csv":
            df.to_csv(output_file, index=False)
        elif args.format == "xlsx":
            df.to_excel(output_file, index=False, sheet_name='Data')
        elif args.format == "json":
            df.to_json(output_file, orient="records", indent=2)

        if not args.quiet:
            print(f"Output saved to: {output_file}")

        return

    # If no action specified, show help
    parser.print_help()


if __name__ == "__main__":
    main()
