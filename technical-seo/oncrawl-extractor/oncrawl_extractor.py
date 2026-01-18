"""
OnCrawl API Suite - Comprehensive Streamlit App
Full-featured OnCrawl API client for data extraction, crawl management, and analysis.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from io import StringIO
import io
import json
import time

st.set_page_config(
    page_title="OnCrawl API Suite",
    page_icon="🕷️",
    layout="wide"
)

st.title("🕷️ OnCrawl API Suite")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Connects directly to OnCrawl via API (no file uploads needed)
    - Extract crawl data using 35+ preset SEO queries or custom OQL
    - Launch, pause, resume and manage crawls
    - Compare crawls to track changes over time
    - Export internal/external link data
    - Schedule automated crawls

    **Getting your API token:**
    1. Log in to [app.oncrawl.com](https://app.oncrawl.com)
    2. Click your profile icon (top right) → **Account Settings**
    3. Scroll to **API Access Tokens** section
    4. Click **Generate new token** (or copy an existing one)
    5. Paste the token in the sidebar of this app

    **Using the app:**
    1. Enter your API token in the sidebar
    2. Select your workspace and project from the dropdowns
    3. Choose a module from the sidebar navigation
    4. Run queries and export data as CSV/Excel

    **Available modules:**
    - **Data Extraction**: 35+ preset queries or custom OQL builder
    - **Crawl Management**: Launch, pause, resume, cancel crawls
    - **Aggregations**: Group data by status code, depth, URL segments
    - **Crawl Comparison**: Find new, lost, and changed pages between crawls
    - **Project Management**: Create/delete projects, monitor quotas
    - **Scheduling**: Set up automated crawls
    - **Link Export**: Export internal or external links
    """)
st.markdown("Comprehensive OnCrawl API client for data extraction, crawl management, and analysis.")

# Constants
BASE_URL = "https://app.oncrawl.com/api/v2"

# Comprehensive field list based on API documentation
AVAILABLE_FIELDS = {
    "Core": [
        "url", "urlpath", "urlsegment1", "urlsegment2", "urlsegment3", "urlsegment4",
        "status_code", "content_type", "fetched", "fetch_date"
    ],
    "SEO Metadata": [
        "title", "title_length", "title_duplicates_count",
        "meta_description", "meta_description_length", "meta_description_duplicates_count",
        "h1", "h1_count", "h2_count", "h3_count",
        "canonical", "canonical_evaluation", "canonical_is_http", "canonical_is_relative"
    ],
    "Indexability": [
        "meta_robots_index", "meta_robots_follow", "meta_robots_noarchive", "meta_robots_nosnippet",
        "robots_txt_denied", "x_robots_tag_index", "x_robots_tag_follow",
        "in_sitemap", "in_sitemap_count"
    ],
    "Redirects": [
        "redirect_location", "final_redirect_location", "redirect_count",
        "is_redirect_loop", "redirect_status_code"
    ],
    "Links": [
        "nb_inlinks", "nb_outlinks", "nb_external_outlinks", "nb_internal_outlinks",
        "nb_follow_inlinks", "nb_nofollow_inlinks",
        "nb_follow_outlinks", "nb_nofollow_outlinks",
        "inrank", "depth"
    ],
    "Performance": [
        "delay_total", "delay_first_byte", "delay_last_byte",
        "size_total", "size_html", "size_download"
    ],
    "Content": [
        "word_count", "text_ratio", "lang", "charset",
        "content_encoding", "content_hash"
    ],
    "Structured Data": [
        "has_microdata", "has_jsonld", "has_opengraph", "has_twitter_card",
        "has_schema_org", "schema_org_types"
    ],
    "Images": [
        "nb_images", "nb_images_without_alt", "nb_images_with_empty_alt"
    ],
    "URL Properties": [
        "url_length", "url_has_params", "url_params_count",
        "url_has_uppercase", "url_has_underscores"
    ],
    "Hreflang": [
        "hreflang_keys", "hreflang_values", "has_hreflang", "hreflang_self_referencing"
    ],
    "AMP": [
        "is_amp", "amphtml_url", "amphtml_valid"
    ],
    "Mobile": [
        "is_mobile_friendly", "viewport_meta"
    ],
    "Security": [
        "is_https", "has_mixed_content"
    ]
}

# Flatten for multiselect
ALL_FIELDS = []
for category, fields in AVAILABLE_FIELDS.items():
    ALL_FIELDS.extend(fields)

# Preset queries - expanded with more use cases
PRESET_QUERIES = {
    # Status Code Issues
    "404 Pages": {
        "description": "All pages returning 404 status code",
        "fields": ["url", "status_code", "depth", "inrank", "nb_inlinks"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 404]}]}
    },
    "5xx Server Errors": {
        "description": "All pages with server errors (500-599)",
        "fields": ["url", "status_code", "depth", "nb_inlinks"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "between", [500, 599]]}]}
    },
    "4xx Client Errors": {
        "description": "All pages with client errors (400-499)",
        "fields": ["url", "status_code", "depth", "nb_inlinks"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "between", [400, 499]]}]}
    },
    # Redirects
    "301 Redirects": {
        "description": "All 301 permanent redirects",
        "fields": ["url", "status_code", "redirect_location", "final_redirect_location", "redirect_count"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 301]}]}
    },
    "302 Redirects": {
        "description": "All 302 temporary redirects",
        "fields": ["url", "status_code", "redirect_location", "final_redirect_location", "redirect_count"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 302]}]}
    },
    "Redirect Chains": {
        "description": "Pages with multiple redirects (>1)",
        "fields": ["url", "status_code", "redirect_count", "redirect_location", "final_redirect_location"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["redirect_count", "gt", 1]}]}
    },
    "Redirect Loops": {
        "description": "Pages caught in redirect loops",
        "fields": ["url", "status_code", "redirect_count", "redirect_location", "final_redirect_location"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["is_redirect_loop", "equals", True]}]}
    },
    "Stale Links (3xx with Inlinks)": {
        "description": "Redirecting pages still receiving internal links",
        "fields": ["url", "status_code", "redirect_location", "nb_inlinks", "inrank"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "between", [300, 399]]}, {"field": ["nb_inlinks", "gt", 0]}]}
    },
    # Indexability
    "Indexable Pages": {
        "description": "All indexable pages",
        "fields": ["url", "status_code", "meta_robots_index", "canonical_evaluation", "title", "h1"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["meta_robots_index", "equals", True]}, {"field": ["robots_txt_denied", "equals", False]}, {"field": ["canonical_evaluation", "equals", "matching"]}]}
    },
    "Non-Indexable Pages": {
        "description": "Pages blocked from indexing",
        "fields": ["url", "status_code", "meta_robots_index", "robots_txt_denied", "canonical_evaluation"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"or": [{"field": ["meta_robots_index", "equals", False]}, {"field": ["robots_txt_denied", "equals", True]}, {"field": ["canonical_evaluation", "equals", "not_matching"]}]}]}
    },
    "Noindex Pages": {
        "description": "Pages with noindex directive",
        "fields": ["url", "status_code", "meta_robots_index", "x_robots_tag_index", "title"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["meta_robots_index", "equals", False]}]}
    },
    "Robots.txt Blocked": {
        "description": "Pages blocked by robots.txt",
        "fields": ["url", "status_code", "robots_txt_denied", "nb_inlinks"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["robots_txt_denied", "equals", True]}]}
    },
    # Canonical Issues
    "Non-Matching Canonicals": {
        "description": "Pages where canonical doesn't match URL",
        "fields": ["url", "canonical", "canonical_evaluation", "status_code"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["canonical_evaluation", "equals", "not_matching"]}]}
    },
    "Missing Canonicals": {
        "description": "Pages without canonical tags",
        "fields": ["url", "canonical", "status_code", "title"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["canonical", "is_empty", True]}]}
    },
    # Content Issues
    "Pages Without Title": {
        "description": "Pages missing title tags",
        "fields": ["url", "status_code", "title", "h1"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["title", "is_empty", True]}]}
    },
    "Pages Without H1": {
        "description": "Pages missing H1 tags",
        "fields": ["url", "status_code", "title", "h1"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["h1", "is_empty", True]}]}
    },
    "Pages Without Description": {
        "description": "Pages missing meta descriptions",
        "fields": ["url", "status_code", "title", "meta_description"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["meta_description", "is_empty", True]}]}
    },
    "Duplicate Titles": {
        "description": "Pages with duplicate titles",
        "fields": ["url", "status_code", "title", "title_duplicates_count"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["title_duplicates_count", "gt", 1]}]}
    },
    "Duplicate Descriptions": {
        "description": "Pages with duplicate meta descriptions",
        "fields": ["url", "status_code", "meta_description", "meta_description_duplicates_count"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["meta_description_duplicates_count", "gt", 1]}]}
    },
    "Short Titles (<30 chars)": {
        "description": "Pages with very short titles",
        "fields": ["url", "title", "title_length", "status_code"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["title_length", "lt", 30]}, {"field": ["title_length", "gt", 0]}]}
    },
    "Long Titles (>60 chars)": {
        "description": "Pages with overly long titles",
        "fields": ["url", "title", "title_length", "status_code"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["title_length", "gt", 60]}]}
    },
    "Thin Content (<300 words)": {
        "description": "Pages with thin content",
        "fields": ["url", "word_count", "title", "status_code"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["word_count", "lt", 300]}]}
    },
    # Link Structure
    "Orphan Pages": {
        "description": "Pages with no internal links",
        "fields": ["url", "status_code", "nb_inlinks", "depth"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["nb_inlinks", "equals", 0]}]}
    },
    "Deep Pages (Depth > 5)": {
        "description": "Pages more than 5 clicks deep",
        "fields": ["url", "status_code", "depth", "nb_inlinks"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["depth", "gt", 5]}]}
    },
    "Low Inrank Pages": {
        "description": "Important pages with low internal PageRank",
        "fields": ["url", "inrank", "nb_inlinks", "depth", "status_code"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["inrank", "lt", 0.1]}]}
    },
    "High Outlink Pages": {
        "description": "Pages with many outgoing links (>100)",
        "fields": ["url", "nb_outlinks", "nb_internal_outlinks", "nb_external_outlinks"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["nb_outlinks", "gt", 100]}]}
    },
    # Performance
    "Slow Pages (>3s)": {
        "description": "Pages loading over 3 seconds",
        "fields": ["url", "status_code", "delay_total", "delay_first_byte", "size_total"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["delay_total", "gt", 3000]}]}
    },
    "Very Slow Pages (>5s)": {
        "description": "Pages loading over 5 seconds",
        "fields": ["url", "status_code", "delay_total", "delay_first_byte", "size_total"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["delay_total", "gt", 5000]}]}
    },
    "Large Pages (>1MB)": {
        "description": "Pages over 1MB in size",
        "fields": ["url", "size_total", "size_html", "delay_total"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["size_total", "gt", 1000000]}]}
    },
    "Slow TTFB (>1s)": {
        "description": "Pages with slow Time to First Byte",
        "fields": ["url", "delay_first_byte", "delay_total", "status_code"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["delay_first_byte", "gt", 1000]}]}
    },
    # Images
    "Images Without Alt": {
        "description": "Pages with images missing alt text",
        "fields": ["url", "nb_images", "nb_images_without_alt", "status_code"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["nb_images_without_alt", "gt", 0]}]}
    },
    # Sitemap
    "Not in Sitemap": {
        "description": "Indexable pages not in sitemap",
        "fields": ["url", "in_sitemap", "meta_robots_index", "canonical_evaluation"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["meta_robots_index", "equals", True]}, {"field": ["in_sitemap", "equals", False]}]}
    },
    "In Sitemap (Non-200)": {
        "description": "Non-200 pages found in sitemap",
        "fields": ["url", "status_code", "in_sitemap"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["in_sitemap", "equals", True]}, {"field": ["status_code", "not_equals", 200]}]}
    },
    # HTTPS/Security
    "HTTP Pages": {
        "description": "Non-HTTPS pages",
        "fields": ["url", "is_https", "status_code"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["is_https", "equals", False]}]}
    },
    "Mixed Content": {
        "description": "HTTPS pages with mixed content",
        "fields": ["url", "is_https", "has_mixed_content", "status_code"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}, {"field": ["has_mixed_content", "equals", True]}]}
    },
    # All Pages
    "All Fetched Pages": {
        "description": "Complete list of all crawled pages",
        "fields": ["url", "status_code", "depth", "nb_inlinks", "inrank"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}]}
    },
    "All 200 Pages": {
        "description": "All pages returning 200 OK",
        "fields": ["url", "title", "h1", "depth", "nb_inlinks", "word_count"],
        "oql": {"and": [{"field": ["fetched", "equals", True]}, {"field": ["status_code", "equals", 200]}]}
    }
}

# Aggregate field options for groupBy
AGGREGATE_FIELDS = [
    "status_code", "depth", "urlsegment1", "urlsegment2", "urlsegment3",
    "content_type", "canonical_evaluation", "meta_robots_index", "in_sitemap",
    "lang", "charset"
]

# Crawl status options
CRAWL_STATUSES = ["pending", "running", "paused", "done", "canceled", "failed", "waiting_validation"]


def get_headers(api_token):
    """Return headers for API authentication."""
    return {
        'Authorization': f'Bearer {api_token}',
        'Content-Type': 'application/json'
    }


# =============================================================================
# API Functions - Account & Workspaces
# =============================================================================

@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
def get_workspaces(api_token):
    """Fetch all workspaces for the account."""
    try:
        # Fetch workspaces directly from /workspaces endpoint
        response = requests.get(
            f"{BASE_URL}/workspaces",
            headers=get_headers(api_token),
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            ws_list = data.get('workspaces', [])
            workspaces = []

            for ws_data in ws_list:
                workspaces.append({
                    'id': ws_data.get('id'),
                    'name': ws_data.get('name', ws_data.get('id', 'Unknown')),
                    'crawl_pages_limit': ws_data.get('crawl_pages_limit'),
                    'crawl_pages_used': ws_data.get('crawl_pages_used')
                })
            return workspaces, None
        else:
            return [], f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return [], str(e)


# =============================================================================
# API Functions - Projects
# =============================================================================

@st.cache_data(ttl=300)
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
                break

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


def get_crawl_config(api_token, config_id):
    """Get a specific crawl configuration."""
    try:
        response = requests.get(
            f"{BASE_URL}/crawl_configs/{config_id}",
            headers=get_headers(api_token),
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get('crawl_config', {}), None
        return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)


# =============================================================================
# API Functions - Crawls
# =============================================================================

@st.cache_data(ttl=60)
def get_crawls(api_token, workspace_id, project_id=None, status=None):
    """Fetch crawls for a workspace, optionally filtered by project and status."""
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
    """Launch a new crawl for a project."""
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
# API Functions - Data Queries
# =============================================================================

@st.cache_data(ttl=600)
def get_crawl_fields(api_token, crawl_id):
    """Get available fields for a crawl."""
    try:
        response = requests.get(
            f"{BASE_URL}/data/crawl/{crawl_id}/pages/fields",
            headers=get_headers(api_token),
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            fields = data.get('fields', [])
            # Return dict of field_name -> field_info
            return {f.get('name'): f for f in fields}, None
        return {}, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return {}, str(e)


def search_crawl_data(api_token, crawl_id, fields, oql_query, url_filter=None, limit=1000, offset=0):
    """Search crawl data with pagination."""
    try:
        query = {
            "fields": fields,
            "oql": oql_query,
            "limit": limit,
            "offset": offset
        }

        if url_filter:
            query["oql"] = {
                "and": [
                    oql_query,
                    {"field": ["url", "contains", url_filter]}
                ]
            }

        response = requests.post(
            f"{BASE_URL}/data/crawl/{crawl_id}/pages",
            headers=get_headers(api_token),
            json=query,
            timeout=120
        )

        if response.status_code == 200:
            data = response.json()
            # API returns 'urls' not 'pages', and 'total_hits' in meta
            urls = data.get('urls', data.get('pages', []))
            meta = data.get('meta', {})
            # Normalize meta to use 'total' key
            if 'total_hits' in meta:
                meta['total'] = meta['total_hits']
            return urls, meta, None
        return [], {}, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return [], {}, str(e)


def export_crawl_data(api_token, crawl_id, fields, oql_query, url_filter=None):
    """Export full crawl data to CSV."""
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
            # Try semicolon separator first (OnCrawl default), then comma
            try:
                df = pd.read_csv(StringIO(csv_content), sep=';', quotechar='"')
                if len(df.columns) == 1 and ',' in df.columns[0]:
                    # Likely comma-separated, retry
                    df = pd.read_csv(StringIO(csv_content), sep=',', quotechar='"')
            except Exception:
                df = pd.read_csv(StringIO(csv_content), sep=',', quotechar='"')
            return df, None
        return None, f"Error {response.status_code}: {response.text[:500]}"
    except Exception as e:
        return None, f"Exception: {str(e)}"


def aggregate_crawl_data(api_token, crawl_id, group_by, oql_query=None, agg_type="count"):
    """Run aggregate query on crawl data."""
    try:
        # Use the /pages/aggs endpoint with correct format
        agg_config = {
            "fields": [{"name": group_by}],
            "value": "url:count"
        }

        if oql_query:
            agg_config["oql"] = oql_query

        query = {"aggs": [agg_config]}

        response = requests.post(
            f"{BASE_URL}/data/crawl/{crawl_id}/pages/aggs",
            headers=get_headers(api_token),
            json=query,
            timeout=120
        )

        if response.status_code == 200:
            data = response.json()
            # Convert new format (cols/rows) to old format (buckets) for compatibility
            aggs = data.get('aggs', [])
            result = []
            for agg in aggs:
                cols = agg.get('cols', [])
                rows = agg.get('rows', [])
                buckets = []
                for row in rows:
                    if len(row) >= 2:
                        buckets.append({
                            'key': row[0],
                            'metrics': {'count': row[1]}
                        })
                result.append({'buckets': buckets})
            return result, None
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

def compare_crawls(api_token, crawl_id, reference_crawl_id, fields, oql_query=None, change_type=None):
    """Compare two crawls (Crawl over Crawl)."""
    try:
        query = {
            "reference": reference_crawl_id,
            "fields": fields
        }

        if oql_query:
            query["oql"] = oql_query
        if change_type:
            query["change"] = change_type  # "new", "lost", "changed", "unchanged"

        response = requests.post(
            f"{BASE_URL}/data/crawl/{crawl_id}/pages/coc",
            headers=get_headers(api_token),
            json=query,
            timeout=120
        )

        if response.status_code == 200:
            data = response.json()
            return data.get('pages', []), data.get('meta', {}), None
        return [], {}, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return [], {}, str(e)


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
            "frequency": frequency,  # "daily", "weekly", "monthly"
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
# Sidebar - API Configuration
# =============================================================================

with st.sidebar:
    st.header("🔑 API Configuration")

    api_token = st.text_input(
        "OnCrawl API Token",
        type="password",
        help="Your OnCrawl API access token"
    )

    if api_token:
        st.success("Token provided")

        # Navigation
        st.header("📍 Navigation")
        page = st.radio(
            "Select Module",
            options=[
                "📊 Data Extraction",
                "🔄 Crawl Management",
                "📈 Aggregations",
                "🔀 Crawl Comparison",
                "📁 Project Management",
                "⏰ Scheduling",
                "🔗 Link Export"
            ]
        )
    else:
        st.info("Enter your API token to get started")
        page = None

# =============================================================================
# Main Content
# =============================================================================

if api_token and page:

    # =========================================================================
    # DATA EXTRACTION PAGE
    # =========================================================================
    if page == "📊 Data Extraction":
        st.header("📊 Data Extraction")

        workspaces, ws_error = get_workspaces(api_token)

        if ws_error:
            st.error(ws_error)
        elif workspaces:
            # Workspace selection
            col1, col2, col3 = st.columns(3)

            with col1:
                workspace_options = {ws['name']: ws['id'] for ws in workspaces}
                selected_ws_name = st.selectbox("Workspace", list(workspace_options.keys()))
                selected_ws_id = workspace_options[selected_ws_name]

            with col2:
                projects, _ = get_projects(api_token, selected_ws_id)
                if projects:
                    project_options = {p['name']: p['id'] for p in projects}
                    selected_proj_name = st.selectbox("Project", list(project_options.keys()))
                    selected_proj_id = project_options[selected_proj_name]
                else:
                    st.warning("No projects found")
                    selected_proj_id = None

            with col3:
                if selected_proj_id:
                    crawls, _ = get_crawls(api_token, selected_ws_id, selected_proj_id, status="done")
                    if crawls:
                        crawl_options = {}
                        for c in crawls:
                            crawl_date = datetime.fromtimestamp(c.get('created_at', 0) / 1000).strftime('%Y-%m-%d %H:%M')
                            urls_count = c.get('fetched_urls', 'N/A')
                            label = f"{crawl_date} ({urls_count} URLs)"
                            crawl_options[label] = c['id']
                        selected_crawl_label = st.selectbox("Crawl", list(crawl_options.keys()))
                        selected_crawl_id = crawl_options[selected_crawl_label]
                    else:
                        st.warning("No completed crawls")
                        selected_crawl_id = None
                else:
                    selected_crawl_id = None

            if selected_crawl_id:
                st.divider()

                # Query type selection
                query_type = st.radio("Query Type", ["Preset Queries", "Custom Query"], horizontal=True)

                if query_type == "Preset Queries":
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        selected_preset = st.selectbox("Select Preset", list(PRESET_QUERIES.keys()))

                    with col2:
                        url_filter = st.text_input("URL Filter (optional)", placeholder="/products/")

                    preset_config = PRESET_QUERIES[selected_preset]
                    st.info(f"**{selected_preset}**: {preset_config['description']}")

                    with st.expander("Fields to extract"):
                        st.write(preset_config['fields'])

                    if st.button("Extract Data", type="primary"):
                        with st.spinner(f"Extracting {selected_preset}..."):
                            df, error = export_crawl_data(
                                api_token, selected_crawl_id,
                                preset_config['fields'], preset_config['oql'],
                                url_filter if url_filter else None
                            )

                        if error:
                            st.error(error)
                        elif df is not None and not df.empty:
                            st.success(f"Extracted {len(df):,} rows")
                            st.dataframe(df, use_container_width=True, height=400)

                            col1, col2 = st.columns(2)
                            with col1:
                                csv = df.to_csv(index=False)
                                st.download_button("Download CSV", csv,
                                    file_name=f"oncrawl_{selected_preset.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv")
                            with col2:
                                output = io.BytesIO()
                                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                    df.to_excel(writer, index=False, sheet_name='Data')
                                st.download_button("Download Excel", output.getvalue(),
                                    file_name=f"oncrawl_{selected_preset.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                        else:
                            st.warning("No data found matching the query.")

                else:  # Custom Query
                    st.subheader("Custom Query Builder")

                    # Field selection by category
                    st.markdown("**Select Fields**")
                    selected_fields = []

                    cols = st.columns(4)
                    for idx, (category, fields) in enumerate(AVAILABLE_FIELDS.items()):
                        with cols[idx % 4]:
                            with st.expander(category):
                                for field in fields:
                                    if st.checkbox(field, key=f"field_{field}"):
                                        selected_fields.append(field)

                    if selected_fields:
                        st.write(f"Selected: {', '.join(selected_fields)}")

                    url_filter = st.text_input("URL Filter (optional)", placeholder="/products/", key="custom_url_filter")

                    # OQL Condition builder
                    st.markdown("**OQL Conditions**")
                    num_conditions = st.number_input("Number of conditions", 1, 10, 1)

                    conditions = []
                    for i in range(int(num_conditions)):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            field = st.selectbox(f"Field {i+1}", ALL_FIELDS, key=f"cfield_{i}")
                        with col2:
                            operator = st.selectbox(f"Operator {i+1}",
                                ["equals", "not_equals", "contains", "not_contains", "gt", "lt", "gte", "lte", "is_empty", "between"],
                                key=f"cop_{i}")
                        with col3:
                            if operator == "is_empty":
                                value = st.selectbox(f"Value {i+1}", [True, False], key=f"cval_{i}")
                            elif operator == "between":
                                value = st.text_input(f"Value {i+1} (comma-sep)", placeholder="100,200", key=f"cval_{i}")
                            else:
                                value = st.text_input(f"Value {i+1}", key=f"cval_{i}")

                        if value is not None and value != "":
                            if operator == "between" and isinstance(value, str):
                                try:
                                    value = [int(v.strip()) for v in value.split(",")]
                                except:
                                    pass
                            elif isinstance(value, str) and value.isdigit():
                                value = int(value)
                            conditions.append({"field": [field, operator, value]})

                    if conditions and selected_fields:
                        custom_oql = {"and": conditions}

                        with st.expander("View OQL Query"):
                            st.json(custom_oql)

                        if st.button("Extract Custom Data", type="primary"):
                            with st.spinner("Extracting data..."):
                                df, error = export_crawl_data(
                                    api_token, selected_crawl_id,
                                    selected_fields, custom_oql,
                                    url_filter if url_filter else None
                                )

                            if error:
                                st.error(error)
                            elif df is not None and not df.empty:
                                st.success(f"Extracted {len(df):,} rows")
                                st.dataframe(df, use_container_width=True, height=400)

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.download_button("Download CSV", df.to_csv(index=False),
                                        file_name=f"oncrawl_custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv")
                                with col2:
                                    output = io.BytesIO()
                                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                        df.to_excel(writer, index=False)
                                    st.download_button("Download Excel", output.getvalue(),
                                        file_name=f"oncrawl_custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                            else:
                                st.warning("No data found.")
        else:
            st.warning("No workspaces found")

    # =========================================================================
    # CRAWL MANAGEMENT PAGE
    # =========================================================================
    elif page == "🔄 Crawl Management":
        st.header("🔄 Crawl Management")

        workspaces, ws_error = get_workspaces(api_token)

        if ws_error:
            st.error(ws_error)
        elif workspaces:
            col1, col2 = st.columns(2)

            with col1:
                workspace_options = {ws['name']: ws['id'] for ws in workspaces}
                selected_ws_name = st.selectbox("Workspace", list(workspace_options.keys()), key="cm_ws")
                selected_ws_id = workspace_options[selected_ws_name]

            with col2:
                projects, _ = get_projects(api_token, selected_ws_id)
                if projects:
                    project_options = {p['name']: p['id'] for p in projects}
                    selected_proj_name = st.selectbox("Project", list(project_options.keys()), key="cm_proj")
                    selected_proj_id = project_options[selected_proj_name]
                else:
                    st.warning("No projects found")
                    selected_proj_id = None

            if selected_proj_id:
                st.divider()

                tab1, tab2, tab3 = st.tabs(["📋 View Crawls", "🚀 Launch Crawl", "⚙️ Configurations"])

                with tab1:
                    status_filter = st.selectbox("Filter by Status", ["All"] + CRAWL_STATUSES)

                    # Clear cache to get fresh data
                    if st.button("🔄 Refresh"):
                        get_crawls.clear()

                    crawls, _ = get_crawls(api_token, selected_ws_id, selected_proj_id,
                                          status=None if status_filter == "All" else status_filter)

                    if crawls:
                        for crawl in crawls[:20]:  # Show last 20
                            crawl_date = datetime.fromtimestamp(crawl.get('created_at', 0) / 1000).strftime('%Y-%m-%d %H:%M')
                            status = crawl.get('status', 'unknown')
                            urls = crawl.get('fetched_urls', 'N/A')
                            crawl_id = crawl['id']

                            status_colors = {
                                'done': '🟢', 'running': '🔵', 'paused': '🟡',
                                'pending': '⚪', 'canceled': '🔴', 'failed': '🔴'
                            }

                            with st.expander(f"{status_colors.get(status, '⚪')} {crawl_date} - {status} ({urls} URLs)"):
                                st.write(f"**Crawl ID:** `{crawl_id}`")
                                st.write(f"**Status:** {status}")
                                st.write(f"**URLs Fetched:** {urls}")

                                col1, col2, col3, col4 = st.columns(4)

                                if status == 'running':
                                    with col1:
                                        if st.button("⏸️ Pause", key=f"pause_{crawl_id}"):
                                            _, err = update_crawl_state(api_token, crawl_id, "pause")
                                            if err:
                                                st.error(err)
                                            else:
                                                st.success("Paused")
                                                st.rerun()
                                    with col2:
                                        if st.button("❌ Cancel", key=f"cancel_{crawl_id}"):
                                            _, err = update_crawl_state(api_token, crawl_id, "cancel")
                                            if err:
                                                st.error(err)
                                            else:
                                                st.success("Canceled")
                                                st.rerun()

                                elif status == 'paused':
                                    with col1:
                                        if st.button("▶️ Resume", key=f"resume_{crawl_id}"):
                                            _, err = update_crawl_state(api_token, crawl_id, "resume")
                                            if err:
                                                st.error(err)
                                            else:
                                                st.success("Resumed")
                                                st.rerun()
                                    with col2:
                                        if st.button("❌ Cancel", key=f"cancel2_{crawl_id}"):
                                            _, err = update_crawl_state(api_token, crawl_id, "cancel")
                                            if err:
                                                st.error(err)
                                            else:
                                                st.success("Canceled")
                                                st.rerun()

                                if status in ['done', 'canceled', 'failed']:
                                    with col4:
                                        if st.button("🗑️ Delete", key=f"del_{crawl_id}"):
                                            success, err = delete_crawl(api_token, crawl_id)
                                            if err:
                                                st.error(err)
                                            else:
                                                st.success("Deleted")
                                                st.rerun()
                    else:
                        st.info("No crawls found")

                with tab2:
                    st.subheader("Launch New Crawl")

                    configs, _ = get_crawl_configs(api_token, selected_proj_id)

                    if configs:
                        config_options = {c.get('name', c['id']): c['id'] for c in configs}
                        selected_config = st.selectbox("Crawl Configuration", list(config_options.keys()))
                        selected_config_id = config_options[selected_config]
                    else:
                        st.info("Using default configuration")
                        selected_config_id = None

                    if st.button("🚀 Launch Crawl", type="primary"):
                        with st.spinner("Launching crawl..."):
                            crawl, err = launch_crawl(api_token, selected_proj_id, selected_config_id)

                        if err:
                            st.error(err)
                        else:
                            st.success(f"Crawl launched! ID: {crawl.get('id')}")
                            get_crawls.clear()

                with tab3:
                    st.subheader("Crawl Configurations")

                    configs, err = get_crawl_configs(api_token, selected_proj_id)

                    if err:
                        st.error(err)
                    elif configs:
                        for config in configs:
                            with st.expander(f"📋 {config.get('name', 'Unnamed')}"):
                                st.json(config)
                    else:
                        st.info("No custom configurations found")

    # =========================================================================
    # AGGREGATIONS PAGE
    # =========================================================================
    elif page == "📈 Aggregations":
        st.header("📈 Aggregate Analysis")

        workspaces, _ = get_workspaces(api_token)

        if workspaces:
            col1, col2, col3 = st.columns(3)

            with col1:
                workspace_options = {ws['name']: ws['id'] for ws in workspaces}
                selected_ws_name = st.selectbox("Workspace", list(workspace_options.keys()), key="agg_ws")
                selected_ws_id = workspace_options[selected_ws_name]

            with col2:
                projects, _ = get_projects(api_token, selected_ws_id)
                if projects:
                    project_options = {p['name']: p['id'] for p in projects}
                    selected_proj_name = st.selectbox("Project", list(project_options.keys()), key="agg_proj")
                    selected_proj_id = project_options[selected_proj_name]
                else:
                    selected_proj_id = None

            with col3:
                if selected_proj_id:
                    crawls, _ = get_crawls(api_token, selected_ws_id, selected_proj_id, status="done")
                    if crawls:
                        crawl_options = {}
                        for c in crawls:
                            crawl_date = datetime.fromtimestamp(c.get('created_at', 0) / 1000).strftime('%Y-%m-%d %H:%M')
                            crawl_options[crawl_date] = c['id']
                        selected_crawl_label = st.selectbox("Crawl", list(crawl_options.keys()), key="agg_crawl")
                        selected_crawl_id = crawl_options[selected_crawl_label]
                    else:
                        selected_crawl_id = None
                else:
                    selected_crawl_id = None

            if selected_crawl_id:
                st.divider()

                group_by_field = st.selectbox("Group By", AGGREGATE_FIELDS)

                if st.button("Run Aggregation", type="primary"):
                    with st.spinner("Running aggregation..."):
                        aggs, err = aggregate_crawl_data(api_token, selected_crawl_id, group_by_field)

                    if err:
                        st.error(err)
                    elif aggs:
                        # Parse aggregation results
                        results = []
                        for agg in aggs:
                            for bucket in agg.get('buckets', []):
                                results.append({
                                    group_by_field: bucket.get('key', 'N/A'),
                                    'count': bucket.get('metrics', {}).get('count', 0)
                                })

                        if results:
                            df = pd.DataFrame(results)
                            df = df.sort_values('count', ascending=False)

                            col1, col2 = st.columns([2, 1])

                            with col1:
                                st.dataframe(df, use_container_width=True)

                            with col2:
                                st.bar_chart(df.set_index(group_by_field)['count'].head(20))

                            st.download_button("Download CSV", df.to_csv(index=False),
                                file_name=f"oncrawl_agg_{group_by_field}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv")
                        else:
                            st.warning("No aggregation data returned")
                    else:
                        st.warning("No data")

    # =========================================================================
    # CRAWL COMPARISON PAGE
    # =========================================================================
    elif page == "🔀 Crawl Comparison":
        st.header("🔀 Crawl Comparison (Crawl over Crawl)")

        workspaces, _ = get_workspaces(api_token)

        if workspaces:
            col1, col2 = st.columns(2)

            with col1:
                workspace_options = {ws['name']: ws['id'] for ws in workspaces}
                selected_ws_name = st.selectbox("Workspace", list(workspace_options.keys()), key="coc_ws")
                selected_ws_id = workspace_options[selected_ws_name]

            with col2:
                projects, _ = get_projects(api_token, selected_ws_id)
                if projects:
                    project_options = {p['name']: p['id'] for p in projects}
                    selected_proj_name = st.selectbox("Project", list(project_options.keys()), key="coc_proj")
                    selected_proj_id = project_options[selected_proj_name]
                else:
                    selected_proj_id = None

            if selected_proj_id:
                crawls, _ = get_crawls(api_token, selected_ws_id, selected_proj_id, status="done")

                if crawls and len(crawls) >= 2:
                    st.divider()

                    crawl_options = {}
                    for c in crawls:
                        crawl_date = datetime.fromtimestamp(c.get('created_at', 0) / 1000).strftime('%Y-%m-%d %H:%M')
                        urls = c.get('fetched_urls', 'N/A')
                        crawl_options[f"{crawl_date} ({urls} URLs)"] = c['id']

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Current Crawl")
                        current_label = st.selectbox("Select current crawl", list(crawl_options.keys()), key="current_crawl")
                        current_crawl_id = crawl_options[current_label]

                    with col2:
                        st.subheader("Reference Crawl")
                        reference_label = st.selectbox("Select reference crawl", list(crawl_options.keys()), index=1, key="ref_crawl")
                        reference_crawl_id = crawl_options[reference_label]

                    st.divider()

                    change_type = st.selectbox("Change Type", ["All Changes", "New Pages", "Lost Pages", "Changed Pages", "Unchanged Pages"])
                    change_map = {
                        "All Changes": None,
                        "New Pages": "new",
                        "Lost Pages": "lost",
                        "Changed Pages": "changed",
                        "Unchanged Pages": "unchanged"
                    }

                    fields = st.multiselect("Fields to compare", ALL_FIELDS, default=["url", "status_code", "title", "depth"])

                    if st.button("Compare Crawls", type="primary"):
                        with st.spinner("Comparing crawls..."):
                            df, err = export_crawl_comparison(
                                api_token, current_crawl_id, reference_crawl_id,
                                fields, change_type=change_map[change_type]
                            )

                        if err:
                            st.error(err)
                        elif df is not None and not df.empty:
                            st.success(f"Found {len(df):,} pages")
                            st.dataframe(df, use_container_width=True, height=400)

                            st.download_button("Download CSV", df.to_csv(index=False),
                                file_name=f"oncrawl_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv")
                        else:
                            st.warning("No comparison data found")
                else:
                    st.warning("Need at least 2 completed crawls to compare")

    # =========================================================================
    # PROJECT MANAGEMENT PAGE
    # =========================================================================
    elif page == "📁 Project Management":
        st.header("📁 Project Management")

        workspaces, _ = get_workspaces(api_token)

        if workspaces:
            workspace_options = {ws['name']: ws['id'] for ws in workspaces}
            selected_ws_name = st.selectbox("Workspace", list(workspace_options.keys()), key="pm_ws")
            selected_ws_id = workspace_options[selected_ws_name]

            # Show workspace info
            ws_info = next((ws for ws in workspaces if ws['id'] == selected_ws_id), {})
            if ws_info.get('crawl_pages_limit'):
                used = ws_info.get('crawl_pages_used', 0)
                limit = ws_info.get('crawl_pages_limit', 0)
                st.progress(used / limit if limit > 0 else 0, f"Pages used: {used:,} / {limit:,}")

            st.divider()

            tab1, tab2 = st.tabs(["📋 View Projects", "➕ Create Project"])

            with tab1:
                if st.button("🔄 Refresh Projects"):
                    get_projects.clear()

                projects, _ = get_projects(api_token, selected_ws_id)

                if projects:
                    for proj in projects:
                        with st.expander(f"📁 {proj['name']}"):
                            st.write(f"**ID:** `{proj['id']}`")
                            st.write(f"**Start URL:** {proj.get('start_url', 'N/A')}")
                            st.write(f"**Created:** {datetime.fromtimestamp(proj.get('created_at', 0) / 1000).strftime('%Y-%m-%d %H:%M') if proj.get('created_at') else 'N/A'}")

                            if st.button("🗑️ Delete Project", key=f"del_proj_{proj['id']}"):
                                success, err = delete_project(api_token, proj['id'])
                                if err:
                                    st.error(err)
                                else:
                                    st.success("Project deleted")
                                    get_projects.clear()
                                    st.rerun()
                else:
                    st.info("No projects found")

            with tab2:
                st.subheader("Create New Project")

                project_name = st.text_input("Project Name", placeholder="My Website Audit")
                start_url = st.text_input("Start URL", placeholder="https://example.com")
                user_agent = st.selectbox("User Agent", ["oncrawl", "googlebot", "bingbot", "custom"])

                if user_agent == "custom":
                    user_agent = st.text_input("Custom User Agent")

                if st.button("Create Project", type="primary"):
                    if project_name and start_url:
                        with st.spinner("Creating project..."):
                            proj, err = create_project(api_token, selected_ws_id, project_name, start_url, user_agent)

                        if err:
                            st.error(err)
                        else:
                            st.success(f"Project created! ID: {proj.get('id')}")
                            get_projects.clear()
                    else:
                        st.warning("Please provide project name and start URL")

    # =========================================================================
    # SCHEDULING PAGE
    # =========================================================================
    elif page == "⏰ Scheduling":
        st.header("⏰ Crawl Scheduling")

        workspaces, _ = get_workspaces(api_token)

        if workspaces:
            col1, col2 = st.columns(2)

            with col1:
                workspace_options = {ws['name']: ws['id'] for ws in workspaces}
                selected_ws_name = st.selectbox("Workspace", list(workspace_options.keys()), key="sched_ws")
                selected_ws_id = workspace_options[selected_ws_name]

            with col2:
                projects, _ = get_projects(api_token, selected_ws_id)
                if projects:
                    project_options = {p['name']: p['id'] for p in projects}
                    selected_proj_name = st.selectbox("Project", list(project_options.keys()), key="sched_proj")
                    selected_proj_id = project_options[selected_proj_name]
                else:
                    selected_proj_id = None

            if selected_proj_id:
                st.divider()

                tab1, tab2 = st.tabs(["📋 View Schedules", "➕ Create Schedule"])

                with tab1:
                    schedules, err = get_schedules(api_token, selected_proj_id)

                    if err:
                        st.error(err)
                    elif schedules:
                        for sched in schedules:
                            freq = sched.get('frequency', 'unknown')
                            hour = sched.get('hour', 0)

                            with st.expander(f"⏰ {freq.capitalize()} at {hour}:00"):
                                st.write(f"**ID:** `{sched['id']}`")
                                st.write(f"**Frequency:** {freq}")
                                st.write(f"**Hour:** {hour}:00 UTC")

                                if freq == 'weekly':
                                    st.write(f"**Day of Week:** {sched.get('day_of_week', 'N/A')}")
                                elif freq == 'monthly':
                                    st.write(f"**Day of Month:** {sched.get('day_of_month', 'N/A')}")

                                if st.button("🗑️ Delete Schedule", key=f"del_sched_{sched['id']}"):
                                    success, err = delete_schedule(api_token, sched['id'])
                                    if err:
                                        st.error(err)
                                    else:
                                        st.success("Schedule deleted")
                                        st.rerun()
                    else:
                        st.info("No schedules configured")

                with tab2:
                    st.subheader("Create New Schedule")

                    configs, _ = get_crawl_configs(api_token, selected_proj_id)

                    if configs:
                        config_options = {c.get('name', c['id']): c['id'] for c in configs}
                        selected_config = st.selectbox("Crawl Configuration", list(config_options.keys()))
                        selected_config_id = config_options[selected_config]
                    else:
                        st.warning("No crawl configurations found. Using default.")
                        selected_config_id = None

                    frequency = st.selectbox("Frequency", ["daily", "weekly", "monthly"])
                    hour = st.slider("Hour (UTC)", 0, 23, 2)

                    day_of_week = None
                    day_of_month = None

                    if frequency == "weekly":
                        day_of_week = st.selectbox("Day of Week",
                            ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
                            index=1)
                        day_of_week = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"].index(day_of_week)
                    elif frequency == "monthly":
                        day_of_month = st.number_input("Day of Month", 1, 28, 1)

                    if st.button("Create Schedule", type="primary"):
                        if selected_config_id:
                            with st.spinner("Creating schedule..."):
                                sched, err = create_schedule(
                                    api_token, selected_proj_id, selected_config_id,
                                    frequency, day_of_week, day_of_month, hour
                                )

                            if err:
                                st.error(err)
                            else:
                                st.success("Schedule created!")
                                st.rerun()
                        else:
                            st.warning("Please select a crawl configuration")

    # =========================================================================
    # LINK EXPORT PAGE
    # =========================================================================
    elif page == "🔗 Link Export":
        st.header("🔗 Link Export")

        workspaces, _ = get_workspaces(api_token)

        if workspaces:
            col1, col2, col3 = st.columns(3)

            with col1:
                workspace_options = {ws['name']: ws['id'] for ws in workspaces}
                selected_ws_name = st.selectbox("Workspace", list(workspace_options.keys()), key="link_ws")
                selected_ws_id = workspace_options[selected_ws_name]

            with col2:
                projects, _ = get_projects(api_token, selected_ws_id)
                if projects:
                    project_options = {p['name']: p['id'] for p in projects}
                    selected_proj_name = st.selectbox("Project", list(project_options.keys()), key="link_proj")
                    selected_proj_id = project_options[selected_proj_name]
                else:
                    selected_proj_id = None

            with col3:
                if selected_proj_id:
                    crawls, _ = get_crawls(api_token, selected_ws_id, selected_proj_id, status="done")
                    if crawls:
                        crawl_options = {}
                        for c in crawls:
                            crawl_date = datetime.fromtimestamp(c.get('created_at', 0) / 1000).strftime('%Y-%m-%d %H:%M')
                            crawl_options[crawl_date] = c['id']
                        selected_crawl_label = st.selectbox("Crawl", list(crawl_options.keys()), key="link_crawl")
                        selected_crawl_id = crawl_options[selected_crawl_label]
                    else:
                        selected_crawl_id = None
                else:
                    selected_crawl_id = None

            if selected_crawl_id:
                st.divider()

                link_type = st.radio("Link Type", ["Internal Links", "External Links"], horizontal=True)

                if st.button("Export Links", type="primary"):
                    with st.spinner("Exporting links..."):
                        df, err = export_links(
                            api_token, selected_crawl_id,
                            "internal" if link_type == "Internal Links" else "external"
                        )

                    if err:
                        st.error(err)
                    elif df is not None and not df.empty:
                        st.success(f"Exported {len(df):,} links")
                        st.dataframe(df, use_container_width=True, height=400)

                        st.download_button("Download CSV", df.to_csv(index=False),
                            file_name=f"oncrawl_links_{link_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv")
                    else:
                        st.warning("No links found")

else:
    # Welcome screen
    st.info("Configure your OnCrawl API token in the sidebar to get started.")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("📊 Data Extraction", expanded=True):
            st.markdown("""
            - **35+ preset queries** for SEO issues
            - Custom OQL query builder
            - Export to CSV/Excel
            """)

    with col2:
        with st.expander("🔄 Crawl Management"):
            st.markdown("""
            - Launch, pause, resume crawls
            - View crawl history
            - Manage configurations
            """)

        with st.expander("📈 Aggregations"):
            st.markdown("""
            - Group by status code, depth, segments
            - Visual charts
            - Export aggregated data
            """)

        with st.expander("🔀 Crawl Comparison"):
            st.markdown("""
            - Compare two crawls
            - Find new, lost, changed pages
            - Track changes over time
            """)

        with st.expander("📁 Project Management"):
            st.markdown("""
            - Create/delete projects
            - View project details
            - Monitor usage quotas
            """)

        with st.expander("⏰ Scheduling"):
            st.markdown("""
            - Daily, weekly, monthly schedules
            - Manage existing schedules
            """)

        with st.expander("🔗 Link Export"):
            st.markdown("""
            - Export internal links
            - Export external links
            """)

    with st.expander("🔑 How to get your API token"):
        st.markdown("""
        1. Log in to [app.oncrawl.com](https://app.oncrawl.com)
        2. Click your profile icon → **Account Settings**
        3. Scroll to **API Access Tokens**
        4. Click **Generate new token** or copy existing
        5. Paste the token in the sidebar

        Your token is stored only in your browser session.
        """)

# Footer
st.markdown("---")
