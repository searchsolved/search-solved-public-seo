"""
OnCrawl Data Extractor - Streamlit App
Extract data from OnCrawl crawls using the API.

Author: Lee Foot
Website: https://leefoot.co.uk
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from io import StringIO
import io
import json

st.set_page_config(
    page_title="OnCrawl Data Extractor",
    page_icon="🕷️",
    layout="wide"
)

st.title("🕷️ OnCrawl Data Extractor")
st.markdown("Extract and analyze data from your OnCrawl crawls.")

# Constants
BASE_URL = "https://app.oncrawl.com/api/v2"

# Preset queries
PRESET_QUERIES = {
    "404 Pages": {
        "description": "All pages returning 404 status code",
        "fields": ["url", "status_code", "depth", "inrank", "nb_inlinks"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "equals", 404]}
            ]
        }
    },
    "301 Redirects": {
        "description": "All pages with 301 redirects and their destinations",
        "fields": ["url", "status_code", "redirect_location", "final_redirect_location", "redirect_count"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "equals", 301]}
            ]
        }
    },
    "302 Redirects": {
        "description": "All pages with 302 temporary redirects",
        "fields": ["url", "status_code", "redirect_location", "final_redirect_location", "redirect_count"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "equals", 302]}
            ]
        }
    },
    "Redirect Chains": {
        "description": "Pages with multiple redirects (redirect count > 1)",
        "fields": ["url", "status_code", "redirect_count", "redirect_location", "final_redirect_location", "is_redirect_loop"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["redirect_count", "gt", 1]}
            ]
        }
    },
    "Redirect Loops": {
        "description": "Pages caught in redirect loops",
        "fields": ["url", "status_code", "redirect_count", "redirect_location", "final_redirect_location"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["is_redirect_loop", "equals", True]}
            ]
        }
    },
    "Stale Links (3xx with Inlinks)": {
        "description": "Redirecting pages that still receive internal links",
        "fields": ["url", "status_code", "redirect_location", "nb_inlinks", "inrank"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "between", [300, 399]]},
                {"field": ["nb_inlinks", "gt", 0]}
            ]
        }
    },
    "Indexable Pages": {
        "description": "All indexable pages (200, index, canonical matching)",
        "fields": ["url", "status_code", "meta_robots_index", "canonical_evaluation", "title", "h1"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "equals", 200]},
                {"field": ["meta_robots_index", "equals", True]},
                {"field": ["robots_txt_denied", "equals", False]},
                {"field": ["canonical_evaluation", "equals", "matching"]}
            ]
        }
    },
    "Non-Indexable Pages": {
        "description": "Pages blocked from indexing",
        "fields": ["url", "status_code", "meta_robots_index", "robots_txt_denied", "canonical_evaluation"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "equals", 200]},
                {"or": [
                    {"field": ["meta_robots_index", "equals", False]},
                    {"field": ["robots_txt_denied", "equals", True]},
                    {"field": ["canonical_evaluation", "equals", "not_matching"]}
                ]}
            ]
        }
    },
    "Pages Without Title": {
        "description": "Pages missing title tags",
        "fields": ["url", "status_code", "title", "h1"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "equals", 200]},
                {"field": ["title", "is_empty", True]}
            ]
        }
    },
    "Pages Without H1": {
        "description": "Pages missing H1 tags",
        "fields": ["url", "status_code", "title", "h1"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "equals", 200]},
                {"field": ["h1", "is_empty", True]}
            ]
        }
    },
    "Duplicate Titles": {
        "description": "Pages with duplicate title tags",
        "fields": ["url", "status_code", "title", "title_duplicates_count"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "equals", 200]},
                {"field": ["title_duplicates_count", "gt", 1]}
            ]
        }
    },
    "Slow Pages (>3s)": {
        "description": "Pages with load time over 3 seconds",
        "fields": ["url", "status_code", "delay_total", "delay_first_byte"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "equals", 200]},
                {"field": ["delay_total", "gt", 3000]}
            ]
        }
    },
    "Orphan Pages": {
        "description": "Pages with no internal links pointing to them",
        "fields": ["url", "status_code", "nb_inlinks", "depth"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "equals", 200]},
                {"field": ["nb_inlinks", "equals", 0]}
            ]
        }
    },
    "Deep Pages (Depth > 5)": {
        "description": "Pages more than 5 clicks from homepage",
        "fields": ["url", "status_code", "depth", "nb_inlinks"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "equals", 200]},
                {"field": ["depth", "gt", 5]}
            ]
        }
    },
    "All Fetched Pages": {
        "description": "Complete list of all fetched pages",
        "fields": ["url", "status_code", "depth", "nb_inlinks", "inrank"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]}
            ]
        }
    }
}

# Available fields for custom queries
AVAILABLE_FIELDS = [
    "url", "urlpath", "status_code", "depth", "nb_inlinks", "nb_outlinks",
    "inrank", "title", "h1", "meta_description", "canonical", "canonical_evaluation",
    "meta_robots_index", "meta_robots_follow", "robots_txt_denied",
    "redirect_location", "final_redirect_location", "redirect_count", "is_redirect_loop",
    "delay_total", "delay_first_byte", "content_type", "word_count",
    "title_length", "meta_description_length", "h1_count", "h2_count",
    "title_duplicates_count", "meta_description_duplicates_count",
    "url_has_params", "fetched"
]


def get_headers(api_token):
    """Return headers for API authentication."""
    return {
        'Authorization': f'Bearer {api_token}',
        'Content-Type': 'application/json'
    }


@st.cache_data(ttl=300)
def get_workspaces(api_token):
    """Fetch all workspaces for the account."""
    try:
        response = requests.get(
            f"{BASE_URL}/account",
            headers=get_headers(api_token),
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            workspace_ids = data.get('account', {}).get('workspace_ids', [])

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
                        'name': ws_data.get('name', ws_id)
                    })
            return workspaces
        return []
    except Exception as e:
        st.error(f"Error fetching workspaces: {str(e)}")
        return []


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

        return all_projects
    except Exception as e:
        st.error(f"Error fetching projects: {str(e)}")
        return []


@st.cache_data(ttl=60)
def get_crawls(api_token, workspace_id, project_id):
    """Fetch all completed crawls for a project."""
    try:
        response = requests.get(
            f"{BASE_URL}/workspaces/{workspace_id}/crawls",
            headers=get_headers(api_token),
            params={
                'filters[project_id]': project_id,
                'filters[status]': 'done'
            },
            timeout=30
        )

        if response.status_code == 200:
            crawls = response.json().get('crawls', [])
            # Sort by creation date, newest first
            crawls = sorted(crawls, key=lambda x: x.get('created_at', 0), reverse=True)
            return crawls
        return []
    except Exception as e:
        st.error(f"Error fetching crawls: {str(e)}")
        return []


def export_crawl_data(api_token, crawl_id, fields, oql_query, url_filter=None):
    """Export data from a crawl using the export API."""
    try:
        # Build the query
        query = {
            "fields": fields,
            "oql": oql_query
        }

        # Add URL filter if provided
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
            timeout=120
        )

        if response.status_code == 200:
            csv_content = response.content.decode('utf-8')
            df = pd.read_csv(StringIO(csv_content), sep=';', quotechar='"')
            return df, None
        else:
            return None, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, f"Error: {str(e)}"


# Sidebar - API Configuration
with st.sidebar:
    st.header("API Configuration")

    api_token = st.text_input(
        "OnCrawl API Token",
        type="password",
        help="Your OnCrawl API access token"
    )

    if api_token:
        st.success("API token provided")

        # Workspace selection
        st.header("Select Data Source")

        workspaces = get_workspaces(api_token)

        if workspaces:
            workspace_options = {ws['name']: ws['id'] for ws in workspaces}
            selected_workspace_name = st.selectbox(
                "Workspace",
                options=list(workspace_options.keys())
            )
            selected_workspace_id = workspace_options[selected_workspace_name]

            # Project selection
            projects = get_projects(api_token, selected_workspace_id)

            if projects:
                project_options = {p['name']: p['id'] for p in projects}
                selected_project_name = st.selectbox(
                    "Project",
                    options=list(project_options.keys())
                )
                selected_project_id = project_options[selected_project_name]

                # Crawl selection
                crawls = get_crawls(api_token, selected_workspace_id, selected_project_id)

                if crawls:
                    crawl_options = {}
                    for c in crawls:
                        crawl_date = datetime.fromtimestamp(
                            c.get('created_at', 0) / 1000
                        ).strftime('%Y-%m-%d %H:%M')
                        urls_count = c.get('fetched_urls', 'N/A')
                        label = f"{crawl_date} ({urls_count} URLs)"
                        crawl_options[label] = c['id']

                    selected_crawl_label = st.selectbox(
                        "Crawl",
                        options=list(crawl_options.keys())
                    )
                    selected_crawl_id = crawl_options[selected_crawl_label]

                    st.success(f"Ready to extract data")
                else:
                    st.warning("No completed crawls found")
                    selected_crawl_id = None
            else:
                st.warning("No projects found")
                selected_crawl_id = None
        else:
            st.warning("No workspaces found or invalid API token")
            selected_crawl_id = None
    else:
        st.info("Enter your API token to get started")
        selected_crawl_id = None

# Main content
if api_token and 'selected_crawl_id' in dir() and selected_crawl_id:

    # Query type selection
    st.subheader("Select Query Type")

    query_type = st.radio(
        "Query Type",
        options=["Preset Queries", "Custom Query"],
        horizontal=True
    )

    if query_type == "Preset Queries":
        # Preset query selection
        col1, col2 = st.columns([2, 1])

        with col1:
            selected_preset = st.selectbox(
                "Select a preset query",
                options=list(PRESET_QUERIES.keys())
            )

        with col2:
            url_filter = st.text_input(
                "URL filter (optional)",
                placeholder="/products/",
                help="Filter results to URLs containing this text"
            )

        preset_config = PRESET_QUERIES[selected_preset]
        st.info(f"**{selected_preset}**: {preset_config['description']}")

        # Show fields that will be extracted
        with st.expander("Fields to extract"):
            st.write(preset_config['fields'])

        if st.button("Extract Data", type="primary"):
            with st.spinner(f"Extracting {selected_preset}..."):
                df, error = export_crawl_data(
                    api_token,
                    selected_crawl_id,
                    preset_config['fields'],
                    preset_config['oql'],
                    url_filter if url_filter else None
                )

            if error:
                st.error(error)
            elif df is not None and not df.empty:
                st.success(f"Extracted {len(df):,} rows")

                # Display data
                st.dataframe(df, use_container_width=True, height=400)

                # Download buttons
                col1, col2 = st.columns(2)

                with col1:
                    csv = df.to_csv(index=False)
                    filename = f"oncrawl_{selected_preset.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    st.download_button(
                        "Download CSV",
                        csv,
                        file_name=filename,
                        mime="text/csv"
                    )

                with col2:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Data')
                    excel_data = output.getvalue()
                    filename = f"oncrawl_{selected_preset.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    st.download_button(
                        "Download Excel",
                        excel_data,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.warning("No data found matching the query.")

    else:
        # Custom query builder
        st.markdown("### Custom Query Builder")

        col1, col2 = st.columns(2)

        with col1:
            selected_fields = st.multiselect(
                "Select fields to extract",
                options=AVAILABLE_FIELDS,
                default=["url", "status_code", "depth"]
            )

        with col2:
            url_filter = st.text_input(
                "URL filter (optional)",
                placeholder="/products/",
                help="Filter results to URLs containing this text"
            )

        # OQL Query builder
        st.markdown("### OQL Conditions")
        st.markdown("Build your query by adding conditions:")

        # Simple condition builder
        conditions = []

        num_conditions = st.number_input("Number of conditions", min_value=1, max_value=10, value=1)

        for i in range(int(num_conditions)):
            col1, col2, col3 = st.columns(3)

            with col1:
                field = st.selectbox(f"Field {i+1}", options=AVAILABLE_FIELDS, key=f"field_{i}")

            with col2:
                operator = st.selectbox(
                    f"Operator {i+1}",
                    options=["equals", "not_equals", "contains", "not_contains", "gt", "lt", "gte", "lte", "is_empty", "between"],
                    key=f"op_{i}"
                )

            with col3:
                if operator == "is_empty":
                    value = st.selectbox(f"Value {i+1}", options=[True, False], key=f"val_{i}")
                elif operator == "between":
                    value = st.text_input(f"Value {i+1} (comma-separated)", placeholder="100,200", key=f"val_{i}")
                elif operator in ["equals", "not_equals"] and field in ["fetched", "meta_robots_index", "meta_robots_follow", "robots_txt_denied", "is_redirect_loop", "url_has_params"]:
                    value = st.selectbox(f"Value {i+1}", options=[True, False], key=f"val_{i}")
                else:
                    value = st.text_input(f"Value {i+1}", key=f"val_{i}")

            if value is not None and value != "":
                if operator == "between" and isinstance(value, str):
                    try:
                        value = [int(v.strip()) for v in value.split(",")]
                    except:
                        pass
                elif isinstance(value, str) and value.isdigit():
                    value = int(value)

                conditions.append({"field": [field, operator, value]})

        # Build OQL
        if conditions:
            custom_oql = {"and": conditions}

            with st.expander("View OQL Query"):
                st.json(custom_oql)

            if st.button("Extract Custom Data", type="primary"):
                if not selected_fields:
                    st.error("Please select at least one field")
                else:
                    with st.spinner("Extracting data..."):
                        df, error = export_crawl_data(
                            api_token,
                            selected_crawl_id,
                            selected_fields,
                            custom_oql,
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
                            st.download_button(
                                "Download CSV",
                                csv,
                                file_name=f"oncrawl_custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )

                        with col2:
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                df.to_excel(writer, index=False, sheet_name='Data')
                            excel_data = output.getvalue()
                            st.download_button(
                                "Download Excel",
                                excel_data,
                                file_name=f"oncrawl_custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    else:
                        st.warning("No data found matching the query.")

else:
    # Welcome screen
    st.info("Configure your OnCrawl API token in the sidebar to get started.")

    with st.expander("Available Preset Queries"):
        for name, config in PRESET_QUERIES.items():
            st.markdown(f"**{name}**: {config['description']}")

    with st.expander("How to get your API token"):
        st.markdown("""
        1. Log in to your OnCrawl account
        2. Go to **Account Settings** → **API**
        3. Generate or copy your API access token
        4. Paste it in the sidebar

        Your token is stored only in your browser session and is never saved.
        """)

    with st.expander("Use Cases"):
        st.markdown("""
        - **Technical SEO Audits**: Extract 404s, redirects, orphan pages
        - **Migration Monitoring**: Track redirect chains and loops
        - **Content Audits**: Find pages without titles, H1s, or descriptions
        - **Performance Analysis**: Identify slow-loading pages
        - **Indexability Checks**: Find non-indexable pages
        """)

# Footer
st.markdown("---")
st.markdown("Built by [Lee Foot](https://leefoot.co.uk) | [GitHub](https://github.com/searchsolved/search-solved-public-seo)")
