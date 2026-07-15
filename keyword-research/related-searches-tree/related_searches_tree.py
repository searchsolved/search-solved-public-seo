# Author: Lee Foot
# Website: https://leefoot.com

"""
Related Searches Tree Builder

Builds hierarchical trees of related searches from Google using the DataForSEO
SERP API. Recursively explores related searches to a configurable depth.
Exports to DOT format for visualisation.

Features:
- Enter seed keyword(s) to explore
- Configurable crawl depth and max results
- Tree visualisation in Streamlit
- Export to DOT format for Graphviz
- Export relationships to CSV
"""

import streamlit as st
import pandas as pd
import requests
import os
from base64 import b64encode

st.set_page_config(page_title="Related Searches Tree", page_icon="🌳", layout="wide")

st.title("Related Searches Tree Builder")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Discovers related searches from Google for any keyword
    - Builds a hierarchical tree showing search relationships
    - Visualises keyword expansion opportunities

    **Requirements:**
    - DataForSEO account (get one at [dataforseo.com](https://dataforseo.com/))

    **How to use:**
    1. Enter your DataForSEO login and password in the sidebar
    2. Enter a seed keyword
    3. Set the crawl depth (how many levels deep to explore)
    4. Click "Build Tree" to start
    5. View and export the tree visualisation

    **Note:** Each API call fetches related searches for one keyword.
    Higher depths use more API credits exponentially.
    Cost is approximately $0.002 per keyword.
    """)

# Location code mapping
LOCATION_CODES = {
    "United Kingdom": 2826,
    "United States": 2840,
    "Australia": 2036,
    "Canada": 2124,
    "Germany": 2276,
    "France": 2250,
    "Spain": 2724,
    "Italy": 2380,
    "Netherlands": 2528,
    "India": 2356,
    "Ireland": 2372,
}

# Sidebar settings
st.sidebar.header("API Settings")

dataforseo_login = st.sidebar.text_input(
    "DataForSEO Login",
    type="password",
    value=os.environ.get('DATAFORSEO_LOGIN', ''),
    help="Your login from dataforseo.com"
)

dataforseo_password = st.sidebar.text_input(
    "DataForSEO Password",
    type="password",
    value=os.environ.get('DATAFORSEO_PASSWORD', ''),
    help="Your password from dataforseo.com"
)

has_credentials = bool(dataforseo_login and dataforseo_password)

st.sidebar.markdown("---")
st.sidebar.header("Search Settings")

location = st.sidebar.selectbox(
    "Location",
    list(LOCATION_CODES.keys()),
    index=0
)

crawl_depth = st.sidebar.slider(
    "Crawl depth",
    min_value=1,
    max_value=3,
    value=2,
    help="How many levels deep to explore. Warning: Higher = exponentially more API calls"
)

max_results = st.sidebar.slider(
    "Max related searches per keyword",
    min_value=5,
    max_value=20,
    value=10,
    help="Maximum number of related searches to fetch per keyword"
)


def _build_auth_headers(login, password):
    """Build DataForSEO Basic auth headers."""
    cred = b64encode(f"{login}:{password}".encode()).decode()
    return {
        'Authorization': f'Basic {cred}',
        'Content-Type': 'application/json'
    }


def get_related_searches(keyword, login, password, location_code):
    """Fetch related searches for a keyword from DataForSEO SERP API."""
    headers = _build_auth_headers(login, password)
    payload = [{
        "keyword": keyword,
        "location_code": location_code,
        "language_code": "en",
        "device": "desktop",
        "depth": 10
    }]

    try:
        response = requests.post(
            'https://api.dataforseo.com/v3/serp/google/organic/live/advanced',
            headers=headers,
            json=payload,
            timeout=60
        )
        data = response.json()

        if data.get("status_code") != 20000:
            msg = data.get("status_message", "Unknown error")
            st.warning(f"API error for '{keyword}': {msg}")
            return []

        items = data["tasks"][0]["result"][0]["items"]
        related = []
        for item in items:
            if item["type"] == "related_searches":
                for rs in item.get("items", []):
                    related.append(rs["title"])
        return related

    except Exception as e:
        st.warning(f"Error fetching related searches for '{keyword}': {str(e)}")
        return []


def build_tree(seed_keywords, login, password, location_code, max_depth, max_results_per_kw, progress_callback=None):
    """Build a tree of related searches."""
    relationships = {'Parent': [], 'Child': []}
    visited = set()
    queue = [(kw, 0) for kw in seed_keywords]
    total_queries = 0

    while queue:
        keyword, depth = queue.pop(0)

        if keyword in visited or depth >= max_depth:
            continue

        visited.add(keyword)
        total_queries += 1

        if progress_callback:
            progress_callback(f"Depth {depth}: Exploring '{keyword}' ({total_queries} queries)")

        related = get_related_searches(keyword, login, password, location_code)

        for r in related[:max_results_per_kw]:
            r_lower = r.lower()
            if r_lower not in visited:
                relationships['Parent'].append(keyword.lower())
                relationships['Child'].append(r_lower)

                if depth + 1 < max_depth:
                    queue.append((r, depth + 1))

    return relationships


def create_ascii_tree(relationships, root_keywords):
    """Create ASCII representation of the tree."""
    df = pd.DataFrame(relationships)
    if df.empty:
        return "No relationships found"

    def render_node(node, prefix="", is_last=True, rendered=None):
        if rendered is None:
            rendered = set()

        if node in rendered:
            return f"{prefix}{'└── ' if is_last else '├── '}{node} (circular)\n"

        rendered.add(node)
        result = f"{prefix}{'└── ' if is_last else '├── '}{node}\n"

        children = df[df['Parent'] == node]['Child'].tolist()
        for i, child in enumerate(children):
            child_is_last = (i == len(children) - 1)
            child_prefix = prefix + ("    " if is_last else "│   ")
            result += render_node(child, child_prefix, child_is_last, rendered.copy())

        return result

    roots = [kw.lower() for kw in root_keywords]
    tree_output = ""
    for i, root in enumerate(roots):
        is_last = (i == len(roots) - 1)
        tree_output += render_node(root, "", is_last)

    return tree_output


def generate_dot(relationships, root_keywords):
    """Generate DOT format for Graphviz."""
    df = pd.DataFrame(relationships)
    if df.empty:
        return None

    dot_lines = ['digraph RelatedSearches {']
    dot_lines.append('    rankdir=TB;')
    dot_lines.append('    node [shape=box, style=filled, fillcolor=lightblue];')

    # Highlight root keywords
    for root in root_keywords:
        dot_lines.append(f'    "{root.lower()}" [fillcolor=lightgreen];')

    # Add edges
    for _, row in df.iterrows():
        parent = row['Parent'].replace('"', '\\"')
        child = row['Child'].replace('"', '\\"')
        dot_lines.append(f'    "{parent}" -> "{child}";')

    dot_lines.append('}')
    return '\n'.join(dot_lines)


# Main content
st.subheader("Seed Keywords")

keyword_input = st.text_area(
    "Enter seed keyword(s) (one per line)",
    height=100,
    placeholder="seo tools\nkeyword research"
)

seed_keywords = [kw.strip() for kw in keyword_input.strip().split('\n') if kw.strip()] if keyword_input else []

if seed_keywords:
    st.info(f"Ready to explore {len(seed_keywords)} seed keyword(s)")

    # Estimate API calls and cost
    est_calls = sum(max_results ** i for i in range(crawl_depth + 1)) * len(seed_keywords)
    est_cost = est_calls * 0.002
    st.caption(f"Estimated API calls: ~{est_calls} (approx. ${est_cost:.2f})")

if st.button("Build Tree", type="primary", disabled=not has_credentials or not seed_keywords):
    if not has_credentials:
        st.error("Please enter your DataForSEO login and password")
    elif not seed_keywords:
        st.error("Please enter at least one seed keyword")
    else:
        location_code = LOCATION_CODES[location]
        status_text = st.empty()
        progress_bar = st.progress(0)

        def update_progress(msg):
            status_text.text(msg)

        with st.spinner("Building related searches tree..."):
            relationships = build_tree(
                seed_keywords,
                dataforseo_login,
                dataforseo_password,
                location_code,
                crawl_depth,
                max_results,
                update_progress
            )

            progress_bar.progress(100)
            status_text.empty()

            if relationships['Parent']:
                df = pd.DataFrame(relationships)
                df = df.drop_duplicates()

                # Remove self-references
                df = df[df['Parent'] != df['Child']]

                # Store in session state
                st.session_state['tree_relationships'] = relationships
                st.session_state['tree_df'] = df
                st.session_state['seed_keywords'] = seed_keywords

                st.success(f"Found {len(df)} relationships!")
            else:
                st.warning("No related searches found")

# Display results
if 'tree_df' in st.session_state:
    df = st.session_state['tree_df']
    relationships = st.session_state['tree_relationships']
    seeds = st.session_state['seed_keywords']

    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Relationships", len(df))
    with col2:
        unique_keywords = set(df['Parent'].tolist() + df['Child'].tolist())
        st.metric("Unique Keywords", len(unique_keywords))
    with col3:
        st.metric("Seed Keywords", len(seeds))

    # Tree visualisation
    st.subheader("Tree Structure")

    tree_text = create_ascii_tree(relationships, seeds)
    st.code(tree_text, language=None)

    # Relationships table
    st.subheader("All Relationships")
    st.dataframe(df, use_container_width=True)

    # Downloads
    st.subheader("Download")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_data = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="related_searches.csv",
            mime="text/csv"
        )

    with col2:
        dot_content = generate_dot(relationships, seeds)
        if dot_content:
            st.download_button(
                label="Download DOT (Graphviz)",
                data=dot_content,
                file_name="related_searches.dot",
                mime="text/plain"
            )

    with col3:
        # All unique keywords
        all_kws = sorted(set(df['Parent'].tolist() + df['Child'].tolist()))
        kw_df = pd.DataFrame({'Keyword': all_kws})
        kw_csv = kw_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="Download Keywords List",
            data=kw_csv,
            file_name="all_keywords.csv",
            mime="text/csv"
        )

    st.info("**Tip:** Import the DOT file into [Graphviz Online](https://dreampuf.github.io/GraphvizOnline/) for interactive visualisation")

else:
    if not has_credentials:
        st.warning("Enter your DataForSEO login and password in the sidebar to get started")

    st.subheader("Example Output")
    example_tree = """
└── seo tools
    ├── free seo tools
    │   ├── google seo tools
    │   └── seo audit tools
    ├── best seo tools
    │   ├── ahrefs
    │   └── semrush
    └── seo tools for beginners
"""
    st.code(example_tree, language=None)

    example_df = {
        "Parent": ["seo tools", "seo tools", "free seo tools", "free seo tools"],
        "Child": ["free seo tools", "best seo tools", "google seo tools", "seo audit tools"]
    }
    st.dataframe(pd.DataFrame(example_df))
