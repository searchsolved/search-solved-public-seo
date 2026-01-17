####################################################################################
#                                                                                  #
#  Related Searches Tree Builder                                                   #
#                                                                                  #
#  Build hierarchical trees of related searches using ValueSERP API.               #
#                                                                                  #
####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://leefoot.com                                                   #
# Contact  : https://leefoot.com/contact                                           #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                       #
####################################################################################

"""
Related Searches Tree Builder

Builds hierarchical trees of related searches from Google using ValueSERP API.
Recursively explores related searches to a configurable depth.
Exports to DOT format for visualization.

Features:
- Enter seed keyword(s) to explore
- Configurable crawl depth and max results
- Tree visualization in Streamlit
- Export to DOT format for Graphviz
- Export relationships to CSV
"""

import streamlit as st
import pandas as pd
import requests
import json
from io import BytesIO

st.set_page_config(page_title="Related Searches Tree", page_icon="🌳", layout="wide")

st.title("Related Searches Tree Builder")
st.markdown("*Created by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://leefoot.com) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · 🦋 [Bluesky](https://bsky.app/profile/leefootseo.bsky.social)*")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Discovers related searches from Google for any keyword
    - Builds a hierarchical tree showing search relationships
    - Visualizes keyword expansion opportunities

    **Requirements:**
    - ValueSERP API key (get one at [valueserp.com](https://www.valueserp.com/))

    **How to use:**
    1. Enter your ValueSERP API key in the sidebar
    2. Enter a seed keyword
    3. Set the crawl depth (how many levels deep to explore)
    4. Click "Build Tree" to start
    5. View and export the tree visualization

    **Note:** Each API call fetches related searches for one keyword.
    Higher depths use more API credits exponentially.
    """)

# Sidebar settings
st.sidebar.header("API Settings")

api_key = st.sidebar.text_input(
    "ValueSERP API Key",
    type="password",
    help="Your API key from valueserp.com"
)

st.sidebar.markdown("---")
st.sidebar.header("Search Settings")

location = st.sidebar.selectbox(
    "Location",
    [
        "United Kingdom",
        "United States",
        "Australia",
        "Canada",
        "Germany",
        "France",
        "Spain",
        "Italy",
        "Netherlands"
    ],
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


def get_related_searches(keyword, api_key, location):
    """Fetch related searches for a keyword from ValueSERP."""
    params = {
        'api_key': api_key,
        'q': keyword,
        'location': location,
        'num': '10'
    }

    try:
        response = requests.get('https://api.valueserp.com/search', params=params)
        data = response.json()

        related = data.get('related_searches', [])
        return [r['query'] for r in related] if related else []

    except Exception as e:
        st.warning(f"Error fetching related searches for '{keyword}': {str(e)}")
        return []


def build_tree(seed_keywords, api_key, location, max_depth, max_results, progress_callback=None):
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

        related = get_related_searches(keyword, api_key, location)

        for r in related[:max_results]:
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

    # Find all roots (parents that are not children)
    all_children = set(df['Child'])
    roots = [kw.lower() for kw in root_keywords]

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

    # Estimate API calls
    est_calls = sum(max_results ** i for i in range(crawl_depth + 1)) * len(seed_keywords)
    st.caption(f"Estimated API calls: ~{est_calls}")

if st.button("Build Tree", type="primary", disabled=not api_key or not seed_keywords):
    if not api_key:
        st.error("Please enter your ValueSERP API key")
    elif not seed_keywords:
        st.error("Please enter at least one seed keyword")
    else:
        status_text = st.empty()
        progress_bar = st.progress(0)

        def update_progress(msg):
            status_text.text(msg)

        with st.spinner("Building related searches tree..."):
            relationships = build_tree(
                seed_keywords,
                api_key,
                location,
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

    # Tree visualization
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

    st.info("**Tip:** Import the DOT file into [Graphviz Online](https://dreampuf.github.io/GraphvizOnline/) for interactive visualization")

else:
    if not api_key:
        st.warning("Enter your ValueSERP API key in the sidebar to get started")

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
