# Author: Lee Foot
# Website: https://leefoot.com

"""
SERP Clustering at Scale - Streamlit App

Clusters keywords based on common SERP URLs to identify content consolidation opportunities.
Supports CSV upload and live SERP fetching via DataForSEO.
"""

import os
import time
import requests
from base64 import b64encode
from collections import defaultdict
from itertools import combinations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="SERP Clustering at Scale",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 SERP Clustering at Scale")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Clusters keywords by SERP similarity
    - Groups keywords that share ranking URLs
    - Scales to thousands of keywords

    **Two ways to get SERP data:**

    **Option 1: Upload SERP CSVs**
    1. Export SERP data from SERP API or a similar tool as CSV
    2. Upload one or more CSV files (columns: search query and result URL)
    3. Configure clustering threshold and run

    **Option 2: Fetch Live SERPs (DataForSEO)**
    1. Enter your DataForSEO API credentials in the sidebar
    2. Paste keywords (one per line) into the text area
    3. Select location and device, then fetch and cluster

    **Best for:**
    - Large-scale keyword research
    - Content consolidation planning
    - Identifying search intent groups
    """)
st.markdown("Cluster keywords by shared SERP URLs to find content consolidation opportunities.")

# ----------------
# DataForSEO Configuration
# ----------------

DATAFORSEO_ENDPOINT = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"
COST_PER_KEYWORD = 0.002  # USD per keyword (10 results)

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
}


# ----------------
# DataForSEO Live Fetching
# ----------------

def fetch_serps_dataforseo(keywords, login, password, location_code, device, progress_bar, status_text):
    """
    Fetch live SERP results from DataForSEO for a list of keywords.
    Returns a DataFrame with columns 'query' and 'link'.
    """
    cred = b64encode(f"{login}:{password}".encode()).decode()
    headers = {
        "Authorization": f"Basic {cred}",
        "Content-Type": "application/json",
    }

    rows = []
    failed = []
    total = len(keywords)

    for i, keyword in enumerate(keywords):
        keyword = keyword.strip()
        if not keyword:
            continue

        status_text.text(f"Fetching SERP {i + 1}/{total}: {keyword}")
        progress_bar.progress((i + 1) / total * 0.8)  # Reserve 0.8-1.0 for clustering

        payload = [{
            "keyword": keyword,
            "location_code": location_code,
            "language_code": "en",
            "device": device,
            "depth": 10,
        }]

        try:
            response = requests.post(
                DATAFORSEO_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            tasks = data.get("tasks", [])
            if not tasks or tasks[0].get("status_code") != 20000:
                error_msg = tasks[0].get("status_message", "Unknown error") if tasks else "No tasks returned"
                failed.append(f"{keyword}: {error_msg}")
                continue

            result = tasks[0].get("result", [])
            if not result:
                failed.append(f"{keyword}: no results")
                continue

            items = result[0].get("items", [])
            organic_results = [item for item in items if item.get("type") == "organic"]

            for item in organic_results:
                url = item.get("url", "")
                if url:
                    rows.append({"query": keyword, "link": url})

        except requests.exceptions.RequestException as e:
            failed.append(f"{keyword}: {e}")

        # Rate limit: 0.5s between requests
        if i < total - 1:
            time.sleep(0.5)

    if failed:
        with st.expander(f"⚠️ {len(failed)} keyword(s) had issues"):
            for msg in failed:
                st.text(msg)

    if not rows:
        st.error("No SERP data retrieved. Check your credentials and keywords.")
        return None

    return pd.DataFrame(rows)


# ----------------
# Clustering Functions
# ----------------

def build_similarity_matrix(query_map, common_urls_threshold):
    """Build similarity matrix between queries based on shared URLs."""
    similarity_matrix = defaultdict(dict)
    queries = list(query_map.keys())

    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            query1, query2 = queries[i], queries[j]
            shared_urls = len(query_map[query1] & query_map[query2])
            if shared_urls >= common_urls_threshold:
                similarity_matrix[query1][query2] = shared_urls
                similarity_matrix[query2][query1] = shared_urls

    return similarity_matrix, queries


def find_connected_components(similarity_matrix, queries):
    """Find connected components in the similarity graph."""
    visited = set()
    components = []

    def dfs(query, component):
        if query in visited:
            return
        visited.add(query)
        component.add(query)
        for neighbor in similarity_matrix[query]:
            dfs(neighbor, component)

    for query in queries:
        if query not in visited and query in similarity_matrix:
            component = set()
            dfs(query, component)
            if len(component) > 1:
                components.append(component)

    return components


def calculate_consolidation_score(cluster_size, avg_shared_urls, connectivity):
    """Calculate consolidation opportunity score (0-100)."""
    base_score = min(40, avg_shared_urls * 4)
    connectivity_bonus = connectivity * 30
    size_bonus = min(20, (cluster_size - 2) * 5)
    return max(0, min(100, round(base_score + connectivity_bonus + size_bonus)))


def get_recommendation(score):
    """Get recommendation based on score."""
    if score >= 80:
        return "Strong consolidation candidate"
    elif score >= 60:
        return "Good consolidation candidate"
    elif score >= 40:
        return "Possible consolidation"
    elif score >= 20:
        return "Weak candidate"
    else:
        return "Keep separate"


def process_clustering(df, common_urls, progress_bar, status_text):
    """Main clustering function."""
    # Prepare data
    status_text.text("Preparing data...")
    df = df.rename(columns={
        df.columns[0]: 'query',
        df.columns[1]: 'link'
    })
    df['query'] = df['query'].str.lower()
    df = df.drop_duplicates(subset=['query', 'link'])

    # Create query map
    status_text.text("Building query map...")
    query_map = df.groupby('query')['link'].apply(set).to_dict()
    progress_bar.progress(0.3)

    # Build similarity matrix
    status_text.text("Calculating similarities...")
    similarity_matrix, queries = build_similarity_matrix(query_map, common_urls)
    progress_bar.progress(0.5)

    # Find clusters
    status_text.text("Finding clusters...")
    clusters = find_connected_components(similarity_matrix, queries)
    progress_bar.progress(0.7)

    # Build results
    status_text.text("Building results...")
    results = []

    for cluster in clusters:
        cluster_list = list(cluster)
        cluster_name = min(cluster_list, key=len)

        # Find shared URLs
        shared_urls = set(query_map[cluster_list[0]])
        for q in cluster_list[1:]:
            shared_urls &= query_map[q]

        # Calculate metrics
        total_shared = 0
        comparisons = 0
        for i in range(len(cluster_list)):
            for j in range(i + 1, len(cluster_list)):
                if cluster_list[j] in similarity_matrix[cluster_list[i]]:
                    total_shared += similarity_matrix[cluster_list[i]][cluster_list[j]]
                    comparisons += 1

        avg_shared = total_shared / comparisons if comparisons > 0 else 0
        possible_connections = len(cluster_list) * (len(cluster_list) - 1) / 2
        connectivity = comparisons / possible_connections if possible_connections > 0 else 0

        score = calculate_consolidation_score(len(cluster_list), avg_shared, connectivity)

        for query in cluster_list:
            results.append({
                'Cluster': cluster_name,
                'Query': query,
                'Cluster Size': len(cluster_list),
                'Shared URLs': len(shared_urls),
                'Avg Shared': round(avg_shared, 2),
                'Connectivity': round(connectivity, 2),
                'Score': score,
                'Recommendation': get_recommendation(score),
                'Sample URLs': ', '.join(list(shared_urls)[:3])
            })

    # Add unclustered queries
    clustered_queries = set(q for c in clusters for q in c)
    for query in query_map.keys():
        if query not in clustered_queries:
            results.append({
                'Cluster': 'NO_CLUSTER',
                'Query': query,
                'Cluster Size': 1,
                'Shared URLs': 0,
                'Avg Shared': 0,
                'Connectivity': 0,
                'Score': 0,
                'Recommendation': 'Keep separate',
                'Sample URLs': ''
            })

    progress_bar.progress(1.0)
    return pd.DataFrame(results)


def process_clustering_from_live(df, common_urls, progress_bar, status_text):
    """
    Clustering function for live-fetched data.
    The DataFrame already has 'query' and 'link' columns.
    """
    status_text.text("Preparing data...")
    df['query'] = df['query'].str.lower()
    df = df.drop_duplicates(subset=['query', 'link'])

    # Create query map
    status_text.text("Building query map...")
    query_map = df.groupby('query')['link'].apply(set).to_dict()
    progress_bar.progress(0.85)

    # Build similarity matrix
    status_text.text("Calculating similarities...")
    similarity_matrix, queries = build_similarity_matrix(query_map, common_urls)
    progress_bar.progress(0.9)

    # Find clusters
    status_text.text("Finding clusters...")
    clusters = find_connected_components(similarity_matrix, queries)
    progress_bar.progress(0.95)

    # Build results
    status_text.text("Building results...")
    results = []

    for cluster in clusters:
        cluster_list = list(cluster)
        cluster_name = min(cluster_list, key=len)

        # Find shared URLs
        shared_urls = set(query_map[cluster_list[0]])
        for q in cluster_list[1:]:
            shared_urls &= query_map[q]

        # Calculate metrics
        total_shared = 0
        comparisons = 0
        for i in range(len(cluster_list)):
            for j in range(i + 1, len(cluster_list)):
                if cluster_list[j] in similarity_matrix[cluster_list[i]]:
                    total_shared += similarity_matrix[cluster_list[i]][cluster_list[j]]
                    comparisons += 1

        avg_shared = total_shared / comparisons if comparisons > 0 else 0
        possible_connections = len(cluster_list) * (len(cluster_list) - 1) / 2
        connectivity = comparisons / possible_connections if possible_connections > 0 else 0

        score = calculate_consolidation_score(len(cluster_list), avg_shared, connectivity)

        for query in cluster_list:
            results.append({
                'Cluster': cluster_name,
                'Query': query,
                'Cluster Size': len(cluster_list),
                'Shared URLs': len(shared_urls),
                'Avg Shared': round(avg_shared, 2),
                'Connectivity': round(connectivity, 2),
                'Score': score,
                'Recommendation': get_recommendation(score),
                'Sample URLs': ', '.join(list(shared_urls)[:3])
            })

    # Add unclustered queries
    clustered_queries = set(q for c in clusters for q in c)
    for query in query_map.keys():
        if query not in clustered_queries:
            results.append({
                'Cluster': 'NO_CLUSTER',
                'Query': query,
                'Cluster Size': 1,
                'Shared URLs': 0,
                'Avg Shared': 0,
                'Connectivity': 0,
                'Score': 0,
                'Recommendation': 'Keep separate',
                'Sample URLs': ''
            })

    progress_bar.progress(1.0)
    return pd.DataFrame(results)


# ----------------
# Display Results
# ----------------

def display_results(results_df, elapsed):
    """Display clustering results with tabs, charts, and download options."""
    clustered = results_df[results_df['Cluster'] != 'NO_CLUSTER']
    st.success(f"Found {clustered['Cluster'].nunique()} clusters in {elapsed:.1f}s!")

    # Results tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 All Results",
        "🏆 Top Opportunities",
        "📈 Statistics",
        "🎯 Clusters"
    ])

    with tab1:
        st.dataframe(results_df, use_container_width=True, height=400)

        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Results CSV",
            data=csv,
            file_name="serp_clusters.csv",
            mime="text/csv",
            use_container_width=True
        )

    with tab2:
        if len(clustered) > 0:
            top_clusters = (
                clustered.groupby('Cluster')
                .first()
                .reset_index()
                .nlargest(20, 'Score')
            )

            st.dataframe(
                top_clusters[['Cluster', 'Cluster Size', 'Score', 'Recommendation', 'Sample URLs']],
                use_container_width=True
            )
        else:
            st.info("No clusters found. Try lowering the minimum shared URLs threshold.")

    with tab3:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Queries", len(results_df))
        with col2:
            st.metric("Clustered", len(clustered))
        with col3:
            st.metric("Clusters", clustered['Cluster'].nunique())
        with col4:
            avg_score = clustered['Score'].mean() if len(clustered) > 0 else 0
            st.metric("Avg Score", f"{avg_score:.0f}")

        # Score distribution
        st.subheader("Score Distribution")
        if len(clustered) > 0:
            fig = px.histogram(
                clustered,
                x='Score',
                nbins=20,
                title='Consolidation Score Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)

        # Cluster size distribution
        st.subheader("Cluster Size Distribution")
        if len(clustered) > 0:
            size_counts = clustered.groupby('Cluster')['Cluster Size'].first().value_counts().sort_index()
            fig = px.bar(
                x=size_counts.index,
                y=size_counts.values,
                labels={'x': 'Cluster Size', 'y': 'Number of Clusters'},
                title='Distribution of Cluster Sizes'
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        if len(clustered) > 0:
            # List individual clusters
            cluster_names = clustered['Cluster'].unique()

            selected_cluster = st.selectbox(
                "Select a cluster to view",
                cluster_names,
                format_func=lambda x: f"{x} (Score: {clustered[clustered['Cluster'] == x]['Score'].iloc[0]})"
            )

            if selected_cluster:
                cluster_data = clustered[clustered['Cluster'] == selected_cluster]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Keywords", len(cluster_data))
                with col2:
                    st.metric("Score", cluster_data['Score'].iloc[0])
                with col3:
                    st.metric("Shared URLs", cluster_data['Shared URLs'].iloc[0])

                st.markdown("**Keywords in cluster:**")
                for query in cluster_data['Query'].tolist():
                    st.markdown(f"- {query}")

                st.markdown("**Sample shared URLs:**")
                urls = cluster_data['Sample URLs'].iloc[0]
                if urls:
                    for url in urls.split(', '):
                        st.markdown(f"- [{url[:60]}...]({url})")
        else:
            st.info("No clusters found. Try lowering the minimum shared URLs threshold.")


# ----------------
# Sidebar
# ----------------

with st.sidebar:
    st.header("⚙️ Configuration")

    common_urls = st.slider(
        "Minimum Shared URLs",
        min_value=1,
        max_value=10,
        value=4,
        help="Minimum URLs in common to cluster keywords"
    )

    st.markdown("---")

    st.header("🔑 DataForSEO API")
    st.markdown("Required for live SERP fetching. Leave blank to use CSV upload only.")

    dataforseo_login = st.text_input(
        "DataForSEO Login",
        value=os.environ.get("DATAFORSEO_LOGIN", ""),
        type="password",
        help="Your DataForSEO API login (email). Can also be set via DATAFORSEO_LOGIN env var.",
    )

    dataforseo_password = st.text_input(
        "DataForSEO Password",
        value=os.environ.get("DATAFORSEO_PASSWORD", ""),
        type="password",
        help="Your DataForSEO API password. Can also be set via DATAFORSEO_PASSWORD env var.",
    )

    location = st.selectbox(
        "Location",
        options=list(LOCATION_CODES.keys()),
        index=0,
        help="Target location for SERP results",
    )

    device = st.selectbox(
        "Device",
        options=["Desktop", "Mobile"],
        index=0,
        help="Device type for SERP results",
    )

    st.markdown("---")
    st.markdown("### 📖 Data Format")
    st.markdown("""
    **CSV Upload:**
    - Column 1: Search query
    - Column 2: Result URL

    **Live Fetch:**
    - One keyword per line
    - Cost: ~$0.002 per keyword
    """)


# ----------------
# Main Content - Input Tabs
# ----------------

input_tab1, input_tab2 = st.tabs(["📤 Upload SERP CSVs", "🌐 Fetch Live SERPs"])

with input_tab1:
    st.markdown("Upload pre-fetched SERP export CSV files (e.g. from SERP API or DataForSEO batch exports).")

    uploaded_files = st.file_uploader(
        "Upload SERP CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload one or more SERP export files with query and URL columns"
    )

    if uploaded_files:
        # Combine all files
        dfs = []
        for f in uploaded_files:
            try:
                df = pd.read_csv(f, usecols=[0, 1], dtype=str)
                dfs.append(df)
            except Exception as e:
                st.warning(f"Could not read {f.name}: {str(e)}")

        if dfs:
            df_combined = pd.concat(dfs, ignore_index=True)
            st.success(f"Loaded {len(df_combined):,} rows from {len(uploaded_files)} file(s)")

            # Preview
            with st.expander("Preview Data"):
                st.dataframe(df_combined.head(20), use_container_width=True)

            unique_queries = df_combined.iloc[:, 0].nunique()
            st.info(f"Found **{unique_queries:,}** unique queries")

            if st.button("🎯 Run Clustering", type="primary", use_container_width=True, key="csv_cluster"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                start_time = time.time()
                results_df = process_clustering(df_combined, common_urls, progress_bar, status_text)
                elapsed = time.time() - start_time

                progress_bar.empty()
                status_text.empty()

                display_results(results_df, elapsed)

    else:
        st.info("Upload SERP export CSV files to get started with this method.")

with input_tab2:
    st.markdown("Fetch live SERP data from DataForSEO. Enter your API credentials in the sidebar.")

    keywords_text = st.text_area(
        "Keywords (one per line)",
        height=200,
        placeholder="keyword 1\nkeyword 2\nkeyword 3\n...",
        help="Enter keywords to fetch SERPs for, one per line.",
    )

    if keywords_text:
        keywords = [k.strip() for k in keywords_text.strip().split("\n") if k.strip()]
        num_keywords = len(keywords)

        if num_keywords > 0:
            estimated_cost = num_keywords * COST_PER_KEYWORD
            st.info(
                f"**{num_keywords:,}** keywords entered. "
                f"Estimated cost: **${estimated_cost:.2f}** "
                f"(${COST_PER_KEYWORD} per keyword)"
            )

            # Check credentials
            has_credentials = bool(dataforseo_login and dataforseo_password)

            if not has_credentials:
                st.warning("Enter your DataForSEO login and password in the sidebar to enable live fetching.")

            if st.button(
                "🌐 Fetch and Cluster",
                type="primary",
                use_container_width=True,
                disabled=not has_credentials,
                key="live_cluster",
            ):
                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()

                # Fetch SERPs
                location_code = LOCATION_CODES[location]
                device_value = device.lower()

                df_live = fetch_serps_dataforseo(
                    keywords=keywords,
                    login=dataforseo_login,
                    password=dataforseo_password,
                    location_code=location_code,
                    device=device_value,
                    progress_bar=progress_bar,
                    status_text=status_text,
                )

                if df_live is not None and not df_live.empty:
                    st.success(
                        f"Retrieved {len(df_live):,} results for "
                        f"{df_live['query'].nunique()} keywords"
                    )

                    # Preview fetched data
                    with st.expander("Preview Fetched Data"):
                        st.dataframe(df_live.head(20), use_container_width=True)

                    # Cluster the results
                    results_df = process_clustering_from_live(
                        df_live, common_urls, progress_bar, status_text
                    )
                    elapsed = time.time() - start_time

                    progress_bar.empty()
                    status_text.empty()

                    display_results(results_df, elapsed)
                else:
                    progress_bar.empty()
                    status_text.empty()
    else:
        st.info("Enter keywords above and configure your DataForSEO credentials in the sidebar to fetch live SERPs.")

    with st.expander("ℹ️ About DataForSEO"):
        st.markdown("""
        [DataForSEO](https://dataforseo.com/) provides live SERP data via API.

        **Pricing:** ~$0.002 per keyword (10 organic results).

        **How it works:**
        1. Enter your API credentials in the sidebar
        2. Paste your keywords above (one per line)
        3. Select your target location and device
        4. Click "Fetch and Cluster" to retrieve SERPs and cluster in one step

        **Rate limiting:** Requests are sent one at a time with a 0.5s delay
        to avoid overloading the API.
        """)

# Footer
st.markdown("---")
st.markdown(
    "Built by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · "
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · "
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)"
)
