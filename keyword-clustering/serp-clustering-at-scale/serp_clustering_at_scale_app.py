# Author: Lee Foot
# Website: https://leefoot.com

"""
SERP Clustering at Scale - Streamlit App

Clusters keywords based on common SERP URLs to identify content consolidation opportunities.
Supports CSV upload and SERP data from the DataForSEO SERP API (Standard queue:
send the searches now, collect the results later).
"""

import os
import time
from collections import defaultdict

import streamlit as st
import pandas as pd
import plotly.express as px

from dataforseo_queue import (
    QueueError,
    collect_results,
    estimate_cost,
    is_unresolved,
    job_from_csv,
    job_to_csv,
    organic_items,
    send_searches,
)

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

    **Option 2: Fetch SERPs from DataForSEO (two steps)**

    *Why two steps:* Google now routes result links through google.com/goto. DataForSEO's
    Standard queue turns those back into real URLs, but a search can take 25 minutes or
    more to finish. So you send the searches, keep the job file, and collect the results later.

    1. Enter your DataForSEO API credentials in the sidebar
    2. **1. Send searches:** paste keywords (one per line), choose the location, device and
       results per keyword, check the estimated cost and click "Send searches". Download the job file.
    3. **2. Collect results** (about 30 minutes later): upload the job file and click
       "Collect and Cluster". The minimum shared URLs setting in the sidebar can be changed afterwards.

    **Best for:**
    - Large-scale keyword research
    - Content consolidation planning
    - Identifying search intent groups
    """)
st.markdown("Cluster keywords by shared SERP URLs to find content consolidation opportunities.")

# ----------------
# DataForSEO Configuration
# ----------------

LANGUAGE_CODE = "en"

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
# DataForSEO Standard queue results
# ----------------

def serps_to_dataframe(ready):
    """
    Turn collected DataForSEO results ({keyword: SERP items}) into a DataFrame with
    columns 'query' and 'link'. Organic URLs left as unresolved google.com/goto links
    are skipped and counted. Returns (df, unresolved).
    """
    rows = []
    unresolved = 0
    for keyword, items in ready.items():
        for item in organic_items(items):
            url = item.get("url", "")
            if is_unresolved(url):
                unresolved += 1
            elif url:
                rows.append({"query": keyword, "link": url})
    return pd.DataFrame(rows, columns=["query", "link"]), unresolved


def decode_bytes(raw_bytes):
    """Decode an uploaded file, trying a few common encodings."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the file.")


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

def display_results(results_df, elapsed, key_prefix="csv"):
    """Display clustering results with tabs, charts, and download options.

    key_prefix keeps widget keys unique when both input methods show results on one page.
    """
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
        st.dataframe(results_df, width="stretch", height=400, key=f"{key_prefix}_all_results")

        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Results CSV",
            data=csv,
            file_name="serp_clusters.csv",
            mime="text/csv",
            width="stretch",
            key=f"{key_prefix}_download_results",
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
                width="stretch",
                key=f"{key_prefix}_top_clusters",
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
            st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_score_chart")

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
            st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_size_chart")

    with tab4:
        if len(clustered) > 0:
            # List individual clusters
            cluster_names = clustered['Cluster'].unique()

            selected_cluster = st.selectbox(
                "Select a cluster to view",
                cluster_names,
                key=f"{key_prefix}_select_cluster",
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
    st.markdown("Required to send searches and collect results. Leave blank to use CSV upload only.")

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

    st.markdown("---")
    st.markdown("### 📖 Data Format")
    st.markdown("""
    **CSV Upload:**
    - Column 1: Search query
    - Column 2: Result URL

    **DataForSEO:**
    - One keyword per line
    - Cost: keywords x $0.0006 x (results per keyword / 10)
    """)


# ----------------
# Main Content - Input Tabs
# ----------------

input_tab1, input_tab2, input_tab3 = st.tabs(["📤 Upload SERP CSVs", "🌐 1. Send searches", "📥 2. Collect results"])

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
                st.dataframe(df_combined.head(20), width="stretch")

            unique_queries = df_combined.iloc[:, 0].nunique()
            st.info(f"Found **{unique_queries:,}** unique queries")

            if st.button("🎯 Run Clustering", type="primary", width="stretch", key="csv_cluster"):
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

has_credentials = bool(dataforseo_login and dataforseo_password)

with input_tab2:
    st.markdown("Send your keywords to DataForSEO. Enter your API credentials in the sidebar.")

    col1, col2, col3 = st.columns(3)
    with col1:
        location = st.selectbox(
            "Location",
            options=list(LOCATION_CODES.keys()),
            index=0,
            help="Target location for SERP results",
        )
    with col2:
        device = st.selectbox(
            "Device",
            options=["Desktop", "Mobile"],
            index=0,
            help="Device type for SERP results",
        )
    with col3:
        depth = st.slider(
            "Results per keyword",
            min_value=10,
            max_value=100,
            value=10,
            step=10,
            help="Number of Google results to fetch per keyword. DataForSEO bills per 10 results.",
        )

    keywords_text = st.text_area(
        "Keywords (one per line)",
        height=200,
        placeholder="keyword 1\nkeyword 2\nkeyword 3\n...",
        help="Enter keywords to fetch SERPs for, one per line.",
    )

    # Exact duplicates are sent once
    keywords = list(dict.fromkeys(k.strip() for k in keywords_text.split("\n") if k.strip()))

    if keywords:
        st.info(
            f"**{len(keywords):,}** keywords entered at {depth} results each. "
            f"Estimated cost: **${estimate_cost(len(keywords), depth):.4f}**"
        )
        if not has_credentials:
            st.warning("Enter your DataForSEO login and password in the sidebar to send searches.")
    else:
        st.info("Enter keywords above and configure your DataForSEO credentials in the sidebar to send searches.")

    if st.button(
        "🌐 Send searches",
        type="primary",
        width="stretch",
        disabled=not has_credentials or not keywords,
        key="send_searches",
    ):
        try:
            with st.spinner("Sending searches to DataForSEO..."):
                job_rows, send_warnings, send_cost = send_searches(
                    keywords, dataforseo_login, dataforseo_password,
                    LOCATION_CODES[location], LANGUAGE_CODE,
                    device=device.lower(), depth=depth,
                )
        except QueueError as e:
            st.error(str(e))
        else:
            for warning in send_warnings:
                st.warning(warning)
            if job_rows:
                st.session_state["clustering_job"] = job_rows
                st.session_state["clustering_send_cost"] = send_cost

    if st.session_state.get("clustering_job"):
        job_rows = st.session_state["clustering_job"]
        st.success(
            f"Queued {len(job_rows)} search(es). Cost reported by DataForSEO: "
            f"${st.session_state['clustering_send_cost']:.4f}"
        )
        st.markdown(
            "DataForSEO can take **25 minutes or more** to finish. Download the job file, "
            "then use **2. Collect results**. While this page stays open, the job is kept "
            "here too, so you can collect without uploading it."
        )
        st.download_button(
            label="📥 Download job file (CSV)",
            data=job_to_csv(job_rows).encode("utf-8"),
            file_name="serp_clustering_job.csv",
            mime="text/csv",
            width="stretch",
        )

    with st.expander("ℹ️ About DataForSEO"):
        st.markdown("""
        [DataForSEO](https://dataforseo.com/) provides SERP data via API.

        **Pricing:** keywords x $0.0006 x (results per keyword / 10), using the Standard queue.

        **Why two steps:** Google now routes result links through google.com/goto. The Standard
        queue turns those back into real URLs, but a search can take 25 minutes or more to finish.

        **How it works:**
        1. Enter your API credentials in the sidebar
        2. Paste your keywords above (one per line)
        3. Select your target location, device and results per keyword
        4. Click "Send searches" and download the job file
        5. About 30 minutes later, open **2. Collect results** and collect and cluster

        **Rate limiting:** Searches are sent in batches of up to 100, and results are
        collected one at a time with a 0.5s delay to avoid overloading the API.
        """)

with input_tab3:
    st.markdown("Collect the finished searches and cluster them. "
                "Searches can take 25 minutes or more to finish after sending.")

    job_rows = st.session_state.get("clustering_job")
    job_file = st.file_uploader(
        "Job file from step 1",
        type=["csv"],
        help="The serp_clustering_job.csv you downloaded when you sent the searches",
    )
    if job_file is not None:
        try:
            job_rows = job_from_csv(decode_bytes(job_file.getvalue()))
        except ValueError as e:
            st.error(str(e))
            job_rows = None
    elif job_rows:
        st.caption(f"Using the job sent from this page ({len(job_rows)} search(es)).")

    if not has_credentials:
        st.warning("Enter your DataForSEO login and password in the sidebar to collect results.")

    if st.button(
        "🎯 Collect and Cluster",
        type="primary",
        width="stretch",
        disabled=not has_credentials or not job_rows,
        key="collect_cluster",
    ):
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()

        def update_progress(done, total, keyword):
            progress_bar.progress((done / total if total else 1.0) * 0.8)  # Reserve 0.8-1.0 for clustering
            status_text.text(f"Checking {done + 1}/{total}: {keyword}" if keyword else "")

        try:
            result = collect_results(job_rows, dataforseo_login, dataforseo_password,
                                     progress_callback=update_progress)
        except QueueError as e:
            progress_bar.empty()
            status_text.empty()
            st.error(str(e))
        else:
            df_live, unresolved = serps_to_dataframe(result["ready"])
            collected = {
                "df_live": df_live,
                "ready": len(result["ready"]),
                "pending": result["pending"],
                "errors": result["errors"],
                "unresolved": unresolved,
                "total": len(job_rows),
                "results_df": None,
                "threshold": None,
                "elapsed": 0.0,
            }
            if not df_live.empty:
                collected["results_df"] = process_clustering_from_live(
                    df_live.copy(), common_urls, progress_bar, status_text
                )
                collected["threshold"] = common_urls
                collected["elapsed"] = time.time() - start_time
            progress_bar.empty()
            status_text.empty()
            st.session_state["clustering_collected"] = collected

    collected = st.session_state.get("clustering_collected")
    if collected:
        if collected["pending"]:
            st.warning(
                f"{collected['ready']} of {collected['total']} search(es) have finished; "
                f"{len(collected['pending'])} are still running at DataForSEO. "
                "Try again in a few minutes. The clusters below cover the finished ones only."
            )
        if collected["errors"]:
            with st.expander(f"⚠️ {len(collected['errors'])} search(es) failed"):
                for msg in collected["errors"]:
                    st.text(msg)
        if collected["unresolved"]:
            st.warning(
                f"{collected['unresolved']} result(s) came back as unresolved google.com/goto links "
                "and were left out of the clustering."
            )

        df_live = collected["df_live"]
        if collected["ready"] and df_live.empty:
            st.error("No SERP URLs to cluster in the finished searches.")
        elif not df_live.empty:
            st.success(
                f"Retrieved {len(df_live):,} results for "
                f"{df_live['query'].nunique()} keywords"
            )

            # Preview fetched data
            with st.expander("Preview Fetched Data"):
                st.dataframe(df_live.head(20), width="stretch")

            # Re-cluster when the minimum shared URLs setting has changed since collecting
            if collected["threshold"] != common_urls:
                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()
                collected["results_df"] = process_clustering_from_live(
                    df_live.copy(), common_urls, progress_bar, status_text
                )
                collected["threshold"] = common_urls
                collected["elapsed"] = time.time() - start_time
                progress_bar.empty()
                status_text.empty()

            display_results(collected["results_df"], collected["elapsed"], key_prefix="queue")

# Footer
st.markdown("---")
st.markdown(
    "Built by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · "
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · "
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)"
)
