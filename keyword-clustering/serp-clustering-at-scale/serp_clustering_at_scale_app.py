"""
SERP Clustering at Scale - Streamlit App

Clusters keywords based on common SERP URLs to identify content consolidation opportunities.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
from collections import defaultdict
from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go
import time

st.set_page_config(
    page_title="SERP Clustering at Scale",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 SERP Clustering at Scale")
st.markdown("Cluster keywords by shared SERP URLs to find content consolidation opportunities.")


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


# Sidebar
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
    st.markdown("### 📖 Data Format")
    st.markdown("""
    Export from ValueSERP or similar:
    - Column 1: Search query
    - Column 2: Result URL

    File should be named:
    `Batch_Results_*.csv`
    """)

# Main content
st.markdown("### 📤 Upload SERP Data")

uploaded_files = st.file_uploader(
    "Upload ValueSERP CSV files",
    type=["csv"],
    accept_multiple_files=True,
    help="Upload one or more SERP export files"
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
        st.success(f"✅ Loaded {len(df_combined):,} rows from {len(uploaded_files)} file(s)")

        # Preview
        with st.expander("Preview Data"):
            st.dataframe(df_combined.head(20), use_container_width=True)

        unique_queries = df_combined.iloc[:, 0].nunique()
        st.info(f"Found **{unique_queries:,}** unique queries")

        if st.button("🎯 Run Clustering", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            start_time = time.time()
            results_df = process_clustering(df_combined, common_urls, progress_bar, status_text)
            elapsed = time.time() - start_time

            progress_bar.empty()
            status_text.empty()

            clustered = results_df[results_df['Cluster'] != 'NO_CLUSTER']
            st.success(f"✅ Found {clustered['Cluster'].nunique()} clusters in {elapsed:.1f}s!")

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
    st.info("👆 Upload SERP export CSV files to get started")

    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        This tool identifies **content consolidation opportunities** by analyzing
        SERP overlap between keywords:

        1. **Upload** SERP data from ValueSERP or similar tools
        2. **Clustering** groups keywords that share many SERP results
        3. **Scoring** ranks clusters by consolidation potential

        **High scores** indicate keywords that Google treats similarly and could
        potentially be targeted with a single, comprehensive page.
        """)

# Footer
st.markdown("---")
st.markdown(
    "Built by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · "
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · "
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)"
)
