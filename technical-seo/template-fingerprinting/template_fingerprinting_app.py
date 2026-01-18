"""
Template Fingerprinting Tool - Streamlit App

Analyzes the HTML structure of pages to automatically identify and group pages
by template type using TF-IDF vectorization and K-Means clustering.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import time

st.set_page_config(
    page_title="Template Fingerprinting Tool",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Template Fingerprinting Tool")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Identifies page templates from crawl data
    - Groups pages by structural similarity
    - Detects template patterns automatically

    **How to use:**
    1. Upload crawl data with HTML
    2. Configure fingerprinting settings
    3. Analyze template patterns
    4. Download template groups

    **Best for:**
    - Large site audits
    - Template-based optimization
    - CMS pattern identification
    """)
st.markdown("Automatically classify pages into template types using HTML structure analysis.")


def fetch_html(url, timeout=10):
    """Fetches HTML content from a URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=timeout, headers=headers)
        return response.text
    except Exception as e:
        return None


def extract_features(html):
    """Extracts structural features from HTML for fingerprinting."""
    if html is None:
        return ""

    soup = BeautifulSoup(html, 'html.parser')
    features = []

    # Extract tag counts
    tag_counts = Counter(tag.name for tag in soup.find_all())
    features.extend([f"{tag}:{count}" for tag, count in tag_counts.items()])

    # Extract class names
    class_counts = Counter(cls for tag in soup.find_all() for cls in tag.get('class', []))
    features.extend([f"class:{cls}" for cls in class_counts])

    # Extract id attributes
    id_counts = Counter(tag.get('id') for tag in soup.find_all() if tag.get('id'))
    features.extend([f"id:{id_val}" for id_val in id_counts])

    # Extract meta tags
    meta_tags = soup.find_all('meta')
    features.extend([f"meta:{tag.get('name', tag.get('property', ''))}" for tag in meta_tags])

    return " ".join(features)


def classify_pages(urls, n_clusters, timeout, progress_bar, status_text):
    """Main function to classify pages by template type."""
    features = []
    successful_urls = []
    failed_urls = []

    total = len(urls)
    for i, url in enumerate(urls):
        progress_bar.progress((i + 1) / total)
        status_text.text(f"Processing {i + 1}/{total}: {url[:60]}...")

        html = fetch_html(url, timeout)
        if html:
            features.append(extract_features(html))
            successful_urls.append(url)
        else:
            failed_urls.append(url)

        time.sleep(0.1)  # Small delay to be respectful

    if len(successful_urls) < n_clusters:
        return None, None, failed_urls, "Not enough successful URLs for the specified number of clusters"

    # Vectorize features
    status_text.text("Vectorizing features...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(features)

    # Perform clustering
    status_text.text(f"Performing K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    # Create results dataframe
    df_result = pd.DataFrame({
        'URL': successful_urls,
        'Cluster': clusters,
        'Page Type': [f"Template Type {i}" for i in clusters]
    })

    # Get top features for each cluster
    cluster_features = {}
    for i in range(n_clusters):
        cluster_mask = clusters == i
        if cluster_mask.sum() > 0:
            cluster_vectors = X[cluster_mask]
            mean_vector = cluster_vectors.mean(axis=0).A1
            top_indices = mean_vector.argsort()[-10:][::-1]
            top_words = [vectorizer.get_feature_names_out()[idx] for idx in top_indices]
            cluster_features[i] = top_words

    return df_result, cluster_features, failed_urls, None


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    n_clusters = st.slider(
        "Number of Template Types",
        min_value=2,
        max_value=20,
        value=5,
        help="How many template types to identify"
    )

    timeout = st.slider(
        "Request Timeout (seconds)",
        min_value=5,
        max_value=30,
        value=10,
        help="Timeout for fetching each URL"
    )

    st.markdown("---")
    st.markdown("### 📤 How to Use")
    st.markdown("""
    1. Upload a CSV with an 'Address' column
    2. Or export from Screaming Frog
    3. Configure number of clusters
    4. Click 'Analyze Templates'
    """)

# Main content area
uploaded_file = st.file_uploader(
    "Upload CSV with URLs",
    type=["csv"],
    help="CSV file should contain an 'Address' column with URLs to analyze"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Find URL column
    url_column = None
    for col in ['Address', 'URL', 'url', 'address', 'Url']:
        if col in df.columns:
            url_column = col
            break

    if url_column is None:
        url_column = df.columns[0]
        st.warning(f"No 'Address' column found. Using '{url_column}' as URL column.")

    urls = df[url_column].dropna().tolist()
    urls = [u for u in urls if isinstance(u, str) and u.startswith('http')]

    st.info(f"Found **{len(urls)}** valid URLs to analyze")

    # Show preview
    with st.expander("Preview URLs"):
        st.dataframe(pd.DataFrame({'URL': urls[:20]}), use_container_width=True)

    if st.button("🔍 Analyze Templates", type="primary", use_container_width=True):
        if len(urls) < n_clusters:
            st.error(f"Need at least {n_clusters} URLs for {n_clusters} clusters. Found only {len(urls)}.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("Analyzing page templates..."):
                df_result, cluster_features, failed_urls, error = classify_pages(
                    urls, n_clusters, timeout, progress_bar, status_text
                )

            progress_bar.empty()
            status_text.empty()

            if error:
                st.error(error)
            else:
                st.success(f"✅ Successfully classified {len(df_result)} pages into {n_clusters} template types!")

                # Show failed URLs if any
                if failed_urls:
                    with st.expander(f"⚠️ Failed to fetch {len(failed_urls)} URLs"):
                        st.dataframe(pd.DataFrame({'Failed URL': failed_urls}))

                # Results tabs
                tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Visualization", "🔧 Cluster Details"])

                with tab1:
                    st.subheader("Classification Results")
                    st.dataframe(df_result, use_container_width=True, height=400)

                    # Download button
                    csv = df_result.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="template_classification_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with tab2:
                    st.subheader("Template Distribution")

                    # Pie chart
                    cluster_counts = df_result['Page Type'].value_counts()
                    fig_pie = px.pie(
                        values=cluster_counts.values,
                        names=cluster_counts.index,
                        title="Template Type Distribution"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                    # Bar chart
                    fig_bar = px.bar(
                        x=cluster_counts.index,
                        y=cluster_counts.values,
                        labels={'x': 'Template Type', 'y': 'Number of Pages'},
                        title="Pages per Template Type"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                with tab3:
                    st.subheader("Cluster Characteristics")

                    for cluster_id, features in cluster_features.items():
                        count = (df_result['Cluster'] == cluster_id).sum()
                        with st.expander(f"Template Type {cluster_id} ({count} pages)"):
                            st.markdown("**Top distinguishing features:**")
                            for feat in features[:10]:
                                st.markdown(f"- `{feat}`")

                            # Show sample URLs from this cluster
                            sample_urls = df_result[df_result['Cluster'] == cluster_id]['URL'].head(5).tolist()
                            st.markdown("**Sample URLs:**")
                            for url in sample_urls:
                                st.markdown(f"- [{url[:60]}...]({url})")

else:
    # Show example input format
    st.info("👆 Upload a CSV file with URLs to get started")

    with st.expander("Example CSV Format"):
        example_df = pd.DataFrame({
            'Address': [
                'https://example.com/product/item-1',
                'https://example.com/category/shoes',
                'https://example.com/blog/post-1',
                'https://example.com/product/item-2',
            ]
        })
        st.dataframe(example_df)
        st.markdown("The CSV should have an 'Address' column (or similar) containing full URLs.")

# Footer
st.markdown("---")
st.markdown(
    "Built by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · "
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · "
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)"
)
