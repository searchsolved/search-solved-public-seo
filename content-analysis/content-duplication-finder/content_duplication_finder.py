import streamlit as st
import pandas as pd
from polyfuzz import PolyFuzz
from io import BytesIO

st.set_page_config(page_title="Content Duplication Finder", page_icon="🔍", layout="wide")

st.title("Content Duplication Finder")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools)")

with st.expander("How to use this tool"):
    st.markdown("""
    **Screaming Frog Setup:**
    1. Crawl your site with Canonicals enabled
    2. Set up a Custom Extraction for the main page content (name it 'Copy 1')
    3. Export the Internal HTML report as CSV

    **Using this tool:**
    1. Upload your Screaming Frog internal_html.csv export
    2. Set the minimum similarity threshold (0.9 = 90% similar)
    3. Optionally filter URLs (e.g., '/product' to only check product pages)
    4. Click 'Find Duplicates' to analyze
    5. Download the results showing duplicate content clusters

    **Required columns:** Address, H1-1, Copy 1
    """)

# Sidebar settings
st.sidebar.header("Settings")
min_sim = st.sidebar.slider("Minimum Similarity Score", min_value=0.5, max_value=1.0, value=0.9, step=0.05,
                            help="Higher = stricter matching. 0.9 means 90% similar content.")
url_filter = st.sidebar.text_input("URL Filter (optional)", value="",
                                   help="Only analyze URLs containing this text. Leave empty for all URLs.")
group_similarity = st.sidebar.slider("Group Link Similarity", min_value=0.5, max_value=1.0, value=0.75, step=0.05,
                                     help="Threshold for grouping similar content into clusters.")

# File upload
uploaded_file = st.file_uploader("Upload Screaming Frog internal_html.csv", type=['csv'])

if uploaded_file is not None:
    try:
        # Try different encodings
        try:
            df = pd.read_csv(uploaded_file, usecols=["Address", "H1-1", "Copy 1"])
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, usecols=["Address", "H1-1", "Copy 1"], encoding='latin-1')

        st.success(f"Loaded {len(df):,} URLs")

        # Apply URL filter if specified
        if url_filter:
            df = df[df['Address'].str.contains(url_filter, na=False)]
            st.info(f"Filtered to {len(df):,} URLs containing '{url_filter}'")

        # Remove rows with missing content
        df_clean = df[df["Copy 1"].notna()].copy()
        missing_count = len(df) - len(df_clean)
        if missing_count > 0:
            st.warning(f"Removed {missing_count:,} URLs with missing content")

        if len(df_clean) < 2:
            st.error("Need at least 2 URLs with content to compare")
            st.stop()

        with st.expander("Preview uploaded data"):
            st.dataframe(df_clean.head(20))

        if st.button("🔍 Find Duplicates", type="primary"):
            with st.spinner("Analyzing content similarity... This may take a few minutes for large sites."):
                # Make lists for comparison
                from_list = list(df_clean['Copy 1'])
                to_list = list(df_clean['Copy 1'])

                # Do the matching with PolyFuzz
                model = PolyFuzz("TF-IDF")
                model.match(from_list, to_list)

                # Group the matches
                model.group(link_min_similarity=group_similarity)

                # Get matches dataframe
                df_matches = model.get_matches()

                # Match URLs and H1s to the results
                df_matches = df_matches.merge(
                    df_clean.drop_duplicates('Copy 1'),
                    how='left',
                    left_on='Group',
                    right_on="Copy 1"
                )
                df_matches.rename(columns={
                    "Address": "Group Address",
                    "H1-1": "Group H1-1",
                    "Copy 1": "Group Copy 1"
                }, inplace=True)

                df_matches = df_matches.merge(
                    df_clean.drop_duplicates('Copy 1'),
                    how='left',
                    left_on='From',
                    right_on="Copy 1"
                )
                df_matches.rename(columns={
                    "Address": "Source URL",
                    "H1-1": "Source H1",
                    "Copy 1": "From Copy 1"
                }, inplace=True)

                df_matches = df_matches.merge(
                    df_clean.drop_duplicates('Copy 1'),
                    how='left',
                    left_on='To',
                    right_on="Copy 1"
                )
                df_matches.rename(columns={
                    "Address": "Matched URL",
                    "H1-1": "Matched H1",
                    "Copy 1": "To Copy 1"
                }, inplace=True)

                # Select and rename final columns
                df_matches = df_matches[["Source URL", "Matched URL", "Similarity", "Group H1-1"]].copy()
                df_matches.rename(columns={"Group H1-1": "Duplicate Cluster Name"}, inplace=True)

                # Filter results
                df_matches = df_matches[df_matches['Source URL'] != df_matches['Matched URL']]  # Remove self-matches
                df_matches = df_matches[df_matches['Similarity'] >= min_sim]  # Apply similarity threshold

                if len(df_matches) == 0:
                    st.success("No duplicate content found above the similarity threshold!")
                else:
                    # Add cluster size
                    df_matches['Pages in Cluster'] = df_matches['Duplicate Cluster Name'].map(
                        df_matches.groupby('Duplicate Cluster Name')['Duplicate Cluster Name'].count()
                    )

                    # Sort by cluster size
                    df_matches = df_matches.sort_values(by=["Pages in Cluster", "Duplicate Cluster Name"], ascending=[False, True])

                    # Round similarity
                    df_matches['Similarity'] = df_matches['Similarity'].round(3)

                    # Display results
                    st.success(f"Found {len(df_matches):,} duplicate content pairs in {df_matches['Duplicate Cluster Name'].nunique():,} clusters")

                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Duplicate Pairs", f"{len(df_matches):,}")
                    with col2:
                        st.metric("Unique Clusters", f"{df_matches['Duplicate Cluster Name'].nunique():,}")
                    with col3:
                        st.metric("Avg Similarity", f"{df_matches['Similarity'].mean():.1%}")

                    # Show results
                    st.subheader("Duplicate Content Clusters")
                    st.dataframe(df_matches, use_container_width=True)

                    # Download button
                    csv = df_matches.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="content_duplication_results.csv",
                        mime="text/csv"
                    )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Make sure your CSV has columns: Address, H1-1, Copy 1")

else:
    st.info("👆 Upload a Screaming Frog internal_html.csv export to get started")

    # Show example
    st.subheader("Example Output")
    example_data = {
        "Source URL": ["/product/widget-a", "/product/widget-b", "/product/gadget-1"],
        "Matched URL": ["/product/widget-a-copy", "/product/widget-b-v2", "/product/gadget-1-old"],
        "Similarity": [0.95, 0.92, 0.91],
        "Duplicate Cluster Name": ["Widget A Product", "Widget B Product", "Gadget Product"],
        "Pages in Cluster": [3, 2, 2]
    }
    st.dataframe(pd.DataFrame(example_data))
