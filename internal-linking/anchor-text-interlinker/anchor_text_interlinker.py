import streamlit as st
import pandas as pd
from io import BytesIO
from rapidfuzz import fuzz
import re

st.set_page_config(page_title="Anchor Text Interlinker", page_icon="🔗", layout="wide")

st.title("Anchor Text Interlinker")
st.markdown("*Created by [Lee Foot](https://leefoot.com)*")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Finds internal linking opportunities based on keyword matches in page content
    - Matches keywords from Ahrefs/SEMrush to content on your pages
    - Suggests anchor text and target pages for internal links

    **Files needed:**
    1. **Screaming Frog crawl** with custom content extraction (page body text)
    2. **Organic keywords export** from Ahrefs/SEMrush with URL, Keyword, Volume, Position

    **How to set up Screaming Frog extraction:**
    1. Configuration > Custom > Extraction
    2. Add CSS/XPath selector for your main content (e.g., `article`, `.content`, `#main`)
    3. Set extraction to "Extract Text"
    4. Crawl and export Internal HTML

    **How it works:**
    1. Extracts keywords from your organic rankings
    2. Searches for those keywords in your page content
    3. Suggests internal links where keyword appears in content but page isn't already linked
    """)

# Sidebar settings
st.sidebar.header("Settings")

max_kw_words = st.sidebar.slider(
    "Max words in keyword",
    min_value=2,
    max_value=10,
    value=6,
    help="Maximum number of words in keywords to match"
)

min_kw_words = st.sidebar.slider(
    "Min words in keyword",
    min_value=1,
    max_value=5,
    value=2,
    help="Skip single-word keywords (too generic)"
)

min_position = st.sidebar.number_input(
    "Max ranking position",
    min_value=1,
    max_value=100,
    value=50,
    help="Only use keywords ranking in top N positions"
)

min_similarity = st.sidebar.slider(
    "Min keyword/H1 similarity",
    min_value=0,
    max_value=100,
    value=50,
    help="Minimum similarity between keyword and target page H1"
)

# File uploads
st.subheader("Upload Your Data")

col1, col2 = st.columns(2)

with col1:
    crawl_file = st.file_uploader(
        "Screaming Frog Crawl with Content Extraction",
        type=['csv'],
        help="Must include Address, H1, and content extraction column"
    )

with col2:
    keywords_file = st.file_uploader(
        "Organic Keywords Export",
        type=['csv'],
        help="Ahrefs/SEMrush export with URL, Keyword, Volume, Position"
    )

if crawl_file and keywords_file:
    try:
        # Load crawl data
        try:
            df_crawl = pd.read_csv(crawl_file, encoding='utf-8')
        except:
            crawl_file.seek(0)
            df_crawl = pd.read_csv(crawl_file, encoding='latin-1')

        # Load keywords
        try:
            df_keywords = pd.read_csv(keywords_file, encoding='utf-8')
        except:
            keywords_file.seek(0)
            df_keywords = pd.read_csv(keywords_file, encoding='latin-1')

        st.success(f"Loaded {len(df_crawl):,} pages and {len(df_keywords):,} keywords")

        # Column mapping
        with st.expander("Map columns (if needed)"):
            crawl_cols = df_crawl.columns.tolist()
            kw_cols = df_keywords.columns.tolist()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Crawl file:**")
                url_col = st.selectbox(
                    "URL column",
                    crawl_cols,
                    index=crawl_cols.index("Address") if "Address" in crawl_cols else 0
                )
                h1_col = st.selectbox(
                    "H1 column",
                    crawl_cols,
                    index=crawl_cols.index("H1-1") if "H1-1" in crawl_cols else 0
                )
                # Find content extraction column
                content_cols = [c for c in crawl_cols if 'extract' in c.lower() or 'content' in c.lower() or 'copy' in c.lower()]
                default_content = crawl_cols.index(content_cols[0]) if content_cols else 0
                content_col = st.selectbox(
                    "Content extraction column",
                    crawl_cols,
                    index=default_content
                )

            with col2:
                st.markdown("**Keywords file:**")
                kw_url_col = st.selectbox(
                    "URL column",
                    kw_cols,
                    index=kw_cols.index("URL") if "URL" in kw_cols else 0
                )
                kw_col = st.selectbox(
                    "Keyword column",
                    kw_cols,
                    index=kw_cols.index("Keyword") if "Keyword" in kw_cols else 0
                )
                vol_col = st.selectbox(
                    "Volume column",
                    kw_cols,
                    index=kw_cols.index("Volume") if "Volume" in kw_cols else 0
                )
                pos_col = st.selectbox(
                    "Position column",
                    kw_cols,
                    index=kw_cols.index("Position") if "Position" in kw_cols else 0
                )

        if st.button("Find Linking Opportunities", type="primary"):
            with st.spinner("Finding internal linking opportunities..."):
                # Prepare crawl data
                df_crawl = df_crawl[[url_col, h1_col, content_col]].copy()
                df_crawl.columns = ["page", "h1", "content"]
                df_crawl = df_crawl[df_crawl["content"].notna()]
                df_crawl["content"] = df_crawl["content"].str.lower()
                df_crawl["h1"] = df_crawl["h1"].fillna("").str.lower()

                # Prepare keywords data
                df_keywords = df_keywords[[kw_url_col, kw_col, vol_col, pos_col]].copy()
                df_keywords.columns = ["target_page", "keyword", "volume", "position"]

                # Convert to numeric
                df_keywords["volume"] = pd.to_numeric(df_keywords["volume"], errors='coerce').fillna(0)
                df_keywords["position"] = pd.to_numeric(df_keywords["position"], errors='coerce').fillna(100)

                # Filter keywords
                df_keywords = df_keywords[df_keywords["position"] <= min_position]
                df_keywords["keyword"] = df_keywords["keyword"].str.lower().str.strip()

                # Count words in keyword
                df_keywords["word_count"] = df_keywords["keyword"].str.count(" ") + 1
                df_keywords = df_keywords[
                    (df_keywords["word_count"] >= min_kw_words) &
                    (df_keywords["word_count"] <= max_kw_words)
                ]

                # Drop duplicates - keep highest volume
                df_keywords = df_keywords.sort_values("volume", ascending=False)
                df_keywords = df_keywords.drop_duplicates(subset=["keyword"], keep="first")

                st.info(f"Processing {len(df_keywords):,} unique keywords across {len(df_crawl):,} pages")

                # Create keyword-to-target lookup
                keyword_targets = df_keywords.set_index("keyword")[["target_page", "volume", "position"]].to_dict('index')

                # Find matches
                opportunities = []
                progress = st.progress(0)

                for idx, row in df_crawl.iterrows():
                    source_page = row["page"]
                    content = row["content"]

                    for keyword, target_info in keyword_targets.items():
                        target_page = target_info["target_page"]

                        # Skip if source and target are the same
                        if source_page == target_page:
                            continue

                        # Check if keyword appears in content
                        # Use word boundary matching for exact phrase
                        pattern = r'\b' + re.escape(keyword) + r'\b'
                        if re.search(pattern, content):
                            # Get target H1 for similarity check
                            target_h1 = df_crawl[df_crawl["page"] == target_page]["h1"].values
                            target_h1 = target_h1[0] if len(target_h1) > 0 else ""

                            # Calculate similarity
                            similarity = fuzz.token_sort_ratio(keyword, target_h1)

                            if similarity >= min_similarity:
                                opportunities.append({
                                    "source_page": source_page,
                                    "anchor_text": keyword,
                                    "target_page": target_page,
                                    "target_h1": target_h1,
                                    "similarity": similarity,
                                    "volume": target_info["volume"],
                                    "position": target_info["position"]
                                })

                    progress.progress((idx + 1) / len(df_crawl))

                progress.empty()

                if opportunities:
                    df_opps = pd.DataFrame(opportunities)

                    # Remove duplicate source/target combinations
                    df_opps = df_opps.drop_duplicates(subset=["source_page", "target_page"], keep="first")

                    # Remove self-links
                    df_opps = df_opps[df_opps["source_page"] != df_opps["target_page"]]

                    # Sort by volume
                    df_opps = df_opps.sort_values("volume", ascending=False)

                    # Display results
                    st.subheader("Results Summary")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Opportunities", f"{len(df_opps):,}")
                    with col2:
                        st.metric("Unique Source Pages", f"{df_opps['source_page'].nunique():,}")
                    with col3:
                        st.metric("Unique Target Pages", f"{df_opps['target_page'].nunique():,}")
                    with col4:
                        st.metric("Unique Keywords", f"{df_opps['anchor_text'].nunique():,}")

                    st.subheader("Internal Linking Opportunities")
                    st.dataframe(df_opps, use_container_width=True)

                    # Top opportunities by volume
                    st.subheader("Top 20 Opportunities by Search Volume")
                    st.dataframe(df_opps.head(20), use_container_width=True)

                    # Download
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df_opps.to_excel(writer, sheet_name='All Opportunities', index=False)
                        df_opps.head(100).to_excel(writer, sheet_name='Top 100', index=False)

                        # Group by source page
                        df_by_source = df_opps.groupby("source_page").agg({
                            "anchor_text": "count",
                            "volume": "sum"
                        }).reset_index()
                        df_by_source.columns = ["Source Page", "Link Count", "Total Volume"]
                        df_by_source = df_by_source.sort_values("Total Volume", ascending=False)
                        df_by_source.to_excel(writer, sheet_name='By Source Page', index=False)

                    st.download_button(
                        label="Download Excel Report",
                        data=output.getvalue(),
                        file_name="internal_linking_opportunities.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.warning("No linking opportunities found. Try adjusting the filters.")

    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("Upload both files to get started")

    st.subheader("Example Output")
    example_data = {
        "Source Page": ["/blog/home-decor-tips/", "/guides/living-room-ideas/"],
        "Anchor Text": ["velvet sofas", "coffee tables"],
        "Target Page": ["/category/velvet-sofas/", "/category/coffee-tables/"],
        "Volume": [2400, 1800],
        "Similarity": [95, 88]
    }
    st.dataframe(pd.DataFrame(example_data))
