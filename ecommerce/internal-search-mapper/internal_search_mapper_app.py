"""
Internal Search Mapper - Streamlit App

Maps internal site search queries to existing category pages using fuzzy matching.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd

try:
    from polyfuzz import PolyFuzz
except ImportError:
    st.error("Please install polyfuzz: pip install polyfuzz")
    st.stop()

st.set_page_config(
    page_title="Internal Search Mapper",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Internal Search Mapper")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")
st.markdown("Map internal site search queries to existing category pages using fuzzy matching.")


def process_mapping(df_ga, df_sf, search_col, h1_col, url_col, min_similarity):
    """Perform fuzzy matching between search terms and page H1s."""
    # Convert to lowercase for matching
    df_sf = df_sf.copy()
    df_ga = df_ga.copy()

    df_sf[h1_col] = df_sf[h1_col].str.lower()
    df_ga[search_col] = df_ga[search_col].str.lower()

    # Drop non-indexable pages if column exists
    if 'Indexability' in df_sf.columns:
        df_sf = df_sf[~df_sf['Indexability'].isin(['Non-Indexable'])]

    # Keep rows without NaN
    df_ga = df_ga[df_ga[search_col].notna()]
    df_sf = df_sf[df_sf[h1_col].notna()]

    # Create lists
    ga_list = list(df_ga[search_col])
    sf_list = list(df_sf[h1_col])

    if not ga_list or not sf_list:
        return None, "No valid data to match"

    # Perform matching with PolyFuzz
    model = PolyFuzz("TF-IDF").match(ga_list, sf_list)
    df_matches = model.get_matches()

    # Keep only matched rows
    df_matches = df_matches[df_matches['To'].notna()]

    # Filter by minimum similarity
    df_matches = df_matches[df_matches['Similarity'] >= min_similarity]

    # Merge GA data back
    df_merged = pd.merge(
        df_matches, df_ga,
        left_on='From', right_on=search_col,
        how='inner'
    )

    # Merge SF data back
    df_final = pd.merge(
        df_merged, df_sf,
        left_on='To', right_on=h1_col,
        how='inner'
    )

    # Clean up columns
    cols_to_drop = [search_col, h1_col]
    for col in cols_to_drop:
        if col in df_final.columns:
            df_final = df_final.drop(columns=[col])

    # Rename columns
    df_final = df_final.rename(columns={
        'From': 'Search Term',
        'To': 'Matched H1',
        url_col: 'Matched URL'
    })

    # Round similarity
    df_final['Similarity'] = df_final['Similarity'].round(3)

    # Sort by search volume if available
    sort_col = None
    for col in df_final.columns:
        if 'search' in col.lower() and ('unique' in col.lower() or 'volume' in col.lower()):
            sort_col = col
            break

    if sort_col:
        df_final = df_final.sort_values(by=sort_col, ascending=False)

    # Drop duplicate search terms
    df_final = df_final.drop_duplicates(subset=['Search Term'])

    return df_final, None


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    min_similarity = st.slider(
        "Minimum Similarity Score",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Only show matches above this similarity threshold"
    )

    st.markdown("---")
    st.markdown("### 📖 Data Export Guide")
    st.markdown("""
    **GA Search Terms:**
    1. Go to Behavior → Site Search → Search Terms
    2. Export as Excel (.xlsx)

    **Screaming Frog Crawl:**
    1. Crawl your category pages
    2. Export internal_html.csv
    """)

# Main content
st.markdown("### 📤 Upload Data Files")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### GA Search Terms Export")
    ga_file = st.file_uploader(
        "Upload GA Excel",
        type=["xlsx", "xls"],
        key="ga",
        help="Export from: Analytics > Behavior > Site Search > Search Terms"
    )

with col2:
    st.markdown("#### Screaming Frog Crawl")
    sf_file = st.file_uploader(
        "Upload SF CSV",
        type=["csv"],
        key="sf",
        help="Export internal_html.csv from Screaming Frog"
    )

if ga_file and sf_file:
    try:
        # Load GA data
        try:
            df_ga = pd.read_excel(ga_file, sheet_name="Dataset1")
        except:
            df_ga = pd.read_excel(ga_file, sheet_name=0)

        # Load SF data
        df_sf = pd.read_csv(sf_file)

        st.success("✅ Files loaded successfully!")

        # Column selection
        st.markdown("### 🔧 Column Mapping")

        col1, col2, col3 = st.columns(3)

        with col1:
            search_col = st.selectbox(
                "Search Term Column (GA)",
                df_ga.columns.tolist(),
                index=0 if 'Search Term' not in df_ga.columns else df_ga.columns.tolist().index('Search Term')
            )

        with col2:
            h1_col = st.selectbox(
                "H1 Column (SF)",
                df_sf.columns.tolist(),
                index=0 if 'H1-1' not in df_sf.columns else df_sf.columns.tolist().index('H1-1')
            )

        with col3:
            url_col = st.selectbox(
                "URL Column (SF)",
                df_sf.columns.tolist(),
                index=0 if 'Address' not in df_sf.columns else df_sf.columns.tolist().index('Address')
            )

        # Preview data
        with st.expander("Preview Uploaded Data"):
            st.markdown("**GA Search Terms:**")
            st.dataframe(df_ga.head(), use_container_width=True)
            st.markdown("**Screaming Frog Crawl:**")
            st.dataframe(df_sf.head(), use_container_width=True)

        if st.button("🔍 Find Matches", type="primary", use_container_width=True):
            with st.spinner("Performing fuzzy matching..."):
                df_result, error = process_mapping(
                    df_ga, df_sf, search_col, h1_col, url_col, min_similarity
                )

            if error:
                st.error(error)
            elif df_result is not None and len(df_result) > 0:
                st.success(f"✅ Found {len(df_result)} search term matches!")

                # Split results
                exact_matches = df_result[df_result['Similarity'] == 1.0]
                partial_matches = df_result[df_result['Similarity'] < 1.0]

                # Results tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    f"📊 All Matches ({len(df_result)})",
                    f"✅ Exact ({len(exact_matches)})",
                    f"🔄 Partial ({len(partial_matches)})",
                    "📈 Stats"
                ])

                with tab1:
                    st.dataframe(df_result, use_container_width=True, height=400)
                    csv = df_result.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download All Matches",
                        data=csv,
                        file_name="search_mapping_all.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with tab2:
                    if len(exact_matches) > 0:
                        st.dataframe(exact_matches, use_container_width=True, height=400)
                        csv = exact_matches.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Exact Matches",
                            data=csv,
                            file_name="search_mapping_exact.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.info("No exact matches found")

                with tab3:
                    if len(partial_matches) > 0:
                        # Sort by similarity descending
                        partial_matches = partial_matches.sort_values('Similarity', ascending=False)
                        st.dataframe(partial_matches, use_container_width=True, height=400)
                        csv = partial_matches.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Partial Matches",
                            data=csv,
                            file_name="search_mapping_partial.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.info("No partial matches found")

                with tab4:
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Total Matches", len(df_result))
                    with col2:
                        st.metric("Exact Matches", len(exact_matches))
                    with col3:
                        st.metric("Avg Similarity", f"{df_result['Similarity'].mean():.2%}")

                    # Similarity distribution
                    st.subheader("Similarity Distribution")
                    import plotly.express as px

                    fig = px.histogram(
                        df_result,
                        x='Similarity',
                        nbins=20,
                        title='Distribution of Match Similarity Scores'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("""
                    ### 💡 Recommendations

                    1. **Exact matches (1.0)**: Users are searching for pages that exist.
                       Consider improving navigation or search suggestions.

                    2. **High similarity (0.8-0.99)**: Close matches that may indicate
                       slight naming differences. Consider updating H1s or adding synonyms.

                    3. **Medium similarity (0.5-0.79)**: Review manually - these may be
                       related products or categories worth linking.
                    """)
            else:
                st.warning("No matches found with the current settings. Try lowering the minimum similarity.")

    except Exception as e:
        st.error(f"Error processing files: {str(e)}")

else:
    st.info("👆 Upload both GA and Screaming Frog files to get started")

    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        This tool helps you **map internal site search queries** to existing category pages:

        1. **Upload** your GA search terms and Screaming Frog crawl data
        2. **Fuzzy matching** finds pages that match search queries
        3. **Identify opportunities** to improve navigation and reduce zero-result searches

        Use cases:
        - Find search queries that should lead to existing pages
        - Identify pages that users can't find
        - Improve search autocomplete suggestions
        """)

# Footer
st.markdown("---")
st.markdown(
    "Built by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · "
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · "
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)"
)
