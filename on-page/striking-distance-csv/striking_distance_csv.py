####################################################################################
#                                                                                  #
#  Striking Distance CSV Edition                                                   #
#                                                                                  #
#  Find striking distance keywords from GSC CSV exports and check if they          #
#  appear in page titles, H1s, and body content.                                   #
#                                                                                  #
####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                                              #
####################################################################################

"""
Striking Distance CSV Edition - Streamlit App

Analyzes GSC keyword data from CSV exports to find striking distance opportunities.
Combines with crawl data to check keyword presence in titles, H1s, and content.

Requirements:
    pip install streamlit pandas tqdm
"""

import streamlit as st
import pandas as pd
import re
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# App Configuration
st.set_page_config(
    page_title="Striking Distance CSV",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Striking Distance CSV Edition")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-181717?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Identifies keywords close to page 1 rankings
    - Calculates striking distance opportunities
    - Prioritizes quick-win optimizations

    **How to use:**
    1. Upload GSC data (queries + positions)
    2. Configure position thresholds
    3. Calculate striking distance
    4. Download prioritized list

    **Best for:**
    - Quick-win SEO identification
    - Page 1 push opportunities
    - Content optimization prioritization
    """)
st.markdown("""
Find quick-win keyword opportunities from your GSC CSV exports.
Upload GSC data and crawl data to identify keywords you're ranking for but not optimizing.
""")

# Sidebar configuration
st.sidebar.header("Position Filters")
POSITION_MIN = st.sidebar.number_input(
    "Minimum Position",
    min_value=1,
    max_value=100,
    value=4,
    help="Minimum ranking position (default: 4)"
)

POSITION_MAX = st.sidebar.number_input(
    "Maximum Position",
    min_value=1,
    max_value=100,
    value=20,
    help="Maximum ranking position (default: 20)"
)

IMPRESSIONS_MIN = st.sidebar.number_input(
    "Minimum Impressions",
    min_value=0,
    max_value=100000,
    value=0,
    help="Minimum impressions threshold"
)

MAX_KEYWORDS_PER_PAGE = st.sidebar.slider(
    "Keywords per Page",
    min_value=1,
    max_value=20,
    value=10,
    help="Maximum keywords to analyze per page"
)

st.sidebar.header("Brand Filters")
brand_input = st.sidebar.text_area(
    "Branded Terms to Exclude",
    value="",
    help="Enter branded terms to filter out, one per line"
)

sort_metric = st.sidebar.selectbox(
    "Sort By",
    options=['clicks', 'impressions'],
    index=0,
    help="Metric to prioritize keywords by"
)

# Parse branded terms
branded_terms = [term.strip().lower() for term in brand_input.split('\n') if term.strip()]

# File uploaders
st.header("Upload Your Data")

col1, col2 = st.columns(2)

with col1:
    st.subheader("GSC Data")
    gsc_file = st.file_uploader(
        "Upload GSC Export (CSV)",
        type=["csv"],
        help="Google Search Console export with query, page, clicks, impressions, position",
        key="gsc"
    )

with col2:
    st.subheader("Crawl Data")
    crawl_file = st.file_uploader(
        "Upload Crawl Export (CSV)",
        type=["csv"],
        help="Screaming Frog export with Address, Title 1, H1-1, and optionally content",
        key="crawl"
    )


def get_top_keywords_by_page(df, sort_metric, max_keywords):
    """Group by page and get top keywords based on sort metric."""
    top_keywords_by_page = (
        df.groupby('page')
        .apply(lambda x: x.nlargest(max_keywords, sort_metric)[['query', sort_metric, 'position']])
        .reset_index(level=0)
        .reset_index(drop=True)
    )
    return top_keywords_by_page


def check_keyword_in_text(keyword, text):
    """Check if keyword appears in text (case insensitive)."""
    if pd.isna(text) or pd.isna(keyword):
        return False
    try:
        escaped_keyword = re.escape(str(keyword).lower())
        return bool(re.search(escaped_keyword, str(text).lower()))
    except Exception:
        return False


def process_keywords(top_keywords, crawl_df, columns_to_check, sort_metric):
    """Process keywords and check their presence in page content."""
    results = []

    # Create page lookup
    page_groups = {}
    for _, row in crawl_df.iterrows():
        page = row.get('Address', '')
        if page:
            page_groups[page] = row

    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(top_keywords)

    for idx, row in top_keywords.iterrows():
        keyword = row['query']
        page = row['page']
        total_metric = row[sort_metric]
        position = row['position']

        keyword_result = {
            'Page': page,
            'Keyword': keyword,
            f'Total {sort_metric.capitalize()}': total_metric,
            'Position': position,
        }

        if page in page_groups:
            page_data = page_groups[page]
            for column in columns_to_check:
                keyword_result[f'In {column}'] = check_keyword_in_text(keyword, page_data.get(column, ''))
        else:
            for column in columns_to_check:
                keyword_result[f'In {column}'] = False

        results.append(keyword_result)

        if (idx + 1) % 100 == 0 or idx == total - 1:
            progress_bar.progress((idx + 1) / total)
            status_text.text(f"Processing keywords: {idx + 1}/{total}")

    progress_bar.progress(1.0)
    status_text.text("Processing complete!")

    return pd.DataFrame(results)


def filter_omnipresent_keywords(df, columns_to_check):
    """Filter out keywords that appear in all checked columns."""
    check_cols = [f'In {col}' for col in columns_to_check if f'In {col}' in df.columns]
    if check_cols:
        mask = df[check_cols].all(axis=1)
        return df[~mask].reset_index(drop=True)
    return df


def create_wide_format_data(df, max_keywords, sort_metric):
    """Transform data to wide format with keywords as columns."""
    page_data_list = []

    for page, group in df.groupby('Page'):
        page_data = {
            'Page': page,
            f'Total {sort_metric.capitalize()}': group[f'Total {sort_metric.capitalize()}'].sum(),
        }

        keyword_count = 0
        for _, row in group.iterrows():
            if keyword_count >= max_keywords:
                break
            keyword_count += 1
            page_data[f'KW{keyword_count}'] = row['Keyword']
            page_data[f'KW{keyword_count} {sort_metric.capitalize()}'] = row[f'Total {sort_metric.capitalize()}']
            page_data[f'KW{keyword_count} Position'] = row['Position']

            # Add presence columns
            for col in df.columns:
                if col.startswith('In '):
                    page_data[f'KW{keyword_count} {col}'] = row[col]

        page_data_list.append(page_data)

    return pd.DataFrame(page_data_list)


if gsc_file is not None and crawl_file is not None:
    try:
        # Read GSC data
        gsc_df = pd.read_csv(gsc_file, dtype=str)

        # Standardize column names
        gsc_col_mapping = {
            'Top queries': 'query',
            'Query': 'query',
            'Queries': 'query',
            'Top pages': 'page',
            'Page': 'page',
            'Pages': 'page',
            'URL': 'page',
            'Clicks': 'clicks',
            'Impressions': 'impressions',
            'CTR': 'ctr',
            'Position': 'position',
            'Average position': 'position'
        }
        gsc_df.rename(columns=gsc_col_mapping, inplace=True)
        gsc_df.columns = gsc_df.columns.str.lower()

        # Convert numeric columns
        gsc_df['clicks'] = pd.to_numeric(gsc_df['clicks'], errors='coerce').fillna(0).astype(int)
        gsc_df['impressions'] = pd.to_numeric(gsc_df['impressions'], errors='coerce').fillna(0).astype(int)
        gsc_df['position'] = pd.to_numeric(gsc_df['position'], errors='coerce').fillna(0)

        st.success(f"Loaded {len(gsc_df):,} rows of GSC data")

        # Read crawl data
        crawl_df = pd.read_csv(crawl_file, dtype=str)
        st.success(f"Loaded {len(crawl_df):,} rows of crawl data")

        # Column selection for crawl data
        st.subheader("Map Crawl Columns")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            url_col = st.selectbox(
                "URL Column",
                options=crawl_df.columns.tolist(),
                index=crawl_df.columns.tolist().index("Address") if "Address" in crawl_df.columns else 0
            )

        with col2:
            title_col = st.selectbox(
                "Title Column",
                options=['(none)'] + crawl_df.columns.tolist(),
                index=crawl_df.columns.tolist().index("Title 1") + 1 if "Title 1" in crawl_df.columns else 0
            )

        with col3:
            h1_col = st.selectbox(
                "H1 Column",
                options=['(none)'] + crawl_df.columns.tolist(),
                index=crawl_df.columns.tolist().index("H1-1") + 1 if "H1-1" in crawl_df.columns else 0
            )

        with col4:
            content_options = [c for c in crawl_df.columns if any(x in c.lower() for x in ['content', 'text', 'body', 'rendered'])]
            content_col = st.selectbox(
                "Content Column (optional)",
                options=['(none)'] + crawl_df.columns.tolist(),
                index=crawl_df.columns.tolist().index(content_options[0]) + 1 if content_options else 0
            )

        # Rename columns for processing
        rename_map = {url_col: 'Address'}
        if title_col != '(none)':
            rename_map[title_col] = 'Title'
        if h1_col != '(none)':
            rename_map[h1_col] = 'H1'
        if content_col != '(none)':
            rename_map[content_col] = 'Content'

        crawl_df = crawl_df.rename(columns=rename_map)

        # Determine columns to check
        columns_to_check = []
        if title_col != '(none)':
            columns_to_check.append('Title')
        if h1_col != '(none)':
            columns_to_check.append('H1')
        if content_col != '(none)':
            columns_to_check.append('Content')

        if not columns_to_check:
            st.error("Please select at least one column to check (Title, H1, or Content)")
            st.stop()

        # Process button
        if st.button("Find Striking Distance Keywords", type="primary"):
            with st.spinner("Analyzing keywords..."):

                # Filter branded terms
                if branded_terms:
                    original_count = len(gsc_df)
                    gsc_df = gsc_df[
                        ~gsc_df['query'].str.lower().apply(lambda x: any(term in str(x) for term in branded_terms))
                    ]
                    filtered_count = original_count - len(gsc_df)
                    st.info(f"Filtered out {filtered_count:,} branded queries")

                # Apply position filters
                gsc_df = gsc_df[
                    (gsc_df['position'] >= POSITION_MIN) &
                    (gsc_df['position'] <= POSITION_MAX) &
                    (gsc_df['impressions'] >= IMPRESSIONS_MIN)
                ]

                if len(gsc_df) == 0:
                    st.error("No keywords found matching the position and impression filters.")
                    st.stop()

                st.info(f"Found {len(gsc_df):,} keywords in striking distance (positions {POSITION_MIN}-{POSITION_MAX})")

                # Get top keywords per page
                top_keywords = get_top_keywords_by_page(gsc_df, sort_metric, MAX_KEYWORDS_PER_PAGE)

                # Check keyword presence
                keyword_presence = process_keywords(top_keywords, crawl_df, columns_to_check, sort_metric)

                # Filter out omnipresent keywords
                keyword_presence = filter_omnipresent_keywords(keyword_presence, columns_to_check)

                if len(keyword_presence) == 0:
                    st.warning("All keywords already appear in all checked locations. No optimization opportunities found.")
                    st.stop()

                # Create wide format
                wide_format_data = create_wide_format_data(keyword_presence, MAX_KEYWORDS_PER_PAGE, sort_metric)

                # Display results
                st.header("Striking Distance Results")

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Pages with Opportunities", f"{len(wide_format_data):,}")
                with col2:
                    st.metric("Total Keywords Found", f"{len(keyword_presence):,}")
                with col3:
                    missing_title = len(keyword_presence[keyword_presence.get('In Title', True) == False]) if 'In Title' in keyword_presence.columns else 0
                    st.metric("Missing from Title", f"{missing_title:,}")
                with col4:
                    missing_h1 = len(keyword_presence[keyword_presence.get('In H1', True) == False]) if 'In H1' in keyword_presence.columns else 0
                    st.metric("Missing from H1", f"{missing_h1:,}")

                # Detailed view
                st.subheader("Detailed Results")

                view_type = st.radio(
                    "View",
                    options=["Wide Format (by page)", "Long Format (by keyword)"],
                    horizontal=True
                )

                if view_type == "Wide Format (by page)":
                    st.dataframe(
                        wide_format_data,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Download
                    output = BytesIO()
                    wide_format_data.to_csv(output, index=False, encoding='utf-8-sig')
                    output.seek(0)

                    st.download_button(
                        label="📥 Download Wide Format (CSV)",
                        data=output,
                        file_name="striking_distance_wide.csv",
                        mime="text/csv"
                    )
                else:
                    st.dataframe(
                        keyword_presence.sort_values('Position'),
                        use_container_width=True,
                        hide_index=True
                    )

                    # Download
                    output = BytesIO()
                    keyword_presence.to_csv(output, index=False, encoding='utf-8-sig')
                    output.seek(0)

                    st.download_button(
                        label="📥 Download Long Format (CSV)",
                        data=output,
                        file_name="striking_distance_long.csv",
                        mime="text/csv"
                    )

    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        st.exception(e)

else:
    if gsc_file is None:
        st.info("👆 Upload your GSC data export to get started.")
    if crawl_file is None:
        st.info("👆 Upload your Screaming Frog crawl export to check keyword presence.")

    st.markdown("""
    ### What you need:

    **GSC Export:**
    - Export from Google Search Console > Performance
    - Should contain: query, page, clicks, impressions, position

    **Crawl Export:**
    - Screaming Frog internal_html.csv export
    - Should contain: Address, Title 1, H1-1
    - Optional: Custom extraction of main content

    ### What this tool does:

    1. Finds keywords ranking in positions 4-20 (or your custom range)
    2. Filters out branded terms you specify
    3. Checks if keywords appear in page titles, H1s, and content
    4. Highlights opportunities where you rank but don't mention the keyword
    5. Exports actionable optimization recommendations
    """)
