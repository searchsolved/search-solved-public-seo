####################################################################################
#                                                                                  #
#  Category Page Title Suggester                                                   #
#                                                                                  #
#  Analyze category pages and suggest optimal title keywords.                      #
#                                                                                  #
####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                                   #
# Contact  : https://www.leefoot.com/contact                                           #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                       #
####################################################################################

"""
Category Page Title Suggester

Analyzes category page titles and GSC performance data to suggest
high-performing keywords that could be added to page titles.

Features:
- Upload site crawl (URLs + titles)
- Upload GSC data (queries, pages, clicks)
- Splits existing title keywords
- Compares with GSC top performers
- Suggests keywords to add to titles
"""

import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Category Title Suggester", page_icon="📝", layout="wide")

st.title("Category Title Suggester")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Analyzes your category page titles to extract keywords
    - Compares with GSC query performance data
    - Suggests high-performing keywords to add to titles

    **Data requirements:**

    1. **Crawl CSV** with columns:
       - URL/Address
       - Page Title

    2. **GSC Data CSV** with columns:
       - Page URL
       - Query
       - Clicks (and optionally impressions, CTR)

    **How to use:**
    1. Upload your crawl export (Screaming Frog, Sitebulb)
    2. Upload GSC performance data export
    3. Set URL filter for category pages
    4. Configure title delimiter
    5. Click "Analyze Titles"

    **Output:** List of pages with suggested keywords ranked by clicks.
    """)

# Sidebar settings
st.sidebar.header("Analysis Settings")

url_filter = st.sidebar.text_input(
    "URL filter (contains)",
    value="/category/",
    help="Only analyze pages with this string in the URL"
)

title_delimiter = st.sidebar.text_input(
    "Title delimiter",
    value="|",
    help="Character that separates keywords in titles (e.g., Keyword 1 | Keyword 2)"
)

branding = st.sidebar.text_input(
    "Brand name (to exclude)",
    value="",
    help="Brand name to remove from analysis"
)

max_suggestions = st.sidebar.slider(
    "Max keyword suggestions per page",
    min_value=3,
    max_value=20,
    value=10,
    help="Maximum keywords to suggest per page"
)

min_clicks = st.sidebar.number_input(
    "Minimum clicks",
    min_value=0,
    max_value=100,
    value=1,
    help="Only suggest keywords with at least this many clicks"
)


# File uploads
st.subheader("1. Upload Crawl Data")

crawl_file = st.file_uploader(
    "Upload crawl CSV (URL + Title)",
    type=['csv'],
    key="crawl_file"
)

crawl_df = None
url_col = None
title_col = None

if crawl_file:
    try:
        crawl_df = pd.read_csv(crawl_file)
        st.success(f"Loaded {len(crawl_df)} pages")

        col1, col2 = st.columns(2)
        with col1:
            url_options = [c for c in crawl_df.columns if 'address' in c.lower() or 'url' in c.lower()]
            default_url_idx = crawl_df.columns.tolist().index(url_options[0]) if url_options else 0
            url_col = st.selectbox(
                "URL column",
                crawl_df.columns.tolist(),
                index=default_url_idx
            )
        with col2:
            title_options = [c for c in crawl_df.columns if 'title' in c.lower()]
            default_title_idx = crawl_df.columns.tolist().index(title_options[0]) if title_options else 0
            title_col = st.selectbox(
                "Title column",
                crawl_df.columns.tolist(),
                index=default_title_idx
            )

    except Exception as e:
        st.error(f"Error reading crawl CSV: {str(e)}")

st.subheader("2. Upload GSC Data")

gsc_file = st.file_uploader(
    "Upload GSC performance CSV",
    type=['csv'],
    key="gsc_file"
)

gsc_df = None
gsc_page_col = None
gsc_query_col = None
gsc_clicks_col = None

if gsc_file:
    try:
        gsc_df = pd.read_csv(gsc_file)
        st.success(f"Loaded {len(gsc_df)} GSC rows")

        col1, col2, col3 = st.columns(3)
        with col1:
            page_options = [c for c in gsc_df.columns if 'page' in c.lower() or 'url' in c.lower()]
            default_page_idx = gsc_df.columns.tolist().index(page_options[0]) if page_options else 0
            gsc_page_col = st.selectbox(
                "Page column",
                gsc_df.columns.tolist(),
                index=default_page_idx
            )
        with col2:
            query_options = [c for c in gsc_df.columns if 'query' in c.lower() or 'keyword' in c.lower()]
            default_query_idx = gsc_df.columns.tolist().index(query_options[0]) if query_options else 0
            gsc_query_col = st.selectbox(
                "Query column",
                gsc_df.columns.tolist(),
                index=default_query_idx
            )
        with col3:
            clicks_options = [c for c in gsc_df.columns if 'click' in c.lower()]
            default_clicks_idx = gsc_df.columns.tolist().index(clicks_options[0]) if clicks_options else 0
            gsc_clicks_col = st.selectbox(
                "Clicks column",
                gsc_df.columns.tolist(),
                index=default_clicks_idx
            )

    except Exception as e:
        st.error(f"Error reading GSC CSV: {str(e)}")


# Run analysis
if st.button("Analyze Titles", type="primary",
             disabled=crawl_df is None or gsc_df is None):

    with st.spinner("Analyzing category page titles..."):
        # Filter crawl to category pages
        df_pages = crawl_df[[url_col, title_col]].copy()
        df_pages = df_pages.rename(columns={url_col: 'page', title_col: 'title'})

        if url_filter:
            df_pages = df_pages[df_pages['page'].str.contains(url_filter, na=False)]

        st.info(f"Analyzing {len(df_pages)} pages matching URL filter")

        # Clean and split titles
        df_pages = df_pages[df_pages['title'].notna()]
        df_pages = df_pages[df_pages['page'].notna()]

        # Expand title keywords into rows
        df_title_kws = df_pages.join(
            df_pages['title'].str.split(title_delimiter, expand=True).add_prefix('title_')
        )

        # Melt to tall format
        df_title_kws['query'] = df_title_kws.apply(
            lambda row: [v for k, v in row.items() if k.startswith('title_') and pd.notna(v)],
            axis=1
        )
        df_title_kws = df_title_kws[['page', 'title', 'query']].explode('query')
        df_title_kws['query'] = df_title_kws['query'].str.strip().str.lower()
        df_title_kws = df_title_kws[df_title_kws['query'].notna()]
        df_title_kws = df_title_kws[df_title_kws['query'] != '']

        # Remove branding
        if branding:
            df_title_kws = df_title_kws[~df_title_kws['query'].str.contains(branding.lower(), na=False)]

        df_title_kws['kw_source'] = 'page_title'

        # Prepare GSC data
        df_gsc = gsc_df[[gsc_page_col, gsc_query_col, gsc_clicks_col]].copy()
        df_gsc = df_gsc.rename(columns={
            gsc_page_col: 'page',
            gsc_query_col: 'query',
            gsc_clicks_col: 'clicks'
        })

        # Filter GSC to matching pages
        if url_filter:
            df_gsc = df_gsc[df_gsc['page'].str.contains(url_filter, na=False)]

        df_gsc['query'] = df_gsc['query'].str.lower()

        # Remove branding
        if branding:
            df_gsc = df_gsc[~df_gsc['query'].str.contains(branding.lower(), na=False)]

        df_gsc['kw_source'] = 'gsc'

        # Filter by min clicks
        df_gsc = df_gsc[df_gsc['clicks'] >= min_clicks]

        # Keep top keywords per page
        df_gsc = df_gsc.sort_values('clicks', ascending=False)
        df_gsc = df_gsc.groupby('page').head(max_suggestions)

        # Merge title keywords with GSC data
        df_merged = pd.merge(
            df_title_kws[['page', 'query', 'kw_source']],
            df_gsc[['page', 'query', 'clicks']],
            on=['page', 'query'],
            how='outer'
        )

        # Mark which keywords are already in title
        df_merged['in_title'] = df_merged['kw_source'] == 'page_title'

        # Find suggestions: high clicks but not in title
        df_suggestions = df_gsc[~df_gsc['query'].isin(df_title_kws['query'])].copy()

        # Add title back
        page_to_title = dict(zip(df_pages['page'], df_pages['title']))
        df_suggestions['current_title'] = df_suggestions['page'].map(page_to_title)

        # Aggregate by page
        df_suggestions = df_suggestions.sort_values(['page', 'clicks'], ascending=[True, False])

        # Store results
        st.session_state['suggestions'] = df_suggestions
        st.session_state['merged_data'] = df_merged
        st.session_state['page_titles'] = page_to_title

        st.success(f"Found {len(df_suggestions)} keyword suggestions!")


# Display results
if 'suggestions' in st.session_state:
    df_suggestions = st.session_state['suggestions']
    page_titles = st.session_state['page_titles']

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pages with Suggestions", df_suggestions['page'].nunique())
    with col2:
        st.metric("Total Suggestions", len(df_suggestions))
    with col3:
        total_clicks = df_suggestions['clicks'].sum()
        st.metric("Total Clicks (Suggested KWs)", f"{int(total_clicks):,}")

    # Results by page
    st.subheader("Keyword Suggestions by Page")

    for page in df_suggestions['page'].unique()[:20]:  # Limit display
        page_data = df_suggestions[df_suggestions['page'] == page]
        current_title = page_titles.get(page, 'Unknown')

        with st.expander(f"**{page[:60]}...** ({len(page_data)} suggestions)"):
            st.write(f"**Current Title:** {current_title}")
            st.write("**Suggested Keywords to Add:**")

            st.dataframe(
                page_data[['query', 'clicks']].head(max_suggestions),
                use_container_width=True
            )

    # Full table
    st.subheader("All Suggestions")
    st.dataframe(df_suggestions, use_container_width=True)

    # Downloads
    st.subheader("Download")

    col1, col2 = st.columns(2)

    with col1:
        csv_data = df_suggestions.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="title_suggestions.csv",
            mime="text/csv"
        )

    with col2:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_suggestions.to_excel(writer, sheet_name='Suggestions', index=False)

            # Summary by page
            summary = df_suggestions.groupby('page').agg({
                'query': 'count',
                'clicks': 'sum'
            }).reset_index()
            summary.columns = ['Page', 'Suggestion Count', 'Total Clicks']
            summary['Current Title'] = summary['Page'].map(page_titles)
            summary = summary.sort_values('Total Clicks', ascending=False)
            summary.to_excel(writer, sheet_name='Summary', index=False)

        st.download_button(
            label="Download Excel",
            data=output.getvalue(),
            file_name="title_suggestions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.subheader("Example Output")

    example_data = {
        "page": ["example.com/category/shoes", "example.com/category/shoes", "example.com/category/bags"],
        "query": ["running shoes sale", "best running shoes", "leather bags"],
        "clicks": [150, 120, 85],
        "current_title": ["Running Shoes | Brand", "Running Shoes | Brand", "Designer Bags | Brand"]
    }
    st.dataframe(pd.DataFrame(example_data))

    st.info("""
    **Interpretation:**
    - "running shoes sale" has 150 clicks but isn't in the title
    - Consider updating to: "Running Shoes Sale | Best Running Shoes | Brand"
    """)
