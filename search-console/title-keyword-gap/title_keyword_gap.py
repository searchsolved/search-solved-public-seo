####################################################################################
#                                                                                  #
#  Title Keyword Gap Finder                                                        #
#                                                                                  #
#  Compare GSC keywords vs page titles to find missing opportunities.              #
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
Title Keyword Gap Finder

Compares Google Search Console keywords against page titles to identify
keywords that are driving impressions but are missing from the page title.
Great for quick-win title optimization opportunities.

Features:
- Upload Screaming Frog crawl and GSC query data
- Find keywords not present in titles
- Highlight matches and misses
- Export to Excel with highlighting
"""

import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Title Keyword Gap Finder", page_icon="🔎", layout="wide")

st.title("Title Keyword Gap Finder")
st.markdown("*Created by 🌐 [Lee Foot](https://www.leefoot.com) · [LinkedIn](https://www.linkedin.com/in/lee-foot/) · 🦋 [Bluesky](https://bsky.app/profile/leefootseo.bsky.social)*")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Compares your Google Search Console queries against page titles
    - Finds keywords that drive impressions but aren't in the title
    - Highlights which keywords are already in your titles
    - Suggests title optimization opportunities

    **How to get the data:**

    **1. Screaming Frog Crawl:**
    - Crawl your site with Screaming Frog
    - Export `internal_html.csv` from the Internal tab
    - Required columns: Address, Title 1

    **2. GSC Query Data:**
    - Go to Search Console > Performance > Pages
    - Export the data (clicks, impressions, queries by page)
    - Or use the GSC API to export query-level data

    **Required GSC columns:**
    - `page` - The landing page URL
    - `query` - The search query
    - `clicks` - Click count
    - `impressions` - Impression count
    """)

# Sidebar settings
st.sidebar.header("Settings")

title_delimiter = st.sidebar.text_input(
    "Title delimiter",
    value="|",
    help="Character used to split brand from title (e.g., | or -)"
)

branding = st.sidebar.text_input(
    "Brand terms to exclude",
    value="",
    help="Brand name(s) to filter out of analysis (comma-separated)"
)

url_filter = st.sidebar.text_input(
    "URL filter (optional)",
    value="",
    help="Only analyze URLs containing this text (e.g., /products/)"
)

max_keywords_per_page = st.sidebar.number_input(
    "Max keywords per page",
    min_value=5,
    max_value=50,
    value=10,
    help="Maximum GSC keywords to show per page"
)

min_impressions = st.sidebar.number_input(
    "Minimum impressions",
    min_value=0,
    max_value=10000,
    value=0,
    help="Only include queries with at least this many impressions"
)

# File uploads
st.subheader("Upload Files")
col1, col2 = st.columns(2)

with col1:
    crawl_file = st.file_uploader(
        "Screaming Frog crawl (CSV)",
        type=['csv'],
        key="crawl",
        help="Export from Screaming Frog with titles"
    )

with col2:
    gsc_file = st.file_uploader(
        "GSC query data (CSV)",
        type=['csv'],
        key="gsc",
        help="GSC export with page, query, clicks, impressions"
    )


def load_csv(file):
    """Load CSV with encoding fallback."""
    try:
        return pd.read_csv(file, encoding='utf-8')
    except:
        file.seek(0)
        return pd.read_csv(file, encoding='latin-1')


def find_column(df, possible_names):
    """Find a column by trying multiple possible names."""
    for name in possible_names:
        for col in df.columns:
            if name.lower() == col.lower() or name.lower() in col.lower():
                return col
    return None


if crawl_file is not None and gsc_file is not None:
    try:
        # Load files
        df_crawl = load_csv(crawl_file)
        df_gsc = load_csv(gsc_file)

        st.success(f"Loaded crawl: {len(df_crawl):,} URLs | GSC: {len(df_gsc):,} queries")

        # Find columns in crawl
        address_col = find_column(df_crawl, ['address', 'url'])
        title_col = find_column(df_crawl, ['title 1', 'title', 'page title'])
        index_col = find_column(df_crawl, ['indexability'])

        # Find columns in GSC
        page_col = find_column(df_gsc, ['page', 'landing page', 'url'])
        query_col = find_column(df_gsc, ['query', 'keyword', 'top queries'])
        clicks_col = find_column(df_gsc, ['clicks', 'click'])
        impressions_col = find_column(df_gsc, ['impressions', 'impression'])
        ctr_col = find_column(df_gsc, ['ctr', 'click through rate'])

        with st.expander("Column Mapping"):
            st.markdown("**Crawl columns:**")
            col1, col2 = st.columns(2)
            with col1:
                address_col = st.selectbox("URL column", df_crawl.columns.tolist(),
                                           index=df_crawl.columns.tolist().index(address_col) if address_col else 0)
            with col2:
                title_col = st.selectbox("Title column", df_crawl.columns.tolist(),
                                         index=df_crawl.columns.tolist().index(title_col) if title_col else 0)

            st.markdown("**GSC columns:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                page_col = st.selectbox("Page column", df_gsc.columns.tolist(),
                                        index=df_gsc.columns.tolist().index(page_col) if page_col else 0)
            with col2:
                query_col = st.selectbox("Query column", df_gsc.columns.tolist(),
                                         index=df_gsc.columns.tolist().index(query_col) if query_col else 0)
            with col3:
                clicks_col = st.selectbox("Clicks column", df_gsc.columns.tolist(),
                                          index=df_gsc.columns.tolist().index(clicks_col) if clicks_col else 0)
            with col4:
                impressions_col = st.selectbox("Impressions column", df_gsc.columns.tolist(),
                                               index=df_gsc.columns.tolist().index(impressions_col) if impressions_col else 0)

        if st.button("Analyze Title Gaps", type="primary"):
            with st.spinner("Analyzing keywords vs titles..."):
                # Prepare crawl data
                df_titles = df_crawl[[address_col, title_col]].copy()
                df_titles.columns = ['page', 'title']
                df_titles = df_titles.dropna(subset=['title'])

                # Apply URL filter
                if url_filter.strip():
                    df_titles = df_titles[df_titles['page'].str.contains(url_filter, na=False)]

                # Prepare GSC data
                df_queries = df_gsc[[page_col, query_col, clicks_col, impressions_col]].copy()
                df_queries.columns = ['page', 'query', 'clicks', 'impressions']

                # Filter by impressions
                if min_impressions > 0:
                    df_queries = df_queries[df_queries['impressions'] >= min_impressions]

                # Filter by URL
                if url_filter.strip():
                    df_queries = df_queries[df_queries['page'].str.contains(url_filter, na=False)]

                # Filter out brand terms
                if branding.strip():
                    brand_terms = [b.strip().lower() for b in branding.split(',') if b.strip()]
                    for term in brand_terms:
                        df_queries = df_queries[~df_queries['query'].str.lower().str.contains(term, na=False)]

                # Sort and limit keywords per page
                df_queries = df_queries.sort_values(['page', 'clicks'], ascending=[True, False])
                df_queries = df_queries.groupby('page').head(max_keywords_per_page)

                # Merge with titles
                df_merged = pd.merge(df_queries, df_titles, on='page', how='inner')

                if len(df_merged) == 0:
                    st.warning("No matching pages found between crawl and GSC data. "
                               "Check that URLs match exactly.")
                else:
                    # Check if query is in title
                    def check_query_in_title(row):
                        query = str(row['query']).strip().lower()
                        title = str(row['title']).strip().lower()

                        # Split title by delimiter
                        if title_delimiter:
                            title_parts = [p.strip() for p in title.split(title_delimiter)]
                        else:
                            title_parts = [title]

                        # Check if query appears in any part
                        for part in title_parts:
                            if query in part:
                                return True
                        return False

                    df_merged['in_title'] = df_merged.apply(check_query_in_title, axis=1)

                    # Calculate totals per page
                    df_merged['total_clicks'] = df_merged.groupby('page')['clicks'].transform('sum')
                    df_merged['total_impressions'] = df_merged.groupby('page')['impressions'].transform('sum')

                    # Sort by potential
                    df_merged = df_merged.sort_values(
                        by=['total_impressions', 'page', 'clicks'],
                        ascending=[False, True, False]
                    )

                    # Display results
                    st.subheader("Results")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Pages Analyzed", f"{df_merged['page'].nunique():,}")
                    with col2:
                        st.metric("Keywords Analyzed", f"{len(df_merged):,}")
                    with col3:
                        in_title = df_merged['in_title'].sum()
                        st.metric("Already in Title", f"{in_title:,}")
                    with col4:
                        not_in_title = len(df_merged) - in_title
                        st.metric("Missing from Title", f"{not_in_title:,}")

                    # Show gaps (keywords NOT in title)
                    st.subheader("Keywords Missing from Titles (Opportunities)")
                    df_gaps = df_merged[df_merged['in_title'] == False].copy()
                    df_gaps = df_gaps.sort_values('impressions', ascending=False)

                    if len(df_gaps) > 0:
                        st.dataframe(
                            df_gaps[['page', 'title', 'query', 'clicks', 'impressions']].head(100),
                            use_container_width=True
                        )

                        # Top missing keywords overall
                        st.subheader("Most Common Missing Keywords")
                        missing_kws = df_gaps['query'].value_counts().head(20).reset_index()
                        missing_kws.columns = ['Keyword', 'Pages Missing It']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.dataframe(missing_kws, use_container_width=True)
                        with col2:
                            st.bar_chart(missing_kws.head(15).set_index('Keyword'))
                    else:
                        st.success("All analyzed keywords are already present in page titles!")

                    # Download options
                    st.subheader("Download")

                    col1, col2 = st.columns(2)

                    with col1:
                        # Full results CSV
                        csv_full = df_merged.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="Download All Results (CSV)",
                            data=csv_full,
                            file_name="title_keyword_analysis.csv",
                            mime="text/csv"
                        )

                    with col2:
                        # Gaps only CSV
                        csv_gaps = df_gaps.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="Download Gaps Only (CSV)",
                            data=csv_gaps,
                            file_name="title_keyword_gaps.csv",
                            mime="text/csv"
                        )

                    # Excel with highlighting
                    st.markdown("---")
                    try:
                        from openpyxl import Workbook
                        from openpyxl.styles import PatternFill

                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_merged.to_excel(writer, index=False, sheet_name='Analysis')

                            # Get the worksheet
                            ws = writer.sheets['Analysis']

                            # Highlight rows where keyword IS in title (green)
                            green_fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")
                            # Highlight rows where keyword is NOT in title (yellow)
                            yellow_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")

                            # Find in_title column index
                            in_title_col = None
                            for idx, cell in enumerate(ws[1], 1):
                                if cell.value == 'in_title':
                                    in_title_col = idx
                                    break

                            if in_title_col:
                                for row_idx in range(2, ws.max_row + 1):
                                    cell_value = ws.cell(row=row_idx, column=in_title_col).value
                                    fill = green_fill if cell_value else yellow_fill
                                    for col_idx in range(1, ws.max_column + 1):
                                        ws.cell(row=row_idx, column=col_idx).fill = fill

                        excel_data = output.getvalue()
                        st.download_button(
                            label="Download Excel with Highlighting",
                            data=excel_data,
                            file_name="title_keyword_analysis.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except ImportError:
                        st.info("Install openpyxl for Excel export with highlighting")

    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("Upload both a crawl file and GSC data to begin")

    st.subheader("Example Output")
    example_data = {
        "Page": ["/products/widget", "/products/widget", "/products/gadget"],
        "Title": ["Best Widget for Home", "Best Widget for Home", "Professional Gadget Tool"],
        "Query": ["widget", "industrial widget", "gadget"],
        "Impressions": [1500, 800, 2000],
        "In Title": [True, False, True]
    }
    st.dataframe(pd.DataFrame(example_data))
    st.markdown("*Yellow highlight = keyword missing from title (opportunity)*")
