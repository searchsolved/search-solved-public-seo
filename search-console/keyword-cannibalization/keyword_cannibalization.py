####################################################################################
#                                                                                  #
#  Keyword Cannibalization Finder                                                  #
#                                                                                  #
#  Identifies keywords where multiple pages compete for the same search query.     #
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
Keyword Cannibalization Finder - Streamlit App

Analyzes Google Search Console data to find keywords where multiple pages
from your site are competing for the same query. Helps identify consolidation
opportunities and internal linking improvements.

Requirements:
    pip install streamlit pandas
"""

import streamlit as st
import pandas as pd
from io import BytesIO

# App Configuration
st.set_page_config(
    page_title="Keyword Cannibalization Finder",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Keyword Cannibalization Finder")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-Follow-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Identifies keyword cannibalization issues
    - Finds queries where multiple pages compete
    - Suggests consolidation opportunities

    **How to use:**
    1. Upload GSC query data with URLs
    2. Configure cannibalization thresholds
    3. Analyze competing pages
    4. Download cannibalization report

    **Best for:**
    - Cannibalization audits
    - Content consolidation planning
    - Internal competition resolution
    """)
st.markdown("""
Find keywords where multiple pages from your site compete for the same search query.
Upload GSC data to identify cannibalization issues and consolidation opportunities.
""")

# Sidebar configuration
st.sidebar.header("Filters")

position_min = st.sidebar.number_input(
    "Minimum Position",
    min_value=1,
    max_value=100,
    value=1,
    help="Only include keywords ranking at this position or lower"
)

position_max = st.sidebar.number_input(
    "Maximum Position",
    min_value=1,
    max_value=100,
    value=20,
    help="Only include keywords ranking at this position or higher"
)

min_impressions = st.sidebar.number_input(
    "Minimum Impressions",
    min_value=0,
    max_value=100000,
    value=0,
    help="Minimum impressions per query-page combination"
)

min_clicks = st.sidebar.number_input(
    "Minimum Clicks",
    min_value=0,
    max_value=10000,
    value=0,
    help="Minimum clicks per query-page combination"
)

min_pages = st.sidebar.number_input(
    "Minimum Competing Pages",
    min_value=2,
    max_value=10,
    value=2,
    help="Minimum number of pages ranking for the same query to flag as cannibalization"
)

st.sidebar.header("Display Options")
show_urls = st.sidebar.checkbox("Show full URLs", value=False)
group_by_query = st.sidebar.checkbox("Group results by query", value=True)


def read_search_console_data(df):
    """Standardize column names from various GSC export formats."""
    # Common column name mappings
    column_mapping = {
        'Top queries': 'query',
        'Query': 'query',
        'Queries': 'query',
        'Top pages': 'page',
        'Page': 'page',
        'Pages': 'page',
        'URL': 'page',
        'Landing Page': 'page',
        'Clicks': 'clicks',
        'Impressions': 'impressions',
        'CTR': 'ctr',
        'Click Through Rate': 'ctr',
        'Position': 'position',
        'Average position': 'position',
        'Avg. position': 'position'
    }

    df = df.rename(columns=column_mapping)
    df.columns = df.columns.str.lower()

    # Check required columns
    required_columns = ['query', 'page', 'clicks', 'impressions', 'position']
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        # Try to find CTR if missing
        if 'ctr' not in df.columns:
            df['ctr'] = 0

        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Convert numeric columns
    df['clicks'] = pd.to_numeric(df['clicks'], errors='coerce').fillna(0).astype(int)
    df['impressions'] = pd.to_numeric(df['impressions'], errors='coerce').fillna(0).astype(int)
    df['position'] = pd.to_numeric(df['position'], errors='coerce').fillna(0)

    # Handle CTR (might be percentage string like "5.2%")
    if 'ctr' in df.columns:
        if df['ctr'].dtype == 'object':
            df['ctr'] = df['ctr'].str.rstrip('%').astype(float) / 100
        df['ctr'] = pd.to_numeric(df['ctr'], errors='coerce').fillna(0)

    return df


def identify_cannibalization(df, position_range, min_impressions, min_clicks, min_pages):
    """Identify queries where multiple pages are competing."""

    # Filter based on position, impressions, and clicks
    filtered_df = df[
        (df['position'] >= position_range[0]) &
        (df['position'] <= position_range[1]) &
        (df['impressions'] >= min_impressions) &
        (df['clicks'] >= min_clicks)
    ].copy()

    if len(filtered_df) == 0:
        return pd.DataFrame()

    # Group by query and page to aggregate metrics
    cannibalization = filtered_df.groupby(['query', 'page']).agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'ctr': 'mean',
        'position': 'mean'
    }).reset_index()

    # Count pages per query
    pages_per_query = cannibalization.groupby('query')['page'].transform('count')
    cannibalization['competing_pages'] = pages_per_query

    # Filter to only queries with multiple pages (cannibalization)
    cannibalized = cannibalization[cannibalization['competing_pages'] >= min_pages].copy()

    # Round position and CTR for display
    cannibalized['position'] = cannibalized['position'].round(1)
    cannibalized['ctr'] = (cannibalized['ctr'] * 100).round(2)

    # Sort by competing pages (descending), then by impressions
    cannibalized = cannibalized.sort_values(
        ['competing_pages', 'impressions'],
        ascending=[False, False]
    )

    return cannibalized


def get_cannibalization_summary(df):
    """Generate summary statistics."""
    if len(df) == 0:
        return {}

    unique_queries = df['query'].nunique()
    total_pages = len(df)
    total_clicks = df['clicks'].sum()
    total_impressions = df['impressions'].sum()
    avg_competing = df.groupby('query')['page'].count().mean()
    max_competing = df['competing_pages'].max()

    return {
        'unique_queries': unique_queries,
        'total_pages': total_pages,
        'total_clicks': total_clicks,
        'total_impressions': total_impressions,
        'avg_competing': round(avg_competing, 1),
        'max_competing': max_competing
    }


def create_query_groups(df):
    """Create a grouped view by query."""
    groups = []

    for query in df['query'].unique():
        query_data = df[df['query'] == query].copy()
        query_data = query_data.sort_values('position')

        # Get the best performing page
        best_page = query_data.iloc[0]

        groups.append({
            'Query': query,
            'Competing Pages': len(query_data),
            'Total Clicks': query_data['clicks'].sum(),
            'Total Impressions': query_data['impressions'].sum(),
            'Best Position': query_data['position'].min(),
            'Worst Position': query_data['position'].max(),
            'Position Spread': round(query_data['position'].max() - query_data['position'].min(), 1),
            'Best Page': best_page['page']
        })

    return pd.DataFrame(groups).sort_values('Total Impressions', ascending=False)


# File uploader
st.header("Upload GSC Data")
uploaded_file = st.file_uploader(
    "Upload your Search Console export (CSV)",
    type=["csv"],
    help="Export from GSC should contain: query, page, clicks, impressions, position columns"
)

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file, dtype=str)
        df = read_search_console_data(df)

        st.success(f"Loaded {len(df):,} rows of GSC data")

        # Show raw data preview
        with st.expander("Preview Raw Data"):
            st.dataframe(df.head(20))

        # Process button
        if st.button("Find Cannibalization Issues", type="primary"):
            with st.spinner("Analyzing keyword cannibalization..."):

                # Identify cannibalization
                cannibalized = identify_cannibalization(
                    df,
                    position_range=(position_min, position_max),
                    min_impressions=min_impressions,
                    min_clicks=min_clicks,
                    min_pages=min_pages
                )

                if len(cannibalized) == 0:
                    st.warning("No cannibalization issues found with the current filters. Try adjusting the filters.")
                    st.stop()

                # Get summary stats
                summary = get_cannibalization_summary(cannibalized)

                # Display results
                st.header("Cannibalization Analysis Results")

                # Summary metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Cannibalized Queries", f"{summary['unique_queries']:,}")
                with col2:
                    st.metric("Total Competing Pages", f"{summary['total_pages']:,}")
                with col3:
                    st.metric("Affected Clicks", f"{summary['total_clicks']:,}")
                with col4:
                    st.metric("Avg Pages per Query", f"{summary['avg_competing']}")
                with col5:
                    st.metric("Max Competing Pages", f"{summary['max_competing']}")

                # Tabs for different views
                tab1, tab2 = st.tabs(["By Query (Grouped)", "Detailed View"])

                with tab1:
                    st.subheader("Cannibalization by Query")
                    query_groups = create_query_groups(cannibalized)

                    # Truncate URLs if needed
                    if not show_urls:
                        query_groups['Best Page'] = query_groups['Best Page'].str.split('/').str[-1].str[:50]

                    st.dataframe(
                        query_groups,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Download grouped results
                    output = BytesIO()
                    query_groups.to_csv(output, index=False, encoding='utf-8-sig')
                    output.seek(0)

                    st.download_button(
                        label="📥 Download Grouped Results (CSV)",
                        data=output,
                        file_name="cannibalization_by_query.csv",
                        mime="text/csv"
                    )

                with tab2:
                    st.subheader("Detailed Cannibalization Data")

                    # Rename columns for display
                    display_df = cannibalized.rename(columns={
                        'query': 'Query',
                        'page': 'Page',
                        'clicks': 'Clicks',
                        'impressions': 'Impressions',
                        'ctr': 'CTR (%)',
                        'position': 'Avg Position',
                        'competing_pages': 'Competing Pages'
                    })

                    # Truncate URLs if needed
                    if not show_urls:
                        display_df['Page'] = display_df['Page'].str.split('/').str[-1].str[:50]

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Download detailed results
                    output = BytesIO()
                    cannibalized.to_csv(output, index=False, encoding='utf-8-sig')
                    output.seek(0)

                    st.download_button(
                        label="📥 Download Detailed Results (CSV)",
                        data=output,
                        file_name="cannibalization_detailed.csv",
                        mime="text/csv"
                    )

                # Recommendations
                st.header("Recommendations")

                high_impact = query_groups[query_groups['Total Impressions'] >= query_groups['Total Impressions'].quantile(0.75)]

                if len(high_impact) > 0:
                    st.markdown("### High-Impact Cannibalization (Top 25% by Impressions)")
                    st.markdown("""
                    These queries have significant search volume and multiple competing pages.
                    Consider:
                    - **Consolidating content** into a single authoritative page
                    - **Adding canonical tags** to point to the primary page
                    - **Improving internal linking** to signal the primary page
                    - **Differentiating content** if pages serve different intents
                    """)

                    for _, row in high_impact.head(5).iterrows():
                        with st.expander(f"🔍 {row['Query']} ({row['Competing Pages']} pages, {row['Total Impressions']:,} impressions)"):
                            query_detail = cannibalized[cannibalized['query'] == row['Query']].sort_values('position')
                            st.dataframe(query_detail[['page', 'clicks', 'impressions', 'position']], hide_index=True)

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Upload a CSV file from Google Search Console to get started.")

    st.markdown("""
    ### How to export data from Google Search Console:

    1. Go to [Google Search Console](https://search.google.com/search-console)
    2. Select your property
    3. Go to **Performance** report
    4. Set date range (recommend 3+ months for better data)
    5. Click **Export** > **Download CSV**

    ### What this tool does:

    - Identifies queries where multiple pages from your site rank
    - Shows which pages are competing for the same keywords
    - Highlights high-impact cannibalization issues
    - Helps prioritize content consolidation efforts

    ### What is Keyword Cannibalization?

    Keyword cannibalization occurs when multiple pages on your site target the same
    keyword and compete against each other in search results. This can:

    - **Dilute ranking signals** across multiple pages
    - **Confuse search engines** about which page to rank
    - **Split clicks and engagement** between pages
    - **Reduce overall organic visibility**

    ### How to fix cannibalization:

    1. **Consolidate**: Merge competing pages into one comprehensive page
    2. **Differentiate**: Adjust content to target different search intents
    3. **Canonicalize**: Use canonical tags to indicate the preferred page
    4. **Redirect**: 301 redirect weaker pages to the strongest one
    5. **Internal link**: Strengthen internal links to your preferred page
    """)
