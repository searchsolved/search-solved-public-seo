"""
Topical Map Visualiser - Streamlit Version

Upload a tagged keyword CSV and view it as an interactive, zoomable
D3.js circle packing chart. Pairs with the Topical Map Generator tool.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from topical_map_chart import CHART_TITLES, METRIC_CHOICES, render_chart

st.set_page_config(page_title="Topical Map Visualiser", page_icon="🫧", layout="wide")

st.title("Topical Map Visualiser")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")
st.warning("**Experimental Tool** - This is a proof of concept and may have limitations.")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Reads a tagged keyword CSV with a two-level topic hierarchy (parent topic > subtopic > keyword)
    - Builds a zoomable D3.js circle packing chart, with circles sized by the metric you choose
    - Lets you download the chart as a standalone HTML file to share or embed

    **Works with the Topical Map Generator:**
    - Feed it the CSV output of the Topical Map Generator tool in this repository
    - Map the columns (e.g. `Parent Topic`, `Niche Topic 1`, `Keyword`) and use the `count` metric

    **Metrics:**
    - `count` - each keyword counts as 1 (no performance data needed)
    - `impressions` / `clicks` - sums a numeric column, e.g. from Google Search Console
    - `first_page_count` - counts keywords with a position of 1 to 10
    - `top_3_count` - counts keywords with a position of 1 to 3
    """)

uploaded_file = st.file_uploader("Upload tagged keywords CSV", type=['csv'])

if uploaded_file is not None:
    try:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin-1')

        st.success(f"Loaded {len(df):,} rows")

        with st.expander("Preview uploaded data"):
            st.dataframe(df.head(20))

        columns = df.columns.tolist()

        st.sidebar.header("Settings")
        metric = st.sidebar.selectbox("Metric", METRIC_CHOICES, index=0,
                                      help="How each keyword contributes to circle size")
        chart_title = st.sidebar.text_input("Chart title", value=CHART_TITLES[metric])

        def default_index(name):
            matches = [i for i, c in enumerate(columns) if c.lower() == name.lower()]
            return matches[0] if matches else 0

        st.sidebar.subheader("Column mapping")
        parent_col = st.sidebar.selectbox("Parent topic column", columns, index=default_index('Parent'))
        child_col = st.sidebar.selectbox("Subtopic column", columns, index=default_index('Child'))
        keyword_col = st.sidebar.selectbox("Keyword column", columns, index=default_index('query'))

        position_col = impressions_col = clicks_col = None
        if metric in ('first_page_count', 'top_3_count'):
            position_col = st.sidebar.selectbox("Position column", columns, index=default_index('position'))
        elif metric == 'impressions':
            impressions_col = st.sidebar.selectbox("Impressions column", columns, index=default_index('impressions'))
        elif metric == 'clicks':
            clicks_col = st.sidebar.selectbox("Clicks column", columns, index=default_index('clicks'))

        if st.button("Generate Chart", type="primary"):
            required = [c for c in [parent_col, child_col, keyword_col, position_col, impressions_col, clicks_col] if c]
            df_clean = df.dropna(subset=required)

            kwargs = {'parent_col': parent_col, 'child_col': child_col, 'keyword_col': keyword_col}
            if position_col:
                kwargs['position_col'] = position_col
            if impressions_col:
                kwargs['impressions_col'] = impressions_col
            if clicks_col:
                kwargs['clicks_col'] = clicks_col

            html = render_chart(df_clean, metric=metric, chart_title=chart_title, **kwargs)

            components.html(html, height=800, scrolling=False)
            st.info("Click a circle to zoom in. Click the background to zoom back out.")

            st.download_button(
                label="Download HTML",
                data=html,
                file_name="topical_map.html",
                mime="text/html",
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a CSV to get started. Expected columns: a parent topic, a subtopic, and a keyword column, plus optional metric columns (impressions, clicks, position).")
