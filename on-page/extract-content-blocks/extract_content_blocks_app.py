"""
Extract Content Blocks - Streamlit App

Uses Claude AI to identify content blocks and their XPath selectors from web pages.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
import time

from content_blocks_core import (
    add_frequency,
    blocks_to_frame,
    extract_from_url,
    xpath_reference,
)

try:
    from anthropic import Anthropic
except ImportError:
    st.error("Please install anthropic: pip install anthropic")
    st.stop()

st.set_page_config(
    page_title="Extract Content Blocks",
    page_icon="🧱",
    layout="wide"
)

st.title("🧱 Extract Content Blocks")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Fetches each page and removes scripts, styles, header, footer and nav
    - Asks Claude to name the major content blocks and give an XPath for each
    - Counts how often each XPath appears across the pages

    **How to use:**
    1. Enter your Anthropic API key in the sidebar
    2. Enter one URL, paste several, or upload a CSV and pick the URL column
    3. Extract content blocks
    4. Download the results, with or without the frequency column

    **Best for:**
    - Content structure audits
    - Template analysis
    - Content migration preparation
    """)
st.markdown("Use AI to identify content blocks and their XPath selectors from web pages.")


def process_urls(urls, client, progress_bar, status_text):
    """Process multiple URLs. Returns (results DataFrame, list of (url, error))."""
    all_rows = []
    errors = []

    for i, url in enumerate(urls):
        progress_bar.progress((i + 1) / len(urls))
        status_text.text(f"Processing {i + 1}/{len(urls)}: {url[:50]}...")

        try:
            rows, info = extract_from_url(client, url)
            all_rows.extend(rows)
        except Exception as e:
            errors.append((url, str(e)))

        time.sleep(1)  # Rate limiting

    return blocks_to_frame(all_rows), errors


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="Enter your Anthropic API key"
    )

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    This tool uses **Claude AI** to:
    - Analyze HTML structure
    - Identify content blocks
    - Generate XPath selectors
    - Find common patterns

    Useful for:
    - Template identification
    - Scraping setup
    - Content audits
    """)

# Main content
input_method = st.radio(
    "Input Method",
    ["Single URL", "Multiple URLs", "Upload CSV"],
    horizontal=True
)

urls = []

if input_method == "Single URL":
    url = st.text_input(
        "Enter URL",
        placeholder="https://example.com",
        help="Enter a full URL to analyze"
    )
    if url:
        urls = [url]

elif input_method == "Multiple URLs":
    urls_text = st.text_area(
        "Enter URLs (one per line)",
        height=150,
        placeholder="https://example.com/page1\nhttps://example.com/page2"
    )
    if urls_text:
        urls = [u.strip() for u in urls_text.split('\n') if u.strip().startswith('http')]

else:  # Upload CSV
    uploaded_file = st.file_uploader(
        "Upload CSV with URLs",
        type=["csv"],
        help="CSV should contain a column with URLs"
    )
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        url_col = st.selectbox("URL Column", df.columns.tolist())
        urls = df[url_col].dropna().tolist()
        urls = [u for u in urls if isinstance(u, str) and u.startswith('http')]

if urls:
    st.info(f"Ready to process **{len(urls)}** URL(s)")

    # Preview URLs
    with st.expander("Preview URLs"):
        for i, url in enumerate(urls[:10]):
            st.markdown(f"{i + 1}. {url}")
        if len(urls) > 10:
            st.markdown(f"... and {len(urls) - 10} more")

if urls and api_key:
    if st.button("🧱 Extract Content Blocks", type="primary", use_container_width=True):
        client = Anthropic(api_key=api_key)
        progress_bar = st.progress(0)
        status_text = st.empty()

        results_df, errors = process_urls(urls, client, progress_bar, status_text)

        progress_bar.empty()
        status_text.empty()

        if errors:
            with st.expander(f"{len(errors)} URL(s) could not be processed"):
                for failed_url, message in errors:
                    st.markdown(f"- {failed_url}: {message}")

        if len(results_df) > 0:
            st.success(f"✅ Extracted {len(results_df)} content blocks from {len(urls)} URLs!")

            # Results tabs
            tab1, tab2, tab3 = st.tabs(["📊 All Blocks", "📈 Frequency Analysis", "🔧 XPath Reference"])

            with tab1:
                st.dataframe(results_df, use_container_width=True, height=400)

                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Results CSV",
                    data=csv,
                    file_name="content_blocks.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                # Same file the command line script writes: one name per XPath
                # and a frequency column, sorted by frequency.
                combined_csv = add_frequency(results_df).to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download With Frequency CSV",
                    data=combined_csv,
                    file_name="combined_output_with_frequency.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with tab2:
                # XPath frequency
                if 'xpath' in results_df.columns:
                    st.subheader("Most Common XPaths")
                    xpath_counts = results_df['xpath'].value_counts().head(15)

                    import plotly.express as px
                    fig = px.bar(
                        x=xpath_counts.values,
                        y=[x[:50] + '...' if len(x) > 50 else x for x in xpath_counts.index],
                        orientation='h',
                        labels={'x': 'Frequency', 'y': 'XPath'},
                        title='Top 15 XPaths by Frequency'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)

                # Block name frequency
                if 'name' in results_df.columns:
                    st.subheader("Most Common Block Types")
                    name_counts = results_df['name'].value_counts().head(10)

                    fig = px.pie(
                        values=name_counts.values,
                        names=name_counts.index,
                        title='Block Type Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with tab3:
                st.subheader("Unique XPaths Reference")

                # One row per XPath with its first name, notes and frequency
                xpath_ref = xpath_reference(results_df)

                st.dataframe(xpath_ref, use_container_width=True, height=400)

                st.markdown("---")
                st.markdown("**Copy XPaths for Scraping:**")

                for _, row in xpath_ref.head(10).iterrows():
                    st.code(row['xpath'], language='xpath')

        else:
            st.warning("No content blocks found. Try different URLs.")

elif urls and not api_key:
    st.warning("⚠️ Please enter your Anthropic API key in the sidebar")

elif not urls:
    st.info("👆 Enter URL(s) to analyze")

    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        This tool uses **Claude AI** to analyze web pages and identify
        major content blocks with their XPath selectors.

        **Use cases:**
        - **Template Analysis**: Identify common patterns across pages
        - **Scraping Setup**: Get XPaths for content extraction
        - **Content Audits**: Understand page structure

        **How it works:**
        1. Fetches the HTML from each URL
        2. Cleans and filters the HTML
        3. Sends to Claude for analysis
        4. Returns structured block information with XPaths
        """)

# Footer
st.markdown("---")
st.markdown(
    "Built by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · "
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · "
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)"
)
