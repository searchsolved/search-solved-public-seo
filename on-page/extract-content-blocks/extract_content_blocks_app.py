"""
Extract Content Blocks - Streamlit App

Uses Claude AI to identify content blocks and their XPath selectors from web pages.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import time

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
st.markdown("Use AI to identify content blocks and their XPath selectors from web pages.")


def fetch_webpage(url, timeout=30):
    """Fetch webpage content."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text, None
    except Exception as e:
        return None, str(e)


def filter_html(html_content):
    """Remove scripts, styles, header, footer, nav to reduce tokens."""
    soup = BeautifulSoup(html_content, 'html.parser')

    for tag in soup(['script', 'style', 'noscript', 'meta', 'link', 'header', 'footer', 'nav']):
        tag.decompose()

    return str(soup)


def extract_blocks(html_content, client):
    """Call Claude to extract content blocks with XPath."""
    prompt = f"""Analyze this HTML and identify major content blocks/sections.

For each block provide:
- name: descriptive name
- xpath: robust XPath expression to select this element
- notes: brief description

Focus on main content areas (hero sections, feature blocks, carousels). Skip small utility elements.

Return ONLY a JSON array with this exact format:
[
  {{"name": "Hero Section", "xpath": "//div[@class='hero']", "notes": "Main hero banner"}},
  {{"name": "Features", "xpath": "//section[@class='features']", "notes": "Feature grid"}}
]

HTML:
{html_content[:50000]}
"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4000,
            system=[{
                "type": "text",
                "text": "You are an expert web scraper. Extract content blocks and provide XPath selectors. Return only valid JSON."
            }],
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.content[0].text

        # Extract JSON
        start = content.find('[')
        end = content.rfind(']') + 1
        if start != -1 and end > start:
            return json.loads(content[start:end]), None

        return [], "No JSON found in response"
    except json.JSONDecodeError as e:
        return [], f"JSON error: {str(e)}"
    except Exception as e:
        return [], str(e)


def process_urls(urls, client, progress_bar, status_text):
    """Process multiple URLs."""
    all_blocks = []

    for i, url in enumerate(urls):
        progress_bar.progress((i + 1) / len(urls))
        status_text.text(f"Processing {i + 1}/{len(urls)}: {url[:50]}...")

        html, error = fetch_webpage(url)
        if error:
            continue

        filtered = filter_html(html)
        blocks, error = extract_blocks(filtered, client)

        if blocks:
            for block in blocks:
                block['url'] = url
                all_blocks.append(block)

        time.sleep(1)  # Rate limiting

    return pd.DataFrame(all_blocks)


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

        results_df = process_urls(urls, client, progress_bar, status_text)

        progress_bar.empty()
        status_text.empty()

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

                # Create deduped reference
                xpath_ref = (
                    results_df.groupby('xpath')
                    .agg({
                        'name': 'first',
                        'notes': 'first',
                        'url': 'count'
                    })
                    .reset_index()
                    .rename(columns={'url': 'frequency'})
                    .sort_values('frequency', ascending=False)
                )

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
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)"
)
