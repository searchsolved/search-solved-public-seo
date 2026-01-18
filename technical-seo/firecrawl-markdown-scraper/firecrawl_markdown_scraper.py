"""
Firecrawl Markdown Scraper - Streamlit App
Scrape URLs and convert to clean markdown using Firecrawl API.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
import requests
import json
from urllib.parse import urlparse
from io import BytesIO
from time import sleep
import zipfile

st.set_page_config(
    page_title="Firecrawl Markdown Scraper",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Firecrawl Markdown Scraper")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Scrapes web pages using Firecrawl
    - Converts HTML to clean markdown
    - Handles JavaScript-rendered content

    **How to use:**
    1. Enter your Firecrawl API key
    2. Upload URLs to scrape
    3. Configure scraping options
    4. Download markdown content

    **Best for:**
    - Content extraction at scale
    - JS-rendered page scraping
    - Clean content conversion
    """)
st.markdown("Convert web pages to clean markdown using Firecrawl API.")

# Sidebar configuration
with st.sidebar:
    st.header("API Configuration")
    api_key = st.text_input("Firecrawl API Key", type="password",
                            help="Get your API key from firecrawl.dev")

    st.header("Scraping Options")
    only_main_content = st.checkbox("Only main content", value=True,
                                     help="Skip navigation, footer, sidebar")
    block_ads = st.checkbox("Block ads", value=True)
    remove_base64 = st.checkbox("Remove base64 images", value=True)

    wait_time = st.slider("Wait for JS (ms)", 0, 10000, 2000,
                          help="Time to wait for dynamic content")

    request_delay = st.slider("Delay between requests (sec)", 1, 15, 5,
                              help="Increase for rate limiting")

    st.header("Output Options")
    include_metadata = st.checkbox("Include metadata header", value=True)


def scrape_url(url, api_key, options, max_retries=3):
    """Scrape a URL using Firecrawl API."""
    api_url = "https://api.firecrawl.dev/v1/scrape"

    payload = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": options.get('only_main_content', True),
        "timeout": 30000,
        "blockAds": options.get('block_ads', True),
        "removeBase64Images": options.get('remove_base64', True),
        "waitFor": options.get('wait_time', 2000)
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=60)

            # Handle rate limiting
            if response.status_code == 429:
                delay = (2 ** attempt) * 10
                sleep(delay)
                continue

            response.raise_for_status()
            data = response.json()

            if data.get('success'):
                return {
                    'success': True,
                    'markdown': data.get('data', {}).get('markdown', ''),
                    'metadata': data.get('data', {}).get('metadata', {})
                }, None
            else:
                return None, data.get('error', 'Unknown error')

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                sleep((2 ** attempt) * 5)
                continue
            return None, str(e)

    return None, "Max retries exceeded"


def get_filename_from_url(url):
    """Generate filename from URL."""
    parsed = urlparse(url)
    path = parsed.path.strip('/')
    if path:
        filename = path.replace('/', '_')
    else:
        filename = parsed.netloc.replace('.', '_')
    return filename[:100]  # Limit length


def format_markdown_with_metadata(markdown, metadata, url, include_meta=True):
    """Format markdown with optional metadata header."""
    if not include_meta:
        return markdown

    header = f"""---
url: {url}
title: {metadata.get('title', 'N/A')}
description: {metadata.get('description', 'N/A')}
---

"""
    return header + markdown


# Main interface
tab1, tab2, tab3 = st.tabs(["Single URL", "Bulk Scraping", "URL List"])

with tab1:
    st.subheader("Scrape Single URL")

    url = st.text_input("Enter URL to scrape", placeholder="https://example.com/page")

    if st.button("Scrape URL", type="primary", disabled=not api_key or not url):
        options = {
            'only_main_content': only_main_content,
            'block_ads': block_ads,
            'remove_base64': remove_base64,
            'wait_time': wait_time
        }

        with st.spinner("Scraping..."):
            result, error = scrape_url(url, api_key, options)

        if error:
            st.error(f"Failed: {error}")
        elif result:
            markdown = result['markdown']
            metadata = result['metadata']

            st.success(f"Scraped {len(markdown)} characters")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Title", metadata.get('title', 'N/A')[:50] + "..." if len(metadata.get('title', '')) > 50 else metadata.get('title', 'N/A'))
            with col2:
                st.metric("Content Length", f"{len(markdown):,} chars")

            # Display markdown
            with st.expander("View Markdown", expanded=True):
                st.code(markdown[:5000] + "..." if len(markdown) > 5000 else markdown, language="markdown")

            # Download
            formatted = format_markdown_with_metadata(markdown, metadata, url, include_metadata)
            filename = get_filename_from_url(url)

            st.download_button("Download Markdown",
                               formatted,
                               f"{filename}.md",
                               "text/markdown")

with tab2:
    st.subheader("Bulk Scraping from File")

    uploaded_file = st.file_uploader("Upload CSV/Excel with URLs", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write(f"Loaded {len(df)} rows")

            url_col = st.selectbox("URL Column", list(df.columns))
            max_urls = st.slider("Maximum URLs to scrape", 1, min(100, len(df)), min(20, len(df)))

            if st.button("Start Bulk Scraping", type="primary", disabled=not api_key):
                urls = df[url_col].dropna().head(max_urls).tolist()

                options = {
                    'only_main_content': only_main_content,
                    'block_ads': block_ads,
                    'remove_base64': remove_base64,
                    'wait_time': wait_time
                }

                results = []
                markdown_files = {}

                progress = st.progress(0)
                status = st.empty()

                for idx, url in enumerate(urls):
                    status.text(f"Scraping {idx + 1}/{len(urls)}: {url[:50]}...")

                    result, error = scrape_url(url, api_key, options)

                    if error:
                        results.append({
                            'url': url,
                            'status': 'Failed',
                            'error': error,
                            'content_length': 0,
                            'title': ''
                        })
                    elif result:
                        markdown = result['markdown']
                        metadata = result['metadata']

                        results.append({
                            'url': url,
                            'status': 'Success',
                            'content_length': len(markdown),
                            'title': metadata.get('title', ''),
                            'error': ''
                        })

                        # Store markdown for download
                        filename = get_filename_from_url(url)
                        formatted = format_markdown_with_metadata(markdown, metadata, url, include_metadata)
                        markdown_files[f"{filename}.md"] = formatted

                    progress.progress((idx + 1) / len(urls))

                    if idx < len(urls) - 1:
                        sleep(request_delay)

                status.text("Complete!")

                # Display results
                results_df = pd.DataFrame(results)

                col1, col2, col3 = st.columns(3)
                col1.metric("Successful", len(results_df[results_df['status'] == 'Success']))
                col2.metric("Failed", len(results_df[results_df['status'] == 'Failed']))
                col3.metric("Total Characters", f"{results_df['content_length'].sum():,}")

                st.dataframe(results_df, use_container_width=True)

                # Download options
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.download_button("Download Results CSV",
                                       results_df.to_csv(index=False),
                                       "scrape_results.csv",
                                       "text/csv")

                with col2:
                    # Create ZIP of all markdown files
                    if markdown_files:
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                            for filename, content in markdown_files.items():
                                zf.writestr(filename, content)
                        zip_buffer.seek(0)

                        st.download_button("Download All Markdown (ZIP)",
                                           zip_buffer.getvalue(),
                                           "markdown_files.zip",
                                           "application/zip")

                with col3:
                    # Combined markdown
                    if markdown_files:
                        combined = "\n\n---\n\n".join([
                            f"# {filename}\n\n{content}"
                            for filename, content in markdown_files.items()
                        ])
                        st.download_button("Download Combined Markdown",
                                           combined,
                                           "combined_content.md",
                                           "text/markdown")

        except Exception as e:
            st.error(f"Error loading file: {e}")

with tab3:
    st.subheader("Paste URL List")

    urls_input = st.text_area("Paste URLs (one per line)",
                               height=200,
                               placeholder="https://example.com/page1\nhttps://example.com/page2")

    if st.button("Scrape All URLs", type="primary", disabled=not api_key or not urls_input):
        urls = [u.strip() for u in urls_input.split('\n') if u.strip()]

        if not urls:
            st.warning("No valid URLs found")
        else:
            options = {
                'only_main_content': only_main_content,
                'block_ads': block_ads,
                'remove_base64': remove_base64,
                'wait_time': wait_time
            }

            results = []
            markdown_files = {}

            progress = st.progress(0)
            status = st.empty()

            for idx, url in enumerate(urls):
                status.text(f"Scraping {idx + 1}/{len(urls)}: {url[:50]}...")

                result, error = scrape_url(url, api_key, options)

                if error:
                    results.append({
                        'url': url,
                        'status': 'Failed',
                        'error': error,
                        'content_length': 0
                    })
                elif result:
                    markdown = result['markdown']
                    metadata = result['metadata']

                    results.append({
                        'url': url,
                        'status': 'Success',
                        'content_length': len(markdown),
                        'title': metadata.get('title', '')
                    })

                    filename = get_filename_from_url(url)
                    formatted = format_markdown_with_metadata(markdown, metadata, url, include_metadata)
                    markdown_files[f"{filename}.md"] = formatted

                progress.progress((idx + 1) / len(urls))

                if idx < len(urls) - 1:
                    sleep(request_delay)

            status.text("Complete!")

            # Display results
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)

            # Download
            if markdown_files:
                col1, col2 = st.columns(2)

                with col1:
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for filename, content in markdown_files.items():
                            zf.writestr(filename, content)
                    zip_buffer.seek(0)

                    st.download_button("Download All (ZIP)",
                                       zip_buffer.getvalue(),
                                       "markdown_files.zip",
                                       "application/zip")

                with col2:
                    combined = "\n\n---\n\n".join([
                        f"# {filename}\n\n{content}"
                        for filename, content in markdown_files.items()
                    ])
                    st.download_button("Download Combined",
                                       combined,
                                       "combined.md",
                                       "text/markdown")

# Help section
with st.expander("How to Use"):
    st.markdown("""
    **Getting Started:**
    1. Get your Firecrawl API key from [firecrawl.dev](https://firecrawl.dev)
    2. Enter your API key in the sidebar
    3. Enter URLs to scrape

    **Scraping Options:**
    - **Only main content**: Removes navigation, footer, sidebar
    - **Block ads**: Removes ad content for cleaner output
    - **Remove base64 images**: Reduces output size
    - **Wait for JS**: Time to wait for dynamic content to load

    **Output:**
    - Clean markdown preserving headings, lists, links, tables
    - Optional metadata header with title and description
    - Individual files or combined output

    **Rate Limiting:**
    - Free tier has rate limits - increase delay between requests
    - Tool automatically retries on rate limit errors

    **Use Cases:**
    - Content migration
    - LLM training data collection
    - Content audits
    - Competitor content analysis
    - Documentation archival
    """)

# Footer
st.markdown("---")
