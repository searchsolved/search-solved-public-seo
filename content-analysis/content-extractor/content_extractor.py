"""
Content Extractor - Extract Main Content and H1 from URLs
Useful for content audits and striking distance analysis.

Author: Lee Foot
Date: January 2025
"""

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(
    page_title="Content Extractor",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Content Extractor")
st.markdown("""
Extract main text content and H1 headings from URLs.
Useful for content audits and striking distance analysis.
""")

# Sidebar configuration
st.sidebar.header("Configuration")

rate_limit = st.sidebar.slider(
    "Delay between requests (seconds)",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.5,
    help="Time to wait between URL requests to avoid rate limiting"
)

max_workers = st.sidebar.slider(
    "Concurrent requests",
    min_value=1,
    max_value=10,
    value=3,
    help="Number of URLs to process simultaneously"
)

user_agent = st.sidebar.text_input(
    "User Agent",
    value="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    help="User agent string for HTTP requests"
)

timeout = st.sidebar.slider(
    "Request timeout (seconds)",
    min_value=5,
    max_value=30,
    value=10,
    help="Maximum time to wait for a response"
)


def fetch_page(url, headers, timeout_val):
    """Fetch a web page content with custom headers."""
    try:
        response = requests.get(url, headers=headers, timeout=timeout_val)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        return response.text, None
    except requests.RequestException as e:
        return None, str(e)


def html_to_text(html_content):
    """Convert HTML content to formatted text."""
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()

    # Replace <br> tags with newlines
    for br in soup.find_all("br"):
        br.replace_with("\n")

    # Append newlines to block-level elements
    for element in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "div"]):
        element.append("\n\n")

    # Get the text
    text = soup.get_text(separator=" ")

    # Normalize whitespace
    return ' '.join(text.split())


def extract_h1(html_content):
    """Extract the first H1 tag from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    h1_tag = soup.find('h1')
    return h1_tag.get_text(strip=True) if h1_tag else None


def extract_title(html_content):
    """Extract the page title from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    title_tag = soup.find('title')
    return title_tag.get_text(strip=True) if title_tag else None


def process_url(url, headers, timeout_val, rate_limit_val):
    """Process a single URL and extract content."""
    time.sleep(random.uniform(rate_limit_val * 0.5, rate_limit_val * 1.5))

    html_content, error = fetch_page(url, headers, timeout_val)

    if html_content:
        h1 = extract_h1(html_content)
        title = extract_title(html_content)
        content = html_to_text(html_content)
        return {
            'URL': url,
            'Title': title,
            'H1': h1,
            'Content': content,
            'Content_Length': len(content) if content else 0,
            'Status': 'Success',
            'Error': None
        }
    else:
        return {
            'URL': url,
            'Title': None,
            'H1': None,
            'Content': None,
            'Content_Length': 0,
            'Status': 'Failed',
            'Error': error
        }


# Input methods
st.subheader("Input URLs")
input_method = st.radio(
    "Choose input method:",
    ["Text Area", "CSV Upload"],
    horizontal=True
)

urls = []

if input_method == "Text Area":
    url_input = st.text_area(
        "Enter URLs (one per line)",
        height=200,
        placeholder="https://example.com/page1\nhttps://example.com/page2"
    )
    if url_input:
        urls = [url.strip() for url in url_input.strip().split('\n') if url.strip()]

else:
    uploaded_file = st.file_uploader(
        "Upload CSV with URLs",
        type=['csv'],
        help="Upload a CSV file containing a column with URLs"
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        url_column = st.selectbox(
            "Select URL column",
            options=df.columns.tolist()
        )

        urls = df[url_column].dropna().astype(str).tolist()

if urls:
    st.info(f"Found {len(urls)} URLs to process")

    if st.button("🚀 Extract Content", type="primary"):
        headers = {'User-Agent': user_agent}

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Process URLs with threading
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_url, url, headers, timeout, rate_limit): url
                for url in urls
            }

            completed = 0
            for future in as_completed(futures):
                url = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        'URL': url,
                        'Title': None,
                        'H1': None,
                        'Content': None,
                        'Content_Length': 0,
                        'Status': 'Error',
                        'Error': str(e)
                    })

                completed += 1
                progress_bar.progress(completed / len(urls))
                status_text.text(f"Processing: {completed}/{len(urls)} URLs")

        progress_bar.empty()
        status_text.empty()

        # Create results dataframe
        results_df = pd.DataFrame(results)

        # Sort by original URL order
        url_order = {url: i for i, url in enumerate(urls)}
        results_df['_order'] = results_df['URL'].map(url_order)
        results_df = results_df.sort_values('_order').drop('_order', axis=1)

        # Display summary
        st.subheader("Results Summary")
        col1, col2, col3 = st.columns(3)

        success_count = len(results_df[results_df['Status'] == 'Success'])
        failed_count = len(results_df[results_df['Status'] != 'Success'])

        with col1:
            st.metric("Total URLs", len(results_df))
        with col2:
            st.metric("Successful", success_count)
        with col3:
            st.metric("Failed", failed_count)

        # Display results table
        st.subheader("Extracted Content")

        # Show preview without full content (too long)
        preview_df = results_df[['URL', 'Title', 'H1', 'Content_Length', 'Status', 'Error']].copy()
        st.dataframe(preview_df, use_container_width=True)

        # Show failed URLs if any
        if failed_count > 0:
            with st.expander(f"View Failed URLs ({failed_count})"):
                failed_df = results_df[results_df['Status'] != 'Success'][['URL', 'Error']]
                st.dataframe(failed_df, use_container_width=True)

        # Download options
        st.subheader("Download Results")

        col1, col2 = st.columns(2)

        with col1:
            # CSV download
            csv_buffer = BytesIO()
            results_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            st.download_button(
                label="📥 Download CSV",
                data=csv_buffer,
                file_name="extracted_content.csv",
                mime="text/csv"
            )

        with col2:
            # Excel download
            excel_buffer = BytesIO()
            results_df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)

            st.download_button(
                label="📥 Download Excel",
                data=excel_buffer,
                file_name="extracted_content.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

else:
    st.info("👆 Enter URLs above to get started.")

    st.markdown("""
    ### How it Works
    1. Enter URLs directly or upload a CSV file
    2. Configure rate limiting to avoid being blocked
    3. Click "Extract Content" to process URLs
    4. Download the results as CSV or Excel

    ### What Gets Extracted
    - **Title** - The page title tag
    - **H1** - The first H1 heading
    - **Content** - Main text content (scripts/nav/footer removed)
    - **Content Length** - Character count of extracted content

    ### Use Cases
    - Content audits
    - Striking distance keyword analysis
    - Bulk content extraction for analysis
    - Page content comparison
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("Built by [Lee Foot](https://leefoot.com) · [Bluesky](https://bsky.app/profile/leefootseo.bsky.social) · [LinkedIn](https://www.linkedin.com/in/lee-foot/)")
