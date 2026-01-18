####################################################################################
#                                                                                  #
#  Link Quality Analyzer                                                           #
#                                                                                  #
#  Extract and analyze internal links from pages, check status codes.              #
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
Link Quality Analyzer

Extracts internal links from pages, checks their HTTP status codes, and
calculates reading metrics for content. Great for finding broken links
and spammy anchor text patterns.

Features:
- Extract links from XML sitemap or URL list
- Configurable CSS selector for content area
- HTTP status code checking
- Anchor text frequency analysis
- Reading score metrics
- Export to CSV
"""

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import time

st.set_page_config(page_title="Link Quality Analyzer", page_icon="🔗", layout="wide")

# Check for textstat
try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

st.title("Link Quality Analyzer")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-181717?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Extracts all internal links from specified pages
    - Checks HTTP status codes for each link
    - Analyzes anchor text patterns (frequency, duplicates)
    - Calculates reading scores for page content
    - Identifies broken links and spammy patterns

    **Input options:**
    1. XML sitemap URL
    2. Upload CSV with URLs
    3. Paste URLs directly

    **Content selector:**
    - Specify a CSS selector to target only the main content area
    - This avoids analyzing navigation, footer, and sidebar links
    - Example: `article`, `main`, `.content`, `#post-content`

    **Output includes:**
    - Source URL and link destination
    - Anchor text used
    - HTTP status codes
    - Anchor frequency analysis
    - Reading score per page
    """)

# Sidebar settings
st.sidebar.header("Settings")

content_selector = st.sidebar.text_input(
    "Content CSS selector",
    value="body",
    help="CSS selector for main content area (e.g., 'article', 'main', '.content')"
)

check_status = st.sidebar.checkbox(
    "Check HTTP status codes",
    value=True,
    help="Makes additional requests to verify each link - slower but more thorough"
)

calculate_reading = st.sidebar.checkbox(
    "Calculate reading scores",
    value=TEXTSTAT_AVAILABLE,
    help="Calculate Flesch reading scores for page content"
)

if not TEXTSTAT_AVAILABLE and calculate_reading:
    st.sidebar.warning("Install textstat for reading scores: pip install textstat")
    calculate_reading = False

st.sidebar.markdown("---")
st.sidebar.header("Request Settings")

delay = st.sidebar.slider(
    "Delay between requests (seconds)",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.5
)

timeout = st.sidebar.number_input(
    "Request timeout (seconds)",
    min_value=5,
    max_value=60,
    value=15
)

max_urls = st.sidebar.number_input(
    "Maximum URLs to process",
    min_value=1,
    max_value=1000,
    value=50
)

user_agent = st.sidebar.text_input(
    "User Agent",
    value="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
)

# URL input
st.subheader("Enter URLs to Analyze")

input_method = st.radio(
    "Input method",
    ["XML Sitemap URL", "Upload CSV", "Paste URLs"],
    horizontal=True
)

urls = []


def fetch_urls_from_sitemap(sitemap_url):
    """Fetch URLs from an XML sitemap."""
    try:
        headers = {'User-Agent': user_agent}
        response = requests.get(sitemap_url, headers=headers, timeout=30)
        soup = BeautifulSoup(response.content, 'xml')
        all_urls = []

        # Check for sitemap index
        sitemaps = soup.find_all('sitemap')
        if sitemaps:
            for sm in sitemaps:
                loc = sm.find('loc')
                if loc:
                    child_urls = fetch_urls_from_sitemap(loc.text)
                    all_urls.extend(child_urls)
        else:
            locs = soup.find_all('loc')
            all_urls = [loc.text for loc in locs]

        # Filter out images
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.svg', '.pdf', '.webp')
        all_urls = [url for url in all_urls if not url.lower().endswith(image_extensions)]

        return all_urls
    except Exception as e:
        st.error(f"Error fetching sitemap: {str(e)}")
        return []


if input_method == "XML Sitemap URL":
    sitemap_url = st.text_input("Enter XML Sitemap URL")
    if sitemap_url and st.button("Fetch URLs"):
        with st.spinner("Fetching URLs from sitemap..."):
            urls = fetch_urls_from_sitemap(sitemap_url)
            if urls:
                st.session_state['urls'] = urls[:max_urls]
                st.success(f"Found {len(urls)} URLs (processing first {min(len(urls), max_urls)})")

    if 'urls' in st.session_state:
        urls = st.session_state['urls']

elif input_method == "Upload CSV":
    url_file = st.file_uploader("Upload CSV with URLs", type=['csv'])
    if url_file is not None:
        try:
            df_urls = pd.read_csv(url_file)
            url_col = st.selectbox("Select URL column", df_urls.columns.tolist())
            urls = df_urls[url_col].dropna().tolist()[:max_urls]
            st.info(f"Found {len(urls)} URLs")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

else:
    url_text = st.text_area("Paste URLs (one per line)", height=200)
    if url_text:
        urls = [u.strip() for u in url_text.strip().split('\n') if u.strip()][:max_urls]
        st.info(f"Found {len(urls)} URLs")


def get_domain(url):
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except:
        return ""


def extract_links_from_page(url, selector):
    """Extract all links from a page within the specified selector."""
    try:
        headers = {'User-Agent': user_agent}
        response = requests.get(url, headers=headers, timeout=timeout)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Get page content for reading score
        page_text = ""
        content_container = soup.select_one(selector)
        if content_container:
            page_text = content_container.get_text(separator=' ', strip=True)
        else:
            page_text = soup.get_text(separator=' ', strip=True)

        # Get H1
        h1 = ""
        h1_tag = soup.find('h1')
        if h1_tag:
            h1 = h1_tag.get_text(strip=True)

        # Find links in content
        links = []
        container = soup.select_one(selector) if selector else soup

        if container:
            for a in container.find_all('a', href=True):
                href = a.get('href', '')
                anchor = a.get_text(strip=True)

                # Skip empty anchors
                if not anchor:
                    anchor = "[IMAGE]" if a.find('img') else "[EMPTY]"

                # Normalize URL
                full_url = urljoin(url, href)

                # Skip certain link types
                if any(skip in href for skip in ['mailto:', 'tel:', 'javascript:', '#']):
                    continue

                links.append({
                    'anchor_text': anchor,
                    'link_url': full_url,
                    'is_internal': get_domain(url) == get_domain(full_url)
                })

        return links, page_text, h1

    except Exception as e:
        return [], "", ""


def check_http_status(url):
    """Check HTTP status code of a URL."""
    try:
        headers = {'User-Agent': user_agent}
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=False)
        return response.status_code
    except requests.exceptions.Timeout:
        return "Timeout"
    except requests.exceptions.ConnectionError:
        return "Connection Error"
    except Exception as e:
        return "Error"


# Main processing
if urls and st.button("Analyze Links", type="primary"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    all_links = []
    page_stats = []

    for i, url in enumerate(urls):
        status_text.text(f"Processing {i+1}/{len(urls)}: {url[:60]}...")
        progress_bar.progress((i + 1) / len(urls))

        links, page_text, h1 = extract_links_from_page(url, content_selector)

        # Calculate reading score
        reading_score = None
        if calculate_reading and page_text and len(page_text.split()) > 50:
            try:
                reading_score = round(textstat.flesch_reading_ease(page_text), 2)
            except:
                pass

        page_stats.append({
            'source_url': url,
            'h1': h1,
            'links_count': len(links),
            'internal_links': sum(1 for l in links if l['is_internal']),
            'external_links': sum(1 for l in links if not l['is_internal']),
            'flesch_score': reading_score
        })

        for link in links:
            all_links.append({
                'source_url': url,
                'anchor_text': link['anchor_text'],
                'link_url': link['link_url'],
                'is_internal': link['is_internal']
            })

        time.sleep(delay)

    # Check status codes if enabled
    if check_status and all_links:
        status_text.text("Checking link status codes...")
        unique_links = list(set(l['link_url'] for l in all_links))

        status_cache = {}
        for i, link_url in enumerate(unique_links):
            if i % 10 == 0:
                progress_bar.progress((i + 1) / len(unique_links))
            status_cache[link_url] = check_http_status(link_url)
            time.sleep(delay / 2)

        for link in all_links:
            link['status_code'] = status_cache.get(link['link_url'], 'Unknown')

    status_text.text("Analysis complete!")

    # Create DataFrames
    df_links = pd.DataFrame(all_links)
    df_pages = pd.DataFrame(page_stats)

    # Display results
    st.subheader("Results Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pages Analyzed", len(page_stats))
    with col2:
        st.metric("Total Links Found", len(all_links))
    with col3:
        internal = sum(1 for l in all_links if l.get('is_internal'))
        st.metric("Internal Links", internal)
    with col4:
        if check_status:
            broken = sum(1 for l in all_links if str(l.get('status_code', '')).startswith('4'))
            st.metric("Broken Links (4xx)", broken)

    # Anchor text analysis
    if all_links:
        st.subheader("Anchor Text Analysis")

        df_anchors = df_links.groupby(['source_url', 'anchor_text']).size().reset_index(name='frequency')
        df_anchors = df_anchors.sort_values('frequency', ascending=False)

        # Find spammy patterns (high frequency identical anchors)
        high_freq = df_anchors[df_anchors['frequency'] > 3]
        if len(high_freq) > 0:
            st.warning(f"Found {len(high_freq)} anchor texts used more than 3 times on a single page")
            st.dataframe(high_freq.head(20))

        # Most common anchors overall
        st.subheader("Most Common Anchor Texts")
        anchor_counts = df_links['anchor_text'].value_counts().head(20)
        st.bar_chart(anchor_counts)

    # Broken links
    if check_status and all_links:
        st.subheader("Link Status Breakdown")
        status_counts = df_links['status_code'].value_counts()
        st.dataframe(status_counts)

        broken_links = df_links[df_links['status_code'].astype(str).str.startswith('4')]
        if len(broken_links) > 0:
            st.subheader("Broken Links (4xx)")
            st.dataframe(broken_links[['source_url', 'anchor_text', 'link_url', 'status_code']])

    # Page stats
    st.subheader("Page Statistics")
    st.dataframe(df_pages, use_container_width=True)

    # Download
    st.subheader("Download")

    col1, col2 = st.columns(2)

    with col1:
        csv_links = df_links.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="Download All Links (CSV)",
            data=csv_links,
            file_name="link_analysis.csv",
            mime="text/csv"
        )

    with col2:
        csv_pages = df_pages.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="Download Page Stats (CSV)",
            data=csv_pages,
            file_name="page_stats.csv",
            mime="text/csv"
        )

elif not urls:
    st.info("Enter URLs to analyze using one of the methods above")

    st.subheader("Example Output")
    example = {
        "Source URL": ["/blog/guide-1", "/blog/guide-1", "/blog/guide-2"],
        "Anchor Text": ["click here", "click here", "best products"],
        "Link URL": ["/products/a", "/products/b", "/products/a"],
        "Status Code": [200, 404, 200],
        "Is Internal": [True, True, True]
    }
    st.dataframe(pd.DataFrame(example))
