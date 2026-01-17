"""
Sitemap URL Extractor - Streamlit App
Extracts all URLs from sitemap indexes and child sitemaps.

Author: Lee Foot
Website: https://leefoot.com
"""

import streamlit as st
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import gzip
import re
import io
from datetime import datetime

st.set_page_config(
    page_title="Sitemap URL Extractor",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ Sitemap URL Extractor")
st.markdown("Extract all URLs from any XML sitemap index and its child sitemaps.")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    sitemap_url = st.text_input(
        "Sitemap URL",
        placeholder="https://example.com/sitemap.xml",
        help="Enter the URL of your sitemap index or sitemap file"
    )

    user_agent = st.text_input(
        "User Agent",
        value="Mozilla/5.0 (compatible; SitemapExtractor/1.0)",
        help="User agent string for requests"
    )

    request_delay = st.slider(
        "Request Delay (seconds)",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.5,
        help="Delay between requests to avoid overwhelming the server"
    )

    include_metadata = st.checkbox(
        "Include Metadata",
        value=True,
        help="Include lastmod, priority, and changefreq if available"
    )


def extract_sitemap_urls(content, is_bytes=False):
    """Extract URLs and metadata from sitemap XML content."""
    urls_data = []

    if is_bytes:
        xml_content = content
        text_content = content.decode('utf-8', errors='ignore')
    else:
        xml_content = content.encode('utf-8')
        text_content = content

    # Try ElementTree parsing first
    try:
        root = ET.fromstring(xml_content)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        # Check if this is a sitemap index
        sitemaps = root.findall(".//sm:sitemap", ns)
        if not sitemaps:
            sitemaps = root.findall(".//sitemap")

        if sitemaps:
            # This is a sitemap index
            for sitemap in sitemaps:
                loc = sitemap.find("sm:loc", ns)
                if loc is None:
                    loc = sitemap.find("loc")
                if loc is not None and loc.text:
                    url_data = {"url": loc.text.strip(), "type": "sitemap"}

                    # Get lastmod if available
                    lastmod = sitemap.find("sm:lastmod", ns)
                    if lastmod is None:
                        lastmod = sitemap.find("lastmod")
                    if lastmod is not None and lastmod.text:
                        url_data["lastmod"] = lastmod.text.strip()

                    urls_data.append(url_data)
            return urls_data, True  # is_index = True

        # This is a regular sitemap
        url_elements = root.findall(".//sm:url", ns)
        if not url_elements:
            url_elements = root.findall(".//url")

        for url_elem in url_elements:
            loc = url_elem.find("sm:loc", ns)
            if loc is None:
                loc = url_elem.find("loc")
            if loc is not None and loc.text:
                url_data = {"url": loc.text.strip(), "type": "url"}

                # Get optional metadata
                lastmod = url_elem.find("sm:lastmod", ns)
                if lastmod is None:
                    lastmod = url_elem.find("lastmod")
                if lastmod is not None and lastmod.text:
                    url_data["lastmod"] = lastmod.text.strip()

                priority = url_elem.find("sm:priority", ns)
                if priority is None:
                    priority = url_elem.find("priority")
                if priority is not None and priority.text:
                    url_data["priority"] = priority.text.strip()

                changefreq = url_elem.find("sm:changefreq", ns)
                if changefreq is None:
                    changefreq = url_elem.find("changefreq")
                if changefreq is not None and changefreq.text:
                    url_data["changefreq"] = changefreq.text.strip()

                urls_data.append(url_data)

        return urls_data, False  # is_index = False

    except ET.ParseError:
        pass

    # Fallback to regex
    loc_pattern = r"<loc>(.*?)</loc>"
    urls = re.findall(loc_pattern, text_content, re.DOTALL)

    # Determine if it's an index by checking for <sitemap> tags
    is_index = "<sitemap>" in text_content.lower() or "<sitemap " in text_content.lower()

    for url in urls:
        url_data = {"url": url.strip(), "type": "sitemap" if is_index else "url"}
        urls_data.append(url_data)

    return urls_data, is_index


def fetch_sitemap(url, user_agent):
    """Fetch sitemap content, handling gzip compression."""
    headers = {"User-Agent": user_agent}

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    # Check if content is gzipped
    if url.endswith('.gz') or response.content.startswith(b'\x1f\x8b'):
        try:
            content = gzip.decompress(response.content)
            return content, True
        except Exception:
            pass

    return response.content, True


def process_sitemap(sitemap_url, user_agent, delay, progress_callback=None):
    """Process sitemap index and all child sitemaps."""
    import time

    all_urls = []
    sitemaps_to_process = [sitemap_url]
    processed_sitemaps = set()

    while sitemaps_to_process:
        current_url = sitemaps_to_process.pop(0)

        if current_url in processed_sitemaps:
            continue

        processed_sitemaps.add(current_url)

        if progress_callback:
            progress_callback(f"Processing: {current_url[:80]}...")

        try:
            content, is_bytes = fetch_sitemap(current_url, user_agent)
            urls_data, is_index = extract_sitemap_urls(content, is_bytes)

            if is_index:
                # Add child sitemaps to queue
                for item in urls_data:
                    if item["url"] not in processed_sitemaps:
                        sitemaps_to_process.append(item["url"])
            else:
                # Add URLs to results
                for item in urls_data:
                    item["source_sitemap"] = current_url
                    all_urls.append(item)

            if delay > 0 and sitemaps_to_process:
                time.sleep(delay)

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error processing {current_url}: {str(e)}")

    return all_urls


# Main app logic
if sitemap_url:
    if st.button("Extract URLs", type="primary"):
        status_container = st.empty()
        progress_bar = st.progress(0)

        def update_status(message):
            status_container.text(message)

        try:
            with st.spinner("Extracting URLs from sitemap..."):
                urls = process_sitemap(
                    sitemap_url,
                    user_agent,
                    request_delay,
                    progress_callback=update_status
                )

            progress_bar.progress(100)

            if urls:
                st.success(f"Extracted {len(urls):,} URLs!")

                # Create DataFrame
                df = pd.DataFrame(urls)

                # Reorder columns
                columns_order = ["url"]
                if include_metadata:
                    if "lastmod" in df.columns:
                        columns_order.append("lastmod")
                    if "priority" in df.columns:
                        columns_order.append("priority")
                    if "changefreq" in df.columns:
                        columns_order.append("changefreq")
                    if "source_sitemap" in df.columns:
                        columns_order.append("source_sitemap")

                # Only include columns that exist
                columns_order = [c for c in columns_order if c in df.columns]
                if not include_metadata:
                    columns_order = ["url"]

                df = df[columns_order]

                # Display summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total URLs", f"{len(df):,}")
                with col2:
                    if "lastmod" in df.columns:
                        has_lastmod = df["lastmod"].notna().sum()
                        st.metric("URLs with lastmod", f"{has_lastmod:,}")
                with col3:
                    unique_sitemaps = df["source_sitemap"].nunique() if "source_sitemap" in df.columns else 1
                    st.metric("Sitemaps Processed", unique_sitemaps)

                # Display data
                st.dataframe(df, use_container_width=True, height=400)

                # Download options
                st.subheader("Download")
                col1, col2, col3 = st.columns(3)

                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        file_name=f"sitemap_urls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                with col2:
                    # URL list only
                    url_list = "\n".join(df["url"].tolist())
                    st.download_button(
                        "Download URL List (TXT)",
                        url_list,
                        file_name=f"sitemap_urls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )

                with col3:
                    # Excel
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='URLs')
                    excel_data = output.getvalue()
                    st.download_button(
                        "Download Excel",
                        excel_data,
                        file_name=f"sitemap_urls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.warning("No URLs found in the sitemap.")

        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching sitemap: {str(e)}")
        except Exception as e:
            st.error(f"Error processing sitemap: {str(e)}")

else:
    st.info("Enter a sitemap URL in the sidebar to get started.")

    # Show examples
    st.subheader("Examples")
    st.markdown("""
    **Common sitemap locations:**
    - `https://example.com/sitemap.xml`
    - `https://example.com/sitemap_index.xml`
    - `https://example.com/sitemaps/sitemap.xml`

    **Supported formats:**
    - XML Sitemaps
    - Sitemap Index files
    - Gzipped sitemaps (.xml.gz)
    """)

# Footer
st.markdown("---")
st.markdown("Built by 🌐 [Lee Foot](https://leefoot.com) · [LinkedIn](https://www.linkedin.com/in/lee-foot/) · 🦋 [Bluesky](https://bsky.app/profile/leefootseo.bsky.social)")
