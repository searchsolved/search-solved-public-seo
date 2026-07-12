# Author: Lee Foot
# Website: https://leefoot.com

####################################################################################
#                                                                                  #
#  Categories Missing From Navigation - Streamlit App                             #
#                                                                                  #
#  Find sitemap URLs that are not linked from your site navigation.               #
#                                                                                  #
####################################################################################
# Author: Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                       #
####################################################################################

"""
Categories Missing From Navigation - Streamlit App

Fetches an XML sitemap and a page (usually the homepage), extracts links
found within a CSS selector for the navigation element, and reports which
sitemap URLs are missing from the navigation.

Requirements:
    pip install streamlit pandas requests beautifulsoup4 lxml
"""

from io import BytesIO

import pandas as pd
import streamlit as st

from categories_missing_from_navigation import (
    DEFAULT_NAV_SELECTOR,
    extract_navigation_links,
    extract_urls_from_sitemap,
    find_missing_urls,
    homepage_from_sitemap,
)

# App Configuration
st.set_page_config(
    page_title="Categories Missing From Navigation",
    page_icon="🧭",
    layout="wide"
)

st.title("🧭 Categories Missing From Navigation")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Fetches all URLs from an XML sitemap (sitemap index files are supported)
    - Fetches your homepage (or any page you choose) and extracts every link inside the navigation element
    - Reports the sitemap URLs that are not linked from the navigation

    **How to use:**
    1. Enter your XML sitemap URL (for example https://www.example.com/sitemap.xml)
    2. Enter the CSS selector for your navigation element (for example `nav`, `#main-nav` or `.header-menu`)
    3. Optionally filter results to URLs containing a string (for example `/category/`)
    4. Run the check and download the results as a CSV

    **Best for:**
    - Finding category pages with no navigation links
    - Auditing internal linking after a navigation redesign
    - Spotting orphaned sections of large e-commerce sites
    """)

# Sidebar configuration
st.sidebar.header("Settings")

nav_selector = st.sidebar.text_input(
    "Navigation CSS Selector",
    value=DEFAULT_NAV_SELECTOR,
    help="CSS selector for the navigation element, for example nav, #main-nav or .header-menu. "
         "Use your browser's inspector to find it."
)

url_filter = st.sidebar.text_input(
    "URL Contains Filter (optional)",
    value="",
    help="Only report missing URLs containing this string, for example /category/. Leave blank for all URLs."
)

# Main inputs
st.header("Enter Site Details")

sitemap_url = st.text_input(
    "XML Sitemap URL",
    placeholder="https://www.example.com/sitemap.xml",
    help="Standard sitemaps and sitemap index files are both supported."
)

page_url = st.text_input(
    "Page to Check (optional)",
    placeholder="https://www.example.com/ (defaults to the homepage of the sitemap domain)",
    help="The page whose navigation will be checked. Leave blank to use the homepage."
)

if st.button("Find Missing URLs", type="primary"):
    if not sitemap_url or not sitemap_url.startswith("http"):
        st.error("Please enter a valid sitemap URL starting with http:// or https://")
        st.stop()

    if not nav_selector.strip():
        st.error("Please enter a CSS selector for the navigation element.")
        st.stop()

    check_page = page_url.strip() or homepage_from_sitemap(sitemap_url)

    # Fetch sitemap URLs
    status_text = st.empty()
    status_text.text("Fetching sitemap...")

    def sitemap_progress(index, total, child_url):
        status_text.text(f"Fetching child sitemap {index}/{total}: {child_url}")

    try:
        sitemap_urls = extract_urls_from_sitemap(sitemap_url, progress_callback=sitemap_progress)
    except Exception as e:
        st.error(f"Failed to fetch sitemap: {e}")
        st.stop()

    if not sitemap_urls:
        st.error("No URLs found in the sitemap.")
        st.stop()

    # Fetch navigation links
    status_text.text(f"Fetching page and extracting navigation links: {check_page}")

    try:
        navigation_urls = extract_navigation_links(check_page, nav_selector.strip())
    except Exception as e:
        st.error(f"Failed to fetch page: {e}")
        st.stop()

    status_text.empty()

    if not navigation_urls:
        st.warning(
            f"No links found for selector '{nav_selector}'. Check the selector in your "
            "browser's inspector. Note: navigation rendered with JavaScript will not be "
            "visible to this tool."
        )
        st.stop()

    # Compare
    missing = find_missing_urls(sitemap_urls, navigation_urls, url_filter.strip() or None)

    st.header("Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sitemap URLs", f"{len(sitemap_urls):,}")
    with col2:
        st.metric("Navigation Links", f"{len(set(navigation_urls)):,}")
    with col3:
        st.metric("Missing From Navigation", f"{len(missing):,}")

    if missing:
        df_missing = pd.DataFrame(missing, columns=["missing_url"])
        st.dataframe(df_missing, use_container_width=True, hide_index=True)

        output = BytesIO()
        df_missing.to_csv(output, index=False, encoding="utf-8-sig")
        output.seek(0)

        st.download_button(
            label="📥 Download Missing URLs (CSV)",
            data=output,
            file_name="missing_from_navigation.csv",
            mime="text/csv"
        )
    else:
        st.success("Every sitemap URL was found in the navigation!")

else:
    st.info("👆 Enter a sitemap URL and navigation selector, then run the check.")

    st.markdown("""
    ### How it works:

    1. All URLs are extracted from your XML sitemap (child sitemaps are fetched with a polite delay)
    2. The homepage (or your chosen page) is fetched and every link inside your navigation selector is extracted
    3. Relative links are resolved to absolute URLs
    4. Sitemap URLs not present in the navigation are reported

    ### Tips:

    - **Finding your selector**: right-click your navigation in the browser, choose Inspect, and note the element's tag, id or class (for example `nav`, `#main-nav`, `.header-menu`)
    - **Filtering**: use the URL contains filter to limit the report to category URLs, for example `/category/`
    - **JavaScript navigation**: this tool reads the raw HTML, so navigation injected by JavaScript will not be detected
    - **Trailing slashes**: comparison is exact, so make sure your sitemap and navigation use consistent URL formats
    """)
