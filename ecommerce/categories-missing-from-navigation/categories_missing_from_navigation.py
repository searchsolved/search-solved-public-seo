# Author: Lee Foot
# Website: https://leefoot.com

####################################################################################
#                                                                                  #
#  Categories Missing From Navigation                                             #
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
Categories Missing From Navigation - Core Logic

Fetches an XML sitemap (including sitemap index files), fetches a page
(usually the homepage), extracts every link found within a user-supplied
CSS selector for the navigation element, and reports which sitemap URLs
are missing from the navigation.

Requests are made politely: a real user agent is sent and a delay is
applied between fetches.
"""

import time
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

DEFAULT_HEADERS = {"User-Agent": USER_AGENT}
DEFAULT_DELAY = 2.0  # seconds between fetches
DEFAULT_TIMEOUT = 30  # seconds per request
DEFAULT_NAV_SELECTOR = "nav"


def fetch(url, timeout=DEFAULT_TIMEOUT):
    """Fetch a URL with a real user agent and return the response."""
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response


def extract_urls_from_sitemap(sitemap_url, delay=DEFAULT_DELAY, progress_callback=None):
    """Extract all URLs from an XML sitemap.

    Handles both standard sitemaps and sitemap index files. For an index,
    each child sitemap is fetched in turn with a polite delay between
    requests.

    Returns a list of page URLs.
    """
    response = fetch(sitemap_url)
    parsed = BeautifulSoup(response.content, "xml")

    urls = []

    if parsed.find("sitemapindex"):
        child_sitemaps = [loc.text.strip() for loc in parsed.select("sitemap > loc")]
        for index, child in enumerate(child_sitemaps, start=1):
            if progress_callback:
                progress_callback(index, len(child_sitemaps), child)
            time.sleep(delay)
            child_response = fetch(child)
            child_parsed = BeautifulSoup(child_response.content, "xml")
            urls.extend(loc.text.strip() for loc in child_parsed.select("url > loc"))
    else:
        urls.extend(loc.text.strip() for loc in parsed.find_all("loc"))

    return urls


def extract_navigation_links(page_url, css_selector, delay=DEFAULT_DELAY):
    """Extract all link URLs found within a CSS selector on a page.

    A polite delay is applied before the fetch so back-to-back calls with
    the sitemap fetch do not hammer the server. Relative URLs are resolved
    to absolute URLs against the page URL.

    Returns a list of absolute URLs.
    """
    time.sleep(delay)
    response = fetch(page_url)
    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    for element in soup.select(css_selector):
        for anchor in element.find_all("a"):
            href = anchor.get("href")
            if href:
                links.append(urljoin(page_url, href.strip()))

    return links


def find_missing_urls(sitemap_urls, navigation_urls, url_filter=None):
    """Return sitemap URLs that do not appear in the navigation.

    Optionally restrict the results to URLs containing url_filter, which is
    useful for limiting the check to category URLs (for example "/category/").
    """
    missing = set(sitemap_urls) - set(navigation_urls)

    if url_filter:
        missing = {url for url in missing if url_filter in url}

    return sorted(missing)


def homepage_from_sitemap(sitemap_url):
    """Derive the homepage URL from a sitemap URL."""
    parts = urlsplit(sitemap_url)
    return f"{parts.scheme}://{parts.netloc}/"
