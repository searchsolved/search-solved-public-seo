####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://leefoot.com                                                   #
# Contact  : https://leefoot.com/contact                                           #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                       #
####################################################################################

import streamlit as st
import re
import json
import time
import random

st.set_page_config(
    page_title="eBay Related Search Scraper | LeeFootSEO",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from streamlit_echarts import st_echarts
from stqdm import stqdm
import pandas as pd
from bs4 import BeautifulSoup
import requests
from user_agent2 import generate_user_agent


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_headers():
    """Generate fresh headers with a new user agent for each request."""
    ua = generate_user_agent(navigator="chrome")
    return {
        'User-Agent': str(ua),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
    }


def is_blocked(response_text):
    """Check if eBay has blocked the request with a CAPTCHA page."""
    blocked_indicators = [
        'Pardon our interruption',
        'please verify yourself',
        'unusual traffic',
        'captcha',
        'security measure',
    ]
    response_lower = response_text.lower()
    return any(indicator.lower() in response_lower for indicator in blocked_indicators)


def make_request(url, session):
    """Make a request with retry logic and delay."""
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(random.uniform(2, 4))
            
            response = session.get(url, headers=get_headers(), timeout=15)
            
            if response.status_code == 200 and not is_blocked(response.text):
                return response
            elif is_blocked(response.text):
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(3, 5))
                    continue
                return None
                
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                continue
            return None
    
    return None


RELATED_SELECTORS = [
    '.srp-related-searches a',
    '.s-answer-region-above-river a',
    '[data-testid="related-searches"] a',
    '.srp-river-answer--RELATED_SEARCHES a',
    '.b-visualnav__links a',
    'section.b-module a[href*="_nkw="]',
]

URL_PATH = "/sch/i.html?_nkw="


def extract_keyword_from_url(href):
    """Extract and decode keyword from eBay URL parameter."""
    match = re.search(r'_nkw=([^&]+)', href)
    if match:
        kw = match.group(1)
        kw = kw.replace('+', ' ')
        kw = requests.utils.unquote(kw)
        return kw.strip()
    return None


def extract_related_searches(soup):
    """Try multiple methods to extract related searches from eBay."""
    related_kws = []
    
    for selector in RELATED_SELECTORS:
        elements = soup.select(selector)
        if elements:
            for el in elements:
                href = el.get('href', '')
                if '_nkw=' in href or 'sch/i.html' in href:
                    kw = extract_keyword_from_url(href)
                    if kw and len(kw) > 1 and kw not in related_kws:
                        related_kws.append(kw)
            if related_kws:
                return related_kws
    
    all_links = soup.find_all('a', href=True)
    for link in all_links:
        href = link.get('href', '')
        if '_nkw=' in href and 'sch/' in href:
            kw = extract_keyword_from_url(href)
            if kw and len(kw) > 1 and kw not in related_kws:
                link_text = link.get_text(strip=True).lower()
                if link_text not in ['shop by category', 'home', 'ebay']:
                    related_kws.append(kw)
    
    scripts = soup.find_all('script', type='application/json')
    for script in scripts:
        try:
            data = json.loads(script.string)
            if isinstance(data, dict):
                for key in ['relatedSearches', 'related_searches', 'suggestions']:
                    if key in data:
                        items = data[key]
                        if isinstance(items, list):
                            for item in items:
                                if isinstance(item, str):
                                    related_kws.append(item)
                                elif isinstance(item, dict) and 'keyword' in item:
                                    related_kws.append(item['keyword'])
        except (json.JSONDecodeError, TypeError):
            continue
    
    return list(set(related_kws))


def get_ebay_url(cctld, keyword):
    """Build eBay search URL"""
    base = f"https://www.ebay{cctld}{URL_PATH}"
    return base + requests.utils.quote(keyword)


# ============================================================================
# UI
# ============================================================================

st.title("🔍 eBay Related Search Scraper")
st.markdown("Discover keyword opportunities by mapping eBay's related search suggestions into an interactive visualization.")

st.markdown("---")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    cctld_options = {
        '🇬🇧 United Kingdom': '.co.uk',
        '🇺🇸 United States': '.com',
        '🇩🇪 Germany': '.de',
        '🇪🇸 Spain': '.es',
        '🇫🇷 France': '.fr',
        '🇳🇱 Netherlands': '.nl',
        '🇦🇺 Australia': '.com.au',
        '🇨🇦 Canada': '.ca',
        '🇮🇹 Italy': '.it',
    }
    
    selected_country = st.selectbox(
        'eBay Marketplace',
        options=list(cctld_options.keys()),
        help='Select which eBay marketplace to search'
    )
    
    ccTLD = cctld_options[selected_country]
    
    st.markdown("---")
    
    st.subheader("ℹ️ How It Works")
    st.markdown("""
    1. **Search** - Enter a seed keyword
    2. **Expand** - We find related searches on eBay
    3. **Go Deeper** - Each related search is expanded
    4. **Visualize** - Results shown as an interactive tree
    """)
    
    st.markdown("---")
    
    st.subheader("⚠️ Note")
    st.caption("eBay may occasionally block requests. If this happens, wait a few minutes and try again with a different keyword.")
    
    st.markdown("---")
    
    st.markdown("""
    **Built by [Lee Foot](https://leefoot.com)**
    
    [🦋 Bluesky](https://bsky.app/profile/leefootseo.bsky.social) · [💼 LinkedIn](https://www.linkedin.com/in/lee-foot/) · [📧 Contact](https://leefoot.com/contact)
    """)


# Main content
col1, col2 = st.columns([1, 2])

with col1:
    with st.form(key='search_form'):
        seed_keyword = st.text_input(
            '🔎 Seed Keyword',
            placeholder='e.g., running shoes',
            help='Enter the keyword you want to explore'
        )
        
        submitted = st.form_submit_button('🚀 Start Scraping', use_container_width=True)

with col2:
    if not submitted:
        st.info("👈 Enter a keyword and click **Start Scraping** to begin")


# ============================================================================
# MAIN SCRAPING LOGIC
# ============================================================================

if submitted:
    if not seed_keyword.strip():
        st.error("⚠️ Please enter a keyword to search")
        st.stop()
    
    # Progress section
    progress_container = st.container()
    
    with progress_container:
        status_text = st.empty()
        progress_bar = st.progress(0)
    
    related_search_kws = []
    source_kws = []
    final_kws = []
    
    session = requests.Session()
    
    # First request
    status_text.text("🔍 Searching eBay for related keywords...")
    progress_bar.progress(10)
    
    search_url = get_ebay_url(ccTLD, seed_keyword)
    response = make_request(search_url, session)
    
    if response is None:
        progress_container.empty()
        st.error("⚠️ eBay blocked the request (CAPTCHA/bot detection)")
        st.info("""
        **Tips to try:**
        - Wait a few minutes and try again
        - Try a different keyword  
        - Try a different marketplace (in sidebar)
        """)
        st.stop()
    
    soup = BeautifulSoup(response.text, "html.parser")
    related_search_kws = extract_related_searches(soup)
    
    if not related_search_kws:
        progress_container.empty()
        st.warning("😕 No related searches found for this keyword")
        st.info("Try a broader or different keyword.")
        
        with st.expander("🔧 Debug Information"):
            st.code(f"URL: {search_url}")
            st.code(f"Status: {response.status_code}")
            st.text_area("Response HTML (first 3000 chars)", response.text[:3000], height=200)
        st.stop()
    
    progress_bar.progress(25)
    status_text.text(f"✅ Found {len(related_search_kws)} related searches! Expanding...")
    
    # Second loop
    blocked_count = 0
    total_kws = len(related_search_kws)
    
    for idx, kw in enumerate(related_search_kws):
        time.sleep(random.uniform(1.5, 3.0))
        
        progress_pct = 25 + int((idx / total_kws) * 65)
        progress_bar.progress(progress_pct)
        status_text.text(f"🔄 Processing: {kw} ({idx + 1}/{total_kws})")
        
        search_url = get_ebay_url(ccTLD, kw)
        response = make_request(search_url, session)
        
        if response is None:
            blocked_count += 1
            if blocked_count >= 3:
                st.warning("⚠️ Multiple requests blocked. Stopping early.")
                break
            continue
        
        soup_lv2 = BeautifulSoup(response.text, "html.parser")
        lv2_related = extract_related_searches(soup_lv2)
        
        for lv2_kw in lv2_related:
            source_kws.append(kw)
            final_kws.append(lv2_kw)
    
    session.close()
    progress_bar.progress(100)
    status_text.text("✅ Complete!")
    time.sleep(0.5)
    progress_container.empty()
    
    # Build dataframe
    if not source_kws:
        df = pd.DataFrame({
            'seed_keyword': [seed_keyword] * len(related_search_kws),
            'related_searches': related_search_kws
        })
    else:
        df = pd.DataFrame({
            'seed_keyword': source_kws,
            'related_searches': final_kws
        })
    
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Results section
    st.markdown("---")
    st.subheader(f"📊 Results for: {seed_keyword}")
    
    # Stats
    unique_l1 = df['seed_keyword'].nunique()
    unique_l2 = df['related_searches'].nunique()
    total_relationships = len(df)
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        st.metric("Level 1 Keywords", unique_l1)
    
    with col_stat2:
        st.metric("Level 2 Keywords", unique_l2)
    
    with col_stat3:
        st.metric("Total Relationships", total_relationships)
    
    # Download and data
    st.markdown("")
    
    col_dl, col_spacer = st.columns([1, 2])
    
    with col_dl:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f'ebay_related_{seed_keyword.replace(" ", "_")}.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    with st.expander("📋 View Data Table"):
        st.dataframe(df, use_container_width=True, height=300)
    
    # Visualization
    st.markdown("---")
    st.subheader("🌳 Keyword Relationship Tree")
    
    # Build tree data
    children_list = []
    
    for int_word in df['seed_keyword'].unique():
        children_list_level_2 = []
        
        for query_2 in df[df['seed_keyword'] == int_word]['related_searches'].unique():
            children_list_level_2.append({"name": query_2})
        
        children_list.append({'name': int_word, 'children': children_list_level_2})
    
    tree = {'name': seed_keyword, 'children': children_list}
    
    # View selector
    view_type = st.radio(
        "Select view",
        ["🔵 Radial Tree", "📊 Vertical Tree", "📁 Text Tree"],
        horizontal=True,
        help="Choose how to visualize the keyword relationships"
    )
    
    st.caption("Click nodes to expand/collapse. Right-click to save as image.")
    
    if view_type == "🔵 Radial Tree":
        opts = {
            "tooltip": {
                "trigger": "item",
                "triggerOn": "mousemove",
            },
            "series": [
                {
                    "type": "tree",
                    "data": [tree],
                    "layout": "radial",
                    "top": "5%",
                    "left": "15%",
                    "bottom": "5%",
                    "right": "15%",
                    "symbolSize": 12,
                    "symbol": "circle",
                    "itemStyle": {
                        "color": "#10B981",
                        "borderColor": "#059669",
                        "borderWidth": 2,
                    },
                    "lineStyle": {
                        "color": "#94A3B8",
                        "width": 1.5,
                        "curveness": 0.5,
                    },
                    "label": {
                        "fontSize": 12,
                    },
                    "emphasis": {
                        "itemStyle": {
                            "color": "#F59E0B",
                            "borderColor": "#D97706",
                        },
                        "lineStyle": {
                            "color": "#F59E0B",
                            "width": 2,
                        },
                    },
                    "expandAndCollapse": True,
                    "initialTreeDepth": 2,
                    "animationDuration": 550,
                    "animationDurationUpdate": 750,
                }
            ],
        }
        
        st_echarts(opts, key=f"radial_tree_{seed_keyword}", height=700)
    
    elif view_type == "📊 Vertical Tree":
        opts = {
            "tooltip": {
                "trigger": "item",
                "triggerOn": "mousemove",
            },
            "series": [
                {
                    "type": "tree",
                    "data": [tree],
                    "layout": "orthogonal",
                    "orient": "TB",
                    "top": "5%",
                    "left": "10%",
                    "bottom": "5%",
                    "right": "10%",
                    "symbolSize": 10,
                    "symbol": "circle",
                    "itemStyle": {
                        "color": "#3B82F6",
                        "borderColor": "#2563EB",
                        "borderWidth": 2,
                    },
                    "lineStyle": {
                        "color": "#94A3B8",
                        "width": 1.5,
                    },
                    "label": {
                        "position": "top",
                        "fontSize": 11,
                        "rotate": -45,
                        "align": "right",
                        "verticalAlign": "middle",
                    },
                    "leaves": {
                        "label": {
                            "position": "bottom",
                            "rotate": -45,
                            "align": "left",
                            "verticalAlign": "middle",
                        }
                    },
                    "emphasis": {
                        "itemStyle": {
                            "color": "#F59E0B",
                            "borderColor": "#D97706",
                        },
                        "lineStyle": {
                            "color": "#F59E0B",
                            "width": 2,
                        },
                    },
                    "expandAndCollapse": True,
                    "initialTreeDepth": 2,
                    "animationDuration": 550,
                    "animationDurationUpdate": 750,
                }
            ],
        }
        
        # Calculate height based on number of nodes
        tree_height = max(700, len(df) * 8)
        st_echarts(opts, key=f"vertical_tree_{seed_keyword}", height=tree_height)
    
    else:  # Text Tree
        st.markdown("")
        
        # Build text tree with expanders
        st.markdown(f"**🌱 {seed_keyword}**")
        
        for l1_keyword in df['seed_keyword'].unique():
            l2_keywords = df[df['seed_keyword'] == l1_keyword]['related_searches'].unique().tolist()
            
            with st.expander(f"📂 {l1_keyword} ({len(l2_keywords)} keywords)"):
                for l2_kw in l2_keywords:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;📄 {l2_kw}")

# Footer
st.markdown("---")
st.markdown(
    "Made by [Lee Foot](https://leefoot.com) · "
    "[🦋 Bluesky](https://bsky.app/profile/leefootseo.bsky.social) · "
    "[💼 LinkedIn](https://www.linkedin.com/in/lee-foot/) · "
    "[📧 Contact](https://leefoot.com/contact)"
)
