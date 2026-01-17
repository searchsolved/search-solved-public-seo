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

# Page config must be first Streamlit command
st.set_page_config(
    page_title="eBay Related Search Scraper | LeeFootSEO",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Imports
from streamlit_echarts import st_echarts
from stqdm import stqdm
import pandas as pd
from bs4 import BeautifulSoup
import requests
from user_agent2 import generate_user_agent

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Root variables */
    :root {
        --primary: #0066FF;
        --primary-light: #E8F0FE;
        --secondary: #FF6B35;
        --dark: #1A1A2E;
        --gray-50: #F8FAFC;
        --gray-100: #F1F5F9;
        --gray-200: #E2E8F0;
        --gray-300: #CBD5E1;
        --gray-500: #64748B;
        --gray-700: #334155;
        --gray-900: #0F172A;
        --success: #10B981;
        --warning: #F59E0B;
        --error: #EF4444;
        --ebay-blue: #0064D2;
        --ebay-red: #E53238;
        --ebay-yellow: #F5AF02;
        --ebay-green: #86B817;
    }
    
    /* Global styles */
    .stApp {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Hero header */
    .hero-container {
        background: linear-gradient(135deg, var(--gray-900) 0%, #2D2D44 100%);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(0, 102, 255, 0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    
    .hero-container::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255, 107, 53, 0.1) 0%, transparent 70%);
        pointer-events: none;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: var(--gray-300);
        margin: 0 0 1.5rem 0;
        position: relative;
        z-index: 1;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 0.5rem 1rem;
        border-radius: 100px;
        font-size: 0.85rem;
        color: var(--gray-200);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        z-index: 1;
    }
    
    .hero-badge a {
        color: var(--primary);
        text-decoration: none;
        transition: color 0.2s;
    }
    
    .hero-badge a:hover {
        color: white;
    }
    
    /* eBay colors decoration */
    .ebay-colors {
        display: flex;
        gap: 4px;
        margin-bottom: 1rem;
    }
    
    .ebay-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }
    
    .ebay-dot.blue { background: var(--ebay-blue); }
    .ebay-dot.red { background: var(--ebay-red); }
    .ebay-dot.yellow { background: var(--ebay-yellow); }
    .ebay-dot.green { background: var(--ebay-green); }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.1);
        border: 1px solid var(--gray-100);
        margin-bottom: 1.5rem;
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--gray-100);
    }
    
    .card-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
    }
    
    .card-icon.blue { background: var(--primary-light); }
    .card-icon.orange { background: #FFF0EB; }
    .card-icon.green { background: #ECFDF5; }
    
    .card-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--gray-900);
        margin: 0;
    }
    
    .card-description {
        font-size: 0.85rem;
        color: var(--gray-500);
        margin: 0;
    }
    
    /* Form styling */
    .stTextInput > div > div > input {
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        border: 2px solid var(--gray-200);
        transition: all 0.2s;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(0, 102, 255, 0.1);
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    /* Button styling */
    .stFormSubmitButton > button {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, var(--primary) 0%, #0052CC 100%);
        border: none;
        color: white;
        width: 100%;
        transition: all 0.2s;
        box-shadow: 0 4px 14px rgba(0, 102, 255, 0.25);
    }
    
    .stFormSubmitButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0, 102, 255, 0.35);
    }
    
    .stDownloadButton > button {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        background: var(--success);
        border: none;
        color: white;
        transition: all 0.2s;
    }
    
    .stDownloadButton > button:hover {
        background: #059669;
        transform: translateY(-1px);
    }
    
    /* Stats cards */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid var(--gray-100);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--gray-900);
        line-height: 1.2;
    }
    
    .stat-value.blue { color: var(--primary); }
    .stat-value.orange { color: var(--secondary); }
    .stat-value.green { color: var(--success); }
    
    .stat-label {
        font-size: 0.85rem;
        color: var(--gray-500);
        margin-top: 0.25rem;
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.1);
        border: 1px solid var(--gray-100);
        margin-top: 1.5rem;
    }
    
    .chart-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--gray-100);
    }
    
    .chart-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--gray-900);
        margin: 0;
    }
    
    .chart-hint {
        font-size: 0.8rem;
        color: var(--gray-500);
        background: var(--gray-50);
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 12px;
        border: none;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        color: var(--gray-700);
        background: var(--gray-50);
        border-radius: 10px;
    }
    
    /* Progress text */
    .progress-text {
        font-size: 0.9rem;
        color: var(--gray-500);
        margin: 0.5rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: var(--gray-400);
        font-size: 0.85rem;
    }
    
    .footer a {
        color: var(--primary);
        text-decoration: none;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* How it works section */
    .how-it-works {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .step {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 1rem;
        background: var(--gray-50);
        border-radius: 10px;
    }
    
    .step-number {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: var(--primary);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.85rem;
        flex-shrink: 0;
    }
    
    .step-content {
        flex: 1;
    }
    
    .step-title {
        font-weight: 600;
        color: var(--gray-900);
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
    }
    
    .step-desc {
        font-size: 0.8rem;
        color: var(--gray-500);
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)


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


# CSS selectors for eBay related searches
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
# UI COMPONENTS
# ============================================================================

# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="ebay-colors">
        <div class="ebay-dot blue"></div>
        <div class="ebay-dot red"></div>
        <div class="ebay-dot yellow"></div>
        <div class="ebay-dot green"></div>
    </div>
    <h1 class="hero-title">eBay Related Search Scraper</h1>
    <p class="hero-subtitle">Discover keyword opportunities by mapping eBay's related search suggestions into an interactive visualization.</p>
    <div class="hero-badge">
        Built by <a href="https://leefoot.com" target="_blank">Lee Foot</a> · 
        <a href="https://bsky.app/profile/leefootseo.bsky.social" target="_blank">@leefootseo</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Main content columns
col1, col2 = st.columns([1, 2])

with col1:
    # Search Card
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-icon blue">🔍</div>
            <div>
                <p class="card-title">Search Settings</p>
                <p class="card-description">Enter your seed keyword</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form(key='search_form'):
        seed_keyword = st.text_input(
            'Seed Keyword',
            placeholder='e.g., running shoes, vintage watches...',
            help='Enter the keyword you want to explore'
        )
        
        cctld_options = {
            '🇬🇧 United Kingdom (.co.uk)': '.co.uk',
            '🇺🇸 United States (.com)': '.com',
            '🇩🇪 Germany (.de)': '.de',
            '🇪🇸 Spain (.es)': '.es',
            '🇫🇷 France (.fr)': '.fr',
            '🇳🇱 Netherlands (.nl)': '.nl',
            '🇦🇺 Australia (.com.au)': '.com.au',
            '🇨🇦 Canada (.ca)': '.ca',
            '🇮🇹 Italy (.it)': '.it',
        }
        
        selected_country = st.selectbox(
            'eBay Marketplace',
            options=list(cctld_options.keys()),
            help='Select which eBay marketplace to search'
        )
        
        ccTLD = cctld_options[selected_country]
        
        submitted = st.form_submit_button('🚀 Start Scraping')
    
    # How it works
    st.markdown("""
    <div class="card" style="margin-top: 1rem;">
        <div class="card-header">
            <div class="card-icon orange">💡</div>
            <div>
                <p class="card-title">How It Works</p>
                <p class="card-description">Three simple steps</p>
            </div>
        </div>
        <div style="display: flex; flex-direction: column; gap: 0.75rem;">
            <div class="step">
                <div class="step-number">1</div>
                <div class="step-content">
                    <div class="step-title">Search eBay</div>
                    <div class="step-desc">We search your keyword and find related searches</div>
                </div>
            </div>
            <div class="step">
                <div class="step-number">2</div>
                <div class="step-content">
                    <div class="step-title">Go Deeper</div>
                    <div class="step-desc">Each related search is expanded to find more keywords</div>
                </div>
            </div>
            <div class="step">
                <div class="step-number">3</div>
                <div class="step-content">
                    <div class="step-title">Visualize</div>
                    <div class="step-desc">Results displayed as an interactive tree diagram</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    results_placeholder = st.empty()
    
    if not submitted:
        # Show placeholder when no search has been made
        results_placeholder.markdown("""
        <div class="card" style="min-height: 400px; display: flex; align-items: center; justify-content: center;">
            <div style="text-align: center; color: var(--gray-400);">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🌳</div>
                <p style="font-size: 1.1rem; margin: 0;">Enter a keyword and click <strong>Start Scraping</strong></p>
                <p style="font-size: 0.9rem; margin-top: 0.5rem;">Your keyword tree will appear here</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# MAIN SCRAPING LOGIC
# ============================================================================

if submitted:
    results_placeholder.empty()
    
    if not seed_keyword.strip():
        st.error("⚠️ Please enter a keyword to search")
        st.stop()
    
    with col2:
        # Progress card
        progress_container = st.container()
        
        with progress_container:
            st.markdown("""
            <div class="card">
                <div class="card-header">
                    <div class="card-icon green">⚡</div>
                    <div>
                        <p class="card-title">Scraping in Progress</p>
                        <p class="card-description">Please wait while we gather data</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            status_text = st.empty()
            progress_bar = st.progress(0)
        
        # Store the data
        related_search_kws = []
        source_kws = []
        final_kws = []
        
        # Use a session for connection pooling
        session = requests.Session()
        
        # First request
        status_text.markdown('<p class="progress-text">🔍 Searching eBay for initial related keywords...</p>', unsafe_allow_html=True)
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
            - Try a different marketplace
            
            This happens when eBay detects automated traffic.
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
                st.code(response.text[:2000], language='html')
            st.stop()
        
        progress_bar.progress(25)
        status_text.markdown(f'<p class="progress-text">✅ Found {len(related_search_kws)} related searches! Expanding each one...</p>', unsafe_allow_html=True)
        
        # Second loop
        blocked_count = 0
        total_kws = len(related_search_kws)
        
        for idx, kw in enumerate(related_search_kws):
            time.sleep(random.uniform(1.5, 3.0))
            
            progress_pct = 25 + int((idx / total_kws) * 65)
            progress_bar.progress(progress_pct)
            status_text.markdown(f'<p class="progress-text">🔄 Processing: <strong>{kw}</strong> ({idx + 1}/{total_kws})</p>', unsafe_allow_html=True)
            
            search_url = get_ebay_url(ccTLD, kw)
            response = make_request(search_url, session)
            
            if response is None:
                blocked_count += 1
                if blocked_count >= 3:
                    st.warning("⚠️ Multiple requests blocked. Stopping to avoid IP ban.")
                    break
                continue
            
            soup_lv2 = BeautifulSoup(response.text, "html.parser")
            lv2_related = extract_related_searches(soup_lv2)
            
            for lv2_kw in lv2_related:
                source_kws.append(kw)
                final_kws.append(lv2_kw)
        
        session.close()
        progress_bar.progress(100)
        status_text.markdown('<p class="progress-text">✅ Complete!</p>', unsafe_allow_html=True)
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
        
        # Stats
        unique_l1 = df['seed_keyword'].nunique()
        unique_l2 = df['related_searches'].nunique()
        total_relationships = len(df)
        
        st.markdown(f"""
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-value blue">{unique_l1}</div>
                <div class="stat-label">Level 1 Keywords</div>
            </div>
            <div class="stat-card">
                <div class="stat-value orange">{unique_l2}</div>
                <div class="stat-label">Level 2 Keywords</div>
            </div>
            <div class="stat-card">
                <div class="stat-value green">{total_relationships}</div>
                <div class="stat-label">Total Relationships</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Download and data table
        col_dl, col_tbl = st.columns([1, 1])
        
        with col_dl:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f'ebay_related_searches_{seed_keyword.replace(" ", "_")}.csv',
                mime='text/csv',
            )
        
        with col_tbl:
            with st.expander("📊 View Data Table"):
                st.dataframe(df, use_container_width=True, height=300)
        
        # Visualization
        st.markdown("""
        <div class="chart-container">
            <div class="chart-header">
                <h3 class="chart-title">🌳 Keyword Relationship Tree</h3>
                <span class="chart-hint">💡 Right-click to save as image</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Build tree data
        children_list = []
        
        for int_word in df['seed_keyword'].unique():
            children_list_level_2 = []
            
            for query_2 in df[df['seed_keyword'] == int_word]['related_searches'].unique():
                children_list_level_2.append({"name": query_2})
            
            children_list.append({'name': int_word, 'children': children_list_level_2})
        
        tree = {'name': seed_keyword, 'children': children_list}
        
        opts = {
            "backgroundColor": "#FFFFFF",
            "tooltip": {
                "trigger": "item",
                "triggerOn": "mousemove",
                "backgroundColor": "rgba(15, 23, 42, 0.95)",
                "borderColor": "transparent",
                "textStyle": {
                    "color": "#F8FAFC",
                    "fontSize": 13,
                    "fontFamily": "DM Sans, sans-serif"
                },
                "padding": [8, 12],
                "borderRadius": 8,
            },
            "series": [
                {
                    "type": "tree",
                    "data": [tree],
                    "layout": "radial",
                    "top": "8%",
                    "left": "20%",
                    "bottom": "8%",
                    "right": "20%",
                    "symbolSize": 14,
                    "symbol": "circle",
                    "itemStyle": {
                        "color": "#0066FF",
                        "borderColor": "#0052CC",
                        "borderWidth": 2,
                        "shadowColor": "rgba(0, 102, 255, 0.3)",
                        "shadowBlur": 8,
                    },
                    "lineStyle": {
                        "color": "#E2E8F0",
                        "width": 1.5,
                        "curveness": 0.5,
                    },
                    "label": {
                        "fontSize": 12,
                        "fontFamily": "DM Sans, sans-serif",
                        "color": "#334155",
                        "fontWeight": 500,
                    },
                    "emphasis": {
                        "itemStyle": {
                            "color": "#FF6B35",
                            "borderColor": "#E55A2B",
                            "shadowColor": "rgba(255, 107, 53, 0.4)",
                            "shadowBlur": 12,
                        },
                        "lineStyle": {
                            "color": "#FF6B35",
                            "width": 2,
                        },
                        "label": {
                            "color": "#1A1A2E",
                            "fontWeight": 700,
                        }
                    },
                    "expandAndCollapse": True,
                    "initialTreeDepth": 2,
                    "animationDuration": 750,
                    "animationDurationUpdate": 500,
                    "animationEasing": "cubicOut",
                }
            ],
        }
        
        st_echarts(opts, key=f"tree_{seed_keyword}", height=800)

# Footer
st.markdown("""
<div class="footer">
    Made with ❤️ by <a href="https://leefoot.com" target="_blank">Lee Foot</a> · 
    <a href="https://bsky.app/profile/leefootseo.bsky.social" target="_blank">Bluesky</a> · 
    <a href="https://www.linkedin.com/in/lee-foot/" target="_blank">LinkedIn</a> · 
    <a href="https://leefoot.com/contact" target="_blank">Contact</a>
</div>
""", unsafe_allow_html=True)
