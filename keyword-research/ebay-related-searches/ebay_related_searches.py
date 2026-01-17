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

st.set_page_config(
    page_title="eBay Related Search Scraper by LeeFootSEO",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)
from streamlit_echarts import st_echarts
from stqdm import stqdm
import pandas as pd
from bs4 import BeautifulSoup
import requests
from user_agent2 import generate_user_agent

# set fake agent
ua = generate_user_agent(navigator="chrome")
header = {
    'User-Agent': str(ua),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-GB,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
}

# Multiple CSS selectors to try (eBay changes these periodically)
RELATED_SELECTORS = [
    '.srp-related-searches a',  # Current common selector
    '.s-answer-region-above-river a',  # Legacy selector
    '[data-testid="related-searches"] a',  # Data attribute selector
    '.srp-river-answer--RELATED_SEARCHES a',  # River answer format
    '.b-visualnav__links a',  # Visual nav links
    'section.b-module a[href*="_nkw="]',  # Generic module with search links
]

url_path = "/sch/i.html?_nkw="

st.title("eBay Related Search Scraper")
st.subheader("Get Related Searches from eBay")
st.write("An app which visualises related searches from eBay")
st.write(
    "Made by [@leefootseo](https://bsky.app/profile/leefootseo.bsky.social) | [Website](https://leefoot.com) | [Contact](https://leefoot.com/contact)")
st.write("")

with st.form(key='columns_in_form_2'):
    seed_keyword = st.text_input('Enter the Keyword to Search eBay')
    ccTLD = st.selectbox(
        'Select Which ccTLD to Search',
        ('.co.uk', '.com', '.de', '.es', '.fr', '.nl', '.com.au', '.ca', '.it'))
    submitted = st.form_submit_button('Submit')


def extract_related_searches(soup):
    """
    Try multiple methods to extract related searches from eBay.
    Returns a list of related search keywords.
    """
    related_kws = []
    
    # Method 1: Try CSS selectors
    for selector in RELATED_SELECTORS:
        elements = soup.select(selector)
        if elements:
            for el in elements:
                text = el.get_text(strip=True)
                href = el.get('href', '')
                # Only include if it looks like a search link
                if text and ('_nkw=' in href or 'sch/i.html' in href):
                    # Clean up the keyword
                    text = text.strip()
                    if text and len(text) > 1 and text not in related_kws:
                        related_kws.append(text)
            if related_kws:
                return related_kws
    
    # Method 2: Extract from URL parameters in links
    all_links = soup.find_all('a', href=True)
    for link in all_links:
        href = link.get('href', '')
        if '_nkw=' in href and 'sch/' in href:
            # Extract keyword from URL
            match = re.search(r'_nkw=([^&]+)', href)
            if match:
                kw = match.group(1).replace('+', ' ').replace('%20', ' ')
                kw = requests.utils.unquote(kw)
                if kw and len(kw) > 1 and kw not in related_kws:
                    # Check if link text suggests it's a related search
                    link_text = link.get_text(strip=True)
                    if link_text and link_text.lower() != 'shop by category':
                        related_kws.append(kw)
    
    # Method 3: Look for JSON data in script tags (eBay sometimes embeds data this way)
    scripts = soup.find_all('script', type='application/json')
    for script in scripts:
        try:
            data = json.loads(script.string)
            # Look for related searches in various possible structures
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
    
    return list(set(related_kws))  # Remove duplicates


def get_ebay_url(cctld, keyword):
    """Build eBay search URL"""
    base = f"https://www.ebay{cctld}{url_path}"
    return base + requests.utils.quote(keyword)


if submitted:
    if not seed_keyword.strip():
        st.error("Please enter a keyword to search")
        st.stop()
    
    # Store the data
    related_search_kws = []
    source_kws = []
    final_kws = []
    
    # First request - get initial related searches
    try:
        search_url = get_ebay_url(ccTLD, seed_keyword)
        response = requests.get(search_url, headers=header, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        related_search_kws = extract_related_searches(soup)
        
        if not related_search_kws:
            st.warning("No related searches found for the initial keyword. eBay may have changed their page structure or blocked the request.")
            st.info("Try a different keyword or check back later.")
            
            # Debug info in expander
            with st.expander("Debug Information"):
                st.write(f"URL requested: {search_url}")
                st.write(f"Response status: {response.status_code}")
                st.write(f"Response length: {len(response.text)} characters")
                # Show a snippet of the HTML
                st.code(response.text[:2000], language='html')
            st.stop()
        
        st.success(f"Found {len(related_search_kws)} related searches for '{seed_keyword}'")
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching eBay: {str(e)}")
        st.stop()

    # Second loop - get related searches for each related keyword
    st.write("Searching eBay for second-level related keywords...")
    
    for kw in stqdm(related_search_kws):
        try:
            search_url = get_ebay_url(ccTLD, kw)
            response = requests.get(search_url, headers=header, timeout=15)
            
            if response.status_code == 200:
                soup_lv2 = BeautifulSoup(response.text, "html.parser")
                lv2_related = extract_related_searches(soup_lv2)
                
                for lv2_kw in lv2_related:
                    source_kws.append(kw)
                    final_kws.append(lv2_kw)
                    
        except requests.exceptions.RequestException:
            continue  # Skip failed requests
    
    if not source_kws:
        st.warning("Could not retrieve second-level related searches.")
        # Still create a basic dataframe with first level results
        df = pd.DataFrame({
            'seed_keyword': [seed_keyword] * len(related_search_kws),
            'related_searches': related_search_kws
        })
    else:
        df = pd.DataFrame({
            'seed_keyword': source_kws,
            'related_searches': final_kws
        })

    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    
    st.success(f"Found {len(df)} total keyword relationships")


    def visualize_autocomplete(df_autocomplete_full):
        df_vis = df_autocomplete_full.copy()
        df_vis['Keyword'] = seed_keyword

        children_list = []
        
        for int_word in df_vis['seed_keyword'].unique():
            children_list_level_2 = []
            
            for query_2 in df_vis[df_vis['seed_keyword'] == int_word]['related_searches'].unique():
                q_lv2_line = {"name": query_2}
                children_list_level_2.append(q_lv2_line)
            
            level2_tree = {'name': int_word, 'children': children_list_level_2}
            children_list.append(level2_tree)

        tree = {'name': seed_keyword, 'children': children_list}

        opts = {
            "backgroundColor": "#F0F2F6",
            "title": {
                "x": 'center',
                "y": 'top',
                "top": "5%",
                "textStyle": {
                    "fontSize": 22,
                },
                "subtextStyle": {
                    "fontSize": 15,
                    "color": '#2ec4b6',
                },
            },
            "series": [
                {
                    "type": "tree",
                    "data": [tree],
                    "layout": "radial",
                    "top": "10%",
                    "left": "25%",
                    "bottom": "5%",
                    "right": "25%",
                    "symbolSize": 20,
                    "itemStyle": {
                        "color": '#2ec4b6',
                    },
                    "label": {
                        "fontSize": 14,
                    },
                    "expandAndCollapse": True,
                    "animationDuration": 550,
                    "animationDurationUpdate": 750,
                }
            ],
        }
        
        st.header(f"eBay Related Searches for: {seed_keyword}")
        st.caption("Right mouse click to save as image.")
        st_echarts(opts, key=seed_keyword, height=1700)


    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')


    csv = convert_df(df)

    st.download_button(
        label="📥 Download your report!",
        data=csv,
        file_name='ebay_related_searches.csv',
        mime='text/csv',
    )

    # Show data table
    with st.expander("View Data Table"):
        st.dataframe(df, use_container_width=True)

    # Visualisation
    visualize_autocomplete(df)
