# Author: Lee Foot
# Website: https://leefoot.com

"""
SERP Crossover Analyzer

Fetches SERP results for multiple keywords and analyses URL crossover.
Identifies keyword cannibalisation and topic overlap opportunities.

Features:
- Fetch SERPs for multiple keywords via DataForSEO SERP API
- Calculate URL crossover percentage between keywords
- Identify which URLs rank for multiple keywords
- Visual crossover matrix
- Export detailed results
"""

import os
import math
import streamlit as st
import pandas as pd
import requests
from base64 import b64encode
from io import BytesIO
from urllib.parse import urlparse

st.set_page_config(page_title="SERP Crossover Analyzer", page_icon="🔀", layout="wide")

st.title("SERP Crossover Analyzer")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Fetches SERP results for multiple keywords
    - Calculates URL crossover percentage between keywords
    - Identifies cannibalisation and topic clustering opportunities

    **Requirements:**
    - DataForSEO account (sign up at [dataforseo.com](https://dataforseo.com/))

    **How to use:**
    1. Enter your DataForSEO login and password in the sidebar
    2. Enter keywords to compare (one per line)
    3. Select location and device
    4. Click "Analyse SERPs"
    5. Review crossover matrix and overlapping URLs

    **Use cases:**
    - Identify keyword cannibalisation (same URL ranking for multiple keywords)
    - Find topic clustering opportunities
    - Analyse SERP similarity for keyword grouping
    """)

LOCATION_CODES = {
    "United Kingdom": 2826,
    "United States": 2840,
    "Australia": 2036,
    "Canada": 2124,
    "Germany": 2276,
    "France": 2250,
    "Spain": 2724,
    "Italy": 2380,
    "Netherlands": 2528,
    "India": 2356,
}

# Sidebar settings
st.sidebar.header("API Settings")

dataforseo_login = st.sidebar.text_input(
    "DataForSEO Login",
    type="password",
    value=os.environ.get('DATAFORSEO_LOGIN', ''),
    help="Your login from dataforseo.com"
)

dataforseo_password = st.sidebar.text_input(
    "DataForSEO Password",
    type="password",
    value=os.environ.get('DATAFORSEO_PASSWORD', ''),
    help="Your password from dataforseo.com"
)

st.sidebar.markdown("---")
st.sidebar.header("Search Settings")

location = st.sidebar.selectbox(
    "Location",
    list(LOCATION_CODES.keys()),
    index=0
)

device = st.sidebar.selectbox(
    "Device",
    ["Desktop", "Mobile", "Tablet"],
    index=0
)

num_results = st.sidebar.slider(
    "Results per SERP",
    min_value=5,
    max_value=100,
    value=10,
    help="Number of organic results to fetch per keyword"
)


def extract_domain(url):
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc.replace('www.', '')
    except Exception:
        return url


def fetch_serp(keyword, login, password, location_name, device_type, num_results):
    """Fetch SERP results for a keyword using the DataForSEO API."""
    cred = b64encode(f"{login}:{password}".encode()).decode()
    headers = {
        'Authorization': f'Basic {cred}',
        'Content-Type': 'application/json'
    }

    payload = [{
        "keyword": keyword,
        "location_code": LOCATION_CODES[location_name],
        "language_code": "en",
        "device": device_type.lower(),
        "depth": num_results
    }]

    try:
        response = requests.post(
            'https://api.dataforseo.com/v3/serp/google/organic/live/advanced',
            headers=headers,
            json=payload,
            timeout=60
        )
        data = response.json()

        items = data["tasks"][0]["result"][0]["items"]
        organic = [item for item in items if item["type"] == "organic"]

        results = []
        for i, result in enumerate(organic[:num_results]):
            results.append({
                'position': result.get('rank_group', i + 1),
                'title': result.get('title', ''),
                'link': result.get('url', ''),
                'domain': extract_domain(result.get('url', ''))
            })

        return results

    except Exception as e:
        st.warning(f"Error fetching SERP for '{keyword}': {str(e)}")
        return []


def calculate_crossover(serp_data):
    """Calculate crossover matrix between keywords."""
    keywords = list(serp_data.keys())

    # Create URL sets for each keyword
    url_sets = {kw: set(r['link'] for r in results) for kw, results in serp_data.items()}

    # Calculate crossover matrix
    matrix = pd.DataFrame(index=keywords, columns=keywords, dtype=float)

    for i, kw1 in enumerate(keywords):
        for j, kw2 in enumerate(keywords):
            if i == j:
                matrix.loc[kw1, kw2] = 100.0
            else:
                common = url_sets[kw1].intersection(url_sets[kw2])
                union = url_sets[kw1].union(url_sets[kw2])
                if union:
                    crossover = (len(common) / len(union)) * 100
                else:
                    crossover = 0.0
                matrix.loc[kw1, kw2] = round(crossover, 1)

    return matrix


def find_overlapping_urls(serp_data):
    """Find URLs that appear in multiple SERPs."""
    url_keywords = {}

    for keyword, results in serp_data.items():
        for result in results:
            url = result['link']
            if url not in url_keywords:
                url_keywords[url] = {
                    'title': result['title'],
                    'domain': result['domain'],
                    'keywords': [],
                    'positions': []
                }
            url_keywords[url]['keywords'].append(keyword)
            url_keywords[url]['positions'].append(result['position'])

    # Filter to URLs appearing in multiple SERPs
    overlapping = []
    for url, data in url_keywords.items():
        if len(data['keywords']) > 1:
            overlapping.append({
                'URL': url,
                'Domain': data['domain'],
                'Title': data['title'],
                'Keywords Count': len(data['keywords']),
                'Keywords': ', '.join(data['keywords']),
                'Positions': ', '.join(map(str, data['positions']))
            })

    df = pd.DataFrame(overlapping)
    if not df.empty:
        df = df.sort_values('Keywords Count', ascending=False)

    return df


# Main content
st.subheader("Enter Keywords")

keyword_input = st.text_area(
    "Keywords to compare (one per line, 2-20 keywords)",
    height=150,
    placeholder="seo tools\nkeyword research\ncontent marketing"
)

keywords = [kw.strip() for kw in keyword_input.strip().split('\n') if kw.strip()] if keyword_input else []

has_credentials = bool(dataforseo_login and dataforseo_password)

if keywords:
    if len(keywords) < 2:
        st.warning("Please enter at least 2 keywords to compare")
    elif len(keywords) > 20:
        st.warning("Please enter no more than 20 keywords")
    else:
        estimated_cost = len(keywords) * 0.002 * math.ceil(num_results / 10)
        st.info(f"Ready to analyse {len(keywords)} keywords. Estimated cost: ${estimated_cost:.3f}")

if st.button("Analyse SERPs", type="primary", disabled=not has_credentials or len(keywords) < 2 or len(keywords) > 20):
    if not has_credentials:
        st.error("Please enter your DataForSEO login and password")
    else:
        serp_data = {}
        progress_bar = st.progress(0)

        for i, keyword in enumerate(keywords):
            st.text(f"Fetching SERP for: {keyword}")
            results = fetch_serp(keyword, dataforseo_login, dataforseo_password, location, device, num_results)
            if results:
                serp_data[keyword] = results
            progress_bar.progress((i + 1) / len(keywords))

        if len(serp_data) >= 2:
            # Calculate crossover
            matrix = calculate_crossover(serp_data)
            overlapping_df = find_overlapping_urls(serp_data)

            # Store results
            st.session_state['serp_data'] = serp_data
            st.session_state['crossover_matrix'] = matrix
            st.session_state['overlapping_urls'] = overlapping_df

            st.success("Analysis complete!")
        else:
            st.error("Need at least 2 successful SERP fetches to analyse")

# Display results
if 'crossover_matrix' in st.session_state:
    matrix = st.session_state['crossover_matrix']
    overlapping_df = st.session_state['overlapping_urls']
    serp_data = st.session_state['serp_data']

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Keywords Analysed", len(serp_data))
    with col2:
        total_urls = sum(len(r) for r in serp_data.values())
        st.metric("Total SERP Results", total_urls)
    with col3:
        st.metric("Overlapping URLs", len(overlapping_df))
    with col4:
        # Average crossover (excluding diagonal)
        values = []
        for i, row in enumerate(matrix.values):
            for j, val in enumerate(row):
                if i != j:
                    values.append(val)
        avg_crossover = sum(values) / len(values) if values else 0
        st.metric("Avg Crossover %", f"{avg_crossover:.1f}%")

    # Crossover matrix
    st.subheader("Crossover Matrix")
    st.caption("Percentage of URL overlap between keyword SERPs (Jaccard similarity)")

    # Style matrix with heatmap
    def color_crossover(val):
        if val == 100:
            return 'background-color: #d9d9d9'
        elif val >= 50:
            return 'background-color: #ff6b6b; color: white'
        elif val >= 25:
            return 'background-color: #ffd93d'
        elif val > 0:
            return 'background-color: #c9e4c5'
        else:
            return ''

    styled_matrix = matrix.style.applymap(color_crossover)
    st.dataframe(styled_matrix, use_container_width=True)

    st.markdown("""
    **Colour key:**
    - Red (50%+): High crossover - potential cannibalisation
    - Yellow (25-50%): Moderate crossover - related topics
    - Green (1-25%): Low crossover - different intent
    - White (0%): No crossover
    """)

    # Overlapping URLs
    if not overlapping_df.empty:
        st.subheader("Overlapping URLs")
        st.caption("URLs that appear in multiple keyword SERPs")
        st.dataframe(overlapping_df, use_container_width=True)
    else:
        st.info("No overlapping URLs found between these keywords")

    # Raw SERP data
    with st.expander("View Raw SERP Data"):
        all_serp = []
        for keyword, results in serp_data.items():
            for r in results:
                all_serp.append({
                    'Keyword': keyword,
                    'Position': r['position'],
                    'Title': r['title'],
                    'URL': r['link'],
                    'Domain': r['domain']
                })
        st.dataframe(pd.DataFrame(all_serp), use_container_width=True)

    # Downloads
    st.subheader("Download")

    col1, col2, col3 = st.columns(3)

    with col1:
        matrix_csv = matrix.to_csv(encoding='utf-8-sig')
        st.download_button(
            label="Download Matrix (CSV)",
            data=matrix_csv,
            file_name="crossover_matrix.csv",
            mime="text/csv"
        )

    with col2:
        if not overlapping_df.empty:
            overlap_csv = overlapping_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="Download Overlaps (CSV)",
                data=overlap_csv,
                file_name="overlapping_urls.csv",
                mime="text/csv"
            )

    with col3:
        # Full Excel export
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            matrix.to_excel(writer, sheet_name='Crossover Matrix')
            if not overlapping_df.empty:
                overlapping_df.to_excel(writer, sheet_name='Overlapping URLs', index=False)
            # All SERP data
            all_serp_df = pd.DataFrame(all_serp)
            all_serp_df.to_excel(writer, sheet_name='Raw SERP Data', index=False)

        st.download_button(
            label="Download Excel Report",
            data=output.getvalue(),
            file_name="serp_crossover_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    if not has_credentials:
        st.warning("Enter your DataForSEO login and password in the sidebar to get started")

    st.subheader("Example Output")

    example_matrix = pd.DataFrame({
        "seo tools": [100.0, 30.0, 10.0],
        "keyword research": [30.0, 100.0, 20.0],
        "content marketing": [10.0, 20.0, 100.0]
    }, index=["seo tools", "keyword research", "content marketing"])

    st.dataframe(example_matrix)

    example_overlap = {
        "URL": ["https://ahrefs.com/seo", "https://semrush.com/"],
        "Domain": ["ahrefs.com", "semrush.com"],
        "Keywords Count": [3, 2],
        "Keywords": ["seo tools, keyword research, content marketing", "seo tools, keyword research"]
    }
    st.dataframe(pd.DataFrame(example_overlap))
