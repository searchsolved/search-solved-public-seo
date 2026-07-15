# Author: Lee Foot
# Website: https://leefoot.com

####################################################################################
#                                                                                  #
#  DataForSEO Google Ads Search Volume Tool                                        #
#                                                                                  #
#  Fetch search volume data from DataForSEO Google Ads API.                        #
#                                                                                  #
####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://leefoot.com                                                   #
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                       #
####################################################################################

"""
DataForSEO Google Ads Search Volume Tool

Fetches search volume, CPC, competition and bid data from the DataForSEO
Google Ads Search Volume API. Batches requests efficiently (up to 700 keywords
per call) to minimise API cost.

Features:
- Enter keywords via text area or CSV upload
- Configurable country and language
- Batched API requests (700 keywords per call, rate limited to 12/min)
- Export with volume, CPC, competition, bid and monthly search data
"""

import streamlit as st
import pandas as pd
import requests
import time
import math
import os
from io import BytesIO

st.set_page_config(page_title="DataForSEO Search Volume", page_icon="📊", layout="wide")

st.title("DataForSEO Search Volume")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Fetches search volume data from the DataForSEO Google Ads Search Volume API
    - Returns volume, CPC (USD), competition, competition index, top-of-page bids and monthly search trends
    - Batches up to 700 keywords per request to minimise cost (DataForSEO charges per request, not per keyword)

    **Requirements:**
    - DataForSEO account (sign up at [dataforseo.com](https://dataforseo.com/))
    - API credits (each request costs a fixed amount regardless of keyword count, so batching is efficient)

    **How to use:**
    1. Enter your DataForSEO login and password in the sidebar
    2. Select your target country
    3. Enter keywords or upload a CSV
    4. Click "Get Search Volume"
    5. Download results

    **Note:** CPC values are returned in USD by default.
    """)

# Sidebar settings
st.sidebar.header("API Settings")

dataforseo_login = st.sidebar.text_input(
    "DataForSEO Login",
    type="password",
    value=os.environ.get("DATAFORSEO_LOGIN", ""),
    help="Your DataForSEO API login (email)"
)

dataforseo_password = st.sidebar.text_input(
    "DataForSEO Password",
    type="password",
    value=os.environ.get("DATAFORSEO_PASSWORD", ""),
    help="Your DataForSEO API password"
)

st.sidebar.markdown("---")
st.sidebar.header("Search Settings")

# Country options: name -> (location_code, language_code)
COUNTRIES = {
    "United Kingdom": (2826, "en"),
    "United States": (2840, "en"),
    "Australia": (2036, "en"),
    "Canada": (2124, "en"),
    "Germany": (2276, "de"),
    "France": (2250, "fr"),
    "Spain": (2724, "es"),
    "Italy": (2380, "it"),
    "Netherlands": (2528, "nl"),
    "Brazil": (2076, "pt"),
    "India": (2356, "en"),
    "Japan": (2392, "ja"),
}

selected_country = st.sidebar.selectbox(
    "Country",
    list(COUNTRIES.keys()),
    index=0
)

location_code, language_code = COUNTRIES[selected_country]

st.sidebar.info("CPC values are returned in USD.")


def fetch_keyword_data(keywords, login, password, location_code, language_code):
    """Fetch keyword data from DataForSEO Google Ads Search Volume API in batches."""
    url = "https://api.dataforseo.com/v3/keywords_data/google_ads/search_volume/live"

    results = []
    batch_size = 700
    total_batches = math.ceil(len(keywords) / batch_size)

    for batch_num, i in enumerate(range(0, len(keywords), batch_size), start=1):
        chunk = keywords[i:i + batch_size]

        payload = [{
            "keywords": chunk,
            "location_code": location_code,
            "language_code": language_code,
        }]

        try:
            response = requests.post(
                url,
                json=payload,
                auth=(login, password),
            )

            if response.status_code == 200:
                resp_json = response.json()
                tasks = resp_json.get("tasks", [])
                if not tasks:
                    st.error("API returned no tasks.")
                    return None

                task = tasks[0]
                if task.get("status_code") != 20000:
                    st.error(f"API Error: {task.get('status_message', 'Unknown error')}")
                    return None

                task_results = task.get("result", [])
                if task_results is None:
                    task_results = []

                for item in task_results:
                    monthly = item.get("monthly_searches") or []
                    monthly_str = "; ".join(
                        f"{m.get('year')}-{m.get('month'):02d}: {m.get('search_volume', 0)}"
                        for m in monthly
                    ) if monthly else ""

                    results.append({
                        "Keyword": item.get("keyword", ""),
                        "Search Volume": item.get("search_volume") or 0,
                        "CPC": item.get("cpc") or 0,
                        "Competition": item.get("competition") or "",
                        "Competition Index": item.get("competition_index") or 0,
                        "Low Top of Page Bid": item.get("low_top_of_page_bid") or 0,
                        "High Top of Page Bid": item.get("high_top_of_page_bid") or 0,
                        "Monthly Searches": monthly_str,
                    })
            else:
                try:
                    error_detail = response.json()
                    error_msg = error_detail.get("status_message", response.text)
                except Exception:
                    error_msg = response.text
                st.error(f"HTTP {response.status_code}: {error_msg}")
                return None

        except Exception as e:
            st.error(f"Error processing batch {batch_num}: {str(e)}")
            return None

        # Rate limit: 12 requests/min. Sleep 5s between batches if more than one.
        if total_batches > 1 and batch_num < total_batches:
            time.sleep(5)

    return results


# Main content
st.subheader("Enter Keywords")

input_method = st.radio(
    "Input method",
    ["Text input", "CSV upload"],
    horizontal=True
)

keywords = []

if input_method == "Text input":
    keyword_text = st.text_area(
        "Enter keywords (one per line)",
        height=200,
        placeholder="keyword 1\nkeyword 2\nkeyword 3"
    )

    if keyword_text:
        keywords = [kw.strip() for kw in keyword_text.strip().split('\n') if kw.strip()]

else:
    keyword_file = st.file_uploader(
        "Upload CSV with keywords",
        type=['csv'],
        help="CSV with a 'keyword' or 'Keyword' column"
    )

    if keyword_file:
        try:
            df_upload = pd.read_csv(keyword_file)
            keyword_col = None
            for col in df_upload.columns:
                if col.lower() == 'keyword':
                    keyword_col = col
                    break
            if not keyword_col:
                keyword_col = df_upload.columns[0]

            keywords = df_upload[keyword_col].dropna().astype(str).tolist()
            st.success(f"Loaded {len(keywords)} keywords from CSV")
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")

if keywords:
    num_requests = math.ceil(len(keywords) / 700)
    st.info(f"Ready to fetch data for {len(keywords)} keywords")
    st.caption(
        f"This will use {num_requests} API request{'s' if num_requests != 1 else ''} "
        f"(up to 700 keywords each). DataForSEO charges per request, not per keyword."
    )

has_credentials = bool(dataforseo_login and dataforseo_password)

if st.button("Get Search Volume", type="primary", disabled=not has_credentials or not keywords):
    if not has_credentials:
        st.error("Please enter your DataForSEO login and password")
    elif not keywords:
        st.error("Please enter some keywords")
    else:
        with st.spinner(f"Fetching data for {len(keywords)} keywords..."):
            progress_bar = st.progress(0)

            results = fetch_keyword_data(
                keywords,
                dataforseo_login,
                dataforseo_password,
                location_code,
                language_code,
            )

            progress_bar.progress(100)

            if results:
                df = pd.DataFrame(results)

                # Store in session state
                st.session_state['keyword_results'] = df

                st.success(f"Successfully fetched data for {len(df)} keywords")

# Display results
if 'keyword_results' in st.session_state:
    df = st.session_state['keyword_results']

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Keywords", len(df))
    with col2:
        total_volume = df['Search Volume'].sum()
        st.metric("Total Volume", f"{total_volume:,}")
    with col3:
        avg_cpc = df['CPC'].mean()
        st.metric("Avg CPC (USD)", f"${avg_cpc:.2f}")
    with col4:
        high_vol = len(df[df['Search Volume'] >= 1000])
        st.metric("High Volume (1000+)", high_vol)

    # Data table
    st.subheader("Results")

    # Sort options
    sort_col = st.selectbox(
        "Sort by",
        ['Search Volume', 'Keyword', 'CPC', 'Competition Index'],
        index=0
    )
    sort_order = st.checkbox("Ascending", value=False)

    display_cols = [
        "Keyword", "Search Volume", "CPC", "Competition",
        "Competition Index", "Low Top of Page Bid", "High Top of Page Bid",
    ]
    df_sorted = df.sort_values(sort_col, ascending=sort_order)
    st.dataframe(df_sorted[display_cols], use_container_width=True)

    # Download options
    st.subheader("Download")

    col1, col2 = st.columns(2)

    with col1:
        csv_data = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"keyword_volumes_{location_code}.csv",
            mime="text/csv"
        )

    with col2:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Keyword Data', index=False)

        st.download_button(
            label="Download Excel",
            data=output.getvalue(),
            file_name=f"keyword_volumes_{location_code}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    if not has_credentials:
        st.warning("Enter your DataForSEO login and password in the sidebar to get started")

    st.subheader("Example Output")
    example_data = {
        "Keyword": ["seo tools", "keyword research", "google analytics"],
        "Search Volume": [12100, 8100, 165000],
        "CPC": [2.50, 3.20, 1.80],
        "Competition": ["HIGH", "HIGH", "MEDIUM"],
        "Competition Index": [85, 72, 65],
        "Low Top of Page Bid": [1.00, 1.50, 0.80],
        "High Top of Page Bid": [4.00, 5.50, 3.00],
    }
    st.dataframe(pd.DataFrame(example_data))
