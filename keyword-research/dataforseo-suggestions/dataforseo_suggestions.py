####################################################################################
#                                                                                  #
#  DataForSEO Keyword Suggestions                                                  #
#                                                                                  #
#  Get keyword suggestions with search volumes from DataForSEO API.                #
#                                                                                  #
####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://leefoot.com                                                   #
# Contact  : https://leefoot.com/contact                                           #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                       #
####################################################################################

"""
DataForSEO Keyword Suggestions

Gets keyword suggestions with search volumes from the DataForSEO API.
Great for keyword research and expanding seed keywords.

Features:
- Single or bulk keyword input
- Location and language selection
- Search volume data
- Export to CSV
"""

import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import json

st.set_page_config(page_title="DataForSEO Keyword Suggestions", page_icon="🔑", layout="wide")

st.title("DataForSEO Keyword Suggestions")
st.markdown("*Created by [Lee Foot](https://leefoot.com)*")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Gets keyword suggestions from DataForSEO's Keyword Suggestions API
    - Returns related keywords with search volumes
    - Supports multiple locations and languages

    **Requirements:**
    - DataForSEO API credentials (login and password)
    - Get free credits at [dataforseo.com](https://dataforseo.com)

    **Pricing:**
    - Keyword Suggestions API costs $0.05 per request (as of 2024)
    - Each seed keyword is one request
    """)

# Sidebar - API credentials
st.sidebar.header("API Credentials")
api_login = st.sidebar.text_input(
    "DataForSEO Login (Email)",
    help="Your DataForSEO account email"
)
api_password = st.sidebar.text_input(
    "DataForSEO Password",
    type="password",
    help="Your DataForSEO API password (not account password)"
)

st.sidebar.markdown("---")
st.sidebar.header("Settings")

# Location options
locations = {
    "United Kingdom": "United Kingdom",
    "United States": "United States",
    "Australia": "Australia",
    "Canada": "Canada",
    "Germany": "Germany",
    "France": "France",
    "Spain": "Spain",
    "Italy": "Italy",
    "Netherlands": "Netherlands",
    "Brazil": "Brazil",
    "India": "India",
    "Japan": "Japan"
}

location = st.sidebar.selectbox(
    "Location",
    list(locations.keys()),
    help="Target location for search volume data"
)

# Language options
languages = {
    "English": "English",
    "German": "German",
    "French": "French",
    "Spanish": "Spanish",
    "Italian": "Italian",
    "Dutch": "Dutch",
    "Portuguese": "Portuguese",
    "Japanese": "Japanese"
}

language = st.sidebar.selectbox(
    "Language",
    list(languages.keys()),
    help="Target language for suggestions"
)

limit = st.sidebar.slider(
    "Max suggestions per keyword",
    min_value=10,
    max_value=1000,
    value=100,
    step=10,
    help="Maximum keyword suggestions to return per seed"
)

include_seed = st.sidebar.checkbox(
    "Include seed keyword in results",
    value=True,
    help="Include the original keyword in the output"
)

# Main input
st.subheader("Enter Seed Keywords")

input_method = st.radio(
    "Input method",
    ["Enter keywords", "Upload CSV"],
    horizontal=True
)

seed_keywords = []

if input_method == "Enter keywords":
    keyword_text = st.text_area(
        "Enter seed keywords (one per line)",
        height=150,
        help="Enter your seed keywords, one per line"
    )
    if keyword_text:
        seed_keywords = [k.strip() for k in keyword_text.strip().split('\n') if k.strip()]
        st.info(f"Found {len(seed_keywords)} seed keywords")

else:
    keyword_file = st.file_uploader(
        "Upload CSV with keywords",
        type=['csv'],
        help="CSV file with a column containing keywords"
    )

    if keyword_file is not None:
        try:
            df_kws = pd.read_csv(keyword_file)
            kw_col = st.selectbox("Select keyword column", df_kws.columns.tolist())
            seed_keywords = df_kws[kw_col].dropna().astype(str).tolist()
            st.info(f"Found {len(seed_keywords)} seed keywords")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")


def get_keyword_suggestions(login, password, keyword, loc, lang, max_results, include_seed_kw):
    """Get keyword suggestions from DataForSEO API."""
    base_url = "https://api.dataforseo.com/v3/dataforseo_labs/google/keyword_suggestions/live"

    post_data = [{
        "keyword": keyword,
        "location_name": loc,
        "language_name": lang,
        "include_serp_info": False,
        "include_seed_keyword": include_seed_kw,
        "limit": max_results
    }]

    try:
        response = requests.post(
            base_url,
            auth=HTTPBasicAuth(login, password),
            headers={"Content-Type": "application/json"},
            data=json.dumps(post_data),
            timeout=60
        )

        response_data = response.json()

        if response_data.get("status_code") == 20000:
            results = []
            tasks = response_data.get('tasks', [])

            if tasks and tasks[0].get('result'):
                result = tasks[0]['result'][0]
                seed = result.get('seed_keyword', keyword)
                items = result.get('items', [])

                for item in items:
                    keyword_info = item.get('keyword_info', {})
                    results.append({
                        'seed_keyword': seed,
                        'suggested_keyword': item.get('keyword', '').replace('-', ' '),
                        'search_volume': keyword_info.get('search_volume', 0),
                        'cpc': keyword_info.get('cpc', 0),
                        'competition': keyword_info.get('competition', 0),
                        'competition_level': keyword_info.get('competition_level', '')
                    })

            return results, None
        else:
            error_msg = response_data.get('status_message', 'Unknown error')
            return None, f"API Error: {error_msg}"

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"
    except Exception as e:
        return None, f"Error: {str(e)}"


# Main processing
if seed_keywords and st.button("Get Keyword Suggestions", type="primary"):
    if not api_login or not api_password:
        st.error("Please enter your DataForSEO API credentials in the sidebar")
        st.stop()

    progress_bar = st.progress(0)
    status_text = st.empty()

    all_results = []
    errors = []

    for i, seed in enumerate(seed_keywords):
        status_text.text(f"Processing {i+1}/{len(seed_keywords)}: {seed}")
        progress_bar.progress((i + 1) / len(seed_keywords))

        results, error = get_keyword_suggestions(
            api_login,
            api_password,
            seed,
            locations[location],
            languages[language],
            limit,
            include_seed
        )

        if results:
            all_results.extend(results)
        if error:
            errors.append({'seed_keyword': seed, 'error': error})

    status_text.text("Processing complete!")

    if all_results:
        df_results = pd.DataFrame(all_results)

        # Remove duplicates
        df_results = df_results.drop_duplicates(subset=['suggested_keyword'])
        df_results = df_results.sort_values('search_volume', ascending=False)

        # Display results
        st.subheader("Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Suggestions", f"{len(df_results):,}")
        with col2:
            st.metric("Unique Keywords", f"{df_results['suggested_keyword'].nunique():,}")
        with col3:
            total_volume = df_results['search_volume'].sum()
            st.metric("Total Search Volume", f"{total_volume:,}")

        # Show top keywords
        st.subheader("Top Keywords by Search Volume")
        st.dataframe(df_results.head(100), use_container_width=True)

        # Volume distribution
        st.subheader("Search Volume Distribution")
        df_results['volume_bucket'] = pd.cut(
            df_results['search_volume'],
            bins=[0, 100, 500, 1000, 5000, 10000, float('inf')],
            labels=['0-100', '100-500', '500-1K', '1K-5K', '5K-10K', '10K+']
        )
        bucket_counts = df_results['volume_bucket'].value_counts().sort_index()
        st.bar_chart(bucket_counts)

        # Errors
        if errors:
            with st.expander(f"Errors ({len(errors)})"):
                st.dataframe(pd.DataFrame(errors))

        # Download
        st.subheader("Download")
        csv_output = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv_output,
            file_name="keyword_suggestions.csv",
            mime="text/csv"
        )

    else:
        st.warning("No keyword suggestions returned. Check your API credentials and try again.")

elif not seed_keywords:
    st.info("Enter seed keywords above to get suggestions")

    st.subheader("Example Output")
    example_data = {
        "Seed Keyword": ["seo tools", "seo tools", "seo tools"],
        "Suggested Keyword": ["free seo tools", "best seo tools", "seo audit tool"],
        "Search Volume": [12100, 8100, 3600],
        "CPC": [2.50, 3.20, 4.10],
        "Competition": [0.85, 0.92, 0.78]
    }
    st.dataframe(pd.DataFrame(example_data))
