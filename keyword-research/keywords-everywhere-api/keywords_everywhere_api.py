####################################################################################
#                                                                                  #
#  Keywords Everywhere API Tool                                                    #
#                                                                                  #
#  Fetch search volume data from Keywords Everywhere API.                          #
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
Keywords Everywhere API Tool

Fetches search volume, CPC, and competition data from the Keywords Everywhere API.
Batches requests efficiently (100 keywords per call) to optimize API usage.

Features:
- Enter keywords via text area or CSV upload
- Configurable country and currency
- Batched API requests (100 keywords per call)
- Export with volume, CPC, competition data
"""

import streamlit as st
import pandas as pd
import requests
from io import BytesIO

st.set_page_config(page_title="Keywords Everywhere API", page_icon="🔑", layout="wide")

st.title("Keywords Everywhere API")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Fetches search volume data from Keywords Everywhere API
    - Returns volume, CPC, and competition for each keyword
    - Batches requests efficiently (100 keywords per call)

    **Requirements:**
    - Keywords Everywhere API key (get one at [keywordseverywhere.com](https://keywordseverywhere.com/))
    - API credits (each keyword uses credits)

    **How to use:**
    1. Enter your API key in the sidebar
    2. Select your target country
    3. Enter keywords or upload a CSV
    4. Click "Get Search Volume"
    5. Download results
    """)

# Sidebar settings
st.sidebar.header("API Settings")

api_key = st.sidebar.text_input(
    "Keywords Everywhere API Key",
    type="password",
    help="Your API key from keywordseverywhere.com"
)

st.sidebar.markdown("---")
st.sidebar.header("Search Settings")

# Country options
COUNTRIES = {
    "United Kingdom": "uk",
    "United States": "us",
    "Australia": "au",
    "Canada": "ca",
    "Germany": "de",
    "France": "fr",
    "Spain": "es",
    "Italy": "it",
    "Netherlands": "nl",
    "Brazil": "br",
    "India": "in",
    "Japan": "jp"
}

CURRENCIES = {
    "uk": "GBP",
    "us": "USD",
    "au": "AUD",
    "ca": "CAD",
    "de": "EUR",
    "fr": "EUR",
    "es": "EUR",
    "it": "EUR",
    "nl": "EUR",
    "br": "BRL",
    "in": "INR",
    "jp": "JPY"
}

selected_country = st.sidebar.selectbox(
    "Country",
    list(COUNTRIES.keys()),
    index=0
)

country_code = COUNTRIES[selected_country]
currency = CURRENCIES[country_code]

st.sidebar.info(f"Currency: {currency}")

# Data source options
data_source = st.sidebar.selectbox(
    "Data Source",
    ["gkp", "cli"],
    format_func=lambda x: "Google Keyword Planner" if x == "gkp" else "Clickstream",
    help="gkp = Google Keyword Planner data, cli = Clickstream data"
)


def fetch_keyword_data(keywords, api_key, country, currency, data_source):
    """Fetch keyword data from Keywords Everywhere API in batches."""
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    results = []
    batch_size = 100

    for i in range(0, len(keywords), batch_size):
        chunk = keywords[i:i + batch_size]

        data = {
            'country': country,
            'currency': currency,
            'dataSource': data_source,
            'kw[]': chunk
        }

        try:
            response = requests.post(
                'https://api.keywordseverywhere.com/v1/get_keyword_data',
                data=data,
                headers=headers
            )

            if response.status_code == 200:
                keywords_data = response.json().get('data', [])

                for idx, element in enumerate(keywords_data):
                    if idx < len(chunk):
                        results.append({
                            'Keyword': chunk[idx],
                            'Volume': element.get('vol', 0),
                            'CPC': element.get('cpc', {}).get('value', 0),
                            'Competition': element.get('competition', 0),
                            'Trend': element.get('trend', [])
                        })
            else:
                error_msg = response.json().get('message', 'Unknown error')
                st.error(f"API Error: {error_msg}")
                return None

        except Exception as e:
            st.error(f"Error processing batch: {str(e)}")
            return None

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
    st.info(f"Ready to fetch data for {len(keywords)} keywords")

    # Estimate API usage
    estimated_credits = len(keywords)
    st.caption(f"Estimated API credits: ~{estimated_credits}")

if st.button("Get Search Volume", type="primary", disabled=not api_key or not keywords):
    if not api_key:
        st.error("Please enter your API key")
    elif not keywords:
        st.error("Please enter some keywords")
    else:
        with st.spinner(f"Fetching data for {len(keywords)} keywords..."):
            progress_bar = st.progress(0)

            results = fetch_keyword_data(
                keywords,
                api_key,
                country_code,
                currency,
                data_source
            )

            progress_bar.progress(100)

            if results:
                df = pd.DataFrame(results)

                # Format trend as string for display
                df['Trend'] = df['Trend'].apply(lambda x: str(x) if x else '')

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
        total_volume = df['Volume'].sum()
        st.metric("Total Volume", f"{total_volume:,}")
    with col3:
        avg_cpc = df['CPC'].mean()
        st.metric("Avg CPC", f"{currency} {avg_cpc:.2f}")
    with col4:
        high_vol = len(df[df['Volume'] >= 1000])
        st.metric("High Volume (1000+)", high_vol)

    # Data table
    st.subheader("Results")

    # Sort options
    sort_col = st.selectbox(
        "Sort by",
        ['Volume', 'Keyword', 'CPC', 'Competition'],
        index=0
    )
    sort_order = st.checkbox("Ascending", value=False)

    df_sorted = df.sort_values(sort_col, ascending=sort_order)
    st.dataframe(df_sorted.drop(columns=['Trend']), use_container_width=True)

    # Download options
    st.subheader("Download")

    col1, col2 = st.columns(2)

    with col1:
        csv_data = df.drop(columns=['Trend']).to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"keyword_volumes_{country_code}.csv",
            mime="text/csv"
        )

    with col2:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.drop(columns=['Trend']).to_excel(writer, sheet_name='Keyword Data', index=False)

        st.download_button(
            label="Download Excel",
            data=output.getvalue(),
            file_name=f"keyword_volumes_{country_code}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    if not api_key:
        st.warning("Enter your Keywords Everywhere API key in the sidebar to get started")

    st.subheader("Example Output")
    example_data = {
        "Keyword": ["seo tools", "keyword research", "google analytics"],
        "Volume": [12100, 8100, 165000],
        "CPC": [2.50, 3.20, 1.80],
        "Competition": [0.85, 0.72, 0.65]
    }
    st.dataframe(pd.DataFrame(example_data))
