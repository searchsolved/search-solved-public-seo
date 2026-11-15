# Author: Lee Foot
# Website: https://leefoot.com

####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                                              #
####################################################################################

import streamlit as st

st.set_page_config(page_title="SERP Keyword Extractor by @LeeFootSEO", page_icon="📈",
                   layout="wide")  # needs to be the first thing after the streamlit import
import math
import os
from base64 import b64encode

import pandas as pd
import requests
from fuzzywuzzy import fuzz
import altair as alt

DATAFORSEO_ENDPOINT = 'https://api.dataforseo.com/v3/serp/google/organic/live/advanced'

# Region name -> (DataForSEO location_code, language_code)
LOCATIONS = {
    "United Kingdom": (2826, "en"),
    "United States": (2840, "en"),
    "Australia": (2036, "en"),
    "France": (2250, "fr"),
    "Canada": (2124, "en"),
    "Germany": (2276, "de"),
    "Italy": (2380, "it"),
    "Spain": (2724, "es"),
}


def parse_organic_results(response_data):
    """Return the organic items from a DataForSEO SERP response.

    Raises ValueError with a readable message when the API reports an error
    or the response has no result.
    """
    if not isinstance(response_data, dict):
        raise ValueError("Unexpected response from DataForSEO")
    if response_data.get('status_code') != 20000:
        raise ValueError(f"API Error: {response_data.get('status_message', 'Unknown error')}")
    try:
        task = response_data["tasks"][0]
    except (KeyError, TypeError, IndexError):
        raise ValueError("No task returned by DataForSEO")
    if task.get('status_code') != 20000:
        raise ValueError(f"Task Error: {task.get('status_message', 'Unknown error')}")
    try:
        items = task["result"][0]["items"] or []
    except (KeyError, TypeError, IndexError):
        raise ValueError("No search results found")
    return [item for item in items if item.get("type") == "organic"]


st.title("SERP Keyword Extractor - Find Related Keywords!")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Searches Google for your seed keyword using the DataForSEO SERP API
    - Splits the titles of the ranking pages into phrases
    - Lists the phrases that appear in several titles, with how often they appear

    **How to use:**
    1. Enter your DataForSEO login and password in the sidebar (sign up at [dataforseo.com](https://dataforseo.com/))
    2. Choose the region, device and number of results to analyse
    3. Enter a seed keyword and click Submit
    4. Download the related keywords as a CSV

    **Best for:**
    - Finding related keywords to dual optimise a page
    - Expanding existing content
    - On-page optimisation research
    """)

# streamlit variables
q = st.text_input('Input Your Search Keyword')

dataforseo_login = st.sidebar.text_input(
    'DataForSEO Login',
    type='password',
    value=os.environ.get('DATAFORSEO_LOGIN', ''),
    help="Your login from dataforseo.com"
)

dataforseo_password = st.sidebar.text_input(
    'DataForSEO Password',
    type='password',
    value=os.environ.get('DATAFORSEO_PASSWORD', ''),
    help="Your password from dataforseo.com"
)

location_select = st.sidebar.selectbox(
    "Select The Region To Search Google From",
    list(LOCATIONS.keys()),
)

device_select = st.sidebar.selectbox(
    "Select The Host Device To Use To Search Google",
    (
        "Desktop",
        "Mobile",
    ),
)
st.sidebar.caption("Tablet is no longer offered: DataForSEO only supports desktop and mobile results.")

minimum_frequency = st.sidebar.slider("Set Minimum Keyword Frequency", min_value=1, max_value=10, value=2)
num_pages = st.sidebar.slider("Set Number of Results to Analyse", min_value=10, max_value=100, value=20, step=10)

# Show estimated cost (DataForSEO bills per 10 results)
estimated_cost = 0.002 * math.ceil(num_pages / 10)
st.sidebar.caption(f"Estimated cost per search: ${estimated_cost:.3f}")

with st.form(key='columns_in_form_2'):
    submitted = st.form_submit_button('Submit')

if submitted:
    if not dataforseo_login or not dataforseo_password:
        st.error("Please enter your DataForSEO login and password in the sidebar")
        st.stop()

    if not q:
        st.error("Please enter a search keyword")
        st.stop()

    st.write("Searching Google for: %s" % q)

    query = []
    title = []

    cred = b64encode(f"{dataforseo_login}:{dataforseo_password}".encode()).decode()
    headers = {
        'Authorization': f'Basic {cred}',
        'Content-Type': 'application/json'
    }

    location_code, language_code = LOCATIONS[location_select]
    payload = [{
        "keyword": q,
        "location_code": location_code,
        "language_code": language_code,
        "device": device_select.lower(),
        "depth": num_pages
    }]

    try:
        with st.spinner('Fetching SERP data...'):
            response = requests.post(DATAFORSEO_ENDPOINT, headers=headers, json=payload, timeout=60)
            response_data = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {str(e)}")
        st.stop()
    except ValueError:
        st.error("Failed to parse API response. Please check your DataForSEO login and password.")
        st.stop()

    try:
        result = parse_organic_results(response_data)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    if response_data.get('cost') is not None:
        st.caption(f"Cost reported by DataForSEO: ${response_data['cost']:.4f}")

    if not result:
        st.info("No organic results found for this keyword.")
        st.stop()

    for var in result:
        title.append(var.get('title') or "")
        query.append(q)

    # make the df
    df = pd.DataFrame(None)
    df['query'] = query
    df['title'] = title

    # standardise delimiters
    df['title'] = df['title'].str.replace("/", "|")
    df['title'] = df['title'].str.replace("-", "|")
    df['title'] = df['title'].str.replace(":", "|")
    df['title'] = df['title'].str.replace("&", "|")
    df['title'] = df['title'].str.replace(",", "|")
    df['title'] = df['title'].str.lower()

    # split on specific delimiter before exploding
    df['title'] = df['title'].str.split('|')

    # explode the column
    df = df.explode('title').reset_index(drop=True)
    df['title'] = df['title'].str.strip()  # strip out leading and trailing spaces
    df['serp_frequency'] = df.groupby(["title"])["title"].transform("count")  # get the count
    df = df.drop_duplicates(subset=['title'], keep="first")  # drop dupes

    # drop rows containing a single word
    df['total_words'] = df['title'].str.count(' ') + 1

    # drop rows below minimum frequency and single word keywords
    df = df[df["serp_frequency"] >= minimum_frequency]
    df = df[df["total_words"] >= 2]

    df = df[~df["title"].str.contains(r"\.\.\.", na=False)]

    if df.empty:
        st.warning('No Opportunity Available for this Keyword - Please Search Again!')
        st.stop()

    df['similarity'] = df.apply(lambda x: fuzz.partial_ratio(x['query'], x['title']), axis=1)

    # clean up the df
    df.rename(columns={"title": "extracted_keywords"}, inplace=True)
    del df['total_words']
    df['query'] = df['query'].str.lower()

    st.subheader("📊 Related Keywords Graph")
    chart = (
        alt.Chart(df)
            .mark_bar()
            .encode(
            alt.X("extracted_keywords:O"),
            alt.Y("serp_frequency"),
            alt.Color("extracted_keywords:O"),
            alt.Tooltip(["extracted_keywords", "serp_frequency"]),
        )
            .properties(width=800, height=400)
            .interactive()
    )
    st.altair_chart(chart)

    st.subheader("🪄 Related Keywords Table")
    st.write(df)

    def convert_df(df):
        return df.to_csv(index=False).encode("utf-8")


    # download results
    st.write("")
    csv = convert_df(df)
    st.download_button(
        label="💾 Download The Data!",
        data=csv,
        file_name="related_serp_keywords.csv",
        mime="text/csv",
    )
