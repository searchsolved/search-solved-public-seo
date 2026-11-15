# Author: Lee Foot
# Website: https://leefoot.com

####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                                              #
####################################################################################

import streamlit as st

st.set_page_config(page_title="SERP Entity Extractor", page_icon="mag",
                   layout="wide")
import os
from base64 import b64encode

import pandas as pd
import requests
from dandelion import DataTXT
import trafilatura
from trafilatura import extract
from trafilatura.settings import use_config
import altair as alt
from stqdm import stqdm

newconfig = use_config()
newconfig.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")

DATAFORSEO_ENDPOINT = 'https://api.dataforseo.com/v3/serp/google/organic/live/advanced'

# Region name -> (DataForSEO location_code, language_code)
LOCATIONS = {
    "United Kingdom": (2826, "en"),
    "United States": (2840, "en"),
    "Australia": (2036, "en"),
    "Brazil": (2076, "pt"),
    "France": (2250, "fr"),
    "Canada": (2124, "en"),
    "Germany": (2276, "de"),
    "Italy": (2380, "it"),
    "Spain": (2724, "es"),
    "New Zealand": (2554, "en"),
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


with st.expander("How do I use this app?"):
    st.write("""
        1. You will need a DataForSEO account from dataforseo.com (pay as you go)
        2. You will need an API key from dandelion.eu - 1,000 free requests per day
        3. Enter your DataForSEO login and password and your Dandelion key, enter a seed keyword, and click submit
        4. This data can be used to optimize pages / expand on existing content""")

st.title("SERP Entity Extractor")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Extracts entities from SERP results
    - Analyzes competitor content for entities
    - Compares entity coverage across rankings

    **How to use:**
    1. Enter your DataForSEO login and password and your Dandelion.eu key in the sidebar
    2. Choose the region, device and number of results pages to analyse
    3. Enter a seed keyword and click Submit
    4. Identify entity gaps vs competitors

    **Best for:**
    - SERP entity analysis
    - Content gap identification
    - Topical authority research
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

dandelion_key = st.sidebar.text_input('Input your Dandelion.eu API Key')

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

accuracy = st.sidebar.slider("Set Entity Accuracy", min_value=10, max_value=100, value=80)
num_pages = st.sidebar.slider("Set Number of Pages to Analyse", min_value=1, max_value=10, value=1)
top_entities = st.sidebar.slider("Set The Number of Entities to Display", min_value=5, max_value=25, value=15)

# Show estimated cost (DataForSEO bills per page of 10 results)
estimated_cost = 0.002 * num_pages
st.sidebar.caption(f"Estimated DataForSEO cost per search: ${estimated_cost:.3f}")

accuracy = accuracy / 100
num_pages = num_pages * 10

# store the SERP Data
query = []
serp_title = []
link = []

# store the entity data
entity = []
confidence = []
title = []
wiki_url = []
categories = []
url_list = []

with st.form(key='columns_in_form_2'):
    submitted = st.form_submit_button('Submit')

if submitted:
    try:
        datatxt = DataTXT(token=dandelion_key)
    except Exception:
        st.info("Missing Dandelion.eu API key! Please visit www.dandelion.eu to get your free API key")
        st.stop()

    if not dataforseo_login or not dataforseo_password:
        st.error("Please enter your DataForSEO login and password in the sidebar")
        st.stop()

    if not q:
        st.info("No results returned. Did you enter a keyword to search?")
        st.stop()

    st.write("Searching Google for: %s" % q)

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

    for var in result:
        if not var.get('url'):
            continue
        serp_title.append(var.get('title') or '')
        link.append(var['url'])
        query.append(q)

    if not link:
        st.info("No organic results found for this keyword.")
        st.stop()

    # make the serp df
    df = pd.DataFrame(None)
    df['Keyword'] = query
    df['Current URL'] = link
    df['title'] = serp_title

    # get the entities
    df['Keyword'] = df['Keyword'].astype(str)
    unique_urls = list(set(df['Current URL']))

    for url in stqdm(unique_urls):

        try:
            downloaded = trafilatura.fetch_url(url)
            txt = extract(downloaded, config=newconfig)

            response = datatxt.nex(txt)

            for annotation in response.annotations:
                entity.append(annotation['spot'])
                confidence.append(annotation['confidence'])
                title.append(annotation['title'])
                wiki_url.append(annotation['uri'])
                categories.append(annotation['label'])
                url_list.append(str(url))
        except Exception:
            pass

    df = pd.DataFrame(None)
    df['url'] = url_list
    try:
        df['entity'] = entity
    except Exception:
        st.info("No Entities Were Found! Please increase the Number of Pages to Search, or adjust your search term")
        st.stop()
    df['confidence'] = confidence
    df['entity_title'] = title
    df['category'] = categories
    df['wiki_url'] = wiki_url

    # drop duplicates
    df['entity'] = df['entity'].str.lower()
    df['entity_title'] = df['entity_title'].str.lower()
    df['category'] = df['category'].str.lower()
    df.drop_duplicates(subset=["url", "entity"], keep="first", inplace=True)

    df['most_frequent'] = df['entity'].map(df.groupby('entity')['entity'].count())
    df.sort_values(["most_frequent", "entity"], ascending=[False, True], inplace=True)

    df_chart = df[["entity", "most_frequent", "confidence"]].copy()
    df_chart = df_chart.sort_values(by="most_frequent", ascending=False)
    df_chart.drop_duplicates(subset="entity", inplace=True)
    df_chart = df_chart[df_chart.confidence >= accuracy]
    df_chart = df_chart.iloc[:top_entities]
    df_chart.drop_duplicates(subset=['entity'], keep="first", inplace=True)

    st.subheader("Top Entities")
    chart = (
        alt.Chart(df_chart)
            .mark_bar()
            .encode(
            alt.X("entity:O", sort=alt.EncodingSortField(field="entity", op="count", order='ascending')),
            alt.Y("most_frequent"),
            alt.Color("entity:O", sort=alt.SortField('count', order='descending')),
            alt.Tooltip(["entity", "most_frequent"]),
        )
            .properties(width=800, height=400)
            .interactive()
    )
    st.altair_chart(chart)

    df["query"] = q
    df = df[["query", "entity", "confidence", "entity_title", "category", "wiki_url", "most_frequent"]]
    df = df[df.confidence >= accuracy]
    df = df.sort_values(by="confidence", ascending=False)
    df.sort_values(["most_frequent", "confidence"], ascending=[False, False], inplace=True)
    df.drop_duplicates(subset="entity", inplace=True)

    def convert_df(df):
        return df.to_csv(index=False).encode("utf-8")

    # download results
    st.write("")
    csv = convert_df(df)
    st.download_button(
        label="Download The Data!",
        data=csv,
        file_name="related_serp_keywords.csv",
        mime="text/csv",
    )

    st.subheader("Related Keywords Table")
    st.write(df, index=False)
