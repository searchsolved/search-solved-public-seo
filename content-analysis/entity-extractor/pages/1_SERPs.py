####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Twitter  : https://x.com/LeeFootSEO                                              #
####################################################################################

import streamlit as st

st.set_page_config(page_title="SERP Entity Extractor", page_icon="mag",
                   layout="wide")
import pandas as pd
import requests
import json
from dandelion import DataTXT
import trafilatura
from trafilatura import extract
from trafilatura.settings import use_config
import altair as alt
from stqdm import stqdm

newconfig = use_config()
newconfig.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")

st.write("Made in [![this is an image link](https://i.imgur.com/iIOA6kU.png)](https://www.streamlit.io/) by [@LeeFootSEO](https://x.com/LeeFootSEO)")

with st.expander("How do I use this app?"):
    st.write("""
        1. You will need an API key from www.ValueSERP.com - they offer 100 searches for free
        2. You will need an API key from dandelion.eu - 1,000 free requests per day
        3. Enter your API keys, enter a seed keyword, and click submit
        4. This data can be used to optimize pages / expand on existing content""")

st.title("SERP Entity Extractor")

# streamlit variables
q = st.text_input('Input Your Search Keyword')
value_serp_key = st.sidebar.text_input('Input your ValueSERP API Key')
dandelion_key = st.sidebar.text_input('Input your Dandelion.eu API Key')

location_select = st.sidebar.selectbox(
    "Select The Region To Search Google From",
    (
        "United Kingdom",
        "United States",
        "Australia",
        "Brazil",
        "France",
        "Canada",
        "Germany",
        "Italy",
        "Spain",
        "New Zealand",
    ),
)

device_select = st.sidebar.selectbox(
    "Select The Host Device To Use To Search Google",
    (
        "Desktop",
        "Mobile",
        "Tablet",
    ),
)

accuracy = st.sidebar.slider("Set Entity Accuracy", min_value=10, max_value=100, value=80)
num_pages = st.sidebar.slider("Set Number of Pages to Analyse", min_value=1, max_value=10, value=1)
top_entities = st.sidebar.slider("Set The Number of Entities to Display", min_value=5, max_value=25, value=15)

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

    st.write("Searching Google for: %s" % q)

    params = {
        'api_key': value_serp_key,
        'q': q,
        'location': location_select,
        'include_fields': 'organic_results',
        'location_auto': True,
        'device': device_select,
        'output': 'json',
        'page': '1',
        'num': num_pages
    }

    response = requests.get('https://api.valueserp.com/search', params)
    response_data = json.loads(response.text)
    result = response_data.get('organic_results')

    try:
        for var in result:
                serp_title.append(var['title'])
                link.append(var['link'])
                query.append(q)
    except TypeError:
        st.info("No results returned. Did you enter a keyword to search?")
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
