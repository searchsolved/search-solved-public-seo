import streamlit as st

st.set_page_config(page_title="SERP Keyword Extractor by @LeeFootSEO", page_icon="📈",
                   layout="wide")  # needs to be the first thing after the streamlit import
import pandas as pd
import requests
import json
from fuzzywuzzy import fuzz
import altair as alt

st.write(
    "[![this is an image link](https://i.imgur.com/Ex8eeC2.png)](https://www.patreon.com/leefootseo) [Become a Patreon for Early Access, Support & More!](https://www.patreon.com/leefootseo)  |  Made in [![this is an image link](https://i.imgur.com/iIOA6kU.png)](https://www.streamlit.io/) by [@LeeFootSEO](https://twitter.com/LeeFootSEO)")

with st.expander("How do I use this app?"):
    st.write("""

        1. You will need an API key from www.ValueSERP.Com - they offer 100 searches for free
        Plenty to test this with!
        2. Enter your API key, enter a seed keyword, and click submit.
        3. This data can be used to dual optimise pages / expand on existing content etc""")

st.title("SERP Keyword Extractor - Find Related Keywords!")

# streamlit variables
q = st.text_input('Input Your Search Keyword')
value_serp_key = st.sidebar.text_input('Input your ValueSERP API Key')

location_select = st.sidebar.selectbox(
    "Select The Region To Search Google From",
    (
        "United Kingdom",
        "United States",
        "Australia",
        "France",
        "Canada",
        "Germany",
        "Italy",
        "Spain",
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

minimum_frequency = st.sidebar.slider("Set Minimum Keyword Frequency", min_value=1, max_value=10, value=2)
num_pages = st.sidebar.slider("Set Number of Results to Analyse", min_value=10, max_value=100, value=20)
minimum_frequency -= 1

with st.form(key='columns_in_form_2'):
    submitted = st.form_submit_button('Submit')

if submitted:
    st.write("Searching Google for: %s" % q)

    query = []
    title = []

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
    if result == None:
        st.info("No Data Received, Please Check Your API Key!")
        st.stop()
    for var in result:
        try:
            title.append(var['title'])
            query.append(q)
        except Exception:
            title.append("")
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

    # drop single word keywords and rows with a frequency of 1
    df = df[~df["serp_frequency"].isin([minimum_frequency])]
    df = df[~df["total_words"].isin([minimum_frequency])]

    df = df[~df["title"].str.contains("\.\.\.", na=False)]

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
