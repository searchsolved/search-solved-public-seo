import streamlit as st

st.set_page_config(page_title="SERP N-gram & Title Extractor", page_icon="📈",
                   layout="wide")

import requests
import json
import trafilatura
from trafilatura import bare_extraction
from trafilatura.settings import use_config

import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
from itertools import chain
stop = stopwords.words('english')

newconfig = use_config()
newconfig.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")

st.title("SERP N-gram & Title Extractor")

st.markdown("*Created by 🌐 [Lee Foot](https://www.leefoot.com) · [LinkedIn](https://www.linkedin.com/in/lee-foot/) · 🦋 [Bluesky](https://bsky.app/profile/leefootseo.bsky.social)*")

with st.expander("How do I use this app?"):
    st.write("""
        1. You will need an API key from www.ValueSERP.com - they offer 100 searches for free
        2. Enter your API key, enter a seed keyword, and click submit
        3. The tool extracts page titles and content from ranking pages
        4. Use this data to optimize your page titles and expand content coverage
    """)

# streamlit variables
kw = st.text_input('Input Your Search Keyword')
value_serp_key = st.sidebar.text_input('Input your ValueSERP API Key', type='password')

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

num_pages = st.sidebar.slider("Set Number of Pages to Analyse", min_value=1, max_value=10, value=1)
num_pages = num_pages * 10

# store the SERP Data
links = []

with st.form(key='columns_in_form_2'):
    submitted = st.form_submit_button('Submit')

if submitted:
    if not value_serp_key:
        st.error("Please enter your ValueSERP API key in the sidebar")
        st.stop()

    if not kw:
        st.error("Please enter a search keyword")
        st.stop()

    # store the data
    links = []

    # set up the request parameters
    params = {
        'api_key': value_serp_key,
        'q': kw,
        'location': location_select,
        'include_fields': 'organic_results',
        'location_auto': True,
        'device': device_select,
        'output': 'json',
        'page': '1',
        'num': num_pages
    }

    # make the http GET request to VALUE SERP
    with st.spinner('Fetching SERP data...'):
        api_result = requests.get('https://api.valueserp.com/search', params)
        response_df = json.loads(api_result.text)
        result = response_df.get('organic_results')

    try:
        for var in result:
            links.append(var['link'])

    except TypeError:
        st.error("No results found. Please check your API key and try again.")
        st.stop()

    # store the extracted data
    text = []
    title = []
    extracted_url = []

    # loop through page 1 urls extracting content and h1s
    with st.spinner('Extracting content from pages...'):
        for url in links:
            try:
                downloaded = trafilatura.fetch_url(url)
                d = bare_extraction(downloaded, config=newconfig, with_metadata=True)
                text.append((d['text']))
                title.append(d['title'])
                extracted_url.append(d['url'])
            except Exception:
                pass


    # make serp extraction df
    df_serp = pd.DataFrame(None)
    df_serp['url'] = extracted_url
    df_serp['text'] = text
    df_serp['title'] = title
    df_serp['query'] = kw

    # get most frequent words from all ages tiles combined]
    df_title = df_serp.groupby(['query'])['title'].apply(' '.join).reset_index()

    # start strip out all special characters from a column
    spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")", "*", "+", ",", ".", "/", ":", ";", "<", "=", ">", "?", "@",
                  "[", "\\", "]", "^", "-", "_", "`", "{", "|", "}", "~", "–"]

    for char in spec_chars:
        df_title['title'] = df_title['title'].str.replace(char, ' ', regex=True)

    # get word counts of page title
    word_str = df_title['title']
    word_str = [words for segments in word_str for words in segments.split()]
    counts = pd.Series(word_str).value_counts()
    counts = counts.to_dict()

    # make the dataframe with the most popular page titles
    df_title = pd.DataFrame(counts.items(), columns=['keyword', 'frequency'])
    df_title = df_title[df_title.frequency > 1]
    df_title['query'] = kw

    # create n-grams dataframe
    df_serp['text'] = df_serp['text'].str.lower()

    for char in spec_chars:
        df_serp['text'] = df_serp['text'].str.replace(char, ' ', regex=True)

    # remove stop words
    df_serp['text'] = df_serp['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    my_l = df_serp['text']
    my_l = [words for segments in my_l for words in segments.split()]
    my_str = " ".join(my_l)

    df_ngrams = pd.DataFrame({'text': [my_str]})


    def find_ngrams(input_list, n):
        return list(zip(*[input_list[i:] for i in range(n)]))


    df_ngrams['bigrams'] = df_ngrams['text'].map(lambda x: find_ngrams(x.split(" "), 2))

    # store the data
    bigram_list = []

    bigrams = df_ngrams['bigrams'].tolist()

    for bi in bigrams:
        bigrams = list(chain(*bigrams))
        bigrams = [(x.lower(), y.lower()) for x, y in bi]
        bigram_counts = Counter(bigrams)
        common = bigram_counts.most_common(10)
        bigram_list.append(common)

    # # make the ngram dataframe
    df_ngrams = pd.DataFrame(None)
    df_ngrams['bigrams'] = bigram_list
    df_ngrams = df_ngrams.explode("bigrams")
    df_ngrams['bigrams'] = df_ngrams['bigrams'].astype(str)

    df_serp_display = df_serp[["title", "url"]]
    df_title_display = df_title[["keyword", "frequency"]]

    st.success("Analysis complete!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Content Bigrams")
        st.write("Most common word pairs from page content")
        st.dataframe(df_ngrams)

    with col2:
        st.subheader("Title Keywords")
        st.write("Words appearing in multiple titles")
        st.dataframe(df_title_display)

    with col3:
        st.subheader("Extracted Titles")
        st.write("Page titles from ranking URLs")
        st.dataframe(df_serp_display)

    # Download buttons
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        csv_ngrams = df_ngrams.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Bigrams", csv_ngrams, "bigrams.csv", "text/csv")

    with col2:
        csv_titles = df_title_display.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Title Keywords", csv_titles, "title_keywords.csv", "text/csv")

    with col3:
        csv_serp = df_serp_display.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download SERP Titles", csv_serp, "serp_titles.csv", "text/csv")
