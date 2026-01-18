####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                                              #
####################################################################################

import streamlit as st

st.set_page_config(page_title="Keyword Entity Extractor", page_icon="mag",
                   layout="wide")

import chardet
import pandas as pd
from dandelion import DataTXT
from stqdm import stqdm

st.title("Keyword Entity Extractor")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-181717?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Batch entity extraction from CSV data
    - Processes multiple text entries at once
    - Exports entity analysis results

    **How to use:**
    1. Upload a CSV with text content
    2. Select the text column to analyze
    3. Click "Extract Entities"
    4. Download results with entities

    **Best for:**
    - Bulk content analysis
    - Large-scale entity audits
    - Content inventory enrichment
    """)

# streamlit variables
uploaded_file = st.file_uploader("Upload your .csv list of keywords / Crawl file")
api_key = st.sidebar.text_input('Please enter your Dandelion API Key')
accuracy = st.sidebar.slider("Set Entity Accuracy", min_value=10, max_value=100, value=80)
accuracy = accuracy / 100

# store the data
entity = []
confidence = []
title = []
wiki_url = []
categories = []
url_list = []

if uploaded_file is not None:

    try:

        result = chardet.detect(uploaded_file.getvalue())
        encoding_value = result["encoding"]
        if encoding_value == "UTF-16":
            white_space = True
        else:
            white_space = False
        df = pd.read_csv(uploaded_file, encoding=encoding_value, delim_whitespace=white_space, on_bad_lines='skip')

        number_of_rows = len(df)

        if number_of_rows == 0:
            st.caption("Your sheet seems empty!")
        with st.expander("View raw data", expanded=False):
            st.write(df)
    except UnicodeDecodeError:
        st.warning("""The file doesn't seem to load. Check the filetype, file format and Schema""")

else:
    st.info("Upload a .csv or .txt file first.")
    st.stop()

with st.form(key='columns_in_form_2'):
    st.subheader("Select The Keyword & URL Columns")
    kw_col = st.selectbox('Select the column containing your KEYWORDS:', df.columns)
    url_col = st.selectbox('Select the column containing your URL:', df.columns)
    submitted = st.form_submit_button('Submit')

if submitted:

    df = df[df[kw_col].notna()]  # drop missing values
    df = df[df[url_col].notna()]  # drop missing values
    df[kw_col] = df[kw_col].astype(str)
    df[url_col] = df[url_col].astype(str)
    df.columns = df.columns.str.strip()
    df.rename(columns={kw_col: "Keyword", url_col: "Current URL"}, inplace=True)
    unique_urls = list(set(df['Current URL']))

    st.write("Unique URLs: ", len(unique_urls))

    for url in stqdm(unique_urls):
        try:
            df_url = df[df['Current URL'].str.contains(url)]
            string = df_url["Keyword"].str.cat(sep=', ')

            datatxt = DataTXT(token=api_key)
            response = datatxt.nex(string)

            for annotation in response.annotations:
                entity.append(annotation['spot'])
                confidence.append(annotation['confidence'])
                title.append(annotation['title'])
                wiki_url.append(annotation['uri'])
                categories.append(annotation['label'])
                url_list.append(url)

        except Exception:
            pass

    df = pd.DataFrame(None)
    df['url'] = url_list
    df['entity'] = entity
    df['confidence'] = confidence
    df['title'] = title
    df['category'] = categories
    df['wiki_url'] = wiki_url
    df = df[df.confidence >= accuracy]

    # drop duplicates
    df.drop_duplicates(subset=["url", "entity"], keep="first", inplace=True)

    def convert_df(df):
        return df.to_csv().encode('utf-8')

    csv = convert_df(df)

    st.download_button(
        label="Download Your Entities!",
        data=csv,
        file_name='extracted_entities.csv',
        mime='text/csv')
