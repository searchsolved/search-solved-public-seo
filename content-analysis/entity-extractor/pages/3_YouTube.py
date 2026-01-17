####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                                              #
####################################################################################

import streamlit as st

st.set_page_config(page_title="YouTube Entity Extractor", page_icon="movie_camera",
                   layout="wide")

import re
import pandas as pd
from dandelion import DataTXT
from youtube_transcript_api import YouTubeTranscriptApi

st.write("Made in [![this is an image link](https://i.imgur.com/iIOA6kU.png)](https://www.streamlit.io/) by [@leefootseo](https://bsky.app/profile/leefootseo.bsky.social)")

st.title("YouTube Entity Extractor")

# streamlit variables
api_key = st.sidebar.text_input('Please enter your Dandelion API Key')
accuracy = st.sidebar.slider("Set Entity Accuracy", min_value=10, max_value=100, value=80)

# clean the input data
accuracy = accuracy / 100

# store the YouTube Data
sub_titles = []

# store the entity data
entity = []
confidence = []
title = []
wiki_url = []
categories = []


with st.form(key='columns_in_form_2'):
    yt = st.text_input('Please Paste in a YouTube URL')
    submitted = st.form_submit_button('Submit')

if submitted:

    # strip out the garbage from the urls
    yt = re.sub(r'^.*?=', '=', yt)
    yt = re.sub(r'\&.*', '&', yt)
    yt = yt.replace("&", "")
    yt = yt.replace("=", "")

    try:
        srt = YouTubeTranscriptApi.get_transcript(yt)
    except Exception:
        st.info("No Transcript Available to Process. Try Another Video!")
        st.stop()

    for text in srt:
        sub_titles.append(text['text'])

    text = " ".join(sub_titles)

    try:
        datatxt = DataTXT(token=api_key)
    except Exception:
        st.write("Please Check API Key! Visit: https://dandelion.eu/ for a Free Key (1,000 Credits per day)")
        st.stop()

    response = datatxt.nex(text)

    for annotation in response.annotations:
        entity.append(annotation['spot'])
        confidence.append(annotation['confidence'])
        title.append(annotation['title'])
        wiki_url.append(annotation['uri'])
        categories.append(annotation['label'])

    df = pd.DataFrame(None)
    df['entity'] = title
    df['confidence'] = confidence
    df['category'] = categories
    df['wiki_url'] = wiki_url

    # drop duplicates
    df['entity'] = df['entity'].str.lower()
    df['category'] = df['category'].str.lower()

    df = df[df.confidence >= accuracy]
    df['# of mentions'] = df['entity'].map(df.groupby('entity')['entity'].count())
    df.drop_duplicates(subset=['entity'], keep="first", inplace=True)
    df.sort_values(["# of mentions", "entity"], ascending=[False, True], inplace=True)

    def convert_df(df):
        return df.to_csv().encode('utf-8')

    csv = convert_df(df)

    st.download_button(
        label="Download Your Entities!",
        data=csv,
        file_name='extracted_entities.csv',
        mime='text/csv')
