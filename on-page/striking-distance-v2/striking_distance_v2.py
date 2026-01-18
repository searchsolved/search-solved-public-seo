####################################################################################
#                                                                                  #
#  Striking Distance Creator V2                                                    #
#                                                                                  #
#  Find keywords close to ranking and check if they appear in title/H1/copy.       #
#                                                                                  #
####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                                              #
####################################################################################

from typing import Union

import chardet
import pandas as pd
import requests
import streamlit as st
import trafilatura
from bs4 import BeautifulSoup
from pandas import DataFrame, Series
from stqdm import stqdm
from trafilatura import bare_extraction
from trafilatura.settings import use_config
from user_agent2 import generate_user_agent


st.set_page_config(
    page_title="Striking Distance Creator V2 by LeeFootSEO",
    page_icon="baseball",
    layout="wide",
)
st.title("Striking Distance Creator V2")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-Follow-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Advanced striking distance analysis
    - Combines position + impression data
    - Scores opportunities by potential impact

    **How to use:**
    1. Upload GSC performance data
    2. Configure scoring weights
    3. Analyze striking distance
    4. Export prioritized opportunities

    **Best for:**
    - Data-driven optimization planning
    - ROI-based prioritization
    - Executive reporting
    """)
st.subheader("Automatic On-page Checker for Keywords Close to Ranking")
st.write("An app which blends keyword data and crawl data to provide actionable insights for keywords close to ranking.")
st.write("Made in [![this is an image link](https://i.imgur.com/iIOA6kU.png)](https://www.streamlit.io/) by [@leefootseo](https://bsky.app/profile/leefootseo.bsky.social)")
st.write("")

# set fake agent
ua = generate_user_agent(navigator="chrome")
header = {'User-Agent': str(ua)}

# set slider values
min_position = st.sidebar.slider("Set The Min Position Cutoff", value=4, max_value=10)
max_position = st.sidebar.slider("Set The Max Position Cutoff", value=20, max_value=20)
min_volume = st.sidebar.slider("Set Minimum Search Volume", value=50, max_value=500000, step=50)

# file uploader
uploaded_file = st.file_uploader("Upload your keyword file", help="""Upload a keyword report from ahrefs / semrush etc""")

if uploaded_file is not None:

    try:

        result = chardet.detect(uploaded_file.getvalue())
        encoding_value = result["encoding"]

        if encoding_value == "UTF-16":
            white_space = True
        else:
            white_space = False

        df_keywords = pd.read_csv(
            uploaded_file,
            encoding=encoding_value,
            delim_whitespace=white_space,
            on_bad_lines='skip',
        )

        number_of_rows = len(df_keywords)

        if number_of_rows == 0:
            st.caption("Your sheet seems empty!")
            st.stop()

        with st.expander("View raw data", expanded=False):
            st.write(df_keywords)

    except UnicodeDecodeError:
        st.warning(
            """
            The file doesn't seem to load. Check the filetype, file format and Schema
            """
        )

else:
    st.stop()

with st.form(key='columns_in_form_2'):
    st.subheader("Please Select The Keyword, URL, Volume and Position Columns")

    kw_col = st.selectbox('Select the keyword column:', df_keywords.columns)
    url_col = st.selectbox('Select the url column:', df_keywords.columns)
    vol_col = st.selectbox('Select the volume column:', df_keywords.columns)
    pos_col = st.selectbox('Select the position column:', df_keywords.columns)

    submitted = st.form_submit_button('Submit')

if submitted:

    df_keywords.rename(columns={url_col: "Current URL", kw_col: "Keyword", vol_col: "Volume", pos_col: "Current position"}, inplace=True)

    # clean the data
    df_keywords = df_keywords[df_keywords["Current URL"].notna()]  # remove nans
    df_keywords = df_keywords[df_keywords["Keyword"].notna()]  # remove nans
    df_keywords.drop(df_keywords.loc[df_keywords['Volume'] < min_volume].index, inplace=True)
    df_keywords.drop(df_keywords.loc[df_keywords['Current position'] <= min_position].index, inplace=True)
    df_keywords.drop(df_keywords.loc[df_keywords['Current position'] >= max_position].index, inplace=True)

    try:
        df_keywords["Volume"] = df_keywords["Volume"].str.replace("0-10", "0")
    except AttributeError:
        pass

    df_keywords = df_keywords.sort_values(by="Volume", ascending=False)  # keep the top opportunity
    df_keywords = df_keywords.astype({"Current URL": str, "Keyword": str, "Current position": int, "Volume": int})

    # make new dataframe to merge search volume back in later
    df_keyword_vol = df_keywords[["Keyword", "Volume"]]

    # store the extracted data
    text = []
    title = []
    extracted_url = []
    h1 = []

    newconfig = use_config()
    newconfig.set("DEFAULT", "EXTRACTION_TIMEOUT", "10")
    newconfig.set("DEFAULT", "USER_AGENTS", str(header))

    st.write("Extracting Content from URLs, Please Wait!")
    # loop through page 1 urls extracting content and h1s
    for url in stqdm(list(set(df_keywords['Current URL']))):

        try:

            downloaded = trafilatura.fetch_url(url)
            d = bare_extraction(downloaded, config=newconfig, with_metadata=True)

            text.append((d['text']))
            extracted_url.append(d['url'])

            # use beautiful soup to extract the h1 and page title
            response = requests.get(url, headers=header)
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')

            h1.append(soup.find('h1').text.strip())
            title.append(soup.find('title').text.strip())

        except Exception:
            pass

        # sanity check, pad lists to same length
        max_list_len = max(len(text), len(title), len(extracted_url), len(h1))

        if len(text) != max_list_len:
            diff = max_list_len - len(text)
            text.extend([""] * diff)

        if len(title) != max_list_len:
            diff = max_list_len - len(title)
            title.extend([""] * diff)

        if len(extracted_url) != max_list_len:
            diff = max_list_len - len(extracted_url)
            extracted_url.extend([""] * diff)

        if len(h1) != max_list_len:
            diff = max_list_len - len(h1)
            h1.extend([""] * diff)

    # make the dataframe with the extracted content
    df_sf = pd.DataFrame(None)
    df_sf['url'] = extracted_url
    df_sf['h1'] = h1
    df_sf['title'] = title
    df_sf['copy'] = text

    # clean ' and - from page title, h1 and copy keywords for better matching
    try:
        df_sf['title'] = df_sf['title'].str.replace("'", '')
        df_sf['title'] = df_sf['title'].str.replace("-", ' ')
        df_sf['h1'] = df_sf['h1'].str.replace("'", '')
        df_sf['h1'] = df_sf['h1'].str.replace("-", ' ')
        df_sf['copy'] = df_sf['copy'].str.replace("'", '')
        df_sf['copy'] = df_sf['copy'].str.replace("-", ' ')
    except AttributeError:
        st.info("Could Not Extract Content from the Server! Try Whitelisting, or Running this App Locally!")
        st.stop()

    # make a copy of the keywords dataframe for grouping - this ensures stats can be merged back in later
    df_keywords_group = df_keywords.copy()

    # groups the URLs (remove the dupes and combines stats)
    df_keywords_group["KWs in Striking Dist."] = 1  # used to count the number of keywords in striking distance
    df_keywords_group = (
        df_keywords_group.groupby("Current URL")
            .agg({"Volume": "sum", "KWs in Striking Dist.": "count"})
            .reset_index()
    )

    # create a new df, combine the merged data with the original data. display in adjacent rows ala grepwords
    df_merged_all_kws = df_keywords_group.merge(
        df_keywords.groupby("Current URL")["Keyword"]
            .apply(lambda x: x.reset_index(drop=True))
            .unstack()
            .reset_index()
    )

    # sort by biggest opportunity
    df_merged_all_kws = df_merged_all_kws.sort_values(
        by="KWs in Striking Dist.", ascending=False
    )

    # reindex the columns to keep just the top five keywords
    cols = "Current URL", "Volume", "KWs in Striking Dist.", 0, 1, 2, 3, 4
    df_merged_all_kws = df_merged_all_kws.reindex(columns=cols)

    # This is the Final Striking Distance DF Which Should go to a Separate Worksheet.
    df_striking: Union[Series, DataFrame, None] = df_merged_all_kws.rename(
        columns={
            "Volume": "Striking Dist. Vol",
            0: "KW1",
            1: "KW2",
            2: "KW3",
            3: "KW4",
            4: "KW5",
        }
    )

    # merges striking distance df with screaming frog df to merge in the title, h1 and category description
    df_striking = pd.merge(df_striking, df_sf, left_on="Current URL", right_on="url", how="inner")

    # set the final column order and merge the keyword data in
    cols = [
        "Current URL",
        "title",
        "h1",
        "copy",
        "Striking Dist. Vol",
        "KWs in Striking Dist.",
        "KW1",
        "KW1 Vol",
        "KW1 in Title",
        "KW1 in H1",
        "KW1 in Copy",
        "KW2",
        "KW2 Vol",
        "KW2 in Title",
        "KW2 in H1",
        "KW2 in Copy",
        "KW3",
        "KW3 Vol",
        "KW3 in Title",
        "KW3 in H1",
        "KW3 in Copy",
        "KW4",
        "KW4 Vol",
        "KW4 in Title",
        "KW4 in H1",
        "KW4 in Copy",
        "KW5",
        "KW5 Vol",
        "KW5 in Title",
        "KW5 in H1",
        "KW5 in Copy",
    ]

    # re-index the columns to place them in a logical order + inserts new blank columns for kw checks.
    df_striking = df_striking.reindex(columns=cols)


    def merger(col1, col2):
        dx = df_striking.merge(df_keyword_vol, how='left', left_on=col1, right_on=col2)
        return dx


    def volume(vol1, vol2):
        vol = df_striking[vol1] = df_striking[vol2]
        df_striking.drop(['Keyword', 'Volume'], axis=1, inplace=True)
        return vol


    df_striking = df_striking.fillna('')

    df_striking = merger("KW1", "Keyword")
    volume("KW1 Vol", "Volume")
    df_striking = merger("KW2", "Keyword")
    volume("KW2 Vol", "Volume")
    df_striking = merger("KW3", "Keyword")
    volume("KW3 Vol", "Volume")
    df_striking = merger("KW4", "Keyword")
    volume("KW4 Vol", "Volume")
    df_striking = merger("KW5", "Keyword")
    volume("KW5 Vol", "Volume")

    # replace nan values with empty strings
    df_striking = df_striking.fillna("")

    df_striking = df_striking.astype({"title": str, "h1": str, "copy": str})

    # drop the title, h1 and category description to lower case so kws can be matched to them
    df_striking["title"] = df_striking["title"].str.lower()
    df_striking["h1"] = df_striking["h1"].str.lower()
    df_striking["copy"] = df_striking["copy"].str.lower()

    # stop the script is dataframe is empty
    rows = len(df_striking)
    if rows == 0:
        st.info("Dataframe is empty! Nothing to do!")
        st.stop()

    # check whether a keyword appears in title, h1 or category description
    df_striking["KW1 in Title"] = df_striking.apply(lambda row: row["KW1"] in row["title"], axis=1)
    df_striking["KW1 in H1"] = df_striking.apply(lambda row: row["KW1"] in row["h1"], axis=1)
    df_striking["KW1 in Copy"] = df_striking.apply(lambda row: row["KW1"] in row["copy"], axis=1)

    df_striking["KW2 in Title"] = df_striking.apply(lambda row: row["KW2"] in row["title"], axis=1)
    df_striking["KW2 in H1"] = df_striking.apply(lambda row: row["KW2"] in row["h1"], axis=1)
    df_striking["KW2 in Copy"] = df_striking.apply(lambda row: row["KW2"] in row["copy"], axis=1)

    df_striking["KW3 in Title"] = df_striking.apply(lambda row: row["KW3"] in row["title"], axis=1)
    df_striking["KW3 in H1"] = df_striking.apply(lambda row: row["KW3"] in row["h1"], axis=1)
    df_striking["KW3 in Copy"] = df_striking.apply(lambda row: row["KW3"] in row["copy"], axis=1)

    df_striking["KW4 in Title"] = df_striking.apply(lambda row: row["KW4"] in row["title"], axis=1)
    df_striking["KW4 in H1"] = df_striking.apply(lambda row: row["KW4"] in row["h1"], axis=1)
    df_striking["KW4 in Copy"] = df_striking.apply(lambda row: row["KW4"] in row["copy"], axis=1)

    df_striking["KW5 in Title"] = df_striking.apply(lambda row: row["KW5"] in row["title"], axis=1)
    df_striking["KW5 in H1"] = df_striking.apply(lambda row: row["KW5"] in row["h1"], axis=1)
    df_striking["KW5 in Copy"] = df_striking.apply(lambda row: row["KW5"] in row["copy"], axis=1)

    # delete true / false values if there is no keyword
    df_striking.loc[df_striking["KW1"] == "", ["KW1 in Title", "KW1 in H1", "KW1 in Copy"]] = ""
    df_striking.loc[df_striking["KW2"] == "", ["KW2 in Title", "KW2 in H1", "KW2 in Copy"]] = ""
    df_striking.loc[df_striking["KW3"] == "", ["KW3 in Title", "KW3 in H1", "KW3 in Copy"]] = ""
    df_striking.loc[df_striking["KW4"] == "", ["KW4 in Title", "KW4 in H1", "KW4 in Copy"]] = ""
    df_striking.loc[df_striking["KW5"] == "", ["KW5 in Title", "KW5 in H1", "KW5 in Copy"]] = ""

    # drops rows if all values evaluate to true. (nothing for the user to do).
    df_striking = df_striking.drop(
        df_striking[
            (df_striking["KW1 in Title"] == True)
            & (df_striking["KW1 in H1"] == True)
            & (df_striking["KW1 in Copy"] == True)
            ].index
    )
    df_striking = df_striking.drop(
        df_striking[
            (df_striking["KW2 in Title"] == True)
            & (df_striking["KW2 in H1"] == True)
            & (df_striking["KW2 in Copy"] == True)
            ].index
    )
    df_striking = df_striking.drop(
        df_striking[
            (df_striking["KW3 in Title"] == True)
            & (df_striking["KW3 in H1"] == True)
            & (df_striking["KW3 in Copy"] == True)
            ].index
    )
    df_striking = df_striking.drop(
        df_striking[
            (df_striking["KW4 in Title"] == True)
            & (df_striking["KW4 in H1"] == True)
            & (df_striking["KW4 in Copy"] == True)
            ].index
    )
    df_striking = df_striking.drop(
        df_striking[
            (df_striking["KW5 in Title"] == True)
            & (df_striking["KW5 in H1"] == True)
            & (df_striking["KW5 in Copy"] == True)
            ].index
    )

    del df_striking['copy']


    def convert_df(df):
        return df.to_csv().encode('utf-8')


    csv = convert_df(df_striking)

    st.download_button(
        label="Download Keywords in Striking Distance",
        data=csv,
        file_name='keywords_in_striking_distance.csv',
        mime='text/csv')
