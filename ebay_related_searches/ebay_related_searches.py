import streamlit as st

st.set_page_config(
    page_title="eBay Related Search Scraper by LeeFootSEO",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)
from streamlit_echarts import st_echarts
from stqdm import stqdm
import pandas as pd
from bs4 import BeautifulSoup
import requests
from user_agent2 import (generate_user_agent)

# set fake agent
ua = generate_user_agent(navigator="chrome")
header = {'User-Agent': str(ua)}

css = "body > div.srp-main.srp-main--isLarge > div.s-answer-region.s-answer-region-above-river > div"
url = "/sch/i.html?_nkw="

# store the data
related_search_kws = []
source_kws = []
final_kws = []

st.title("eBay Related Search Scraper")
st.subheader("Get Related Searches from Ebay")
st.write(
    "An app which visualises related searches from eBay")
st.write(
"[![this is an image link](https://i.imgur.com/Ex8eeC2.png)](https://www.patreon.com/leefootseo) [Become a Patreon for Early Access, Support & More!](https://www.patreon.com/leefootseo)  |  Made in [![this is an image link](https://i.imgur.com/iIOA6kU.png)](https://www.streamlit.io/) by [@LeeFootSEO](https://twitter.com/LeeFootSEO)")
st.write("")

with st.form(key='columns_in_form_2'):
    seed_keyword = st.text_input('Enter the Keyword to Search eBay')
    submitted = st.form_submit_button('Submit')
    ccTLD = st.selectbox(
        'Select Which ccTLD to Search',
        ('.com', '.co.uk', '.de', '.es', '.fr', '.nl'))

if submitted:

    response = requests.get("http://www.ebay" + ccTLD + url + seed_keyword, headers=header)
    soup = BeautifulSoup(response.text, "html.parser")
    for related in soup.select(css):
        result_str = related.get_text(separator=' ')
        result_str = result_str.replace("  ", "@")
        result_str = result_str.replace(" ", ",")
        result_str = result_str.replace("@", " ")
        result_str = result_str.replace("Related:,", "")
        related_search_kws = result_str.split(",")

    # second loop
    st.write("Searching eBay for Related keywords")
    for i in stqdm(related_search_kws):
        print(i)
        response = requests.get("http://www.ebay" + ccTLD + url + i)
        soup = BeautifulSoup(response.text, "html.parser")
        for related in soup.select(css):
            result_str = related.get_text(separator=' ')
            result_str = result_str.replace("  ", "@")
            result_str = result_str.replace(" ", ",")
            result_str = result_str.replace("@", " ")
            result_str = result_str.replace("Related:,", "")

            source_kws.append(i)
            final_kws.append(result_str)

    df = pd.DataFrame(None)
    df['seed_keyword'] = source_kws
    df['related_searches'] = final_kws

    try:
        df['related_searches'] = df['related_searches'].str.split(',')
    except Exception:
        st.info("Error: No Related Searches Were Found, Try a Broader Keyword!")
        st.stop()

    df = df.explode('related_searches').reset_index(drop=True)


    def visualize_autocomplete(df_autocomplete_full):
        df_autocomplete_full['Keyword'] = seed_keyword

        for query in df_autocomplete_full['Keyword'].unique():
            df_autocomplete_full = df_autocomplete_full[df_autocomplete_full['Keyword'] == query]
            children_list = []
            children_list_level_1 = []

            for int_word in df_autocomplete_full['seed_keyword']:
                q_lv1_line = {"name": int_word}
                if not q_lv1_line in children_list_level_1:
                    children_list_level_1.append(q_lv1_line)

                children_list_level_2 = []

                for query_2 in df_autocomplete_full[df_autocomplete_full['seed_keyword'] == int_word][
                    'related_searches']:
                    q_lv2_line = {"name": query_2}
                    children_list_level_2.append(q_lv2_line)

                level2_tree = {'name': int_word, 'children': children_list_level_2}

                if not level2_tree in children_list:
                    children_list.append(level2_tree)

                tree = {'name': query, 'children': children_list}

                opts = {
                    "backgroundColor": "#F0F2F6",

                    "title": {
                        "x": 'center',
                        "y": 'top',
                        "top": "5%",

                        "textStyle": {
                            "fontSize": 22,

                        },
                        "subtextStyle": {
                            "fontSize": 15,
                            "color": '#2ec4b6',

                        },
                    },

                    "series": [
                        {
                            "type": "tree",
                            "data": [tree],
                            "layout": "radial",
                            "top": "10%",
                            "left": "25%",
                            "bottom": "5%",
                            "right": "25%",
                            "symbolSize": 20,
                            "itemStyle": {
                                "color": '#2ec4b6',
                            },
                            "label": {
                                "fontSize": 14,
                            },

                            "expandAndCollapse": True,
                            "animationDuration": 550,
                            "animationDurationUpdate": 750,
                        }
                    ],
                }
            st.header(f"eBay Related Searches for: {query}")
            st.caption("Right mouse click to save as image.")
            st_echarts(opts, key=query, height=1700)


    # add download button
    def convert_df(df):  # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')


    csv = convert_df(df)

    st.download_button(
        label="📥 Download your report!",
        data=csv,
        file_name='ebay_related_searches.csv',
        mime='text/csv', )

    # visualisation
    visualize_autocomplete(df)
