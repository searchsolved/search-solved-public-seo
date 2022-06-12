import streamlit as st

# region format
st.set_page_config(page_title="Keyword Difficulty Finder", page_icon="🔎",
                   layout="wide")  # needs to be the first thing after the streamlit import

from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import pandas as pd
import requests
from io import BytesIO
import chardet
from stqdm import stqdm

st.write(
    "Made in [![this is an image link](https://i.imgur.com/iIOA6kU.png)](https://www.streamlit.io/) with :heart: by [@LeeFootSEO](https://twitter.com/LeeFootSEO)")
st.title("Keyword Difficulty Finder")

# streamlit variables
uploaded_file = st.file_uploader("Upload your Keyword report")

value_serp_key = st.sidebar.text_input('Enter your ValueSERP Key')  #good
device = st.sidebar.radio("Select the device to search Google", ('Mobile', 'Desktop', 'Tablet'))
location_select = st.sidebar.selectbox('Select the location to search Google', ('United States', 'United Kingdom', 'Australia', 'India', 'Spain', 'Italy', 'Canada', 'Germany', 'Ireland', 'France', 'Holland'))
threads = st.sidebar.slider("Set number of threads", value=10, min_value=1, max_value=20)
common_urls = st.sidebar.slider("Set number of common urls to match", value=3, min_value=2, max_value=5)
max_diff = st.sidebar.slider("Set maximum Keyword difficulty", value=10, min_value=0, max_value=99)
filter_questions = st.sidebar.checkbox('Select only question keywords?', value=False)

if uploaded_file is not None:

    try:

        result = chardet.detect(uploaded_file.getvalue())
        encoding_value = result["encoding"]

        if encoding_value == "UTF-16":
            white_space = True
        else:
            white_space = False

        df_comp = pd.read_csv(uploaded_file, encoding=encoding_value, delim_whitespace=white_space, on_bad_lines='skip')
        number_of_rows = len(df_comp)

        if number_of_rows == 0:
            st.caption("Your sheet seems empty!")

        with st.expander("↕ View raw data", expanded=False):
            st.write(df_comp)

    except UnicodeDecodeError:
        st.warning("""🚨 The file doesn't seem to load. Check the filetype, file format and Schema""")

else:
    st.info("👆 Upload a .csv or .txt file first.")
    st.stop()

with st.form(key='columns_in_form_2'):
    st.subheader("Please Select the Keyword Column")
    kw_col = st.selectbox('Select the Keyword column:', df_comp.columns)

    submitted = st.form_submit_button('Submit')

if submitted:
    df_comp.rename(columns={kw_col: "Keyword"}, inplace=True)
    if 'Difficulty' in df_comp.columns:
        df_comp['Difficulty'] = df_comp['Difficulty'].fillna("0").astype(int)
        df_comp = df_comp[df_comp.Difficulty <= max_diff]

    if filter_questions:
        q_words = "who |what |where |why |when |how |is |are |does |do |can "
        df_comp = df_comp[df_comp['Keyword'].str.contains(q_words)]

    # make the initial keyword list
    kws = list(set(df_comp['Keyword']))
    with st.expander("↕ View keywords to process", expanded=False):
        st.write(kws)
    st.info("Getting SERP Data, this may take some time! Small batches are recommended!")

    # create allintitle and phrase matches
    df_comp['allintitle_kw_temp'] = "allintitle: " + df_comp['Keyword']
    df_comp['quoted_kw_temp'] = '"' + df_comp['Keyword'] + '"'

    # extend kw list with allintitle and phrase matches
    kws.extend(df_comp['allintitle_kw_temp'].tolist())
    kws.extend(df_comp['quoted_kw_temp'].tolist())

    # delete the helper columns
    del df_comp['allintitle_kw_temp']
    del df_comp['quoted_kw_temp']

    # store the main data
    search_q = []
    total_results = []

    # store the serp cluster df data
    link_l = []
    query_padded_len = []

    counter = 0

    def get_serp(kw):
        global counter
        counter += 1
        global list_counter
        print(f'Searching Google for: {kw}', "(", counter, "of", len(kws), ")")

        params = {
            'api_key': value_serp_key,
            'q': kw,
            'location': location_select,
            'include_fields': ['organic_results', 'search_information'],
            'location_auto': True,
            'device': "Desktop",
            'output': 'json',
            'page': '1',
            'num': '10'
        }

        response = requests.get('https://api.valueserp.com/search', params)
        response_data = json.loads(response.text)
        result = response_data.get('search_information')
        total_results.append((result['total_results']))
        search_q.append(kw)
        query_padded_len.append(kw)

        result_links = response_data.get('organic_results')

        for var in result_links:
            try:
                link_l.append(var['link'])
            except Exception:
                link_l.append("")

            max_list_len = max(len(link_l), len(query_padded_len))

            if len(query_padded_len) != max_list_len:
                diff = max_list_len - len(query_padded_len)
                query_padded_len.extend([kw] * diff)

    # use stqdm with concurrent futures
    try:
        with stqdm(total=len(kws)) as pbar:
            with ThreadPoolExecutor(max_workers=threads) as ex:
                futures = [ex.submit(get_serp, url) for url in kws]
                for future in as_completed(futures):
                    result = future.result()
                    pbar.set_description("Searching Keywords: ")
                    pbar.update(1)
    except Exception:
        pass



    print("Finished searching!")
    df_results = pd.DataFrame(None)
    df_results['Keyword'] = search_q
    df_results['Total Results'] = total_results

    # make two dfs from the results df to merge back into the main df vlookup style
    try:
        df_results_allintitle = df_results[df_results['Keyword'].str.contains("allintitle")].copy()
    except AttributeError:
        st.warning("No Matching Keywords After Filtering Options Set. Please Check and Try Again!")
        st.stop()
    df_results_phrase = df_results[df_results['Keyword'].str.contains('"')].copy()

    # rename the column names ready for the vlookup style match
    df_results.rename(columns={"Total Results": "Search Results"}, inplace=True)
    df_results_phrase.rename(columns={"Total Results": "Quoted Results"}, inplace=True)
    df_results_allintitle.rename(columns={"Total Results": "Allintite Results"}, inplace=True)

    # remove the keyword modifiers for matching
    df_results_phrase['Keyword'] = df_results_phrase['Keyword'].apply(lambda x: x.replace('\"', ''))
    df_results_allintitle['Keyword'] = df_results_allintitle['Keyword'].apply(lambda x: x.replace('allintitle: ', ''))

    # do the vlookup merge
    df_comp = pd.merge(df_comp, df_results[['Keyword', 'Search Results']], on='Keyword', how='left')
    df_comp = pd.merge(df_comp, df_results_phrase[['Keyword', 'Quoted Results']], on='Keyword', how='left')
    df_comp = pd.merge(df_comp, df_results_allintitle[['Keyword', 'Allintite Results']], on='Keyword', how='left')

    # make the cluster df
    df_cluster = pd.DataFrame(None)
    df_cluster["keyword"] = query_padded_len
    df_cluster["link"] = link_l

    # remove the keyword modifiers before clustering
    df_cluster['keyword'] = df_cluster['keyword'].apply(lambda x: x.replace('\"', ''))
    df_cluster['keyword'] = df_cluster['keyword'].apply(lambda x: x.replace('allintitle: ', ''))
    df_cluster.drop_duplicates(subset=["keyword", "link"], keep="first", inplace=True)

    # -------------- start cluster logic
    # sort and group kws, drop any under min threshold
    df_cluster_cluster = df_cluster.sort_values(by="keyword", ascending=True)
    df_cluster = df_cluster.groupby('link')['keyword'].apply(','.join).reset_index()
    df_cluster["temp_count"] = (df_cluster["keyword"].str.count(",") + 1)
    df_cluster = df_cluster[df_cluster["temp_count"] >= common_urls]

    # keep only the longest version of each keyword group
    list1 = df_cluster['keyword']
    substrings = {w1 for w1 in list1 for w2 in list1 if w1 in w2 and w1 != w2}
    longest_str = set(list1) - substrings
    longest_str = list(longest_str)

    df_cluster = df_cluster[df_cluster['keyword'].isin(longest_str)]

    # drop any cluster groups under the minimum threshold
    df_cluster['temp_count'] = df_cluster['keyword'].map(df_cluster.groupby('keyword')['keyword'].count())
    df_cluster = df_cluster[df_cluster["temp_count"] >= common_urls]

    # make the cluster column & delete helper column
    df_cluster['serp_cluster'] = df_cluster['keyword']

    # explode the keyword list
    df_cluster['keyword'] = df_cluster['keyword'].str.split(',')
    df_cluster = df_cluster.explode('keyword')

    df_cluster['temp_count'] = df_cluster['keyword'].astype(str).map(len)
    df_cluster = df_cluster.sort_values(by="temp_count", ascending=True)
    df_cluster['serp_cluster'] = df_cluster.groupby('serp_cluster')['keyword'].transform('first')
    df_cluster.sort_values(['serp_cluster', "keyword"], ascending=[True, True], inplace=True)

    del df_cluster['temp_count']
    df_cluster.drop_duplicates(subset="keyword", inplace=True)

    # clean the final df
    try:
        df_cluster['keyword'] = df_cluster['keyword'].str.strip()
        df_cluster.drop_duplicates(subset=["keyword", "serp_cluster"], keep="first", inplace=True)
        df_cluster.rename(columns={"keyword": "Keyword", "serp_cluster": "Serp Cluster"}, inplace=True)
        df_comp = pd.merge(df_comp, df_cluster[['Keyword', 'Serp Cluster']], on='Keyword', how='left')
        df_comp['Serp Cluster'] = df_comp['Serp Cluster'].fillna("zzz_no_cluster")
    except AttributeError:
        df_comp['Serp Cluster'] = "zzz_no_cluster"

    # pop the columns to the front
    col = df_comp.pop('Allintite Results')
    df_comp.insert(0, col.name, col)
    df_comp = df_comp.sort_values(by="Allintite Results", ascending=True)

    col = df_comp.pop('Quoted Results')
    df_comp.insert(0, col.name, col)
    df_comp = df_comp.sort_values(by="Quoted Results", ascending=True)

    col = df_comp.pop('Search Results')
    df_comp.insert(0, col.name, col)
    df_comp = df_comp.sort_values(by="Search Results", ascending=True)

    col = df_comp.pop('Keyword')
    df_comp.insert(0, col.name, col)
    df_comp = df_comp.sort_values(by="Keyword", ascending=True)

    col = df_comp.pop('Serp Cluster')
    df_comp.insert(0, col.name, col)

    # check if cluster size = 1 and overwrite with no_cluster
    df_comp['cluster_count'] = df_comp['Serp Cluster'].map(df_comp.groupby('Serp Cluster')['Serp Cluster'].count())

    df_comp.loc[df_comp["cluster_count"] == 1, "Serp Cluster"] = "zzz_no_cluster"
    del df_comp['cluster_count']
    df_comp = df_comp.sort_values(by="Serp Cluster", ascending=True)
    df_comp.fillna({"Search Results": 0, "Quoted Results": 0, "Allintite Results": 0}, inplace=True)
    # make question dataframe ------------------------------------------------------------------------------------------
    q_words_filter = "who |what |where |why |when |how |is |are |does |do |can "
    df_questions = df_comp[df_comp['Keyword'].str.contains(q_words_filter)]

    # save to Excel sheet
    dfs = [df_comp, df_questions]  # make a list of the dataframe to use as a sheet

    # Function to save all dataframes to one single excel
    @st.cache
    def dfs_tabs(df_list, sheet_list, file_name):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        for dataframe, sheet in zip(df_list, sheet_list):
            dataframe.to_excel(writer, sheet_name=sheet, startrow=0, startcol=0, index=False)
        writer.save()
        processed_data = output.getvalue()
        return processed_data

    # list of sheet names
    sheets = ['Competitive Analysis', 'Questions Only']

    df_xlsx = dfs_tabs(dfs, sheets, 'competitor_analysis.xlsx')
    st.download_button(label='📥 Download Current Result',
                       data=df_xlsx,
                       file_name='df_serp_difficulty_checks.xlsx')
