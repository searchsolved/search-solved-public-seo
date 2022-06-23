import streamlit as st

# region format
st.set_page_config(page_title="BERT Semantic Interlinking App", page_icon="🔗",
                   layout="wide")  # needs to be the first thing after the streamlit import

from io import BytesIO
from streamlit_echarts import st_echarts
from urllib.parse import urlparse
import chardet
import pandas as pd
from sentence_transformers import SentenceTransformer, util

finish = False

beta_limit = 10000

@st.cache(allow_output_mutation=True)
def get_model():
    #model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')  # highest semantic scoring card
    # model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

    return model


model = get_model()

st.write(
    "Made in [![this is an image link](https://i.imgur.com/iIOA6kU.png)](https://www.streamlit.io/) by [@LeeFootSEO](https://twitter.com/LeeFootSEO) / [![this is an image link](https://i.imgur.com/bjNRJra.png)](https://www.buymeacoffee.com/leefootseo) [Support My Work! Buy me a coffee!](https://www.buymeacoffee.com/leefootseo)")

st.title("BERT Semantic Interlinking Tool")
st.subheader("Upload a crawl film to find semantically relevant pages to interlink. (Beta limited to 10,000 rows)")
accuracy_slide = st.sidebar.slider("Set Cluster Accuracy: 0-100", value=75)
min_cluster_size = st.sidebar.slider("Set Minimum Cluster Size: 0-100", value=2)
source_filter = st.sidebar.text_input('Filter Source URL Type')
destination_filter = st.sidebar.text_input('Filter Destination URL Type')
min_similarity = accuracy_slide / 100

uploaded_file = st.file_uploader(
    "Upload your crawl file",
    help="""Upload a Screaming Frog internal_html.csv file""")

if uploaded_file is not None:

    try:

        result = chardet.detect(uploaded_file.getvalue())
        encoding_value = result["encoding"]

        if encoding_value == "UTF-16":
            white_space = True
        else:
            white_space = False

        df = pd.read_csv(
            uploaded_file,
            encoding=encoding_value,
            delim_whitespace=white_space,
            error_bad_lines=False,
        )

        number_of_rows = len(df)

        if number_of_rows > beta_limit:
            df = df[:beta_limit]
            st.caption("🚨 Imported rows over the beta limit, limiting to first " + str(beta_limit) + " rows.")

        if number_of_rows == 0:
            st.caption("Your sheet seems empty!")

        with st.expander("↕️ View raw data", expanded=False):
            st.write(df)

    except UnicodeDecodeError:
        st.warning(
            """
            🚨 The file doesn't seem to load. Check the filetype, file format and Schema

            """
        )

else:
    st.stop()

with st.form(key='columns_in_form_2'):
    st.subheader("Please Select the Column to Match (Recommend H1 / Title or Extracted Content)")
    kw_col = st.selectbox('Select the keyword column:', df.columns)
    submitted = st.form_submit_button('Submit')
    if submitted:
        df[kw_col] = df[kw_col].str.encode('ascii', 'ignore').str.decode('ascii')
        df.drop_duplicates(subset=kw_col, inplace=True)
        st.info("Finding Interlinking Opportunities, This May Take a While! Please Wait!")

        # store the data
        cluster_name_list = []
        corpus_sentences_list = []
        df_all = []

        corpus_set = set(df[kw_col])
        corpus_set_all = corpus_set

        cluster = True

        while cluster:

            corpus_sentences = list(corpus_set)
            check_len = len(corpus_sentences)
            corpus_embeddings = model.encode(corpus_sentences, batch_size=256, show_progress_bar=True,
                                             convert_to_tensor=True)
            clusters = util.community_detection(corpus_embeddings, min_community_size=2, threshold=0.70,
                                                init_max_size=len(corpus_embeddings))

            for keyword, cluster in enumerate(clusters):
                for sentence_id in cluster[0:]:
                    corpus_sentences_list.append(corpus_sentences[sentence_id])
                    cluster_name_list.append("Cluster {}, #{} Elements ".format(keyword + 1, len(cluster)))

            df_new = pd.DataFrame(None)
            df_new['source_h1'] = cluster_name_list
            df_new[kw_col] = corpus_sentences_list

            df_all.append(df_new)
            have = set(df_new[kw_col])

            corpus_set = corpus_set_all - have
            remaining = len(corpus_set)

            if check_len == remaining:
                break

        df_new = pd.concat(df_all)
        df = df.merge(df_new.drop_duplicates(kw_col), how='left', on=kw_col)

        # ------------------------------ rename the clusters to the shortest keyword -----------------------------------

        df['length'] = df[kw_col].astype(str).map(len)
        df = df.sort_values(by="length", ascending=True)
        df['source_h1'] = df.groupby('source_h1')[kw_col].transform('first')
        df.sort_values(['source_h1', kw_col], ascending=[True, True], inplace=True)
        df['source_h1'] = df['source_h1'].fillna("zzz_no_cluster")
        del df['length']

        col = df.pop(kw_col)
        df.insert(0, col.name, col)
        col = df.pop('source_h1')
        df.insert(0, col.name, col)
        df2 = df[["Address", kw_col]].copy()
        df2.rename(columns={"Address": "source_url", kw_col: "source_h1"}, inplace=True)

        df2 = df2.loc[:, ~df2.columns.duplicated()].copy()
        if 'source_url' not in df2.columns:
            df2['source_url'] = df2['source_h1']

        df = df.merge(df2.drop_duplicates('source_h1'), how='left', on="source_h1")  # merge on first instance only
        df = df[["source_url", "source_h1", "Address", kw_col]]
        try:
            df.drop_duplicates(subset=["Address", "source_url"], keep="first", inplace=True)
        except AttributeError:
            st.warning("No Results Found! Try Matching on a Different Column! (Recommend H1 or Extracted Content)")
            st.stop()

        try:
            df = df[df["Address"].str.contains(destination_filter, na=False)]
        except AttributeError:
            st.warning("No Results Found! Try Matching on a Different Column! (Recommend H1 / Title or Extracted Content)")
            st.stop()

        df = df[df["source_url"].str.contains(source_filter, na=False)]

        df = df[~df["Address"].str.contains("zzz_no_cluster", na=False)]
        df.rename(columns={"Address": "destination_url", kw_col: "destination_url_h1"}, inplace=True)
        df['source_h1'] = df['source_h1'].str.lower()
        df['destination_url_h1'] = df['destination_url_h1'].str.lower()
        df['check'] = df['source_url'] == df['destination_url']
        df = df[~df["check"].isin([True])]
        del df['check']
        finish = True

# make excel output and visualise results ------------------------------------------------------------------------------
if finish == True:

    df_list = []
    sheet_list = []

    # clean special characters
    spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
                  "*", "+", ",", ".", "/", ":", ";", "<",
                  "=", ">", "?", "@", "[", "\\", "]", "^",
                  "`", "{", "|", "}", "~", "–"]

    df['source_h1'] = df['source_h1'].str.encode('ascii', 'ignore').str.decode('ascii')

    # make the dataframe for visualisation
    df_autocomplete_full = df.copy()

    # extracts the domain from the address column if present
    extracted_domain = df['source_url'].iloc[0]
    url = extracted_domain
    o = urlparse(url)
    domain = o.netloc
    df_autocomplete_full['seed'] = domain

    filt = list(set(df['source_h1']))

    df_list.append(df)
    sheet_list.append("All Results")

    for i in filt:

        worksheet_name = i.replace(" ", "_")
        for char in spec_chars:
            worksheet_name = worksheet_name.replace(char, "")
            worksheet_name = worksheet_name.replace("  ", "_")

        worksheet_name = worksheet_name[0:31]
        sheet_list.append(worksheet_name)
        df_list.append(df[df['source_h1'].str.contains(i)].copy())

    # save to Excel sheet
    def dfs_tabs(df_list, sheet_list, file_name):  # function to save all dataframes to one single excel doc

        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        for dataframe, sheet in zip(df_list, sheet_list):
            dataframe.to_excel(writer, sheet_name=sheet, startrow=0, startcol=0, index=False)

        writer.save()
        processed_data = output.getvalue()
        return processed_data

    df_xlsx = dfs_tabs(df_list, sheet_list, 'serp-cluster-output.xlsx')
    st.download_button(label='📥 Download BERT Interlinking Opportunities', data=df_xlsx, file_name='bert_interlinking_opportunities.xlsx')

    # visualise result -----------------------------------------------------------------------------------------------------
    def visualize_autocomplete(df_autocomplete_full):

        query = df_autocomplete_full['seed'].iloc[0]

        for query in df_autocomplete_full['seed'].unique():
            df_autocomplete_full = df_autocomplete_full[df_autocomplete_full['seed'] == query]
            children_list = []
            children_list_level_1 = []

            for int_word in df_autocomplete_full['source_h1']:
                q_lv1_line = {"name": int_word}
                if not q_lv1_line in children_list_level_1:
                    children_list_level_1.append(q_lv1_line)

                children_list_level_2 = []

                for query_2 in df_autocomplete_full[df_autocomplete_full['source_h1'] == int_word][
                    'destination_url_h1']:
                    q_lv2_line = {"name": query_2}
                    children_list_level_2.append(q_lv2_line)

                level2_tree = {'name': int_word, 'children': children_list_level_2}

                if not level2_tree in children_list:
                    children_list.append(level2_tree)

                tree = {'name': query, 'children': children_list}

                opts = {
                    "backgroundColor": "#F0F2F6",


                    "title": {
                        # "subtext": "https://tools.alekseo.com/askey.html",
                        # "text": f"Questions Map for: «{query}»",
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
            st.caption("Right mouse click to save as image.")
            st_echarts(opts, key=query, height=1700)

    st.header("Visualising First 100 Results")
    df_autocomplete_full = df_autocomplete_full[:100]
    visualize_autocomplete(df_autocomplete_full)
