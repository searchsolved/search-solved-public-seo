import streamlit as st

finish = False
# region format
st.set_page_config(page_title="BERT Semantic Interlinking App", page_icon="🔗",
                   layout="wide")  # needs to be the first thing after the streamlit import

import time
import chardet
import pandas as pd
from sentence_transformers import SentenceTransformer, util

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

        # if remove_dupes:
        #     df.drop_duplicates(subset="Keyword", inplace=True)

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
#

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

        # -------------------------------------- download the csv ----------------------------------------------------------

if finish == True:
    st.success('Finished!')
    st.markdown("### **🎈 Download your results!**")
    st.write("")


    def convert_df(df):  # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv(index=False).encode('utf-8')


    csv = convert_df(df)

    st.download_button(
        label="📥 Download your report!",
        data=csv,
        file_name='your_interlinking_opportunities.csv',
        mime='text/csv')
