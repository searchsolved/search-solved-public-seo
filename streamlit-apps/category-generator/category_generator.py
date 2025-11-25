####################################################################################
#                                                                                  #
#  Automatic Category Page Suggester                                               #
#                                                                                  #
#  Analyze crawl data to suggest new category pages based on product inventory.    #
#                                                                                  #
####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Twitter  : https://x.com/LeeFootSEO                                              #
####################################################################################

import streamlit as st
import collections
import re
import string
import chardet
from polyfuzz import PolyFuzz

import pandas as pd
import requests
from nltk.util import ngrams
from stqdm import stqdm

st.set_page_config(
    page_title="Automatic Category Page Suggester by LeeFootSEO",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)
st.title("Automatic Category Page Suggester")
st.subheader("Automatically align your inventory with search demand.")
st.write(
    "An app which reviews opportunities to create new categories to align with search demand.")
st.write(
    "Made in [![this is an image link](https://i.imgur.com/iIOA6kU.png)](https://www.streamlit.io/) by [@LeeFootSEO](https://x.com/LeeFootSEO)")
st.write("")

# set the layout columns
col1, col2 = st.columns(2)

# set variables
min_product_match_exact = st.sidebar.slider("Minimum products to match to (Exact match): 0-100", value=3, min_value=1)
min_product_match_fuzzy = st.sidebar.slider("Minimum products to match to (Fuzzy match): 0-100", value=3, min_value=1)

min_sim_match = st.sidebar.slider("Minimum similarity to an existing category: 0-100", value=96)
min_cpc = st.sidebar.slider("Minimum CPC in $", value=0)
min_search_vol = st.sidebar.text_input('Set the minimum search volume', 100)
kwe_key = st.sidebar.text_input('Paste your Keywords Everywhere API key')

country_kwe = st.sidebar.selectbox('Set the country to pull search data from',
                                   ('us', 'uk', 'au', 'ca', 'in', 'nz', 'za'), index=0)

currency_kwe = st.sidebar.selectbox('Set the currency to use for CPC data', (
    'usd', 'gbp', 'bdt', 'bgn', 'bhd', 'bif', 'bmd', 'bnd', 'bob', 'brl', 'bsd', 'btn', 'bwp', 'byr', 'bzd', 'cad',
    'chf', 'clp',
    'cny', 'cop', 'crc', 'cup', 'cve', 'czk', 'djf', 'dkk', 'dop', 'dzd', 'eek', 'egp', 'etb', 'eur', 'fjd', 'fkp',
    'ghs', 'gmd', 'gnf', 'gtq', 'gyd', 'hkd', 'hnl', 'hrk', 'huf', 'idr', 'ils', 'inr', 'iqd', 'isk', 'jod', 'jpy',
    'kes',
    'kgs', 'khr', 'kmf', 'kpw', 'krw', 'kwd', 'kyd', 'kzt', 'lkr', 'mad', 'mdl', 'mkd', 'mmk', 'mnt', 'mop', 'mro',
    'mur',
    'mvr', 'mwk', 'mxn', 'myr', 'nad', 'ngn', 'nio', 'nok', 'npr', 'nzd', 'omr', 'pab', 'pen', 'pgk', 'php', 'pkr',
    'pln',
    'qar', 'ron', 'rub', 'rwf', 'sar', 'sbd', 'scr', 'sdg', 'sek', 'sgd', 'shp', 'skk', 'sll', 'sos', 'std', 'svc',
    'syp',
    'szl', 'thb', 'tnd', 'top', 'try', 'ttd', 'twd', 'tzs', 'ugx', 'uyu', 'uzs', 'vef', 'vnd', 'vuv', 'wst', 'xaf',
    'xcd', 'xof', 'xpf', 'yer', 'zar', 'zmk'
), index=0)

keep_longest_word = st.sidebar.checkbox('Keep The Longest Word Suggestion', value=True)
enable_fuzzy_product_match = st.sidebar.checkbox('Enable Fuzzy Product Matching? (Slow)', value=False)

min_search_vol = int(min_search_vol)
data_source_kwe = 'gkp'  # gkp = google keyword planner only // cli = clickstream data + keyword planner

http_or_https_gsc = "https://"  # http prefix // default = https://
parms = "page=|p=|utm_medium|sessionid|affiliateid|sort=|order=|type=|categoryid=|itemid=|viewItems=|query" \
        "=|search=|lang="  # drop common parameter urls

# -------------------------------- read the keyword to a dataframe called df_crawl ----------------------------------
with col1:
    uploaded_file = st.file_uploader("Upload your inlinks.csv crawl file",
                                     help="""Which reports does the tool currently support?""")

    if uploaded_file is not None:
        try:
            result = chardet.detect(uploaded_file.getvalue())
            encoding_value = result["encoding"]

            if encoding_value == "UTF-16":
                white_space = True
            else:
                white_space = False

            df_all_inlinks = pd.read_csv(
                uploaded_file,
                encoding=encoding_value,
                delim_whitespace=white_space,
                dtype=str,
                on_bad_lines='skip',
            )
            number_of_rows = len(df_all_inlinks)
            if number_of_rows == 0:
                st.caption("Your sheet seems empty!")

        except UnicodeDecodeError:
            st.warning("""The file doesn't seem to load. Check the filetype, file format and Schema""")

    else:
        st.stop()

# -------------------------------- read the crawl file to a dataframe called df_crawl ----------------------------------
with col2:
    uploaded_crawl_file = st.file_uploader("Upload your internal_html.csv crawl export",
                                           help="""Which reports does the tool currently support?""")
    if uploaded_crawl_file is not None:

        try:
            result = chardet.detect(uploaded_crawl_file.getvalue())
            encoding_value = result["encoding"]
            if encoding_value == "UTF-16":
                white_space = True
            else:
                white_space = False

            df_internal_html = pd.read_csv(
                uploaded_crawl_file,
                encoding=encoding_value,
                delim_whitespace=white_space,
                on_bad_lines='skip',
                dtype=str,
            )

            number_of_rows = len(df_internal_html)
            if number_of_rows == 0:
                st.caption("Your sheet seems empty!")

        except UnicodeDecodeError:
            st.warning("""The file doesn't seem to load. Check the filetype, file format and Schema""")

    else:
        st.stop()

with col2:
    with st.form(key='columns_in_form_2'):
        st.subheader("Map the Product & Category Columns")
        product_extract_col = st.selectbox('Select the PRODUCT column:', df_internal_html.columns)
        category_extract_col = st.selectbox('Select the CATEGORY column:', df_internal_html.columns)
        submitted_crawl_btn = st.form_submit_button('Submit')

# ------------------------------------- start rest of code -------------------------------------------------------
if submitted_crawl_btn == True:

    # ------------------------------------- clean up the crawl files -------------------------------------------------------

    df_all_inlinks = df_all_inlinks.rename(columns={"From": "Source", "To": "Destination"})

    df_internal_html = df_internal_html[
        ~df_internal_html["Indexability"].isin(["Non-Indexable"])]  # keep indexable urls
    df_internal_html['H1-1'] = df_internal_html['H1-1'].str.lower()
    df_internal_html['H1-1'] = df_internal_html['H1-1'].str.encode('ascii', 'ignore').str.decode('ascii')
    df_internal_html['Title-1'] = df_internal_html['Title 1'].str.encode('ascii', 'ignore').str.decode('ascii')

    # ---------------------------------- work out the page type from extractors --------------------------------------------

    df1 = df_internal_html[df_internal_html[product_extract_col].notna()].copy()
    df2 = df_internal_html[df_internal_html[category_extract_col].notna()].copy()
    df1.rename(columns={product_extract_col: "Page Type"}, inplace=True)
    df2.rename(columns={category_extract_col: "Page Type"}, inplace=True)
    df1["Page Type"] = "Product Page"
    df2["Page Type"] = "Category Page"
    df_internal_html = pd.concat([df1, df2])

    # ------------------------------------ extract the domain from the crawl -----------------------------------------------

    extracted_domain = df_internal_html["Address"]
    extracted_domain = extracted_domain.str.split("/").str[2]
    url = extracted_domain.iloc[0]
    url_slash = http_or_https_gsc + url + "/"  # adds a trailing slash to the domain to query the gsc api

    # -------------------------------- make the products & category dataframes ---------------------------------------------

    df_sf_products = df_internal_html[df_internal_html['Page Type'].str.contains("Product Page")].copy()
    df_sf_categories = df_internal_html[df_internal_html['Page Type'].str.contains("Category Page")].copy()

    df_sf_products.drop_duplicates(subset="H1-1", inplace=True)  # drop duplicate values (drop pagination pages etc)
    df_sf_categories.drop_duplicates(subset="H1-1", inplace=True)  # drop duplicate values (drop pagination pages etc)
    df_sf_categories = df_sf_categories[~df_sf_categories["Address"].str.contains(parms, na=False)]
    df_all_inlinks.drop_duplicates(subset=["Source", "Destination"], keep="first", inplace=True)
    df_all_inlinks = pd.merge(df_all_inlinks, df_sf_categories, left_on="Source", right_on="Address", how="left")
    df_all_inlinks = df_all_inlinks[df_all_inlinks["Page Type"].isin(["Category Page"])]

    df_all_inlinks = df_all_inlinks[["Destination", "Source"]]

    df_sf_products = pd.merge(df_sf_products, df_all_inlinks, left_on="Address", right_on="Destination", how="left")
    df_sf_products.rename(columns={"Source": "Parent URL", "Address": "Product URL"}, inplace=True)
    df_sf_products = df_sf_products[df_sf_products["Parent URL"].notna()]  # Only Keep Rows which are not NaN

    # ---------------------------- group dataframes & make lists for n-gramming --------------------------------------------

    df_product_group = (df_sf_products.groupby("Product URL").agg({"Parent URL": "first"}).reset_index())

    category_extractor_list = list(set(df_product_group["Parent URL"]))
    len_product_list = len(category_extractor_list)

    # ---------------------------------------- start n-gram routine --------------------------------------------------------

    ngram_loop_count = 1
    start_num = 0
    appended_data = []

    for i in stqdm(range(0, ngram_loop_count)):
        print("Calculating ngrams")
        while ngram_loop_count != len_product_list:
            df_kwe = df_sf_products[
                df_sf_products["Parent URL"].str.contains(category_extractor_list[start_num], na=False)]
            text = str(df_kwe["H1-1"])

            # clean up the corpus before ngramming
            text = "".join(c for c in text if not c.isdigit())  # removes all numbers
            text = re.sub("<.*>", "", text)
            text = re.sub(r"\b[a-zA-Z]\b", "", text)
            punctuationNoFullStop = "[" + re.sub("\.", "", string.punctuation) + "]"
            text = re.sub(punctuationNoFullStop, "", text)

            # first get individual words
            tokenized = text.split()
            oneNgrams = ngrams(tokenized, 1)
            twoNgrams = ngrams(tokenized, 2)
            threeNgrams = ngrams(tokenized, 3)
            fourNgrams = ngrams(tokenized, 4)
            fiveNgrams = ngrams(tokenized, 5)
            sixNgrams = ngrams(tokenized, 6)
            sevenNgrams = ngrams(tokenized, 7)
            oneNgramsFreq = collections.Counter(oneNgrams)
            twoNgramsFreq = collections.Counter(twoNgrams)
            threeNgramsFreq = collections.Counter(threeNgrams)
            fourNgramsFreq = collections.Counter(fourNgrams)
            fiveNgramsFreq = collections.Counter(fiveNgrams)
            sixNgramsFreq = collections.Counter(sixNgrams)
            sevenNgramsFreq = collections.Counter(sevenNgrams)

            # combines the above collection counters so they can be placed in a dataframe.
            ngrams_combined_list = (
                    twoNgramsFreq.most_common(100)
                    + threeNgramsFreq.most_common(100)
                    + fourNgramsFreq.most_common(100)
                    + fiveNgramsFreq.most_common(100)
                    + sixNgramsFreq.most_common(100)
                    + sevenNgramsFreq.most_common(100)
            )

            # create the final dataframe
            df_ngrams = pd.DataFrame(ngrams_combined_list, columns=["Keyword", "Frequency"])
            df_ngram_frequency = pd.DataFrame(ngrams_combined_list, columns=["Keyword", "Frequency"])
            df_ngrams["Parent Category"] = category_extractor_list[start_num]
            data = df_ngrams
            appended_data.append(data)
            start_num = start_num + 1
            ngram_loop_count = ngram_loop_count + 1

    df_ngrams = pd.concat(appended_data)  # concat the list of dataframes
    ngram_count = df_ngrams.shape[0]  # get the row count
    with col1:
        st.info('Total keywords generated via ngrams: ' + str(ngram_count))

    df_ngrams = df_ngrams.sort_values(by="Frequency", ascending=False)
    df_ngrams["Keyword"] = [' '.join(entry) for entry in df_ngrams["Keyword"]]
    cols = "Parent Category", "Keyword", "Frequency"
    df_ngrams = df_ngrams.reindex(columns=cols)

    # ---------------------------------------- pre-filtering ---------------------------------------------------------------

    df_ngrams = df_ngrams[~df_ngrams['Keyword'].astype(str).str.startswith(("and", "with", "for", "mm ", "cm ", "of"))]
    df_ngrams = df_ngrams[~df_ngrams['Keyword'].astype(str).str.endswith(("and", "with", "for", "mm ", "cm ", "of"))]

    df_sf_products["H1-1"] = df_sf_products["H1-1"].astype(str)
    df_sf_products["H1-1"] = df_sf_products["H1-1"].str.lower()

    df_product_set = set(df_sf_products["H1-1"])  # make a set, then a list
    target_keyword_list = list(df_product_set)
    keyword_list = list(df_ngrams["Keyword"])  # make the keyword list

    # ---------------------- keep only suggestions which match to products x times -----------------------------------------

    with col1:
        st.write("Exact matching to a minimum of", min_product_match_exact, "products ..")
    check_list_exact = []
    for i in stqdm(keyword_list):
        check_freq = sum(i in s for s in target_keyword_list)
        check_list_exact.append(check_freq)

    # search in a fuzzy match
    if enable_fuzzy_product_match:
        with col1:
            st.write("Fuzzy matching keywords to a minimum of", min_product_match_fuzzy, "products ..")
        check_list_fuzzy = []
        for keywords in stqdm(keyword_list):
            check_list_fuzzy.append(
                sum(all(keyword in target for keyword in keywords.split())
                    for target in target_keyword_list)
            )

    df_ngrams["Matching Products (Exact)"] = check_list_exact

    if enable_fuzzy_product_match:
        df_ngrams["Matching Products (Fuzzy)"] = check_list_fuzzy

    df_ngrams = df_ngrams[df_ngrams["Matching Products (Exact)"] >= min_product_match_exact]

    if enable_fuzzy_product_match:
        df_ngrams = df_ngrams[df_ngrams["Matching Products (Fuzzy)"] >= min_product_match_fuzzy]

    rows = df_ngrams.shape[0]
    ngram_loop_count = 1
    start = 1
    end = 100
    df_data = []

    # ------------------------ fuzz match suggested keywords to existing categories ----------------------------------------

    df_ngrams = df_ngrams[df_ngrams["Keyword"].notna()]  # Only Keep Rows which are not NaN
    df_sf_categories = df_sf_categories[df_sf_categories["H1-1"].notna()]  # Only Keep Rows which are not NaN
    df_keyword_list = list(df_ngrams["Keyword"])  # create lists from dfs
    df_sf_cats_list = list(df_sf_categories["H1-1"])  # create lists from dfs
    model = PolyFuzz("TF-IDF").match(df_keyword_list, df_sf_cats_list)  # do the matching

    st.write("Checking if suggestions match to an existing category ..")
    df_fuzz = model.get_matches()  # make the polyfuzz dataframes
    df_ngrams = pd.merge(df_ngrams, df_fuzz, left_on="Keyword", right_on="From")
    df_ngrams.rename(columns={"To": "Matched Category", "clicks": "Clicks", "impressions": "Impressions"},
                     inplace=True)

    if kwe_key != "":
        # -------------------------- check available keywords everywhere credits------------------------------------------------

        df_ngrams.drop_duplicates(subset=["Keyword"], keep="first", inplace=True)
        creds_required = df_ngrams.shape[0]

        my_headers = {
            'Accept': 'application/json',
            'Authorization': 'Bearer ' + kwe_key
        }
        response = requests.get('https://api.keywordseverywhere.com/v1/account/credits', headers=my_headers)
        if response.status_code == 200:
            creds_available = response.content.decode('utf-8')
            creds_available = creds_available.split()
            creds_available = int(creds_available[1])
            with col1:
                st.write("This operation will require", creds_required, "Keywords Everywhere API credits. \nYou have",
                         creds_available,
                         "credits remaining.")
            if creds_available < creds_required:
                st.write("Not enough keywords everywhere credits available!")
                st.stop
        else:
            st.write("An error occurred\n\n", response.content.decode('utf-8'))

        # ---------------------- get search volume with keywords everywhere -----------------------------------------------------

        loops = int(creds_required / 100)
        if loops == 1:
            loops += 1
        fixed_loops = loops * 100  # fixes the total loop counter displayed value
        ngram_loop_count_100 = ngram_loop_count * 100

        with col1:
            st.write("Fetching search volume & CPC data from Keywords Everywhere.")
        for i in stqdm(range(0, loops)):

            while ngram_loop_count != loops:
                keywords = list(df_ngrams["Keyword"][start:end])
                keywords_set = set(keywords)
                keywords = list(keywords_set)
                my_data = {
                    'country': country_kwe,
                    'currency': currency_kwe,
                    'dataSource': data_source_kwe,
                    'kw[]': keywords
                }
                my_headers = {
                    'Accept': 'application/json',
                    'Authorization': 'Bearer ' + kwe_key
                }
                response = requests.post(
                    'https://api.keywordseverywhere.com/v1/get_keyword_data', data=my_data, headers=my_headers)
                try:
                    keywords_data = response.json()['data']
                except KeyError:
                    print("Couldn't retrieve data from Keywords Everywhere. Check credits...")
                    pass

                vol = []
                cpc = []

                for element in keywords_data:
                    vol.append(element["vol"])
                    cpc.append(element["cpc"]["value"])

                rows = zip(keywords, vol, cpc)
                df_kwe = pd.DataFrame(rows, columns=["Keyword", "Search Volume", "CPC"])
                data = df_kwe
                df_data.append(data)
                start = start + 100
                end = end + 100
                ngram_loop_count += +1
                ngram_loop_count_100 += 100

        with col1:
            st.success("Got search volume and CPC data successfully!")

        df_kwe = pd.concat(df_data)
        df_kwe["Search Volume"] = df_kwe["Search Volume"].astype(int)
        df_kwe["CPC"] = df_kwe["CPC"].astype(float)
        df_kwe = df_kwe[df_kwe["Search Volume"] > min_search_vol]
        df_kwe = df_kwe[df_kwe["CPC"] > min_cpc]
        df_kwe = pd.merge(df_kwe, df_ngrams, on="Keyword", how='left')
        df_kwe = df_kwe.sort_values(by="Parent Category", ascending=True)

        # ------------------------------------- clean up the final dataframe ---------------------------------------------------

        df_kwe["Similarity"] = df_kwe["Similarity"] * 100
        df_kwe.fillna({"Similarity": 0}, inplace=True)
        df_kwe["Similarity"] = df_kwe["Similarity"].astype(int)
        df_kwe = df_kwe[df_kwe["Similarity"] <= min_sim_match]
        df_kwe["Matched Category"] = df_kwe["Matched Category"].str.lower()
        df_kwe.drop_duplicates(subset=["Matched Category", "Keyword"], keep="first", inplace=True)
    else:
        df_kwe = df_ngrams

    cols = (
        "Parent Category",
        "Keyword",
        "Search Volume",
        "CPC",
        "Matching Products (Exact)",
        "Matching Products (Fuzzy)",
        "Similarity",
        "Matched Category",
    )
    df_kwe = df_kwe.reindex(columns=cols)

    # --------------------------- keep the longest word and discard the fragments ------------------------------------------

    if keep_longest_word == True:
        with col1:
            st.write("\nKeeping Longest Word and Discarding Fragments ..")

        list1 = df_kwe["Keyword"]
        substrings = {w1 for w1 in list1 for w2 in list1 if w1 in w2 and w1 != w2}
        longest_word = set(list1) - substrings
        longest_word = list(longest_word)
        shortest_word_list = list(set(list1) - set(longest_word))
        with col1:
            with st.expander("Click to see which short words were discarded .."):
                st.write("Discarded the following short words:\n", shortest_word_list)
            df_kwe = df_kwe[~df_kwe['Keyword'].isin(shortest_word_list)]

    # ------------------------------ merge in page title for matched category ----------------------------------------------

    df_mini = df_internal_html[["H1-1", "Title 1"]]
    df_mini = df_mini.rename(columns={"H1-1": "Matched Category", "Title 1": "Matched Category Page Title"})
    df_kwe = pd.merge(df_kwe, df_mini[['Matched Category', 'Matched Category Page Title']], on='Matched Category',
                      how='left')

    # ---------------------- remove keyword suggestions if matched to an existing category in any order --------------------

    df_kwe['Matched Category Page Title Lower'] = df_kwe['Matched Category Page Title'].str.lower()
    df_kwe = df_kwe.astype({"Keyword": "str", "Matched Category": "str", "Matched Category Page Title Lower": "str"})
    col = "Keyword"


    def ismatch(s):
        A = set(s[col].split())
        B = set(s['Matched Category Page Title Lower'].split())
        return A.intersection(B) == A


    df_kwe['KW Matched'] = df_kwe.apply(ismatch, axis=1)

    # --------------------------------------------- handling pluralised words ------------------------------------------

    df_kwe["Keyword + s"] = df_kwe["Keyword"] + "s"  # make new temp column to run the same check on the pluralised word
    col = "Keyword + s"  # updates the column to run function on
    df_kwe['KW Matched 2'] = df_kwe.apply(ismatch, axis=1)
    df_kwe = df_kwe[~df_kwe["KW Matched"].isin([True])]  # drop rows which are matched
    df_kwe = df_kwe[~df_kwe["KW Matched 2"].isin([True])]  # drop rows which are matched
    df_kwe.drop_duplicates(subset=["Parent Category", "Keyword"], keep="first",
                           inplace=True)  # drop if both values dupes

    # ---------------------- Set the Final Column Order ------------------------------------------------------------------

    cols = (
        "Parent Category",
        "Keyword",
        "Search Volume",
        "CPC",
        "Matching Products (Exact)",
        "Matching Products (Fuzzy)",
        "Matched Category",
        "Similarity",
        "Matched Category Page Title",
    )
    df_kwe = df_kwe.reindex(columns=cols)

    if enable_fuzzy_product_match == False:
        del df_kwe['Matching Products (Fuzzy)']
    if kwe_key == "":
        del df_kwe['Search Volume']
        del df_kwe['CPC']

    # ---------------------- Export Final Dataframe to CSV -------------------------------------------------------------

    keyword_volume_count = df_kwe.shape[0]
    df_kwe.sort_values(["Parent Category", "Keyword"], ascending=[True, True], inplace=True)


    @st.cache_data
    def convert_df(df):  # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv(index=False).encode('utf-8')


    csv = convert_df(df_kwe)
    st.download_button(
        label="Download your Landing Page Suggester Report!",
        data=csv,
        file_name='category_opportunities.csv',
        mime='text/csv')
