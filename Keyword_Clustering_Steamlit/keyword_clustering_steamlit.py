import streamlit as st
from polyfuzz import PolyFuzz
import pandas as pd
import sys
import chardet
from tqdm import tqdm

import os
import json

# region format
st.set_page_config(page_title="Keyword Clustering App", page_icon="✨", layout="wide")

c30, c31, c32 = st.columns([2.5, 1, 3])

#with c30:
#    st.image("logo_new_2.png", width=400)
#    st.header("")

with c32:

    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")

    st.write(
        "&nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp Made in [![this is an image link](https://i.imgur.com/iIOA6kU.png)](https://www.streamlit.io/)&nbsp, with :heart: by [@DataChaz](https://www.charlywargnier.com/) | Script by [@LeeFootSEO](https://twitter.com/LeeFootSEO)"
    )

c30, c32 = st.columns([1.9, 1])

with st.expander("⚡ App features", expanded=False):

    st.write(
        """
Simply upload your SEMRush, Ahrefs or Google Search Console report and get:

1.   The cluster name
1.   The cluster size (how many keywords in a given cluster)
1.   The estimated search volume per cluster
1.   The average keyword difficulty per cluster
1.   The average CPC per cluster
1.   The estimated traffic per cluster

        """
    )

    st.markdown("")

    st.write(
        """
The tool is still in Beta, with possible rough edges! [![Gitter](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/DataChaz/keyword-clustering-app) for bug report, questions, or suggestions.

        """
    )

    st.markdown("")

with st.expander("🔥 New in V2!", expanded=False):

    st.write(
        """
-   Clusters and calculates AdWords keyword export with metrics 
-   Accepts .txt and .csv files. Header or headerless! (Excel + OpenOffice formats coming soon!)
-   Auto-detects character encoding on CSVs (including UTF-16 Support)
        """
    )

    st.markdown("")

with st.expander("❔ Why cluster your keywords?", expanded=False):

    st.write(
        """
        Grouping your keywords into topic clusters provides mighty insights for your SEO strategy! It can be useful to:

-   Build topic clusters around your parent keyword
-   Find out what the main keyword and secondary keywords your pages should target 
-   Find new themes and content ideas
-   Bulk filter-out junk keywords from your keyword lists
-   Find your competitors' most profitable keyword clusters

        """
    )

    st.markdown("")

st.write("")

c28, c29, c30, c31 = st.columns([1, 0.3, 1.5, 1])

# with c29:
#     selectbox = st.selectbox(
#         "Select filetype",
#         [".csv", ".txt"],
#         # [".csv", ".txt", ".xlsx", ".xls", ".xlsm", ".xlsb"],
#     )

with c30:

    uploaded_file = st.file_uploader(
        "Upload your SEMRush, Ahrefs or GSC report (.csv or .txt only)",
        help="""

Which reports does the tool currently support? (.csv or .txt only)

-   Google Search Console coverage report
-   SEMrush organic positions report
-   Ahrefs Site Explorer keyword export (V1 or V2)
-   Ahrefs Keyword Explorer export

""",
    )

    if uploaded_file is not None:

        try:
            # if selectbox == ".xlsx":
            # if selectbox == ".csv":

            result = chardet.detect(uploaded_file.getvalue())
            # Get encoding value
            encoding_value = result["encoding"]

            if encoding_value == "UTF-16":
                white_space = True
            else:
                white_space = False

            df_1 = pd.read_csv(
                uploaded_file,
                encoding=encoding_value,
                nrows=10000,
                delim_whitespace=white_space,
                error_bad_lines=False,
            )

            row_count = df_1.count()

            index = df_1.index
            number_of_rows = len(index)

            if number_of_rows == 0:

                st.caption("Your sheet seems empty!")

            elif number_of_rows == 1:
                st.caption(
                    "The sheet you uploaded is "
                    + str(number_of_rows)
                    + " row (cap for our Beta is at 10K rows). Its file encoding is '"
                    + str(encoding_value)
                    + "'"
                )
            else:
                st.caption(
                    "The sheet you uploaded is "
                    + str(number_of_rows)
                    + " rows (cap for our Beta is at 10K rows). Its file encoding is '"
                    + str(encoding_value)
                    + "'"
                )

            # if number_of_rows >= 10000:
            #
            #     st.warning(
            #         f"⚠️ Only the first 10K rows will be clustered. Increased allowance is coming, stay tuned!"
            #     )

            with st.expander("↕️ View raw data", expanded=False):

                #  st.caption("File encoding is '" + str(encoding_value) + "'")

                # row_count = df_1.count()
                # row_count
                #
                # index = df_1.index
                # number_of_rows = len(index)
                #
                # st.caption("File encoding is '" + str(number_of_rows) + "'")
                #
                # if number_of_rows >= 10000:
                #
                #     st.warning(
                #         f"⚠️ Only the first 10K rows will be clustered. Increased allowance  is coming! Stay tuned! 😊)"
                #     )

                st.write(df_1)

            # elif selectbox == ".xlsx":
        #
        #     df_1 = pd.read_excel(uploaded_file, engine="openpyxl", nrows=10000)
        #     with st.expander("↕️ View raw data", expanded=False):
        #         st.write(df_1)

        except UnicodeDecodeError:
            st.warning(
                # "🚨 The file doesn't seem to load. Check the filetype, file format and Schema.
                """

🚨 The file doesn't seem to load. Check the filetype, file format and Schema

Which reports does the tool currently support? (.csv or .txt only)

-   Google Search Console coverage report
-   SEMrush organic positions report
-   Ahrefs Site Explorer keyword export (V1 or V2)
-   Ahrefs Keyword Explorer export

"""
            )

    #         except UnicodeDecodeError:
    #             st.warning(
    #                 "🚨 The file doesn't seem to load. Check the filetype, file format and Schema."
    #             )
    #             st.stop()
    #
    #         help = """
    #
    # Which reports does the tool currently support? (.csv or .txt only)
    #
    # -   Google Search Console coverage report
    # -   SEMrush organic positions report
    # -   Ahrefs Site Explorer keyword export (V1 or V2)
    # -   Ahrefs Keyword Explorer export
    #
    # """

    #     st.stop()

    # except:
    #     st.warning(
    #         "🚨 The file doesn't seem to load - Please make sure the right filetype is selected."
    #     )
    #     st.stop()

    # else:
    #     st.stop()

    else:
        st.info(
            f"""
                👆 Upload a .csv or .txt file first. Try a sample 👉 [Ahrefs Keyword Explorer export](https://raw.githubusercontent.com/CharlyWargnier/CSVHub/main/Sample_files_for_keyword_clustering_app/next_500_rows.csv)
                """
        )
        st.stop()


# -------------------------- check if single column import / and write header if missing -------------------------------

# check the number of columns
col_len = len(df_1.columns)
col_name = df_1.columns[0]

if col_len == 1 and df_1.columns[0] != "Keyword":
    df_1.columns = ["Keyword"]

if col_len == 1 and df_1.columns[0] != "keyword":
    df_1.columns = ["Keyword"]

# -------------------------- detect if import file is adwords and remove the first two rows ----------------------------
adwords_check = False
if col_name == "Search terms report":
    df_1.columns = df_1.iloc[1]
    df_1 = df_1[1:]
    df_1 = df_1.reset_index(drop=True)

    new_header = df_1.iloc[0]  # grab the first row for the header
    df_1 = df_1[1:]  # take the data less the header row
    df_1.columns = new_header  # set the header row as the df header
    adwords_check = True

# ------------------------------------------- Set the Variables --------------------------------------------------------

# rename the parent cluster name using the keyword with the highest search volume (recommended)
parent_by_vol = True
drop_site_links = False
drop_image_links = False
sim_match_percent = 0.99
url_filter = ""
min_volume = 0  # set the minimum search volume / impressions to filter on

# --------------------------------- Check if csv data is gsc and set bool ----------------------------------------------

if "Impressions" in df_1.columns:
    gsc_data = True

# ----------------- standardise the column names between ahrefs v1/v2/semrush/gsc keyword exports ----------------------

df_1.rename(
    columns={
        "Current position": "Position",
        "Current URL": "URL",
        "Current URL inside": "Page URL inside",
        "Current traffic": "Traffic",
        "KD": "Difficulty",
        "Keyword Difficulty": "Difficulty",
        "Search Volume": "Volume",
        "page": "URL",
        "query": "Keyword",
        "Top queries": "Keyword",
        "query": "Keyword",
        "Query": "Keyword",
        "Impressions": "Volume",
        "Clicks": "Traffic",
        "Search term": "Keyword",
        "Impr.": "Volume",
    },
    inplace=True,
)

if col_len > 1:
    # --------------------------------- clean the data pre-grouping ----------------------------------------------------

    if url_filter:
        st.write("Processing only URLs containing:", url_filter)

    try:
        df_1 = df_1[df_1["URL"].str.contains(url_filter, na=False)]
    except KeyError:
        pass

    # ========================= clean strings out of numerical columns (adwords) ========================================

    try:
        df_1["Volume"] = df_1["Volume"].str.replace(",", "").astype(int)
        df_1["Traffic"] = df_1["Traffic"].str.replace(",", "").astype(int)
        df_1["Conv. value / click"] = (
            df_1["Conv. value / click"].str.replace(",", "").astype(float)
        )
        df_1["All conv. value"] = (
            df_1["All conv. value"].str.replace(",", "").astype(float)
        )
        df_1["CTR"] = df_1["CTR"].replace(" --", "0", regex=True)
        df_1["CTR"] = df_1["CTR"].str.replace("\%", "").astype(float)
        df_1["Cost"] = df_1["Cost"].astype(float)
        df_1["Conversions"] = df_1["Conversions"].astype(int)
        df_1["Cost"] = df_1["Cost"].round(2)
        df_1["All conv. value"] = df_1["All conv. value"].astype(float)
        df_1["All conv. value"] = df_1["All conv. value"].round(2)

    except Exception:
        pass

    df_1 = df_1[
        ~df_1["Keyword"].str.contains("Total: ", na=False)
    ]  # remove totals rows
    df_1 = df_1[df_1["Keyword"].notna()]  # keep only rows which are NaN
    df_1 = df_1[df_1["Volume"].notna()]  # keep only rows which are NaN
    df_1["Volume"] = df_1["Volume"].astype(str)
    df_1["Volume"] = df_1["Volume"].apply(lambda x: x.replace("0-10", "0"))
    df_1["Volume"] = df_1["Volume"].astype(float).astype(int)

    # drop sitelinks

    if drop_site_links:
        try:
            df_1 = df_1[
                ~df_1["Page URL inside"].str.contains("Sitelinks", na=False)
            ]  # drop sitelinks
        except KeyError:
            pass
        try:
            if gsc_data:
                df_1 = df_1.sort_values(by="Traffic", ascending=False)
                df_1.drop_duplicates(subset="Keyword", keep="first", inplace=True)
        except NameError:
            pass

    if drop_image_links:
        try:
            df_1 = df_1[
                ~df_1["Page URL inside"].str.contains("Image pack", na=False)
            ]  # drop image pack
        except KeyError:
            pass

    df_1 = df_1[df_1["Volume"] > min_volume]

# ------------------------------------- do the grouping ----------------------------------------------------------------

df_1_list = df_1.Keyword.tolist()  # create list from df
model = PolyFuzz("TF-IDF")

cluster_tags = df_1_list[::]
cluster_tags = set(cluster_tags)
cluster_tags = list(cluster_tags)

print("Cleaning up the cluster tags.. Please be patient!")
substrings = {w1 for w1 in tqdm(cluster_tags) for w2 in cluster_tags if w1 in w2 and w1 != w2}
longest_word = set(cluster_tags) - substrings
longest_word = list(longest_word)
shortest_word_list = list(set(cluster_tags) - set(longest_word))

try:
    model.match(df_1_list, shortest_word_list)

except ValueError:
    st.warning("🚨 Your sheet seems empty!")
    # sys.exit()
    st.stop()

except AttributeError:
    st.warning(
        "🚨 It seems that there's something wrong with your sheet. Have you uploaded the right file?"
    )
    st.stop()

try:
    model.group(link_min_similarity=sim_match_percent)
    df_matched = model.get_matches()

    # ------------------------------- clean the data post-grouping ---------------------------------------------------------

    df_matched.rename(
        columns={"From": "Keyword", "Group": "Cluster Name"}, inplace=True
    )  # renaming multiple columns

    # merge keyword volume / CPC / Pos / URL etc data from original dataframe back in
    df_matched = pd.merge(df_matched, df_1, on="Keyword", how="left")

    # rename traffic (acs) / (desc) to 'Traffic for standardisation
    df_matched.rename(
        columns={"Traffic (desc)": "Traffic", "Traffic (asc)": "Traffic"}, inplace=True
    )

    if col_len > 1:

        # fill in missing values
        df_matched.fillna({"Traffic": 0, "CPC": 0}, inplace=True)
        df_matched["Traffic"] = df_matched["Traffic"].round(0)

        # ------------------------- group the data and merge in original stats -------------------------------------------------
        if not adwords_check:
            try:
                # make dedicated grouped dataframe
                df_grouped = (
                    df_matched.groupby("Cluster Name")
                    .agg(
                        {
                            "Volume": sum,
                            "Difficulty": "median",
                            "CPC": "median",
                            "Traffic": sum,
                        }
                    )
                    .reset_index()
                )
            except Exception:
                df_grouped = (
                    df_matched.groupby("Cluster Name")
                    .agg({"Volume": sum, "Traffic": sum})
                    .reset_index()
                )

            df_grouped = df_grouped.rename(
                columns={
                    "Volume": "Cluster Volume",
                    "Difficulty": "Cluster KD (Median)",
                    "CPC": "Cluster CPC (Median)",
                    "Traffic": "Cluster Traffic",
                }
            )

            df_matched = pd.merge(
                df_matched, df_grouped, on="Cluster Name", how="left"
            )  # merge in the group stats

        if adwords_check:

            df_grouped = (
                df_matched.groupby("Cluster Name")
                .agg(
                    {
                        "Volume": sum,
                        "CTR": "median",
                        "Cost": sum,
                        "Traffic": sum,
                        "All conv. value": sum,
                        "Conversions": sum,
                    }
                )
                .reset_index()
            )

            df_grouped = df_grouped.rename(
                columns={
                    "Volume": "Cluster Volume",
                    "CTR": "Cluster CTR (Median)",
                    "Cost": "Cluster Cost (Sum)",
                    "Traffic": "Cluster Traffic",
                    "All conv. value": "All conv. value (Sum)",
                    "Conversions": "Cluster Conversions (Sum)",
                }
            )

            df_matched = pd.merge(
                df_matched, df_grouped, on="Cluster Name", how="left"
            )  # merge in the group stats

            del df_matched["To"]
            del df_matched["Similarity"]

        # ---------------------------- clean and sort the final output -----------------------------------------------------

        try:
            df_matched.drop_duplicates(
                subset=["URL", "Keyword"], keep="first", inplace=True
            )  # drop if both kw & url are duped

        except KeyError:
            pass

    if not adwords_check:
        cols = (
            "Keyword",
            "Cluster Name",
            "Cluster Size",
            "Cluster Volume",
            "Cluster KD (Median)",
            "Cluster CPC (Median)",
            "Cluster Traffic",
            "Volume",
            "Difficulty",
            "CPC",
            "Traffic",
            "URL",
        )

        df_matched = df_matched.reindex(columns=cols)

        try:
            if gsc_data:
                cols = (
                    "Keyword",
                    "Cluster Name",
                    "Cluster Size",
                    "Cluster Volume",
                    "Cluster Traffic",
                    "Volume",
                    "Traffic",
                )
                df_matched = df_matched.reindex(columns=cols)
        except NameError:
            pass

    # count cluster size
    df_matched["Cluster Size"] = df_matched["Cluster Name"].map(
        df_matched.groupby("Cluster Name")["Cluster Name"].count()
    )

    df_matched.loc[df_matched["Cluster Size"] > 1, "Clustered?"] = True
    df_matched["Clustered?"] = df_matched["Clustered?"].fillna(False)

    # ------------ get the keyword with the highest search volume to replace the auto generated tag name with --------------

    if col_len > 1:
        if parent_by_vol:
            df_matched["vol_max"] = df_matched.groupby(["Cluster Name"])[
                "Volume"
            ].transform(max)
            # this sort is mandatory for the renaming to work properly by floating highest values to the top of the cluster
            df_matched.sort_values(
                ["Cluster Name", "Cluster Volume", "Volume"],
                ascending=[False, True, False],
                inplace=True,
            )
            df_matched["exact_vol_match"] = (
                df_matched["vol_max"] == df_matched["Volume"]
            )
            df_matched.loc[
                df_matched["exact_vol_match"] == True, "highest_ranked_keyword"
            ] = df_matched["Keyword"]
            df_matched["highest_ranked_keyword"] = df_matched[
                "highest_ranked_keyword"
            ].fillna(method="ffill")
            df_matched["Cluster Name"] = df_matched["highest_ranked_keyword"]
            del df_matched["vol_max"]
            del df_matched["exact_vol_match"]
            del df_matched["highest_ranked_keyword"]
    if adwords_check:
        df_matched = df_matched.rename(
            columns={
                "Volume": "Impressions",
                "Traffic": "Clicks",
                "Cluster Traffic": "Cluster Clicks (Sum)",
            }
        )

    # -------------------------------------- final output ------------------------------------------------------------------
    # sort on cluster size
    df_matched.sort_values(
        ["Cluster Size", "Cluster Name", "Cluster Volume"],
        ascending=[False, True, False],
        inplace=True,
    )

except KeyError:
    pass

# endregion V2

# region last snippet V2

try:
    if gsc_data:
        df_matched.rename(
            columns={
                "Cluster Volume": "Cluster Impressions",
                "Cluster Traffic": "Cluster Clicks",
                "Traffic": "Clicks",
                "Volume": "Impressions",
            },
            inplace=True,
        )
except NameError:
    pass

if col_len == 1:
    cols = "Keyword", "Cluster Name", "Cluster Size", "Clustered?"
    df_matched = df_matched.reindex(columns=cols)

info_filter = "what|where|why|when|who|how|which|tip|guide|tutorial|ideas|example|learn|wiki|in mm|in cm|in ft|in feet"
comm_invest_filter = "best|vs|list|compare|review|list|top|difference between"
trans_filter = "purchase|bargain|cheap|deal|value|closeout|buy|shop|price|coupon|discount|price|pricing|delivery|shipping|order|returns|sale|amazon|target|ebay|walmart|cost of|how much"
    
# - add in intent markers
colname = df_matched.columns[1]
df_matched.loc[df_matched[colname].str.contains(info_filter), "Informational"] = "Informational"
df_matched.loc[df_matched[colname].str.contains(comm_invest_filter), "Commercial Investigation"] = "Commercial Investigation"
df_matched.loc[df_matched[colname].str.contains(trans_filter), "Transactional"] = "Transactional"

# find keywords from one column in another in any order and count the frequency
df_matched['Cluster Name'] = df_matched['Cluster Name'].str.strip()
df_matched['Keyword'] = df_matched['Keyword'].str.strip()

df_matched['First Word'] = df_matched['Cluster Name'].str.split(" ").str[0]
df_matched['Second Word'] = df_matched['Cluster Name'].str.split(" ").str[1]
df_matched['Third Word'] = df_matched['Cluster Name'].str.split(" ").str[2]
df_matched['Forth Word'] = df_matched['Cluster Name'].str.split(" ").str[3]

df_matched['Total Keywords'] = df_matched['First Word'].str.count(' ') + 1

def ismatch(s):
    A = set(s["First Word"].split())
    B = set(s['Keyword'].split())
    return A.intersection(B) == A

df_matched['Found'] = df_matched.apply(ismatch, axis=1)

df_matched = df_matched. fillna('')

def ismatch(s):
    A = set(s["Second Word"].split())
    B = set(s['Keyword'].split())
    return A.intersection(B) == A
df_matched['Found 2'] = df_matched.apply(ismatch, axis=1)

df_matched = df_matched. fillna('')

def ismatch(s):
    A = set(s["Third Word"].split())
    B = set(s['Keyword'].split())
    return A.intersection(B) == A
df_matched['Found 3'] = df_matched.apply(ismatch, axis=1)

df_matched = df_matched. fillna('')

def ismatch(s):
    A = set(s["Forth Word"].split())
    B = set(s['Keyword'].split())
    return A.intersection(B) == A
df_matched['Found 4'] = df_matched.apply(ismatch, axis=1)

# todo - document this algo. Essentially if it matches on the second word only, it renames the cluster to the second word
# clean up code nd variable names

df_matched.loc[(df_matched["Found"] == False) & (df_matched["Found 2"] == True), "Cluster Name"] = df_matched["Second Word"]

df_matched.loc[(df_matched["Found"] == False) & (df_matched["Found 2"] == False) & (df_matched["Found 3"] == True), "Cluster Name"] = df_matched["Third Word"]

df_matched.loc[(df_matched["Found"] == False) & (df_matched["Found 2"] == False) & (df_matched["Found 3"] == False) & (df_matched["Found 4"] == True), "Cluster Name"] = df_matched["Forth Word"]

df_matched.loc[(df_matched["Found"] == False) & (df_matched["Found 2"] == False) & (df_matched["Found 3"] == False) & (df_matched["Found 4"] == False), "Cluster Name"] = "zzz_no_cluster_available"


# count cluster_size
df_matched['Cluster Size'] = df_matched['Cluster Name'].map(df_matched.groupby('Cluster Name')['Cluster Name'].count())
df_matched.loc[df_matched["Cluster Size"] == 1, "Cluster Name"] = "zzz_no_cluster_available"

#delete the helper cols
del df_matched['First Word']
del df_matched['Second Word']
del df_matched['Third Word']
del df_matched['Forth Word']

del df_matched['Total Keywords']
del df_matched['Found']
del df_matched['Found 2']
del df_matched['Found 3']
del df_matched['Found 4']

# convert empty strings to NaN and fill with no_cluster_available
df_matched["Cluster Name"] = df_matched["Cluster Name"].replace(r'^\s*$', np.nan, regex=True)
df_matched = df_matched.fillna("zzz_no_cluster_available") 

# check if keywords / cluster names are exclusively numbers and bumps them out of the cluster
df_matched['Number Check'] = df_matched['Cluster Name'].str.isdigit()
df_matched['Number Check 2'] = df_matched['Keyword'].str.isdigit()

# check if keywords / cluster names are 1 char in length and bumps them from the cluster

df_matched["Length Check"] = df_matched["Cluster Name"].str.len()
df_matched["Length Check 2"] = df_matched["Cluster Name"].str.len()


df_matched.loc[(df_matched["Length Check"] == 1), "Cluster Name"] = "zzz_no_cluster_available"
df_matched.loc[(df_matched["Length Check 2"] == 1), "Cluster Name"] = "zzz_no_cluster_available"

df_matched.loc[(df_matched["Number Check"] == True), "Cluster Name"] = "zzz_no_cluster_available"
df_matched.loc[(df_matched["Number Check 2"] == True), "Cluster Name"] = "zzz_no_cluster_available"

#df_matched = df_matched.sort_values(by="Cluster Name", ascending=True)
df_matched.sort_values(["Cluster Name", "Keyword"], ascending=[True, True], inplace=True,) 

# remove double and triple white space etc
df_matched['Cluster Name'] = (df_matched['Cluster Name'].str.split()).str.join(' ')
df_matched['Keyword'] = (df_matched['Keyword'].str.split()).str.join(' ')


# delete helper columns
del df_matched["Number Check"]
del df_matched["Number Check 2"]
del df_matched["Length Check"]
del df_matched["Length Check 2"]

format_dictionary = {
    "Cluster Size": "{:.0f}",
    "Cluster KD (Median)": "{:.0f}",
    "Cluster CPC (Median)": "{:.0f}",
    "Cluster Traffic": "{:.0f}",
    "Cluster Volume": "{:.0f}",
}

st.markdown("### **🎈 Check & download clustered results!**")
st.write("")

# Format as per format_dictionary
df_matched = df_matched.reset_index(drop=True)
df_matchedstyled = df_matched.style.format(format_dictionary)
# CSVButton = download_button(df_matched, "report.csv", "📥 Download your report!")
# st.dataframe(df_matchedstyled, height=1000)

def convert_df(df):
# IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

csv = convert_df(df_matched)

st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='large_df.csv',
        mime='text/csv',)
