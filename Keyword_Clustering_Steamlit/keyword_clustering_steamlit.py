import pandas as pd
import sys
from polyfuzz import PolyFuzz

import streamlit as st
import os

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  df_1 = pd.read_csv(uploaded_file)
  st.write(df_1)

# rename the parent cluster name using the keyword with the highest search volume (recommended)
parent_by_vol = True
drop_site_links = False
drop_image_links = False
sim_match_percent = 0.99
url_filter = ""

# --------------------------------- Check if csv data is gsc and set bool ----------------------------------------------

if 'Impressions' in df_1.columns:
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
        "Impressions": "Volume",
        "Clicks": "Traffic"
    },
    inplace=True,
)

# --------------------------------- clean the data pre-grouping --------------------------------------------------------
if url_filter:
    print("Processing only URLs containing:", url_filter)

try:
    df_1 = df_1[df_1["URL"].str.contains(url_filter, na=False)]
except KeyError:
    pass

df_1 = df_1[df_1["Keyword"].notna()]  # keep only rows which are NaN
df_1 = df_1[df_1["Volume"].notna()]  # keep only rows which are NaN
df_1["Volume"] = df_1["Volume"].astype(str)
df_1["Volume"] = df_1["Volume"].apply(lambda x: x.replace("0-10", "0"))
df_1["Volume"] = df_1["Volume"].astype(float).astype(int)


if drop_site_links:
    try:
        df_1 = df_1[~df_1["Page URL inside"].str.contains("Sitelinks", na=False)]  # drop sitelinks
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
        df_1 = df_1[~df_1["Page URL inside"].str.contains("Image pack", na=False)]  # drop image pack
    except KeyError:
        pass

# ------------------------------------- do the grouping ----------------------------------------------------------------

df_1_list = df_1.Keyword.tolist()  # create list from df
model = PolyFuzz("TF-IDF")
try:
    model.match(df_1_list, df_1_list)
except ValueError:
    print("Empty Dataframe, Can't Match - Check the URL Filter!")
    sys.exit()
model.group(link_min_similarity=sim_match_percent)
df_matched = model.get_matches()

# ------------------------------- clean the data post-grouping ---------------------------------------------------------

df_matched.rename(columns={"From": "Keyword", "Group": "Cluster Name"}, inplace=True)  # renaming multiple columns

# merge keyword volume / CPC / Pos / URL etc data from original dataframe back in
df_matched = pd.merge(df_matched, df_1, on="Keyword", how="left")

# rename traffic (acs) / (desc) to 'Traffic for standardisation
df_matched.rename(columns={"Traffic (desc)": "Traffic", "Traffic (asc)": "Traffic"}, inplace=True)

# fill in missing values
df_matched.fillna({"Traffic": 0, "CPC": 0}, inplace=True)
df_matched['Traffic'] = df_matched['Traffic'].round(0)

# ------------------------- group the data and merge in original stats -------------------------------------------------

try:
    # make dedicated grouped dataframe
    df_grouped = (df_matched.groupby("Cluster Name").agg(
        {"Volume": sum, "Difficulty": "median", "CPC": "median", "Traffic": sum}).reset_index())
except Exception:
    df_grouped = (df_matched.groupby("Cluster Name").agg(
        {"Volume": sum, "Traffic": sum}).reset_index())


df_grouped = df_grouped.rename(
    columns={"Volume": "Cluster Volume", "Difficulty": "Cluster KD (Median)", "CPC": "Cluster CPC (Median)",
             "Traffic": "Cluster Traffic"})

df_matched = pd.merge(df_matched, df_grouped, on="Cluster Name", how="left")  # merge in the group stats

# ---------------------------- clean and sort the final output ---------------------------------------------------------

try:
    df_matched.drop_duplicates(subset=["URL", "Keyword"], keep="first", inplace=True)  # drop if both kw & url are duped
except KeyError:
    pass

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
        cols = "Keyword", "Cluster Name", "Cluster Size", "Cluster Volume", "Cluster Traffic", "Volume", "Traffic"
        df_matched = df_matched.reindex(columns=cols)
except NameError:
    pass

# count cluster size
df_matched['Cluster Size'] = df_matched['Cluster Name'].map(df_matched.groupby('Cluster Name')['Cluster Name'].count())

df_matched.loc[df_matched['Cluster Size'] > 1, 'Clustered?'] = True
df_matched['Clustered?'] = df_matched['Clustered?'].fillna(False)

# ------------ get the keyword with the highest search volume to replace the auto generated tag name with --------------

if parent_by_vol:
    df_matched['vol_max'] = df_matched.groupby(['Cluster Name'])['Volume'].transform(max)
    # this sort is mandatory for the renaming to work properly by floating highest values to the top of the cluster
    df_matched.sort_values(["Cluster Name", "Cluster Volume", "Volume"], ascending=[False, True, False], inplace=True)
    df_matched['exact_vol_match'] = df_matched['vol_max'] == df_matched['Volume']
    df_matched.loc[df_matched['exact_vol_match'] == True, 'highest_ranked_keyword'] = df_matched['Keyword']
    df_matched['highest_ranked_keyword'] = df_matched['highest_ranked_keyword'].fillna(method='ffill')
    df_matched['Cluster Name'] = df_matched['highest_ranked_keyword']
    del df_matched['vol_max']
    del df_matched['exact_vol_match']
    del df_matched['highest_ranked_keyword']

# -------------------------------------- final output ------------------------------------------------------------------
# sort on cluster size
df_matched.sort_values(["Cluster Size", "Cluster Name", "Cluster Volume"], ascending=[False, True, False], inplace=True)

try:
    if gsc_data:
        df_matched.rename(columns={"Cluster Volume": "Cluster Impressions", "Cluster Traffic": "Cluster Clicks", "Traffic": "Clicks", "Volume": "Impressions"}, inplace=True)
except NameError:
    pass

st.write(df_matched)
if st.button('save dataframe'):
    open('df_matched.csv', 'w').write(df_matched.to_csv())
    csv = df_matched.to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    href = f'Download CSV File'
    st.markdown(href, unsafe_allow_html=True)
