import pandas as pd
from polyfuzz import PolyFuzz
import time

# todo add options to
# only redirect to canonical url - generate warning to respect canonical when crawling staging site.
# Option to match on content
# set priority match order e.g. content > url > title > h1

# start timing the script
startTime = time.time()

# ---------------------------------------------------------------------------------------------- read in the crawl files
df_live = pd.read_csv(
    "/python_scripts/migration_mapper/pre.csv",
    usecols=["Address", "Status Code", "Title 1", "H1-1"],
    dtype={'Address': str, 'Status Code': int, 'Title 1': str, 'H1-1': str}
)

df_staging = pd.read_csv(
    "/python_scripts/migration_mapper/post.csv",
    usecols=["Address", "Status Code", "Title 1", "H1-1", "Indexability"],
    dtype={'Address': str, 'Status Code': int, 'Title 1': str, 'H1-1': str}
)

# -------------------------------------------------------------------------------------------------- drop duplicate rows

df_live.drop_duplicates(subset="Address", inplace=True)
df_staging.drop_duplicates(subset="Address", inplace=True)

# make dataframe containing non-redirectable urls - 3xx & 5xx - this is exported to show urls which weren't redirected
df_3xx = df_live[(df_live['Status Code'] >= 300) & (df_live['Status Code'] <= 308)]
df_5xx = df_live[(df_live['Status Code'] >= 500) & (df_live['Status Code'] <= 599)]
df_3xx_5xx = pd.concat([df_3xx, df_5xx])

# keep 2xx and 4xx status codes for redirecting
df_live_200 = df_live[(df_live['Status Code'] >= 200) & (df_live['Status Code'] <= 226)]
df_live_400 = df_live[(df_live['Status Code'] >= 400) & (df_live['Status Code'] <= 499)]
df_live = pd.concat([df_live_200, df_live_400])

# --------------------------------------------------------------------------------------------------------- 404 handling

"""As 404s won't have a field populated for the h1 and title, the will need to be matched on the URL. 
THis section populates the empty values, (NaN's) with the URL. This is preferable to an empty string as an empty
string will match 100% to another empty value. By populating something unique, we can ensure that doesn't 
happen, and that the URL column will always be the highest matching value"""

df_live["Title 1"] = df_live["Title 1"].fillna(df_live.Address)
df_live["H1-1"] = df_live["H1-1"].fillna(df_live.Address)
df_staging["Title 1"] = df_staging["Title 1"].fillna(df_staging.Address)
df_staging["H1-1"] = df_staging["H1-1"].fillna(df_staging.Address)

# instantiate PolyFuzz model, choose TF-IDF as the similarity measure and match the two lists.
model_address = PolyFuzz("TF-IDF").match(list(df_live["Address"]), list(df_staging["Address"]))
model_title = PolyFuzz("TF-IDF").match(list(df_live["Title 1"]), list(df_staging["Title 1"]))
model_h1 = PolyFuzz("TF-IDF").match(list(df_live["H1-1"]), list(df_staging["H1-1"]))

# make the polyfuzz dataframes
df_pf_url = model_address.get_matches()
df_pf_title = model_title.get_matches()
df_pf_h1 = model_h1.get_matches()

# rename the similarity cols
df_pf_url.rename(columns={"Similarity": "URL Similarity", "From": "From (Address)", "To": "To Address"}, inplace=True)
df_pf_title.rename(columns={"Similarity": "Title Similarity", "From": "From (Title)", "To": "To Title"}, inplace=True)
df_pf_h1.rename(columns={"Similarity": "H1 Similarity", "From": "From (H1)", "To": "To H1"}, inplace=True)

# make title df for merge from new_df
df_new_title = df_staging[['Title 1', 'Address']]

# make h1 df for merge from new_df
df_new_h1 = df_staging[['H1-1', 'Address']]

# merge the URL column back in
df_pf_title_merge = pd.merge(df_pf_title, df_new_title, left_on="To Title", right_on="Title 1", how="inner")
df_pf_h1_merge = pd.merge(df_pf_h1, df_new_h1, left_on="To H1", right_on="H1-1", how="inner")

# merge back into df_live
df_final = pd.merge(df_live, df_pf_url, left_on="Address", right_on="From (Address)", how="inner")
df_final = df_final.merge(df_pf_title_merge.drop_duplicates('Title 1'), how='left', left_on='Title 1', right_on="From (Title)")
df_final = df_final.merge(df_pf_h1_merge.drop_duplicates('H1-1'), how='left', left_on='H1-1', right_on="From (H1)")

# rename the columns
df_final.rename(
    columns={
        "Address_x": "URL - Source",
        "To Address": "URL - URL Match",
        "Address_y": "URL - Title Match",
        "Address": "URL - H1 Match",
    },
    inplace=True,
)

# get the max value across the three cells
df_final['Best Match On'] = df_final[["URL Similarity", "Title Similarity", "H1 Similarity"]].idxmax(axis=1)

# Check Value in Max Column and Replace with a Different Cell
df_final.loc[df_final['Best Match On'] == "Title Similarity", 'Highest Match Similarity'] = df_final[
    'Title Similarity']
df_final.loc[df_final['Best Match On'] == "Title Similarity", 'Best Matching URL'] = df_final['URL - Title Match']
df_final.loc[df_final['Best Match On'] == "H1 Similarity", 'Highest Match Similarity'] = df_final['H1 Similarity']
df_final.loc[df_final['Best Match On'] == "H1 Similarity", 'Best Matching URL'] = df_final['URL - H1 Match']
df_final.loc[df_final['Best Match On'] == "URL Similarity", 'Highest Match Similarity'] = df_final['URL Similarity']
df_final.loc[df_final['Best Match On'] == "URL Similarity", 'Best Matching URL'] = df_final['URL - URL Match']
df_final.drop_duplicates(subset="URL - Source", inplace=True)

"""This routine gets the minimum value of the matches. The idea is that when the min and max values are removed, 
only the middle value remains. (e.g. the Second Highest Matching URL)."""

# get the max value across the three cells
df_final['Lowest Match On'] = df_final[["URL Similarity", "Title Similarity", "H1 Similarity"]].idxmin(axis=1)

# this populates a column with all available values. Matching values are subtracted to leave correct value in place.
df_final['Middle Match On'] = "URL Similarity Title Similarity H1 Similarity"
df_final['Middle Match On'] = df_final.apply(lambda x: x['Middle Match On'].replace((x['Best Match On']), ''), 1)
df_final['Middle Match On'] = df_final.apply(lambda x: x['Middle Match On'].replace((x['Lowest Match On']), ''), 1)
df_final['Middle Match On'] = df_final['Middle Match On'].str.strip()  # strip out the whitespace

df_final.loc[df_final['Middle Match On'] == "Title Similarity", 'Middle Match URL'] = df_final["URL - Title Match"]
df_final.loc[df_final['Middle Match On'] == "H1 Similarity", 'Middle Match URL'] = df_final["URL - H1 Match"]
df_final.loc[df_final['Middle Match On'] == "URL Similarity", 'Middle Match URL'] = df_final["URL - URL Match"]

# rename the secondary match column
df_final.rename(columns={"Middle Match URL": "Second Highest Match"}, inplace=True)
df_final.rename(columns={"Middle Match On": "Second Match On"}, inplace=True)

# re-order / index the highest match dataframe columns
new_cols = (
    "URL - Source",
    "Status Code",
    "Best Matching URL",
    "Best Match On",
    "Highest Match Similarity",
    "Highest Match Source Text",
    "Highest Match Destination Text",
    "Second Highest Match",
    "Second Match On",
    "Second Highest Match Similarity",
    "Second Match Source Text",
    "Second Match Destination Text",
    "Lowest Match On",
    "Lowest Match Similarity",
    "Lowest Match Source Text",
    "Lowest Match Destination Text",
    "URL - URL Match",
    "URL - H1 Match",
    "URL - Title Match",
    "From (H1)",
    "To H1",
    "From (Title)",
    "To Title",
    "URL Similarity",
    "H1 Similarity",
    "Title Similarity",
)

df_final = df_final.reindex(columns=new_cols)

# # Check Value in Max Column and Replace with a Different Cell
df_final.loc[df_final["Best Match On"] == "H1 Similarity", "Highest Match Source Text"] = df_final["From (H1)"]
df_final.loc[df_final["Best Match On"] == "H1 Similarity", "Highest Match Destination Text"] = df_final["To H1"]
df_final.loc[df_final["Best Match On"] == "Title Similarity", "Highest Match Source Text"] = df_final["From (Title)"]
df_final.loc[df_final["Best Match On"] == "Title Similarity", "Highest Match Destination Text"] = df_final["To Title"]
df_final.loc[df_final["Best Match On"] == "URL Similarity", "Highest Match Source Text"] = df_final["URL - Source"]
df_final.loc[df_final["Best Match On"] == "URL Similarity", "Highest Match Destination Text"] = df_final["URL - URL Match"]

# # Check Value in Max Column and Replace with a Different Cell
df_final.loc[df_final["Second Match On"] == "H1 Similarity", "Second Match Source Text"] = df_final["From (H1)"]
df_final.loc[df_final["Second Match On"] == "H1 Similarity", "Second Match Destination Text"] = df_final["To H1"]
df_final.loc[df_final["Second Match On"] == "Title Similarity", "Second Match Source Text"] = df_final["From (Title)"]
df_final.loc[df_final["Second Match On"] == "Title Similarity", "Second Match Destination Text"] = df_final["To Title"]
df_final.loc[df_final["Second Match On"] == "URL Similarity", "Second Match Source Text"] = df_final["URL - Source"]
df_final.loc[df_final["Second Match On"] == "URL Similarity", "Second Match Destination Text"] = df_final["URL - URL Match"]

# # Check Value in Max Column and Replace with a Different Cell
df_final.loc[df_final["Lowest Match On"] == "H1 Similarity", "Lowest Match Source Text"] = df_final["From (H1)"]
df_final.loc[df_final["Lowest Match On"] == "H1 Similarity", "Lowest Match Destination Text"] = df_final["To H1"]
df_final.loc[df_final["Lowest Match On"] == "Title Similarity", "Lowest Match Source Text"] = df_final["From (Title)"]
df_final.loc[df_final["Lowest Match On"] == "Title Similarity", "Lowest Match Destination Text"] = df_final["To Title"]
df_final.loc[df_final["Lowest Match On"] == "URL Similarity", "Lowest Match Source Text"] = df_final["URL - Source"]
df_final.loc[df_final["Lowest Match On"] == "URL Similarity", "Lowest Match Destination Text"] = df_final["URL - URL Match"]

# get missing similarity scores
df_final.loc[df_final["Second Match On"] == "H1 Similarity", "Second Highest Match Similarity"] = df_final[
    "H1 Similarity"]
df_final.loc[df_final["Second Match On"] == "Title Similarity", "Second Highest Match Similarity"] = df_final[
    "Title Similarity"]
df_final.loc[df_final["Second Match On"] == "URL Similarity", "Second Highest Match Similarity"] = df_final[
    "URL Similarity"]

df_final.loc[df_final["Lowest Match On"] == "H1 Similarity", "Lowest Match Similarity"] = df_final["H1 Similarity"]
df_final.loc[df_final["Lowest Match On"] == "Title Similarity", "Lowest Match Similarity"] = df_final["Title Similarity"]
df_final.loc[df_final["Lowest Match On"] == "URL Similarity", "Lowest Match Similarity"] = df_final["URL Similarity"]

# check if both url recommendations are the same
df_final["Double Matched?"] = df_final['Best Matching URL'].str.lower() == df_final['Second Highest Match'].str.lower()

# rename highest match values for final output
df_final['Best Match On'] = df_final['Best Match On'].apply(lambda x: x.replace("Title Similarity", "Page Title"))
df_final['Best Match On'] = df_final['Best Match On'].apply(lambda x: x.replace("H1 Similarity", "H1 Heading"))
df_final['Best Match On'] = df_final['Best Match On'].apply(lambda x: x.replace("URL Similarity", "URL"))

# set the final column order - bin off the helper columns
cols = (
    "URL - Source",
    "Status Code",
    "Best Matching URL",
    "Best Match On",
    "Highest Match Similarity",
    "Highest Match Source Text",
    "Highest Match Destination Text",
    "Second Highest Match",
    "Second Match On",
    "Second Highest Match Similarity",
    "Second Match Source Text",
    "Second Match Destination Text",
    "Double Matched?",
)

df_final = df_final.reindex(columns=cols)

df_final.sort_values(["Highest Match Similarity", "Double Matched?"], ascending=[False, False], inplace=True)

df_final.to_csv('/python_scripts/migration_mapper/auto-migration-mapped-all-output.csv', index=False)
df_3xx_5xx.to_csv('/python_scripts/migration_mapper/auto-migration-non-redirectable-urls.csv', index=False)

print("URLs Migrated in {0} seconds!".format(time.time() - startTime))
