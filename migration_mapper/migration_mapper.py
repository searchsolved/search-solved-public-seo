import pandas as pd
from polyfuzz import PolyFuzz
import time

# todo add options to
# canonicalised URLs to follow canonical
# don't map to non-indexable destination page. (Warning for staging crawls)
# match on content
# set priority match order e.g. content > url > title > h1
# ignore existing redirects (just change all status codes to 300

# start timing the script
startTime = time.time()

# read in the old crawl file, specify columns and datatypes
df_sf_old = pd.read_csv(
    "/python_scripts/migration_tool/internal_html_existing.csv",
    usecols=["Address", "Status Code", "Title 1", "H1-1"],
    dtype=str,
)

# read in the old crawl file, specify columns and datatypes
df_sf_new = pd.read_csv(
    "/python_scripts/migration_tool/internal_html_staging.csv",
    usecols=["Address", "Status Code", "Title 1", "H1-1", "Indexability"],
    dtype=str,
)

# this drops non-indexable urls from the destination crawl. (So nothing is mapped to 4xx / 3xx /non-canonical pages etc)
# df_sf_new = df_sf_new[~df_sf_new["Indexability"].isin(["Non-Indexable"])]

# drop duplicate rows
df_sf_old.drop_duplicates(subset="Address", inplace=True)
df_sf_new.drop_duplicates(subset="Address", inplace=True)

# make dataframe containing non-redirectable urls - 3xx & 5xx - this is exported to show urls which weren't redirected
df_3xx_export = df_sf_old[
    (
        df_sf_old["Status Code"].isin(
            [
                "300",
                "301",
                "302",
                "303",
                "304",
                "305",
                "307",
                "308",
                "500",
                "501",
                "502",
                "503",
                "504",
                "505",
                "506",
                "507",
                "508",
                "510",
                "511",
                "599",
            ]
        )
    )
]

# drops any rows which are not 2xx or 4xx - source 3xx will be exported to it's own file for review
df_sf_old = df_sf_old[
    (
        df_sf_old["Status Code"].isin(
            [
                "200",
                "201",
                "202",
                "203",
                "204",
                "205",
                "206",
                "207",
                "208",
                "226",
                "400",
                "401",
                "402",
                "403",
                "404",
                "405",
                "406",
                "407",
                "408",
                "409",
                "410",
                "411",
                "412",
                "413",
                "414",
                "415",
                "416",
                "417",
                "418",
                "421",
                "422",
                "423",
                "424",
                "426",
                "428",
                "429",
                "431",
                "444",
                "451",
                "499",
            ]
        )
    )
]

# this fills in title / H1-1 NaN for source 404 pages with the url instead.
df_sf_old["Title 1"] = df_sf_old["Title 1"].fillna(df_sf_old.Address)
df_sf_old["H1-1"] = df_sf_old["H1-1"].fillna(df_sf_old.Address)
df_sf_new["Title 1"] = df_sf_new["Title 1"].fillna(df_sf_new.Address)
df_sf_new["H1-1"] = df_sf_new["H1-1"].fillna(df_sf_new.Address)

# create lists from dfs
df_sf_old_address_list = list(df_sf_old["Address"])
df_sf_new_address_list = list(df_sf_new["Address"])
df_sf_old_title_list = list(df_sf_old["Title 1"])
df_sf_new_title_list = list(df_sf_new["Title 1"])
df_sf_old_h1_list = list(df_sf_old["H1-1"])
df_sf_new_h1_list = list(df_sf_new["H1-1"])

# instantiate PolyFuzz model, choose TF-IDF as the similarity measure and match the two lists.
model_address = PolyFuzz("TF-IDF").match(df_sf_old_address_list, df_sf_new_address_list)
model_title = PolyFuzz("TF-IDF").match(df_sf_old_title_list, df_sf_new_title_list)
model_h1 = PolyFuzz("TF-IDF").match(df_sf_old_h1_list, df_sf_new_h1_list)

# make the polyfuzz dataframes
df_pf_address = model_address.get_matches()
df_pf_title = model_title.get_matches()
df_pf_h1 = model_h1.get_matches()

# rename the similarity cols
df_pf_address.rename(columns={"Similarity": "URL Similarity", "From": "From (Address)", "To": "To Address"}, inplace=True)
df_pf_title.rename(columns={"Similarity": "Title Similarity", "From": "From (Title)", "To": "To Title"}, inplace=True)
df_pf_h1.rename(columns={"Similarity": "H1 Similarity", "From": "From (H1)", "To": "To H1"}, inplace=True)

# make title df for merge from new_df
df_new_title = df_sf_new[['Title 1', 'Address']]

# make title df for merge from new_df
df_new_h1 = df_sf_new[['H1-1', 'Address']]

# merge the URL column back in
df_pf_title_merge = pd.merge(df_pf_title, df_new_title, left_on="To Title", right_on="Title 1", how="inner")
df_pf_h1_merge = pd.merge(df_pf_h1, df_new_h1, left_on="To H1", right_on="H1-1", how="inner")

# merge back into df_sf_old
df_final = pd.merge(df_sf_old, df_pf_address, left_on="Address", right_on="From (Address)", how="inner")
df_final = df_final.merge(df_pf_title_merge.drop_duplicates('Title 1'),how='left',left_on='Title 1', right_on="From (Title)")
df_final = df_final.merge(df_pf_h1_merge.drop_duplicates('H1-1'),how='left',left_on='H1-1', right_on="From (H1)")

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
df_final['Highest Match On'] = df_final[["URL Similarity", "Title Similarity", "H1 Similarity"]].idxmax(axis=1)
# Check Value in Max Column and Replace with a Different Cell
df_final.loc[df_final['Highest Match On'] == "Title Similarity", 'Highest Match Similarity'] = df_final['Title Similarity']
df_final.loc[df_final['Highest Match On'] == "Title Similarity", 'Highest Match URL'] = df_final['URL - Title Match']
df_final.loc[df_final['Highest Match On'] == "H1 Similarity", 'Highest Match Similarity'] = df_final['H1 Similarity']
df_final.loc[df_final['Highest Match On'] == "H1 Similarity", 'Highest Match URL'] = df_final['URL - H1 Match']
df_final.loc[df_final['Highest Match On'] == "URL Similarity", 'Highest Match Similarity'] = df_final['URL Similarity']
df_final.loc[df_final['Highest Match On'] == "URL Similarity", 'Highest Match URL'] = df_final['URL - URL Match']
df_final.drop_duplicates(subset="URL - Source", inplace=True)

# this routine gets the minimum value of the matches. The idea is that when the min and max values are removed, only
# the middle value remains. (e.g. the Second Highest Matching URL).
# get the max value across the three cells
df_final['Lowest Match On'] = df_final[["URL Similarity", "Title Similarity", "H1 Similarity"]].idxmin(axis=1)

# this populates a column with all available values. Matching values are subtracted to leave correct value in place.
df_final['Middle Match On'] = "URL Similarity Title Similarity H1 Similarity"

df_final['Middle Match On'] = df_final.apply(lambda x : x['Middle Match On'].replace((x['Highest Match On']), ''), 1)
df_final['Middle Match On'] = df_final.apply(lambda x : x['Middle Match On'].replace((x['Lowest Match On']), ''), 1)

# strip out the whitespace
df_final['Middle Match On'] = df_final['Middle Match On'].str.strip()
df_final.to_csv('/python_scripts/test-192.csv')
df_final.loc[df_final['Middle Match On'] == "Title Similarity", 'Middle Match URL'] = df_final["URL - Title Match"]
df_final.loc[df_final['Middle Match On'] == "H1 Similarity", 'Middle Match URL'] = df_final["URL - H1 Match"]
df_final.loc[df_final['Middle Match On'] == "URL Similarity", 'Middle Match URL'] = df_final["URL - URL Match"]
df_final.to_csv('/python_scripts/test-197.csv')
# rename the secondary match column
df_final.rename(columns={"Middle Match URL": "Second Highest Match"}, inplace=True)  # renaming multiple columns
df_final.rename(columns={"Middle Match On": "Second Highest Match On"}, inplace=True)  # renaming multiple columns

# re-order / index the highest match dataframe columns
new_cols = (
    "URL - Source",
    "Status Code",
    "Highest Match URL",
    "Highest Match On",
    "Highest Match Similarity",
    "Highest Match Source Text",
    "Highest Match Destination Text",
    "Second Highest Match",
    "Second Highest Match On",
    "Second Highest Match Similarity",
    "Second Highest Match Source Text",
    "Second Highest Match Destination Text",
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
df_final.loc[df_final["Highest Match On"] == "H1 Similarity", "Highest Match Source Text"] = df_final["From (H1)"]
df_final.loc[df_final["Highest Match On"] == "H1 Similarity", "Highest Match Destination Text"] = df_final["To H1"]
df_final.loc[df_final["Highest Match On"] == "Title Similarity", "Highest Match Source Text"] = df_final["From (Title)"]
df_final.loc[df_final["Highest Match On"] == "Title Similarity", "Highest Match Destination Text"] = df_final["To Title"]
df_final.loc[df_final["Highest Match On"] == "URL Similarity", "Highest Match Source Text"] = df_final["URL - Source"]
df_final.loc[df_final["Highest Match On"] == "URL Similarity", "Highest Match Destination Text"] = df_final["URL - URL Match"]

# # Check Value in Max Column and Replace with a Different Cell
df_final.loc[df_final["Second Highest Match On"] == "H1 Similarity", "Second Highest Match Source Text"] = df_final["From (H1)"]
df_final.loc[df_final["Second Highest Match On"] == "H1 Similarity", "Second Highest Match Destination Text"] = df_final["To H1"]
df_final.loc[df_final["Second Highest Match On"] == "Title Similarity", "Second Highest Match Source Text"] = df_final["From (Title)"]
df_final.loc[df_final["Second Highest Match On"] == "Title Similarity", "Second Highest Match Destination Text"] = df_final["To Title"]
df_final.loc[df_final["Second Highest Match On"] == "URL Similarity", "Second Highest Match Source Text"] = df_final["URL - Source"]
df_final.loc[df_final["Second Highest Match On"] == "URL Similarity", "Second Highest Match Destination Text"] = df_final["URL - URL Match"]

# # Check Value in Max Column and Replace with a Different Cell
df_final.loc[df_final["Lowest Match On"] == "H1 Similarity", "Lowest Match Source Text"] = df_final["From (H1)"]
df_final.loc[df_final["Lowest Match On"] == "H1 Similarity", "Lowest Match Destination Text"] = df_final["To H1"]
df_final.loc[df_final["Lowest Match On"] == "Title Similarity", "Lowest Match Source Text"] = df_final["From (Title)"]
df_final.loc[df_final["Lowest Match On"] == "Title Similarity", "Lowest Match Destination Text"] = df_final["To Title"]
df_final.loc[df_final["Lowest Match On"] == "URL Similarity", "Lowest Match Source Text"] = df_final["URL - Source"]
df_final.loc[df_final["Lowest Match On"] == "URL Similarity", "Lowest Match Destination Text"] = df_final["URL - URL Match"]

# get missing similarity scores
df_final.loc[df_final["Second Highest Match On"] == "H1 Similarity", "Second Highest Match Similarity"] = df_final["H1 Similarity"]
df_final.loc[df_final["Second Highest Match On"] == "Title Similarity", "Second Highest Match Similarity"] = df_final["Title Similarity"]
df_final.loc[df_final["Second Highest Match On"] == "URL Similarity", "Second Highest Match Similarity"] = df_final["URL Similarity"]

df_final.loc[df_final["Lowest Match On"] == "H1 Similarity", "Lowest Match Similarity"] = df_final["H1 Similarity"]
df_final.loc[df_final["Lowest Match On"] == "Title Similarity", "Lowest Match Similarity"] = df_final["Title Similarity"]
df_final.loc[df_final["Lowest Match On"] == "URL Similarity", "Lowest Match Similarity"] = df_final["URL Similarity"]

# check if both url recommendations are the same
df_final["Best / Second Best URLs Match?"] = df_final['Highest Match URL'].str.lower() == df_final['Second Highest Match'].str.lower()

# rename highest match values for final output
df_final['Highest Match On'] = df_final['Highest Match On'].apply(lambda x: x.replace("Title Similarity", "Page Title"))
df_final['Highest Match On'] = df_final['Highest Match On'].apply(lambda x: x.replace("H1 Similarity", "H1 Heading"))
df_final['Highest Match On'] = df_final['Highest Match On'].apply(lambda x: x.replace("URL Similarity", "URL"))

# last re-index - set the final column order - bin off thehelper columns

cols = (
    "URL - Source",
    "Status Code",
    "Highest Match URL",
    "Highest Match On",
    "Highest Match Similarity",
    "Highest Match Source Text",
    "Highest Match Destination Text",
    "Second Highest Match",
    "Second Highest Match On",
    "Second Highest Match Similarity",
    "Second Highest Match Source Text",
    "Second Highest Match Destination Text",
    "Lowest Match On",
    "Lowest Match Similarity",
    "Lowest Match Source Text",
    "Lowest Match Destination Text",
    "Best / Second Best URLs Match?",
)
df_final = df_final.reindex(columns=cols)

# routine to count how many urls matched to 100%
all_rows = df_final.shape[0]

# round the sim scores
df_final["Highest Match Similarity"] = df_final["Highest Match Similarity"].round(2)

# routine to check 100% matches
match_100 = df_final["Highest Match Similarity"] == 1.000000
df_100 = pd.DataFrame(match_100)
df_100_count = df_100[df_100["Highest Match Similarity"].isin([True])]
df_100_count = df_100_count.shape[0]

# routine to check 90+% matches
match_90 = df_final.loc[(df_final["Highest Match Similarity"] >= 0.90) & (df_final["Highest Match Similarity"] <= 0.99)]
df_90 = pd.DataFrame(match_90)
df_90_count = df_90[~df_90["Highest Match Similarity"].isin([True])]
df_90_count = df_90_count.shape[0]

# routine to check 80+% matches
match_80 = df_final.loc[(df_final["Highest Match Similarity"] >= 0.80) & (df_final["Highest Match Similarity"] <= 0.89)]
df_80 = pd.DataFrame(match_80)
df_80_count = df_80[~df_80["Highest Match Similarity"].isin([True])]
df_80_count = df_80_count.shape[0]

# routine to check 70+% matches
match_70 = df_final.loc[(df_final["Highest Match Similarity"] >= 0.70) & (df_final["Highest Match Similarity"] <= 0.79)]
df_70 = pd.DataFrame(match_70)
df_70_count = df_70[~df_70["Highest Match Similarity"].isin([True])]
df_70_count = df_70_count.shape[0]

# routine to check 60+% matches
match_60 = df_final.loc[(df_final["Highest Match Similarity"] >= 0.60) & (df_final["Highest Match Similarity"] <= 0.69)]
df_60 = pd.DataFrame(match_60)
df_60_count = df_60[~df_60["Highest Match Similarity"].isin([True])]
df_60_count = df_60_count.shape[0]

# routine to check 50+% matches
match_50 = df_final.loc[(df_final["Highest Match Similarity"] >= 0.50) & (df_final["Highest Match Similarity"] <= 0.59)]
df_50 = pd.DataFrame(match_50)
df_50_count = df_50[~df_50["Highest Match Similarity"].isin([True])]
df_50_count = df_50_count.shape[0]

# routine to check 40+% matches
match_40 = df_final.loc[(df_final["Highest Match Similarity"] >= 0.40) & (df_final["Highest Match Similarity"] <= 0.49)]
df_40 = pd.DataFrame(match_40)
df_40_count = df_40[~df_40["Highest Match Similarity"].isin([True])]
df_40_count = df_40_count.shape[0]

# routine to check 30+% matches
match_30 = df_final.loc[(df_final["Highest Match Similarity"] >= 0.30) & (df_final["Highest Match Similarity"] <= 0.39)]
df_30 = pd.DataFrame(match_30)
df_30_count = df_30[~df_30["Highest Match Similarity"].isin([True])]
df_30_count = df_30_count.shape[0]

# routine to check 20+% matches
match_20 = df_final.loc[(df_final["Highest Match Similarity"] >= 0.20) & (df_final["Highest Match Similarity"] <= 0.29)]
df_20 = pd.DataFrame(match_20)
df_20_count = df_20[~df_20["Highest Match Similarity"].isin([True])]
df_20_count = df_20_count.shape[0]

# routine to check 10+% matches
match_10 = df_final.loc[(df_final["Highest Match Similarity"] >= 0.10) & (df_final["Highest Match Similarity"] <= 0.19)]
df_10 = pd.DataFrame(match_10)
df_10_count = df_10[~df_10["Highest Match Similarity"].isin([True])]
df_10_count = df_10_count.shape[0]

# routine to check 00+% matches
match_00 = df_final.loc[(df_final["Highest Match Similarity"] >= 0.00) & (df_final["Highest Match Similarity"] <= 0.09)]
df_00 = pd.DataFrame(match_00)
df_00_count = df_00[~df_00["Highest Match Similarity"].isin([True])]
df_00_count = df_00_count.shape[0]

# calculate percentage matches
percent_match_100 = (df_100_count / all_rows * 100)
percent_match_90 = (df_90_count / all_rows * 100)
percent_match_80 = (df_80_count / all_rows * 100)
percent_match_70 = (df_70_count / all_rows * 100)
percent_match_60 = (df_60_count / all_rows * 100)
percent_match_50 = (df_50_count / all_rows * 100)
percent_match_40 = (df_40_count / all_rows * 100)
percent_match_30 = (df_30_count / all_rows * 100)
percent_match_20 = (df_20_count / all_rows * 100)
percent_match_10 = (df_10_count / all_rows * 100)
percent_match_00 = (df_00_count / all_rows * 100)

# calculate percentage matches
percent_match_100 = round(percent_match_100, 2)
percent_match_90 = round(percent_match_90, 2)
percent_match_80 = round(percent_match_80, 2)
percent_match_70 = round(percent_match_70, 2)
percent_match_60 = round(percent_match_60, 2)
percent_match_50 = round(percent_match_50, 2)
percent_match_40 = round(percent_match_40, 2)
percent_match_30 = round(percent_match_30, 2)
percent_match_20 = round(percent_match_20, 2)
percent_match_10 = round(percent_match_10, 2)
percent_match_00 = round(percent_match_00, 2)

# print out the match results
print("          __          __")
print("  _______/  |______ _/  |_  ______")
print(" /  ___/\   __\__  \\   __\/  ___/")
print(" \___ \  |  |  / __ \|  |  \___ \ ")
print("/____  > |__| (____  /__| /____  >")
print("     \/            \/          \/")

print("\nURLs Matched with 100% Similarity: ", percent_match_100,"%")
print("URLs Matched with 90% Similarity:  ", percent_match_90,"%")
print("URLs Matched with 80% Similarity:  ", percent_match_80,"%")
print("URLs Matched with 70% Similarity:  ", percent_match_70,"%")
print("URLs Matched with 60% Similarity:  ", percent_match_60,"%")
print("URLs Matched with 50% Similarity:  ", percent_match_50,"%")
print("URLs Matched with 40% Similarity:  ", percent_match_40,"%")
print("URLs Matched with 30% Similarity:  ", percent_match_30,"%")
print("URLs Matched with 20% Similarity:  ", percent_match_20,"%")
print("URLs Matched with 10% Similarity:  ", percent_match_10,"%")
print("URLs Matched with 00% Similarity:  ", percent_match_00,"%")
print("")

# sort the final export
df_final.sort_values(["Highest Match Similarity", "Best / Second Best URLs Match?"], ascending=[False, False], inplace=True,)

# make high confidence dataframe (100% match, double matched - 1st & secondary urls are the same)
df_high_confidence = df_final.copy()
df_high_confidence = df_high_confidence[df_high_confidence["Highest Match Similarity"].isin([1])]
df_high_confidence = df_high_confidence[df_high_confidence["Best / Second Best URLs Match?"].isin([True])]

# make low confidence dataframe
df_low_confidence = df_final.copy()
df_low_confidence = pd.concat([df_low_confidence,df_high_confidence]).drop_duplicates(keep=False)

# count rows for final stats
count_high_confidence = df_high_confidence.shape[0]
count_low_confidence = df_low_confidence.shape[0]
df_3xx_export_count = df_3xx_export.shape[0]

# print more stats
high_confidence_percent = (count_high_confidence / all_rows * 100)
low_confidence_percent = (count_low_confidence / all_rows * 100)
df_3xx_export_percent = (df_3xx_export_count / all_rows * 100)

high_confidence_percent = round(high_confidence_percent, 2)
low_confidence_percent = round(low_confidence_percent, 2)
df_3xx_export_percent = round(df_3xx_export_percent, 2)

print("URLs Doubled Matched @ 100% Confidence:", high_confidence_percent, "%", "\U0001F40D \U0001F525")
print("URLs To Be Manually Reviewed: ", low_confidence_percent, "%")
print("Dropped 3xx & 5xx Source URLs:", df_3xx_export_percent, "%")

# df_low_confidence.to_csv('/python_scripts/auto-migration-mapped-manual-review.csv', index=False)
# df_high_confidence.to_csv('/python_scripts/auto-migration-mapped-high-confidence.csv', index=False)
df_final.to_csv('/python_scripts/migration_tool/auto-migration-mapped-all-output.csv', index=False)
df_3xx_export.to_csv('/python_scripts/migration_tool/auto-migration-non-redirectable-urls.csv', index=False)

print("")
print(all_rows, "URLs Migrated in {0} seconds!".format(time.time() - startTime))
