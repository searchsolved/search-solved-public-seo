####################################################################################
# Website  : https://leefoot.co.uk/                                                #
# Contact  : https://leefoot.co.uk/hire-me/                                        #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Twitter  : https://twitter.com/LeeFootSEO                                        #
####################################################################################

Internal Search Report Mapper V2 by Lee Foot 11/01/2021 @LeeFootSEO
Takes the Google Analytics search terms report and merges with a Screaming Frog crawl file to find opportunities to map
internal searches to.

1) Search Terms Report must be exported as an Excel file.
2) Screaming Frog Crawl should ideally contain just category pages, (unless it makes sense for you to map to products).
3) Paths are hardcoded, please enter your own for GA, Screaming Frog and the output csv files
4) V2 Now contains fuzzy matching!"""


from polyfuzz import PolyFuzz
from glob import glob
import pandas as pd

# set the folder paths HERE for your input files!
ga_path = "C:\python_scripts\Internal Search Mapper"  # enter the path the the folder that contains your GA search terms export
sf_path = "C:\python_scripts\Internal Search Mapper"  # enter the path of the folder that contains internal_html.csv file
export_path = "C:\python_scripts\Internal Search Mapper"  # enter the path to export the output.csv file to

# adds file names to the paths.
ga_path_file = ga_path + "/Analytics*.xlsx"
sf_path_file = sf_path + "/internal_html.csv"
export_exact = export_path + "/search-mapping-exact-matches.csv"
export_partial = export_path + "/search-mapping-partial-matches.csv"
export_all = export_path + "/search-mapping-all-matches.csv"

# imports GA data using a wildcard match
for f in glob(ga_path_file):
    df_ga = pd.read_excel((f), sheet_name="Dataset1")

# import screaming frog internal_html columns
df_sf = pd.read_csv(sf_path_file, encoding="utf8")[["H1-1", "Address", "Indexability"]]

# convert to lower case for matching
df_sf["H1-1"] = df_sf["H1-1"].str.lower()
df_ga["Search Term"] = df_ga[
    "Search Term"
].str.lower()  # convert to lower case for matching

# drop non-indexable pages
try:
    df_sf = df_sf[~df_sf["Indexability"].isin(["Non-Indexable"])]
except Exception:
    pass

# delete the helper column
del df_sf["Indexability"]

# keep rows which are not NaN
df_ga = df_ga[df_ga["Search Term"].notna()]
df_sf = df_sf[df_sf["H1-1"].notna()]

# create lists from dfs
ga_list = list(df_ga["Search Term"])
sf_list = list(df_sf["H1-1"])

# instantiate PolyFuzz model, choose TF-IDF as the similarity measure and match the two lists.
model = PolyFuzz("TF-IDF").match(ga_list, sf_list)

# make the polyfuzz dataframe
df_pf_matches = model.get_matches()
# keep only rows which are not NaN
df_pf_matches = df_pf_matches[df_pf_matches["To"].notna()]

# merge original ga search term data back into polyfuzz df
df_pf_matches_df_ga = pd.merge(
    df_pf_matches, df_ga, left_on="From", right_on="Search Term", how="inner"
)

# create final_df + merge original screaming frog data back in
final_df = pd.merge(df_pf_matches_df_ga, df_sf, left_on="To", right_on="H1-1")

# delete redundant columns
del final_df["Search Term"]
del final_df["H1-1"]

# sort by opportunity
final_df = final_df.sort_values(by="Total Unique Searches", ascending=False)

# Round Float to two decimal places
final_df = final_df.round(decimals=2)

# rename the cols
final_df.rename(
    columns={"From": "Search Term", "To": "Matched H1", "Address": "Matched URL"},
    inplace=True,
)

# set new column order for final df
cols = [
    "Search Term",
    "Matched H1",
    "Similarity",
    "Matched URL",
    "Total Unique Searches",
    "Results Page Views/Search",
    "% Search Exits",
    "% Search Refinements",
    "Time After Search",
    "Avg. Search Depth",
]

# re-index columns into a logical order
final_df = final_df.reindex(columns=cols)

# drop duplicate keywords
final_df.drop_duplicates(subset=["Search Term"], inplace=True)

# export the final csv
final_df_exact = final_df.loc[final_df["Similarity"] == 1]
final_df_partial = final_df.loc[final_df["Similarity"] != 1].copy()

final_df_partial.sort_values(
    ["Similarity", "Total Unique Searches"],
    ascending=[False, False],
    inplace=True,
)

final_df_exact.to_csv(export_exact, index=False)
final_df_partial.to_csv(export_partial, index=False)
final_df.to_csv(export_all, index=False)
