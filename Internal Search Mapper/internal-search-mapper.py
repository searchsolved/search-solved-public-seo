""" Internal Search Report Mapper V1.1 by Lee Foot 07/01/2021 @LeeFootSEO
Takes the Google Analytics search terms report and merges with a Screaming Frog crawl file to find opportunities to map
internal searches to.

1) Search Terms Report must be exported as an Excel file
2) Screaming Frog Crawl should only contain category pages, (unless you want to map internal searches to products!)
3) Paths are hardcoded, please enter your own for GA, Screaming Frog and the Output.CSV"""

from glob import glob
import pandas as pd

# set the folder paths HERE for your input files!
ga_path = "C:\path-to-folder-with-ga-export"  # enter the path the the folder that contains your GA search terms export
sf_path = "C:\path-to-folder-with-sf-export"  # enter the path of the folder that contains internal_html.csv file
export_path = "C:\path-to-export-folder"  # enter the path to export the output.csv file to

# adds file names to the paths.
ga_path_file = ga_path + "/Analytics*.xlsx"
sf_path_file = sf_path + "/internal_html.csv"
export_path_file = export_path + "/output.csv"

# imports GA data using a wildcard match
for f in glob(ga_path_file):
    df_ga = pd.read_excel((f), sheet_name="Dataset1")

# imports screaming frog columns
df_sf = pd.read_csv(sf_path_file, encoding="utf8")[["H1-1", "Address", "Indexability"]]

# convert to lower case for matching
df_sf["H1-1"] = df_sf["H1-1"].str.lower()
df_ga["Search Term"] = df_ga[
    "Search Term"
].str.lower()  # convert to lower case for matching
df_ga.drop_duplicates(subset=["Search Term"], inplace=True)  # drop duplicates

# drop non-indexable pages
try:
    df_sf = df_sf[~df_sf["Indexability"].isin(["Non-Indexable"])]
except Exception:
    pass

# delete the helper column
del df_sf["Indexability"]  

# merge the dataframe
final_df = pd.merge(df_sf, df_ga, left_on="H1-1", right_on="Search Term", how="inner")

# sort by opportunity
final_df = final_df.sort_values(by="Total Unique Searches", ascending=False)

# Round Float to two decimal places
final_df = final_df.round(decimals=2)  

# renames the cols
final_df.rename(columns={"H1-1": "Matched H1", "Address": "Matched URL"}, inplace=True)

# set new column order for final df
cols = [
    "Search Term",
    "Matched H1",
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

# export the final csv
final_df.to_csv(export_path_file, index=False)
