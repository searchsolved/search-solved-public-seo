""" Internal Search Report Mapper V1 by Lee Foot 07/01/2021 @LeeFootSEO
Takes the Google Analytics Search Terms Report and merges with a Screaming Frog crawl file to find opportunities to Map
internal searches to.

1) Search Terms Report must be Exported as an Excel file
2) Screaming Frog Crawl should only contain category pages (unless you really want to map internal searches to products!
3) Paths are hardcoded, please enter your own for GA, Screaming Frog and the Output.CSV
"""
from glob import glob  # Used to parse wildcard for csv import
from fuzzywuzzy import fuzz
import pandas as pd

# imports using GA data using a wildcard match
for f in glob('/python_scripts/Internal Search Mapper/Analytics*.xlsx'):  # ENTER YOUR PATH TO THE SEARCH TERM REPORT
    df_ga = pd.read_excel((f), sheet_name='Dataset1')

df_sf = pd.read_csv('/YOUR-PATH-HERE/internal_html.csv', encoding='utf8')[['H1-1', 'Address',
                                                                                                 'Indexability']]


df_sf['H1-1'] = df_sf['H1-1'].str.lower()  # convert to lower case for matching
df_ga['Search Term'] = df_ga['Search Term'].str.lower()# convert to lower case for matching
df_ga.drop_duplicates(subset=['Search Term'], inplace=True)  #drop duplicates

try:  # drop non-indexable pages
    df_sf = df_sf[~df_sf['Indexability'].isin(['Non-Indexable'])]  # Drop Non-indexable Pages
except Exception:
    pass

del df_sf['Indexability']  # delete the helper column

# merge the dataframe
final_df = pd.merge(df_sf, df_ga, left_on="H1-1", right_on="Search Term", how="inner")

# sort by opportunity
final_df = final_df.sort_values(by='Total Unique Searches', ascending=False)

# round floats to 2 decimal places
final_df = final_df.round(decimals=2)  # Round Float to two decimal places

# Renames cols
final_df.rename(columns={'H1-1': 'Matched H1', 'Address': 'Matched URL'}, inplace=True)

# reindex the columns in a new order
cols = ['Search Term', 'Matched H1', 'Matched URL','Total Unique Searches', 'Results Page Views/Search',
            '% Search Exits', '% Search Refinements', 'Time After Search', 'Avg. Search Depth']

# Re-indexes Columns To Place Them In A Logical Order + Inserts New Blank Columns for KW Checks.
final_df = final_df.reindex(columns=cols)

# export the final csv
final_df.to_csv('/YOUR-PATH-HERE/output.csv', index=False)  # ENTER YOUR PATH HERE!

# export the final csv
 # ENTER YOUR PATH HERE!