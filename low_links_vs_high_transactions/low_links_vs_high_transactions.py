"""Script by @LeeFootSEO -
For early access to scripts / request features
Join the Patreon: https://www.patreon.com/leefootseo <-------------------------

Internal Links Vs High Transactions Report by @LeeFootSEO Jan 2021
Fuses the 'Top linked pages – internally' Export from Search Console with the Landing Page Report from Google Analytics.

How to Use:

1) Download The Data from GA and GSC

Google Search Console  >  Links  >  Top linked pages – internally - [Export]
Google Analytics > Behavior  >  Site Content  >  Landing Pages) - [Export as a Excel File]

2) Place Exports in the Same Folder

3) Run the Script!
"""

from glob import glob  # Used to parse wildcard for csv import
from urllib.parse import urlparse

import pandas as pd

# SET ALL VARIABLES HERE!

# set the folder paths HERE for your input files!
ga_path = "/python_scripts/low-internal-links"  # enter the path to the folder that contains your GA Landing Page Report
gsc_path = "/python_scripts/low-internal-links"  # enter the path of the folder that contains the top linked pages internally report
export_path = "/python_scripts/low-internal-links"  # enter the path to export the output.csv file to

# adds file names to the paths. (Don't change the import file names, these will automatically import the exports for you
ga_path_file = ga_path + "/Analytics*.xlsx"
gsc_path_file = gsc_path + "/*target*.csv"

export_file = export_path + "/low-internal-links-transactions.csv"

# set lowest percentage of links to keep (default = 10%)
keep = 10

# START THE MAIN PROGRAM

# import the top internally linked pages report.csv
for f in glob(gsc_path_file):
    df_gsc = pd.read_csv((f))

# Extracts the Root Domain from SC Data to Append the Domain to the GA Report.
extracted_domain = df_gsc["Target page"]
extracted_domain = extracted_domain.iloc[0]
url = extracted_domain
o = urlparse(url)
domain = o.netloc
domain = ("https://") + domain
print("Domain is:", domain)
print("Imported: Google Search Console Data.")

# import the GA Landing Page Report (Behavior  >  Site Content  >  Landing Pages) - Export as a Excel File!
for f in glob(ga_path_file):
    df_ga = pd.read_excel((f), sheet_name="Dataset1")

# Adds the Domain Prefix Back In
df_ga["Landing Page"] = domain + df_ga["Landing Page"].astype(str)

print("Imported: Google Analytics Organic Landing Page Data.")

# merge the dataframes
df_combined = pd.merge(
    df_gsc, df_ga, left_on="Target page", right_on="Landing Page", how="inner"
)

# delete the extra landing page column left over from the merge
del df_combined["Landing Page"]

# delete any additional columns prior to csv export
del df_combined["% New Sessions"]
del df_combined["New Users"]
del df_combined["Bounce Rate"]
del df_combined["Pages/Session"]
del df_combined["Avg. Session Duration"]
del df_combined["E-commerce Conversion Rate"]

# drop rows of X-type (Useful to drop paginated pages / URL types you'd like to exclude from analysis)
# df_combined = df_combined[~df_combined["Target page"].str.contains("page", na=False)]

# drop rows with 0 transactions
df_combined = df_combined[df_combined["Transactions"] != 0]

# round all floats to 2 decimal places
df_combined = df_combined.round(2)

# routine to just get X% of the lowest internal links
max = df_combined["Internal links"].max()
lowest_perc = max / keep
lowest_perc = int(lowest_perc)

# drop rows with 0 transactions
df_combined = df_combined[df_combined["Internal links"] <= lowest_perc]

# sort the values
df_combined.sort_values(
    ["Internal links", "Transactions"],
    ascending=[True, False],
    inplace=True,
)

df_combined.to_csv(export_file, index=False)
