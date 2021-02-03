import requests as req
import pandas as pd
from io import StringIO
from urllib.parse import urlparse
from polyfuzz import PolyFuzz

###### start - Set All Variables HERE - start ######

# set the location of the screaming frog crawl file here
df_sf_path = "/python_scripts/archive-org-broken-link-mapping/internal_html.csv"

# set the location to EXPORT the csv file to
output_loc = "/python_scripts/archive-org-broken-link-mapping/output.csv"

# set HTTP Header to: Googlebot Desktop
headers = {'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'}

# set HTTP Header to: Googlebot Smartphone
#headers = {'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.120 Mobile Safari/537.36 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'}

# set crawl delay  # not used at present
crawl_delay = 1

###### end - Set Variables - end ######

# read in the crawl file
df_sf = pd.read_csv(df_sf_path)[["Address", "Status Code", "Indexability"]]

# extract the domain name from the crawl
extracted_domain = df_sf["Address"]
extracted_domain = extracted_domain.iloc[0]
url = extracted_domain
o = urlparse(url)
domain = o.netloc

print("Imported Crawl File ..")
print("Detected Domain = ", domain)

# append extracted domain to archive.org txt url
archive_url = "http://web.archive.org/cdx/search/cdx?url=" + domain + "*&output=txt"

# get the response
print("Downloading URLs from Archive.org ..")
resp = req.get(archive_url, headers=headers)

# make the df and set the column names
df_archive = pd.read_csv(
    StringIO(resp.text),
    sep=" ",
    names=["1", "2", "Address", "Content Type", "5", "6", "7"],
    dtype=str,
)

count_row = df_archive.shape[0]
print("Downloaded", count_row, "URLs from Archive.org ..")

# delete the unused columns
del df_archive["1"]
del df_archive["2"]
del df_archive["5"]
del df_archive["6"]
del df_archive["7"]

# replace port 80 urls before de-duping
df_archive["Address"] = df_archive["Address"].str.replace("\:80", "")

# drop all parameter urls (OPTIONAL!)
df_archive["Address"] = df_archive["Address"].str.replace("\?", "")

# drop duplicate urls
df_archive.drop_duplicates(subset="Address", inplace=True)

# keep only text/http content
df_archive = df_archive[df_archive["Content Type"].isin(["text/html"])]
df_archive["Address"].str.lower()

# delete the helper column
del df_archive["Content Type"]

# drop additional non-html pages incorrectly flagged as text/html
df_archive = df_archive[
    ~df_archive["Address"].str.contains(
        ".css|.js|.jpg|.png|.jpeg|.pdf|.JPG|.ico|ver=|.gif|.txt"
    )
]

# drop marketing tags
df_archive = df_archive[~df_archive["Address"].str.contains("utm|gclid")]

# drop any additional rows (custom) (Drops brand pages and pagination)
df_archive = df_archive[~df_archive["Address"].str.contains("/brand|/page")]

# drop basket related URLs [Should be blocked in robots.txt anyway!]
df_archive = df_archive[
    ~df_archive["Address"].str.contains("basket|checkout|add|account")
]

# make the final df
df_final = pd.merge(df_archive, df_sf, on="Address", how="left")

print("Dropping URLs Found in the Crawl File ..")

# drop archive.org rows which matched the crawl file
df_final = df_final[~df_final["Indexability"].isin(["Indexable"])]
df_final = df_final[~df_final["Indexability"].isin(["Non-Indexable"])]

# delete the helper column
del df_final["Indexability"]

# count the rows
count_row = df_final.shape[0]

print("Getting Live HTTP Status Codes for", count_row, "URLs Using Requests ..")

# get remaining status codes using requests library
def url_access(x):
    return req.head(x).status_code

df_final["Status Code"] = df_final["Address"].apply(url_access)

# drop urls if already redirected
df_final = df_final[~df_final["Status Code"].isin(["301"])]
df_final = df_final[~df_final["Status Code"].isin(["302"])]

print("Automatically Mapping URLs ..")

# create lists from dfs
df_final_list = list(df_final["Address"])
df_sf_list = list(df_sf["Address"])

# instantiate PolyFuzz model, choose TF-IDF as the similarity measure and match the two lists.
model = PolyFuzz("TF-IDF").match(df_final_list, df_sf_list)

# make the polyfuzz dataframe
df_matches = model.get_matches()

count_row = df_final.shape[0]
print("Total Opportunity:", count_row, "URLs")

df_stats = pd.merge(df_matches, df_final, left_on="From", right_on="Address", how="inner")

# sort on similarity
df_stats = df_stats.sort_values(by="Similarity", ascending=False)

df_stats["Status Code"] = df_stats["Status Code"].astype(str)

# replace status codes with plain English
df_stats["Status Code"] = df_stats["Status Code"].str.replace("200", "200 - Orphaned")
df_stats["Status Code"] = df_stats["Status Code"].str.replace("404", "404 - Not Redirected")

# delete address column appended from the merge
del df_stats["Address"]

# Defines the New Columns
cols = ["From", "Status Code", "To", "Similarity"]

# Re-indexes Columns To Place Them In A Logical Order + Inserts New Blank Columns for KW Checks.
df_stats = df_stats.reindex(columns=cols)

# rename the cols
df_stats.rename(columns={"From": "Archive URL", "To": "Suggested URL"}, inplace=True)

# output the final csv
df_stats.to_csv(output_loc, index=False)

print("Finished!")