import concurrent.futures
import time
from io import StringIO
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd
import requests as req
import waybackpy
from bs4 import BeautifulSoup
from polyfuzz import PolyFuzz

startTime = time.time()

# set the user agent here
user_agent = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
CONNECTIONS = 4  # increase this number to run more workers at the same time.

# import the Screaming Frog crawl File (internal_html.csv)
df_sf = pd.read_csv('/python_scripts/archive_mapper_v3/internal_html.csv', usecols=["Address", "H1-1"], dtype={'Address': 'str', 'H1-1': 'str'})

# extract the domain name from the crawl
extracted_domain = df_sf["Address"]
extracted_domain = extracted_domain.iloc[0]
url = extracted_domain
o = urlparse(url)
domain = o.netloc
print("Detected Domain = ", domain)

# append extracted domain to archive.org txt url
archive_url = "http://web.archive.org/cdx/search/cdx?url=" + domain + "*&output=txt"

# requests - set the session user agent
session = req.Session()
session.headers.update({'User-Agent': user_agent})

# get the response
print("Downloading URLs from the Wayback Machine .. Please be patient!")
resp = session.get(archive_url)

# make the dataframe and set the column names
df_archive = pd.read_csv(
    StringIO(resp.text),
    sep=" ",
    names=["1", "2", "Address", "Content Type", "5", "6", "7"],
    dtype=str,
)

# clean the dataframe
count_row = df_archive.shape[0]
print("Downloaded", count_row, "URLs from Archive.org ..")
df_archive["Address"] = df_archive["Address"].str.replace("\:80", "") # replace port 80 urls before de-duping
df_archive.drop_duplicates(subset="Address", inplace=True)  # drop duplicate urls
df_archive = df_archive[df_archive["Content Type"].isin(["text/html"])]  # keep only text/http content
df_archive = df_archive[
    ~df_archive["Address"].str.contains(
        ".css|.js|.jpg|.png|.jpeg|.pdf|.JPG|.PNG|.CSS|.JS|.JPEG|.PDF|.ICO|.GIF|.TXT|.ico|ver=|.gif|.txt|utm|gclid|:80|\?|#|@"
    )
]

df_archive = df_archive[df_archive["Address"].notna()]  # keep only non NaN
df_archive["Match"] = df_archive["Address"].str.contains('|'.join(df_sf['Address']), case=False)  # drop url if in crawl
df_archive = df_archive[~df_archive["Match"].isin([True])]
df_archive = df_archive.reindex(columns=['Address'])  # reindex the columns
remaining_count = df_archive.shape[0]  # calculate and print how many rows remain after filtering
print("Filtered to", remaining_count, "qualifying URLs!")

# fetch the latest version of the archive.org url (so the h1's can be extracted with requests)
url_list = list(df_archive['Address'])  # create list from address column

archive_url_list = []

def get_archive_url(url):
    target_url = waybackpy.Url(url, user_agent)
    newest_archive = target_url.newest()
    return newest_archive

def concurrent_calls():
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONNECTIONS) as executor:
        f1 = (executor.submit(get_archive_url, url) for url in url_list)
        for future in concurrent.futures.as_completed(f1):
            try:
                data = future.result().archive_url
            except Exception as e:
                data = ('error', e)
            finally:
                archive_url_list.append(data)
                print(data)

if __name__ == '__main__':
    concurrent_calls()
    print(archive_url_list)

# extract original url from recovered archive URL (for de-duping) and clean the data
print(archive_url_list)
df_archive_urls = pd.DataFrame(archive_url_list)
df_archive_urls["Recovered Archive URL"] = (df_archive_urls[0]).astype(str)
df_archive_urls["Original URL"] = (df_archive_urls["Recovered Archive URL"].str.split("/").str[5:])
df_archive_urls["Original URL"] = df_archive_urls["Original URL"].str.join(",")
df_archive_urls["Original URL"] = df_archive_urls["Original URL"].str.replace(',', '/')
df_archive_urls.drop_duplicates(subset=['Recovered Archive URL'], keep="first", inplace=True)  # drop duplicates

# # extract h1s from archive url df and make into a list
archive_url_list = df_archive_urls['Recovered Archive URL']
archive_h1_list = []
def get_archive_h1(h1_url):
    html = urlopen(h1_url)
    bsh = BeautifulSoup(html.read(), 'html.parser')
    return bsh.h1.text.strip()

def concurrent_calls():
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONNECTIONS) as executor:
        f1 = (executor.submit(get_archive_h1, h1_url) for h1_url in archive_url_list)
        for future in concurrent.futures.as_completed(f1):
            try:
                data = future.result()
                archive_h1_list.append(data)
            except Exception:
                archive_h1_list.append("No Data Received!")
                pass
            # finally:
            #     archive_h1_list.append(data)
                #print(data)

if __name__ == '__main__':
    concurrent_calls()
    print(archive_h1_list)

# clean up the dataframe
df_archive_urls['H1'] = archive_h1_list  # add list to dataframe column
df_archive_urls = df_archive_urls[~df_archive_urls["H1"].isin(["Got an HTTP 301 response at crawl time"])]  # drop
df_archive = pd.merge(df_archive, df_archive_urls, left_on="Address", right_on="Original URL", how="inner")

# start polyfuzz to merge in archive.org data with original screaming frog dataframe
df_archive = df_archive[df_archive["H1"].notna()]
df_sf = df_sf[df_sf["H1-1"].notna()]
df_sf_list = list(df_sf["H1-1"])
df_archive_list = list(df_archive["H1"])

# instantiate PolyFuzz model, choose TF-IDF as the similarity measure and match the two lists.
model = PolyFuzz("TF-IDF").match(df_archive_list, df_sf_list)
df_matches = model.get_matches()  # make the polyfuzz dataframe
df_sf_mini = df_sf[['H1-1', 'Address']]  # make mini dataframe
df_archive = pd.merge(df_archive, df_matches, left_on="H1", right_on="From", how="inner")  # merge back in
df_archive = pd.merge(df_archive, df_sf_mini, left_on="To", right_on="H1-1")

# clean up the dataframe
df_archive.drop_duplicates(subset=['Address_x'], keep="first", inplace=True)  # drop duplicate rows
df_archive = df_archive.reindex(columns=['Address_x', 'Address_y', 'Similarity'])  # reindex the columns
df_archive.rename(columns={'Address_x': 'Recovered Archive URL', "Address_y": "Live URL"}, inplace=True)  # rename cols
final_count = df_archive.shape[0]  # count the final dataframe rows

# export final output
print("Total Opportunity", final_count, "URLs")
df_archive.to_csv('/python_scripts/urls-to-redirect-archive-org.csv', index=False)
print("The script took {0} seconds!".format(time.time() - startTime))