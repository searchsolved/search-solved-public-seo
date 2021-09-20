import concurrent.futures
import logging
import sys
import time
from io import StringIO
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd
import requests as req
import waybackpy
from bs4 import BeautifulSoup
from polyfuzz import PolyFuzz
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

startTime = time.time()

# set the user agent here
#user_agent = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"
CONNECTIONS = 8  # increase this number to run more workers at the same time.

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

df_archive.to_csv('/python_scripts/53.csv')

# clean the dataframe
count_row = df_archive.shape[0]
print("Downloaded", count_row, "URLs from Archive.org ..")
df_archive["Address"] = df_archive["Address"].str.replace("\:80", "") # replace port 80 urls before de-duping
df_archive.drop_duplicates(subset="Address", inplace=True)  # drop duplicate urls
df_archive = df_archive[df_archive["Content Type"].isin(["text/html"])]  # keep only text/http content
df_archive = df_archive[
    ~df_archive["Address"].str.contains(
        ".css|.js|.jpg|.png|.jpeg|.pdf|.JPG|.PNG|.CSS|.JS|.JPEG|.PDF|.ICO|.GIF|.TXT|.ico|ver=|.gif|.txt|utm|gclid|:80|\?|#|@|.eot|.svg|.ttf|.woff|.XMLDOM|.XMLHTTP"
    )
]

df_archive = df_archive[df_archive["Address"].notna()]  # keep only non NaN
df_archive["Match"] = df_archive["Address"].str.contains('|'.join(df_sf['Address']), case=False)  # drop url if in crawl
df_archive = df_archive[~df_archive["Match"].isin([True])]
df_archive = df_archive.reindex(columns=['Address'])  # reindex the columns
remaining_count = df_archive.shape[0]  # calculate and print how many rows remain after filtering
print("Filtered to", remaining_count, "qualifying URLs!")

if remaining_count == 0:
    print("No valid URLs to redirect. Check Wayback Machine status or try again!")
    sys.exit(0)
# fetch the latest version of the archive.org url (so the h1's can be extracted with requests)
url_list = list(df_archive['Address'])  # create list from address column
print(url_list)
df_archive.to_csv('/python_scripts/df_archive-line-75.csv')  # todo remove this

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

# extract Extracted Archive URL from Extracted Archive URL (for de-duping) and clean the data
print(archive_url_list)
df_archive_urls = pd.DataFrame(archive_url_list, columns=["Extracted Archive URL"])
df_archive_urls["Extracted Archive URL"] = (df_archive_urls["Extracted Archive URL"]).astype(str)
df_archive_urls["Extracted Archive URL"] = (df_archive_urls["Extracted Archive URL"].str.split("/").str[5:])
df_archive_urls["Extracted Archive URL"] = df_archive_urls["Extracted Archive URL"].str.join(",")
df_archive_urls["Extracted Archive URL"] = df_archive_urls["Extracted Archive URL"].str.replace(',', '/')
df_archive_urls.drop_duplicates(subset=['Extracted Archive URL'], keep="first", inplace=True)  # drop duplicates

df_archive_urls.to_csv('/python_scripts/df_archive_urls.csv')  # todo delete this

# todo this code needs fixing - threading appends to the list in the order the task completes rather than chronologically

# extract h1s from archive url df and make into a list
archive_url_list = list(df_archive_urls["Extracted Archive URL"])
archive_h1_list = []
og_url_list = []
def get_archive_h1(h1_url):
    html = urlopen(h1_url)
    bsh = BeautifulSoup(html.read(), 'lxml')
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

if __name__ == '__main__':
    concurrent_calls()
    print(archive_h1_list)

# clean up the dataframe

# try:
#     df_archive_urls['OG URL List'] = og_url_list
# except ValueError:
#     print("\nCouldn't fetch H1's from Wayback Machine. Please try again later ..")
#     sys.exit()
#     pass
print("\archive_h1_list")
print(og_url_list)

df_archive_urls['H1'] = archive_h1_list  # add list to dataframe column
df_archive.to_csv('/python_scripts/df_archive-130.csv')  # todo remove
df_archive_urls = df_archive_urls[~df_archive_urls["H1"].isin(["Got an HTTP 301 response at crawl time"])]  # drop
df_archive = pd.merge(df_archive, df_archive_urls, left_on="Address", right_on="Extracted Archive URL", how="inner")
df_archive.to_csv('/python_scripts/df_archive-133.csv')  # todo remove

# start polyfuzz to merge in archive.org data with original screaming frog dataframe
df_archive = df_archive[df_archive["H1"].notna()]
df_archive = df_archive[~df_archive["H1"].str.contains("No Data Received!", na=False)]

df_archive.to_csv('/python_scripts/df_archive-138.csv')  # todo remove [ archive h1s wrong here]

df_sf = df_sf[df_sf["H1-1"].notna()]
df_sf_list = list(df_sf["H1-1"])
df_archive_list = list(df_archive["H1"])

# instantiate PolyFuzz model, choose TF-IDF as the similarity measure and match the two lists.
model = PolyFuzz("TF-IDF").match(df_archive_list, df_sf_list)
df_matches = model.get_matches()  # make the polyfuzz dataframe
df_matches.to_csv('/python_scripts/df_matches-156.csv')  #todo remove

# make mini dfs for easier matching
df_archive_mini = df_archive[["H1", "Extracted Archive URL"]]
df_sf_mini = df_sf[["H1-1", "Address"]]

df_matches = pd.merge(df_matches, df_archive_mini, left_on="From", right_on="H1")
df_matches = pd.merge(df_matches, df_sf_mini, left_on="To", right_on="H1-1")

df_matches.to_csv('/python_scripts/df_matches.csv')  #todo remove - h1s are good here
#df_matches.drop_duplicates(subset=['Extracted Archive URL'], keep="first", inplace=True)
df_matches.rename(columns={"H1": "Archive H1", "Extracted Archive URL": "Archive URL", "H1-1": "Matched H1", "Address": "Matched URL"}, inplace=True)
cols = "Archive URL", "Archive H1", "Similarity", "Matched URL", "Matched H1", "Final HTTP Status"
df_matches = df_matches.reindex(columns=cols)
df_matches.to_csv('/python_scripts/df_matches-170.csv') #todo remove
# check http status of recovered archive.org url using requests

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
logging.basicConfig(level=logging.DEBUG)
s = req.Session()
retries = Retry(total=5, backoff_factor=16, status_forcelist=[429])
s.mount('http://', HTTPAdapter(max_retries=retries))
archive_url_list = list(df_matches['Archive URL']) # make the list
print("Getting HTTP Status of Source URL..")

count = 0
status_list = []
for url in archive_url_list:
    try:
        status_list.append(s.get(url, headers=headers))
    except Exception:
        status_list.append("Error Getting Status")

df_matches['Final HTTP Status'] = status_list

# export final output
df_matches.to_csv('/python_scripts/urls-to-redirect-archive-org.csv', index=False)
print("The script took {0} seconds!".format(time.time() - startTime))