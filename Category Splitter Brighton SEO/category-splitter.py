# ***********************************************************************************************************************
# ************************************************* Created by Lee Foot *************************************************
# ***************************************************** @LeeFootSEO *****************************************************
# ******************************************************August 2021 *****************************************************
# ***********************************************************************************************************************

import collections
import re
import string
import sys
from datetime import date
import pandas as pd
import requests
import searchconsole
from dateutil.relativedelta import relativedelta
from nltk.util import ngrams
from polyfuzz import PolyFuzz
import time
import os

PATH = os.getcwd()
startTime = time.time()

# ---------------------- Set Filtering Variables ------------------------------------------------------------------------

min_product_match = 3  # set minimum matching products in order for a category to be suggested
min_sim_match = 96  # similarity match percentage to keep // default = 96%
keep_longest_word = True  # group the final keyword suggestions and keep the longest word (remove fragments)
check_gsc_impressions = False  # recommend enabling if you have GSC access to pre-vet queries.
min_search_vol = 1
min_cpc = 0
# ---------------------- Set Screaming Frog Variables -------------------------------------------------------------------
product_extract_col = "product 1"  # set customer extraction column name for products // default = product 1
category_extract_col = "category 1"  # set customer extraction column name for categories // default = category 1
http_or_https_gsc = "https://"  # http prefix // default = https://
parms = "page=|p=|utm_medium|sessionid|affiliateid|sort=|order=|type=|categoryid=|itemid=|viewItems=|query" \
        "=|search=|lang="  # drop common parameter urls

# ---------------------- Set Search Console Variables -------------------------------------------------------------------
date_range_gsc = -3  # set date range variable for search console in months // default = -3
country_gsc = "gbr"  # set the country filter
client_config_path = PATH + "/client_secrets.json"  # path to client_secrets.json
credentials_path = PATH + "/credentials.json"  # path to client_secrets.json
# ---------------------- Set Keywords Everywhere Variables --------------------------------------------------------------
# https://api.keywordseverywhere.com/docs/#/  All Settings Explained
country_kwe = 'uk'
currency_kwe = 'gbp'
data_source_kwe = 'cli'  # gkp = google keyword planner only // cli = clickstream data + keyword planner
with open(PATH + '/kwe_key.txt', 'r') as file:  # read in the Keywords Everywhere API Key
    kwe_key = file.read()
# ---------------------- Read in the x2 Crawl Exports -------------------------------------------------------------------
try:
    df_internal_html = pd.read_csv(
        PATH + "/internal_html.csv",
        usecols=["Address", "Indexability", "H1-1", "Title 1", category_extract_col, product_extract_col],
        encoding="utf-8",
        error_bad_lines=False,
        dtype=({"Address": "str", "Indexability": "str", "H1-1": "str", "Title 1": "str", category_extract_col: "str",
                product_extract_col: "str"}))
except ValueError:
    print("Make sure crawl file contains: Address, Indexability, H1-1, Title 1 and", category_extract_col, "and",
          product_extract_col, "columns.")
    sys.exit()

try:  # try to read in all_inlinks.csv
    df_all_inlinks = pd.read_csv(
        PATH + "/all_inlinks.csv",
        usecols=["Type", "Source", "Destination", "Status Code"],
        encoding="utf-8",
        error_bad_lines=False,
        dtype=({"Type": "str", "Source": "str", "Destination": "str", "Status Code": "str"}))

except FileNotFoundError:  # try to read in inlinks.csv (manual inlinks to product page export - faster export)
    df_all_inlinks = pd.read_csv(
        PATH + "/inlinks.csv",
        usecols=["Type", "From", "To", "Status Code"],
        encoding="utf-8",
        error_bad_lines=False,
        dtype=({"Type": "str", "From": "str", "To": "str", "Status Code": "str"}))

# ---------------------- Clean Up the Crawl Files -----------------------------------------------------------------------
df_all_inlinks = df_all_inlinks.rename(columns={"From": "Source", "To": "Destination"})
df_internal_html = df_internal_html[~df_internal_html["Indexability"].isin(["Non-Indexable"])]  # keep indexable urls
df_internal_html['H1-1'] = df_internal_html['H1-1'].str.lower()
df_internal_html['H1-1'] = df_internal_html['H1-1'].str.encode('ascii', 'ignore').str.decode('ascii')
df_internal_html['Title-1'] = df_internal_html['Title 1'].str.encode('ascii', 'ignore').str.decode('ascii')
df_all_inlinks = df_all_inlinks[df_all_inlinks["Status Code"].isin(["200"])]  # keep only 200 pages
df_all_inlinks = df_all_inlinks[df_all_inlinks["Type"].isin(["Hyperlink"])]  # keep only hyperlink pages
# ---------------------- Work out the Page Type from Extractors ---------------------------------------------------------
df1 = df_internal_html[df_internal_html[product_extract_col].notna()].copy()
df2 = df_internal_html[df_internal_html[category_extract_col].notna()].copy()
df1.rename(columns={product_extract_col: "Page Type"}, inplace=True)
df2.rename(columns={category_extract_col: "Page Type"}, inplace=True)
df1["Page Type"] = "Product Page"
df2["Page Type"] = "Category Page"
df_internal_html = pd.concat([df1, df2])

# ---------------------- Extract the Domain from the Crawl --------------------------------------------------------------
extracted_domain = df_internal_html["Address"]
extracted_domain = extracted_domain.str.split("/").str[2]
url = extracted_domain.iloc[0]
url_slash = http_or_https_gsc + url + "/"  # adds a trailing slash to the domain to query the gsc api
print("Domain is: ", url_slash)
# ---------------------- Make Products & Category Dataframes ------------------------------------------------------------
df_sf_products = df_internal_html[df_internal_html['Page Type'].str.contains("Product Page")].copy()

df_sf_categories = df_internal_html[df_internal_html['Page Type'].str.contains("Category Page")].copy()
df_sf_products.drop_duplicates(subset="H1-1", inplace=True)  # drop duplicate values (drop pagination pages etc)
df_sf_categories.drop_duplicates(subset="H1-1", inplace=True)  # drop duplicate values (drop pagination pages etc)
df_sf_categories = df_sf_categories[~df_sf_categories["Address"].str.contains(parms, na=False)]
df_all_inlinks.drop_duplicates(subset=["Source", "Destination"], keep="first", inplace=True)
df_all_inlinks = pd.merge(df_all_inlinks, df_sf_categories, left_on="Source", right_on="Address", how="left")
df_all_inlinks = df_all_inlinks[df_all_inlinks["Page Type"].isin(["Category Page"])]
cols = "Destination", "Source"
df_all_inlinks = df_all_inlinks.reindex(columns=cols)
df_sf_products = pd.merge(df_sf_products, df_all_inlinks, left_on="Address", right_on="Destination", how="left")
cols = "Source", "Address", "H1-1"
df_sf_products = df_sf_products.reindex(columns=cols)
df_sf_products.rename(columns={"Source": "Parent URL", "Address": "Product URL"}, inplace=True)
df_sf_products = df_sf_products[df_sf_products["Parent URL"].notna()]  # Only Keep Rows which are not NaN

# ---------------------- Group Dataframs & Make Lists for N-Gramming ----------------------------------------------------
df_product_group = (df_sf_products.groupby("Product URL").agg({"Parent URL": "first"}).reset_index())
category_extractor_list = list(df_product_group["Parent URL"])
category_extractor_set = set(category_extractor_list)
category_extractor_list = list(category_extractor_set)
len_product_list = len(category_extractor_list)

# ---------------------- Start N-gram Routine ---------------------------------------------------------------------------
ngram_loop_count = 1
start_num = 0
appended_data = []
while ngram_loop_count != len_product_list:
    df_kwe = df_sf_products[df_sf_products["Parent URL"].str.contains(category_extractor_list[start_num], na=False)]
    text = str(df_kwe["H1-1"])

    # clean up the corpus before ngramming
    text = "".join(c for c in text if not c.isdigit())  # removes all numbers
    text = re.sub("<.*>", "", text)
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    punctuationNoFullStop = "[" + re.sub("\.", "", string.punctuation) + "]"
    text = re.sub(punctuationNoFullStop, "", text)

    # first get individual words
    print("Calculating ngrams:", ngram_loop_count, "of", len_product_list - 1, category_extractor_list[start_num])
    tokenized = text.split()
    oneNgrams = ngrams(tokenized, 1)
    twoNgrams = ngrams(tokenized, 2)
    threeNgrams = ngrams(tokenized, 3)
    fourNgrams = ngrams(tokenized, 4)
    fiveNgrams = ngrams(tokenized, 5)
    sixNgrams = ngrams(tokenized, 6)
    sevenNgrams = ngrams(tokenized, 7)
    oneNgramsFreq = collections.Counter(oneNgrams)
    twoNgramsFreq = collections.Counter(twoNgrams)
    threeNgramsFreq = collections.Counter(threeNgrams)
    fourNgramsFreq = collections.Counter(fourNgrams)
    fiveNgramsFreq = collections.Counter(fiveNgrams)
    sixNgramsFreq = collections.Counter(sixNgrams)
    sevenNgramsFreq = collections.Counter(sevenNgrams)

    # Combines the above collection counters so they can be placed in a dataframe.
    ngrams_combined_list = (
            twoNgramsFreq.most_common(100)
            + threeNgramsFreq.most_common(100)
            + fourNgramsFreq.most_common(100)
            + fiveNgramsFreq.most_common(100)
            + sixNgramsFreq.most_common(100)
            + sevenNgramsFreq.most_common(100)
    )

    # Create the Final DataFrame
    df_ngrams = pd.DataFrame(ngrams_combined_list, columns=["Keyword", "Frequency"])
    df_ngram_frequency = pd.DataFrame(ngrams_combined_list, columns=["Keyword", "Frequency"])
    df_ngrams["Parent Category"] = category_extractor_list[start_num]
    data = df_ngrams
    appended_data.append(data)
    start_num = start_num + 1
    ngram_loop_count = ngram_loop_count + 1

df_ngrams = pd.concat(appended_data)  # concat the list of dataframes
ngram_count = df_ngrams.shape[0]  # get the row count
print("Total keywords generated via ngrams:", ngram_count)

df_ngrams = df_ngrams.sort_values(by="Frequency", ascending=False)
df_ngrams["Keyword"] = [' '.join(entry) for entry in df_ngrams["Keyword"]]
cols = "Parent Category", "Keyword", "Frequency"
df_ngrams = df_ngrams.reindex(columns=cols)

print(f'N-Grams Generated in {time.time() - startTime:.2f} Seconds')
# ---------------------- Pre-Filtering ----------------------------------------------------------------------------------

# todo make this more idiomatic
df_ngrams = df_ngrams[~df_ngrams['Keyword'].astype(str).str.startswith('and')]
df_ngrams = df_ngrams[~df_ngrams['Keyword'].astype(str).str.startswith('with')]
df_ngrams = df_ngrams[~df_ngrams['Keyword'].astype(str).str.startswith('for')]
df_ngrams = df_ngrams[~df_ngrams['Keyword'].astype(str).str.startswith('mm')]
df_ngrams = df_ngrams[~df_ngrams['Keyword'].astype(str).str.startswith('cm')]
df_ngrams = df_ngrams[~df_ngrams['Keyword'].astype(str).str.startswith('of')]
df_ngrams = df_ngrams[~df_ngrams['Keyword'].astype(str).str.endswith(' and')]
df_ngrams = df_ngrams[~df_ngrams['Keyword'].astype(str).str.endswith(' with')]
df_ngrams = df_ngrams[~df_ngrams['Keyword'].astype(str).str.endswith(' for')]
df_ngrams = df_ngrams[~df_ngrams['Keyword'].astype(str).str.endswith(' mm')]
df_ngrams = df_ngrams[~df_ngrams['Keyword'].astype(str).str.endswith(' cm')]
df_ngrams = df_ngrams[~df_ngrams['Keyword'].astype(str).str.endswith(' of')]

# ---------------------- Keep Only Suggestions Which Match to Products X Times ------------------------------------------
df_sf_products["H1-1"] = df_sf_products["H1-1"].astype(str)
df_sf_products["H1-1"] = df_sf_products["H1-1"].str.lower()
df_product_set = set(df_sf_products["H1-1"])  # make a set, then a list
df_product_list = list(df_product_set)
df_kw_list = list(df_ngrams["Keyword"])  # make the keyword list

print("\nMatching N-grams to a minimum of", min_product_match, "products ..")

check_list = []
for i in df_kw_list:
    check_freq = sum(i in s for s in df_product_list)
    check_list.append(check_freq)

df_ngrams["Matching Products"] = check_list  # append product count list and final clean up
df_ngrams = df_ngrams[df_ngrams["Matching Products"] >= min_product_match]
matched_product_row = df_ngrams.shape[0]
print("N-grams matched to a minimum of", min_product_match, "products:", matched_product_row)

rows = df_ngrams.shape[0]
ngram_loop_count = 1
start = 1
end = 100
df_data = []
print(f'N-Grams Matched to Keywords in {time.time() - startTime:.2f} Seconds')
# ---------------------- Fuzz Match Suggested Keywords to Existing Categories -------------------------------------------
df_ngrams = df_ngrams[df_ngrams["Keyword"].notna()]  # Only Keep Rows which are not NaN
df_sf_categories = df_sf_categories[df_sf_categories["H1-1"].notna()]  # Only Keep Rows which are not NaN
df_keyword_list = list(df_ngrams["Keyword"])  # create lists from dfs
df_sf_cats_list = list(df_sf_categories["H1-1"])  # create lists from dfs
model = PolyFuzz("TF-IDF").match(df_keyword_list, df_sf_cats_list)  # do the matching

print("\nFuzzy matching keywords to existing categories ...")
df_fuzz = model.get_matches()  # make the polyfuzz dataframes
df_ngrams = pd.merge(df_ngrams, df_fuzz, left_on="Keyword", right_on="From")
df_ngrams.rename(columns={"To": "Matched Category", "clicks": "Clicks", "impressions": "Impressions"},
                 inplace=True)
print(f'N-Grams Fuzzy Matched to Existing Categories in {time.time() - startTime:.2f} Seconds')
# ---------------------- Authenticate with Google Search Console --------------------------------------------------------
if check_gsc_impressions == True:
    # populates a string pattern to query GSC using sc-domain:domain-name.com instead of the homepage url
    sc_domain = url
    sc_domain = sc_domain.replace("https://", "")
    sc_domain = sc_domain.replace("http://", "")
    sc_domain = sc_domain.replace("www.", "")
    sc_domain = "sc-domain:" + sc_domain
    try:
        print("\nReading client_secrets.json & credentials.json..")  # authenticate with GSC
        three_months = date.today() + relativedelta(months=date_range_gsc)
        end_date = date.today()
        account = searchconsole.authenticate(
            client_config=client_config_path,
            credentials=credentials_path,
        )
        # print(account.webproperties) # uncomment to print the list of available sc accounts
        account_select = account.webproperties
        webproperty = account[(url_slash)]  # connect to the gsc property
        print("Pulling GSC Data. Please Be Patient!")
        gsc_data = (
            webproperty.query.range(three_months, end_date)
                .dimension(
                "query",
                "page",
                "country",
            )
                .get()
        )
    except AttributeError:
        three_months = date.today() + relativedelta(months=date_range_gsc)
        end_date = date.today()
        # print(account.webproperties)  # uncomment to print the list of available sc accounts
        account_select = account.webproperties
        webproperty = account[(sc_domain)]  # connect to the gsc property using the sc:domain
        gsc_data = (
            webproperty.query.range(three_months, end_date)
                .dimension(
                "query",
                "page",
                "country",
            )
                .get()
        )

    except Exception:
        print("No Data Received from Search Console API! Stopped!")
        sys.exit(1)

    df_gsc_data = pd.DataFrame(gsc_data)  # make the dataframe
    df_gsc_data = df_gsc_data[df_gsc_data["country"].str.contains(country_gsc)]  # keep only country specific traffic

    # match to gsc data and remove any keyword which aren't found
    df_ngrams = pd.merge(df_ngrams, df_gsc_data, left_on="Keyword", right_on="query", how="left")
    df_ngrams = df_ngrams.sort_values(by="impressions", ascending=False)
    df_ngrams.drop_duplicates(subset=["Keyword"], keep="first", inplace=True)
    df_ngrams = df_ngrams[df_ngrams["impressions"].notna()]
    req_kw_credits = df_ngrams.shape[0]
    print(f'N-Grams Matched to Search Console in {time.time() - startTime:.2f} Seconds')

df_ngrams.drop_duplicates(subset=["Keyword"], keep="first", inplace=True)
creds_required = df_ngrams.shape[0]
# ---------------------- Check Available with Keywords Everywhere Credits------------------------------------------------
my_headers = {
    'Accept': 'application/json',
    'Authorization': 'Bearer ' + kwe_key
}
response = requests.get('https://api.keywordseverywhere.com/v1/account/credits', headers=my_headers)
if response.status_code == 200:
    creds_available = response.content.decode('utf-8')
    creds_available = creds_available.split()
    creds_available = int(creds_available[1])
    print("\nStarting Keyword Everywhere API Checks")
    print("This operation will require", creds_required, "API credits. \nYou have", creds_available,
          "credits remaining.")
    if creds_available < creds_required:
        print("Not enough keywords everywhere credits available!")
        sys.exit(1)
else:
    print("An error occurred\n\n", response.content.decode('utf-8'))

# ---------------------- Get Search Volume with Keywords Everywhere -----------------------------------------------------

loops = int(creds_required / 100)
if loops == 1:
    loops += 1
fixed_loops = loops * 100  # fixes the total loop counter displayed value
ngram_loop_count_100 = ngram_loop_count * 100

print("\nFetching Search Volume & CPC with Keywords Everywhere ( 000 - 100 ) of", fixed_loops)
while ngram_loop_count != loops:
    print("Fetching Search Volume & CPC with Keywords Everywhere (", ngram_loop_count_100, "-",
          ngram_loop_count_100 + 100,
          ') of', fixed_loops)

    keywords = list(df_ngrams["Keyword"][start:end])
    keywords_set = set(keywords)
    keywords = list(keywords_set)
    my_data = {
        'country': country_kwe,
        'currency': currency_kwe,
        'dataSource': data_source_kwe,
        'kw[]': keywords
    }
    my_headers = {
        'Accept': 'application/json',
        'Authorization': 'Bearer ' + kwe_key
    }
    response = requests.post(
        'https://api.keywordseverywhere.com/v1/get_keyword_data', data=my_data, headers=my_headers)
    try:
        keywords_data = response.json()['data']
    except KeyError:
        print("Couldn't retrieve data from Keywords Everywhere. Check credits...")
        pass

    vol = []
    cpc = []

    for element in keywords_data:
        vol.append(element["vol"])
        cpc.append(element["cpc"]["value"])

    rows = zip(keywords, vol, cpc)
    df_kwe = pd.DataFrame(rows, columns=["Keyword", "Search Volume", "CPC"])
    data = df_kwe
    df_data.append(data)
    start = start + 100
    end = end + 100
    ngram_loop_count += +1
    ngram_loop_count_100 += 100
df_kwe = pd.concat(df_data)
df_kwe["Search Volume"] = df_kwe["Search Volume"].astype(int)
df_kwe["CPC"] = df_kwe["CPC"].astype(float)
df_kwe = df_kwe[df_kwe["Search Volume"] > min_search_vol]
df_kwe = df_kwe[df_kwe["CPC"] > min_cpc]
df_kwe = pd.merge(df_kwe, df_ngrams, on="Keyword", how='left')
df_kwe = df_kwe.sort_values(by="Parent Category", ascending=True)

print(f'\nKeyword Search Volume Fetched in {time.time() - startTime:.2f} Seconds')
# ---------------------- Clean up the Final Dataframe -------------------------------------------------------------------
cols = "Parent Category", "Keyword", "Search Volume", "CPC", "Matching Products", "Similarity", "Matched Category"
df_kwe = df_kwe.reindex(columns=cols)
df_kwe["Similarity"] = df_kwe["Similarity"] * 100
df_kwe["Similarity"] = df_kwe["Similarity"].astype(int)
df_kwe = df_kwe[df_kwe["Similarity"] <= min_sim_match]
df_kwe["Matched Category"] = df_kwe["Matched Category"].str.lower()
df_kwe.drop_duplicates(subset=["Matched Category", "Keyword"], keep="first", inplace=True)

# ---------------------- Keep the Longest Word and Discard the Fragments ------------------------------------------------
if keep_longest_word == True:
    print("\nKeeping Longest Word and Discarding Fragments ..")

    list1 = df_kwe["Keyword"]
    substrings = {w1 for w1 in list1 for w2 in list1 if w1 in w2 and w1 != w2}
    longest_word = set(list1) - substrings
    longest_word = list(longest_word)
    shortest_word_list = list(set(list1) - set(longest_word))
    print("Discarded the following short words:\n", shortest_word_list)
    df_kwe = df_kwe[~df_kwe['Keyword'].isin(shortest_word_list)]
print(f'\nFinal Dataframe Cleaned up in {time.time() - startTime:.2f} Seconds')
# ---------------------- Merge in Page Title for Matched Category -------------------------------------------------------
df_mini = df_internal_html[["H1-1", "Title 1"]]
df_mini = df_mini.rename(columns={"H1-1": "Matched Category", "Title 1": "Matched Category Page Title"})
df_kwe = pd.merge(df_kwe, df_mini[['Matched Category', 'Matched Category Page Title']], on='Matched Category',
                  how='left')

# ---------------------- Remove Keyword Suggestions if Matched to An Existing Category in any Order ---------------------
df_kwe['Matched Category Page Title Lower'] = df_kwe['Matched Category Page Title'].str.lower()
df_kwe = df_kwe.astype({"Keyword": "str", "Matched Category": "str", "Matched Category Page Title Lower": "str"})
col = "Keyword"


def ismatch(s):
    A = set(s[col].split())
    B = set(s['Matched Category Page Title Lower'].split())
    return A.intersection(B) == A


df_kwe['KW Matched'] = df_kwe.apply(ismatch, axis=1)

df_kwe["Keyword + s"] = df_kwe["Keyword"] + "s"  # make new temp column to run the same check on the pluralised word
col = "Keyword + s"  # updates the column to run function on
df_kwe['KW Matched 2'] = df_kwe.apply(ismatch, axis=1)
df_kwe = df_kwe[~df_kwe["KW Matched"].isin([True])]  # drop rows which are matched
df_kwe = df_kwe[~df_kwe["KW Matched 2"].isin([True])]  # drop rows which are matched
df_kwe.drop_duplicates(subset=["Parent Category", "Keyword"], keep="first", inplace=True)  # drop if both values dupes

# ---------------------- Set the Final Column Order --------------------------------------------------------------------
cols = (
    "Parent Category",
    "Keyword",
    "Search Volume",
    "CPC",
    "Matching Products",
    "Similarity",
    "Matched Category",
    "Matched Category Page Title",
    "Recommended Action",
    "Suggested Page Title",
    "New Subcategory URL",
)

df_kwe = df_kwe.reindex(columns=cols)

# ---------------------- Export Final Dataframe to CSV ------------------------------------------------------------------
keyword_volume_count = df_kwe.shape[0]
df_kwe.sort_values(["Parent Category", "Keyword"], ascending=[True, True], inplace=True)
total_vol = sum(df_kwe['Search Volume'])
print("\n-----------------------Final Stats-----------------")
print("Total Subcategory Suggestions Generated with N-grams:", ngram_count)
print("Subcategory Suggestions Matched to Minimum of:", min_product_match, "Products:", matched_product_row)
print("Unique Subcategory Suggestions After De-duplication:", creds_required)
print("Subcategories with Search Volume >", min_search_vol, "and a CPC >", min_cpc, ":", keyword_volume_count)
print("Total Subcategory Volume:", total_vol)
percent_diff = (keyword_volume_count - ngram_count) / ngram_count * 100
percent_diff = (round(percent_diff, 2))
print("\nDiscarded:", abs(percent_diff),"% of Keywords!")

print(f'\nCompleted in {time.time() - startTime:.2f} Seconds')

df_kwe.to_csv(PATH + '/category-splitter-' + str(url) + ".csv", index=False)