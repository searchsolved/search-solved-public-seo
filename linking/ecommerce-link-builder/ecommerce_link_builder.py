####################################################################################
# Website  : https://leefoot.co.uk/                                                #
# Contact  : https://leefoot.co.uk/hire-me/                                        #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Twitter  : https://twitter.com/LeeFootSEO                                        #
####################################################################################

import time
import json
import pandas as pd
import requests
import os
startTime = time.time()

# get the current working directory and print
path = os.getcwd()
print(path)

# read in the zenserp.com key to scrape the serps
with open(path +'/zenserp_key.txt', 'r') as file:  # read in the Keywords Everywhere API Key
    zenserp_key = file.read()

# read in the list of brands
with open(path +'/brands.txt', 'r') as file:  # read in the Keywords Everywhere API Key
    brands = file.read().splitlines()

# make a temp dataframe to append the word 'stockists'
df = pd.DataFrame(brands, columns=["brand"])
df['stockists'] = " Stockists"
df['brand'] = df['brand'] + df['stockists']
total = len(df['brand'])
search_terms = df['brand'].tolist()  # dump search term queries to a list (to loop through with the Search Console API)

# make empty list and dataframe to store the extracted data
df_final = pd.DataFrame(None)
url_list = []
description_list = []
title_list = []
query_list = []
df_position = []
count = 0
for i in search_terms:
    count = count + 1
    print("Searching:", i.strip(), count, "of", total)
    headers = {"apikey": zenserp_key}
    params = (
        ("q", i),
        ("device", "desktop"),
        ("search_engine", "google.co.uk"),
        ("location", "London,England,United Kingdom"),
        ("gl", "GB"),
        ("hl", "en"),
        ("apikey", zenserp_key),
    )

    response = requests.get('https://app.zenserp.com/api/v2/search', headers=headers, params=params);

    # Get JSON Data
    d = response.json()
    json_str = json.dumps(d)  # dumps the json object into an element
    resp = json.loads(json_str)  # load the json to a string
    organic = (resp['organic'])

    # get the length of the list to iterate over in the loop
    list_len = len(organic)
    pos_counter = 0
    counter = 0
    while counter != list_len:
        access = (organic[counter])
        pos_counter = pos_counter + 1
        df_position.append(pos_counter)
        try:
            my_url = (access['url'])
            url_list.append(my_url)
        except Exception:
            url_list.append("MISSING")
            pass

        try:
            my_description = (access['description'])
            description_list.append(my_description)
        except Exception:
            description_list.append("MISSING")
            pass

        try:
            my_title = (access['title'])
            title_list.append(my_title)
        except Exception:
            title_list.append("MISSING")
            pass

        query = (resp['query'])
        q_access = (query['q'])
        query_list.append(q_access)

        counter = counter +1

# add lists to dataframe columns
df_final['query'] = query_list
df_final['url'] = url_list
df_final['title'] = title_list
df_final['description'] = description_list
df_final['position'] = df_position

# clean the data!
df_final = df_final[df_final.position == 1]  # keep position 1 result for each search only
df_final = df_final[~df_final["url"].isin(['MISSING'])]
df_final = df_final[~df_final["description"].isin(['MISSING'])]
df_final = df_final[~df_final["title"].isin(['MISSING'])]
df_final["temp_url"] = df_final.loc[:, ["url"]]

# Remove homepages. Ensures consistancy by adding extra ///// which are removed so all domains are stripped back
df_final["temp_url"] = df_final["temp_url"] + "/////"
df_final["temp_url"] = df_final["temp_url"].str.replace("//////", "")
df_final["temp_url"] = df_final["temp_url"].str.replace("/////", "")
df_final['url_depth'] = df_final["temp_url"].str.count("/")
df_final = df_final[~df_final['url_depth'].isin(["2"])]  # depth 2 = homepage link

# clean and sort the columns
cols = "query", "url", "title", "description"
df_final = df_final.reindex(columns=cols)
df_final.drop_duplicates(subset=['url'], keep="first", inplace=True)

# export the data
df_final.to_csv(path + '/brand_links_output.csv')
print(f'\nCompleted in {time.time() - startTime:.2f} Seconds')
