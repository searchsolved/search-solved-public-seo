import concurrent.futures
import itertools
import json
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import requests
import sys

# set variables
value_serp_key = "YOUR KEY HERE"

location_select = "United Kingdom"
device_select = "Desktop"
threads = 20
max_diff = 10
common_urls = 2
filter_questions = True

# read in the csv
df_comp = pd.read_csv('/python_scripts/your-csv-here.csv')

if 'Difficulty' in df_comp.columns:
    # do the difficulty calculation
    df_comp['Difficulty'] = df_comp['Difficulty'].fillna("0").astype(int)
    df_comp = df_comp[df_comp.Difficulty <= max_diff]

if filter_questions:
    q_words = "who |what |where |why |when |how |is |are |does |do |can "
    df_comp = df_comp[df_comp['Keyword'].str.contains(q_words)]

# make the initial keyword list
kws = df_comp['Keyword'].to_list()

# create allintitle and phrase matches
df_comp['allintitle_kw_temp'] = "allintitle: " + df_comp['Keyword']
df_comp['quoted_kw_temp'] = '"' + df_comp['Keyword'] + '"'

# extend kw list with allintitle and phrase matches
kws.extend(df_comp['allintitle_kw_temp'].tolist())
kws.extend(df_comp['quoted_kw_temp'].tolist())

# delete the helper columns
del df_comp['allintitle_kw_temp']
del df_comp['quoted_kw_temp']

# store the main data
search_q = []
total_results = []

# store the serp cluster df data
link_l = []
query_padded_len = []

counter = 0


def get_serp(kw):
    global counter
    counter += 1
    global list_counter
    print(f'Searching Google for: {kw}', "(", counter, "of", len(kws), ")")

    params = {
        'api_key': value_serp_key,
        'q': kw,
        'location': location_select,
        'include_fields': ['organic_results', 'search_information'],
        'location_auto': True,
        'device': device_select,
        'output': 'json',
        'page': '1',
        'num': '10'
    }

    response = requests.get('https://api.valueserp.com/search', params)
    response_data = json.loads(response.text)
    result = response_data.get('search_information')
    total_results.append((result['total_results']))
    search_q.append(kw)
    query_padded_len.append(kw)

    result_links = response_data.get('organic_results')

    for var in result_links:
        try:
            link_l.append(var['link'])
        except Exception:
            link_l.append("")

        max_list_len = max(len(link_l), len(query_padded_len))

        if len(query_padded_len) != max_list_len:
            diff = max_list_len - len(query_padded_len)
            query_padded_len.extend([kw] * diff)


# use concurrent futures
def concurrent_calls():
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        f1 = executor.map(get_serp, kws)
        for future in f1:
            return future


try:
    concurrent_calls()
except TypeError:
    print("No API Key Found!")
    sys.exit()

print("Finished searching!")
df_results = pd.DataFrame(None)
df_results['Keyword'] = search_q
df_results['Total Results'] = total_results

# make two dfs from the results df to merge back into the main df vlookup style
df_results_allintitle = df_results[df_results['Keyword'].str.contains("allintitle")].copy()
df_results_phrase = df_results[df_results['Keyword'].str.contains('"')].copy()

# rename the column names ready for the vlookup style match
df_results.rename(columns={"Total Results": "Search Results"}, inplace=True)
df_results_phrase.rename(columns={"Total Results": "Quoted Results"}, inplace=True)
df_results_allintitle.rename(columns={"Total Results": "Allintite Results"}, inplace=True)

# remove the keyword modifiers for matching
df_results_phrase['Keyword'] = df_results_phrase['Keyword'].apply(lambda x: x.replace('\"', ''))
df_results_allintitle['Keyword'] = df_results_allintitle['Keyword'].apply(lambda x: x.replace('allintitle: ', ''))

# do the vlookup merge
df_comp = pd.merge(df_comp, df_results[['Keyword', 'Search Results']], on='Keyword', how='left')
df_comp = pd.merge(df_comp, df_results_phrase[['Keyword', 'Quoted Results']], on='Keyword', how='left')
df_comp = pd.merge(df_comp, df_results_allintitle[['Keyword', 'Allintite Results']], on='Keyword', how='left')

# make the cluster df
df = pd.DataFrame(None)
df["keyword"] = query_padded_len
df["url"] = link_l

# remove the keyword modifiers before clustering
df['keyword'] = df['keyword'].apply(lambda x: x.replace('\"', ''))
df['keyword'] = df['keyword'].apply(lambda x: x.replace('allintitle: ', ''))
df.drop_duplicates(subset=["keyword", "url"], keep="first", inplace=True)

# start serp clustering
kw1_kw2_scores = []

for kw1, kw2 in combinations(df['keyword'].unique(), 2):
    kw1_urls = df[df['keyword'] == kw1]['url']
    kw2_urls = df[df['keyword'] == kw2]['url']

    kw1_kw2_score = kw1, kw2, len(set(kw1_urls).intersection(set(kw2_urls)))
    kw1_kw2_scores.append(kw1_kw2_score)

final_df = pd.DataFrame(kw1_kw2_scores, columns=['keyword1', 'keyword2', 'num_common_urls']).sort_values(
    'num_common_urls', ascending=False)

final_df = final_df.loc[final_df['num_common_urls'] >= common_urls, ['keyword1', 'keyword2']]
final_df.reset_index(inplace=True, drop=True)

final_df_dedup = final_df[~pd.DataFrame(np.sort(final_df[['keyword1', 'keyword2']], axis=1)).duplicated()]
final_df_dedup['hub'] = np.where(final_df_dedup['keyword1'].map(len) < final_df_dedup['keyword2'].map(len),
                                 final_df_dedup['keyword1'], final_df_dedup['keyword2'])

df_ul = final_df_dedup.sort_values("hub", ascending=True)

df_ul.reset_index(inplace=True, drop=True)
len_df = len(df_ul)
df_ul['counter'] = np.nan

related_rows = defaultdict(list)

for i in range(len_df):
    row = df_ul.iloc[i, 1:3]  # Series of current row
    for j in range(len_df):
        try:
            next_row = df_ul.iloc[j + 1, 1:3]  # Series of all next rows
            set_row = frozenset(row)
            set_next_row = frozenset(next_row)
            intersection = set_row.intersection(set_next_row)  # set that shows common elements

            if ((len(intersection) > 0) and (df_ul.loc[j + 1, "counter"] != 0)) or ((len(set_next_row.intersection(
                    {l for k, l_values in related_rows.items() for l in itertools.chain(*l_values)}))) > 1) and (
                    df_ul.loc[j + 1, "counter"] != 0):
                hub = df_ul.loc[i, "hub"]
                df_ul.loc[j + 1, "hub"] = hub
                df_ul.loc[j + 1, "counter"] = 0
                related_rows[hub].append(list(set_row.union(set_next_row)))  # problem here?
        except:
            break

df_ul = df_ul.loc[:, df_ul.columns != "counter"]
df_cluster = pd.DataFrame(related_rows.items(), columns=["serp_cluster", "keyword"])
df_cluster['keyword'] = [', '.join(item for sublist in x for item in sublist) for x in df_cluster["keyword"]]

try:
    df_cluster = df_cluster.set_index('serp_cluster').keyword.str.split(',', expand=True).stack().reset_index(
        'serp_cluster')
except AttributeError:
    pass

df_cluster.rename(columns={0: 'keyword'}, inplace=True)

# clean the final df
try:
    df_cluster['keyword'] = df_cluster['keyword'].str.strip()
    df_cluster.drop_duplicates(subset=["keyword", "serp_cluster"], keep="first", inplace=True)
    df_cluster.rename(columns={"keyword": "Keyword", "serp_cluster": "Serp Cluster"}, inplace=True)
    df_comp = pd.merge(df_comp, df_cluster[['Keyword', 'Serp Cluster']], on='Keyword', how='left')
    df_comp['Serp Cluster'] = df_comp['Serp Cluster'].fillna("zzz_no_cluster")
except AttributeError:
    df_comp['Serp Cluster'] = "zzz_no_cluster"

# pop the columns to the front
col = df_comp.pop('Allintite Results')
df_comp.insert(0, col.name, col)
df_comp = df_comp.sort_values(by="Allintite Results", ascending=True)

col = df_comp.pop('Quoted Results')
df_comp.insert(0, col.name, col)
df_comp = df_comp.sort_values(by="Quoted Results", ascending=True)

col = df_comp.pop('Search Results')
df_comp.insert(0, col.name, col)
df_comp = df_comp.sort_values(by="Search Results", ascending=True)

col = df_comp.pop('Keyword')
df_comp.insert(0, col.name, col)
df_comp = df_comp.sort_values(by="Keyword", ascending=True)

col = df_comp.pop('Serp Cluster')
df_comp.insert(0, col.name, col)

# check if cluster size = 1 and overwrite with no_cluster
df_comp['cluster_count'] = df_comp['Serp Cluster'].map(df_comp.groupby('Serp Cluster')['Serp Cluster'].count())

df_comp.loc[df_comp["cluster_count"] == 1, "Serp Cluster"] = "zzz_no_cluster"
del df_comp['cluster_count']
df_comp = df_comp.sort_values(by="Serp Cluster", ascending=True)

# make question dataframe ----------------------------------------------------------------------------------------------
q_words = "who|what|where|why|when|how|is|are|does|do|can"
df_questions = df_comp[df_comp['Keyword'].str.contains(q_words)]

# save the result
df_comp.to_csv("/python_scripts/output.csv")
