####################################################################################
#                                                                                  #
#  SERP Clustering at Scale                                                        #
#                                                                                  #
#  Clusters keywords based on common SERP URLs from ValueSERP batch exports.       #
#                                                                                  #
####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Twitter  : https://x.com/LeeFootSEO                                              #
####################################################################################

"""
SERP Clustering Script
Version: 2.0

Reads batch file exports from www.valueserp.com and clusters keywords based on common URLs.

Usage:
    1. Export SERP data from ValueSERP in CSV format
    2. Place CSV files in a folder
    3. Update FOLDER_LOCATION to point to your folder
    4. Run the script

Requirements:
    pip install pandas tqdm
"""

import glob
import os
import time

import pandas as pd
from tqdm import tqdm

start_time = time.time()

# ----------------
# Configuration Variables
# ----------------

CONSOLIDATE_QUERIES = True
COMMON_URLS = 4  # minimum number of common URLs to match on. Default = 4
FOLDER_LOCATION = os.path.join(os.getcwd(), 'valueserp_exports')  # folder containing exported CSV files
FILE_PREFIX = "/Batch_Results_*.csv"  # file prefix for ValueSERP exports
EXPORT_CSV_FILE_PATH = os.path.join(os.getcwd(), 'serp_cluster_results.csv')  # output file path


# ----------------
# Read and Clean Data
# ----------------

def get_csv_file_paths(folder_location, file_prefix):
    """
    Retrieves paths for CSV files from a specified folder location with a given file prefix.

    Args:
    folder_location (str): The directory path where the CSV files are stored.
    file_prefix (str): The prefix for the CSV files to be matched.

    Returns:
    list: A list of file paths matching the given file prefix in the specified folder location.
    """
    file_pattern = f"{folder_location}{file_prefix}"
    return glob.glob(file_pattern)


def read_csv_files(file_paths):
    """
    Reads multiple CSV files from the provided file paths and concatenates them into a single DataFrame.

    Args:
    file_paths (list): List of file paths of the CSV files to be read.

    Returns:
    pandas.DataFrame: A DataFrame containing concatenated data from the read CSV files.
    """
    df = pd.concat((pd.read_csv(file, usecols=["search.q", "result.organic_results.link"],
                                dtype="str", index_col=None, header=0)
                    for file in tqdm(file_paths, desc="Reading in CSV files...")),
                   axis=0, ignore_index=True)

    return df


def rename_columns(df):
    """
    Renames specific columns of a DataFrame to more descriptive names.

    Args:
    df (pandas.DataFrame): The DataFrame whose columns are to be renamed.

    Returns:
    pandas.DataFrame: The DataFrame with renamed columns.
    """
    return df.rename(columns={"search.q": "query", "result.organic_results.link": "link"})


def normalize_query_strings(df):
    """
    Converts all query strings in the DataFrame to lowercase for normalization.

    Args:
    df (pandas.DataFrame): DataFrame containing the query strings.

    Returns:
    pandas.DataFrame: DataFrame with query strings normalized to lowercase.
    """
    return df.assign(query=lambda x: x['query'].str.lower())


def filter_data(df):
    """
    Filters the DataFrame to only include rows where the link appears more than once and removes duplicates.

    Args:
    df (pandas.DataFrame): DataFrame to be filtered.

    Returns:
    pandas.DataFrame: Filtered DataFrame with duplicate entries removed.
    """
    mask = df['link'].map(df['link'].value_counts()) > 1
    return df[mask].drop_duplicates(subset=["query", "link"])


# ----------------
# Data Mapping and Transformation
# ----------------

def create_query_map(df):
    """
    Creates a mapping of queries to sets of links from the DataFrame.

    Args:
    df (pandas.DataFrame): DataFrame containing query and link data.

    Returns:
    dict: A dictionary mapping each query to a set of associated links.
    """
    return df.groupby('query')['link'].apply(set).to_dict()


def invert_query_map(query_map):
    """
    Inverts a query map to a link map, where each link maps to a set of queries.

    Args:
    query_map (dict): A dictionary mapping queries to sets of links.

    Returns:
    dict: A dictionary mapping each link to a set of associated queries.
    """
    link_map = {}
    for query, links in query_map.items():
        for link in links:
            if link in link_map:
                link_map[link].add(query)
            else:
                link_map[link] = {query}
    return link_map


# ----------------
# Clustering and Analysis
# ----------------

def find_common_links(query_map, common_urls):
    """
    Identifies common links shared between pairs of queries from a query map.

    Args:
    query_map (dict): A dictionary mapping queries to sets of links.
    common_urls (int): Minimum number of common URLs to consider a pair for inclusion.

    Returns:
    pandas.DataFrame: DataFrame of query pairs with their common links and respective counts.
    """
    link_map = invert_query_map(query_map)
    common_link_pairs = {}

    for link, queries in tqdm(link_map.items(), desc='Processing links...'):
        for query1 in queries:
            for query2 in queries:
                if query1 != query2:
                    pair = tuple(sorted([query1, query2]))
                    if pair not in common_link_pairs:
                        common_link_pairs[pair] = set()
                    common_link_pairs[pair].add(link)

    # Filter pairs by the number of common URLs
    common_pairs = [(pair, len(links), links) for pair, links in common_link_pairs.items() if
                    len(links) >= common_urls]
    return pd.DataFrame(common_pairs, columns=['query', '#_common_urls', 'common_urls'])


def assign_cluster_names(df):
    """
    Assigns unique cluster names to each row in the DataFrame.

    Args:
    df (pandas.DataFrame): DataFrame containing queries and related data.

    Returns:
    pandas.DataFrame: DataFrame with an added column for cluster names.
    """
    df['serp_cluster'] = [f'group {i + 1}' for i in range(df.shape[0])]
    return df.explode("query")


def group_queries_by_cluster(df):
    """
    Groups queries by their assigned cluster names and consolidates them.

    Args:
    df (pandas.DataFrame): DataFrame containing queries and their assigned clusters.

    Returns:
    pandas.DataFrame: DataFrame with consolidated cluster information.
    """
    tqdm.pandas(desc="Consolidating Clusters")
    grouped_data = df.groupby('query')['serp_cluster'] \
        .progress_apply('|'.join) \
        .reset_index(name='serp_clusters')
    return grouped_data


def merge_cluster_data(original_df, grouped_data):
    """
    Merges cluster data with the original DataFrame to provide a comprehensive view.

    Args:
    original_df (pandas.DataFrame): The original DataFrame before clustering.
    grouped_data (pandas.DataFrame): DataFrame containing consolidated cluster information.

    Returns:
    pandas.DataFrame: Merged DataFrame with both original data and cluster information.
    """
    tqdm.pandas(desc="Merging Cluster Data")
    expanded_data = grouped_data.assign(serp_cluster=grouped_data['serp_clusters'].str.split('|')) \
        .explode('serp_cluster')[['serp_cluster', 'query']]

    # Mapping cluster names
    cluster_names = expanded_data.groupby('serp_cluster').first()['query'].to_dict()
    expanded_data['serp_cluster'] = expanded_data['serp_cluster'].map(cluster_names)
    expanded_data.drop_duplicates(subset=["serp_cluster", "query"], inplace=True)

    return pd.merge(original_df, expanded_data, on='query', suffixes=('_left', '_right'))


def rename_clusters_shortest_kw(df):
    """
    Renames clusters based on the shortest keyword in each cluster.

    Args:
    df (pandas.DataFrame): DataFrame containing queries and their respective clusters.

    Returns:
    pandas.DataFrame: DataFrame with clusters renamed based on the shortest keyword.
    """
    df['query_len'] = df['query'].str.len()
    idx = df.groupby('serp_cluster_right')['query_len'].idxmin()
    shortest_query_map = df.loc[idx, ['serp_cluster_right', 'query']].set_index('serp_cluster_right')['query']
    df['serp_cluster'] = df['serp_cluster_right'].map(shortest_query_map)
    df.drop(columns=['serp_cluster_right', 'query_len'], inplace=True)

    return df


# ----------------
# Data Post-Processing
# ----------------

def count_cluster_sizes(df):
    """
    Counts the number of keywords in each cluster and adds this information to the DataFrame.

    Args:
    df (pandas.DataFrame): DataFrame containing queries and their clusters.

    Returns:
    pandas.DataFrame: Updated DataFrame with the count of keywords in each cluster.
    """
    df.drop_duplicates(subset=["serp_cluster", "query"], inplace=True)
    df.loc[:, '#_kws_in_cluster'] = df.groupby('serp_cluster')['serp_cluster'].transform('count')

    return df


def sort_for_export(df):
    """
    Prepares the DataFrame for export by sorting and filtering the data.

    Args:
    df (pandas.DataFrame): DataFrame to be prepared for export.

    Returns:
    pandas.DataFrame: Sorted and filtered DataFrame ready for export.
    """
    df = df[['serp_cluster', 'query', '#_kws_in_cluster', '#_common_urls', 'common_urls']].drop_duplicates(
        subset=["query"])
    df['#_kws_in_cluster'] = df.groupby('serp_cluster')['serp_cluster'].transform('count')
    df = df[df['#_kws_in_cluster'] != 1]
    return df.sort_values(["serp_cluster", "query", "#_kws_in_cluster"], ascending=[True, True, False])


# ----------------
# Export the Data
# ----------------

def export_to_csv(df, file_path):
    """
    Exports the given DataFrame to a CSV file at the specified file path.

    Args:
    df (pandas.DataFrame): DataFrame to be exported.
    file_path (str): Path where the CSV file will be saved.

    Returns:
    None
    """
    df.to_csv(file_path, index=False)


# ----------------
# Main Processing Function
# ----------------

def process_serps(consolidate=CONSOLIDATE_QUERIES):
    """
    Main function to process Search Engine Result Pages (SERPs).

    Args:
    consolidate (bool): Flag to determine whether to consolidate query clusters.

    Returns:
    None
    """

    # Step 1: Get CSV file paths
    file_paths = get_csv_file_paths(FOLDER_LOCATION, FILE_PREFIX)

    if not file_paths:
        print(f"No CSV files found in {FOLDER_LOCATION}")
        print("Please ensure your ValueSERP exports are in the correct folder.")
        return

    # Step 2: Read CSV files into a DataFrame
    df = read_csv_files(file_paths)

    # Step 3: Clean the data
    df = rename_columns(df)
    df = normalize_query_strings(df)
    df = filter_data(df)

    # Step 4: Create a map of queries
    query_map = create_query_map(df)

    # Step 5: Find common links
    df = find_common_links(query_map, COMMON_URLS)

    # Step 6: Assign cluster names
    df = assign_cluster_names(df)

    # Step 7: Optionally consolidate clusters
    if consolidate:
        grouped_data = group_queries_by_cluster(df)
        df = merge_cluster_data(df, grouped_data)

    # Step 8: Rename clusters based on shortest keyword
    df = rename_clusters_shortest_kw(df)

    # Step 9: Count cluster sizes
    df = count_cluster_sizes(df)

    # Step 10: Sort data for export
    df = sort_for_export(df)

    # Step 11: Export the processed DataFrame to a CSV file
    export_to_csv(df, EXPORT_CSV_FILE_PATH)
    print(f"Results exported to: {EXPORT_CSV_FILE_PATH}")


# ----------------
# Execute the Main Function
# ----------------

if __name__ == "__main__":
    process_serps()
    print('The script took {0} seconds!'.format(time.time() - start_time))
