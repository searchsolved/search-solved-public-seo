####################################################################################
#                                                                                  #
#  SERP Clustering at Scale                                                        #
#                                                                                  #
#  Clusters keywords based on common SERP URLs from ValueSERP batch exports.       #
#  Supports multiple clustering strategies with consolidation scoring.              #
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
SERP Clustering Script - Improved Version
Version: 3.0

A script to identify content consolidation opportunities based on shared URLs in search results.
Supports overlapping clusters and multiple clustering strategies.

Features:
- Multi-strategy clustering (connected components, cliques, core-based)
- Consolidation scoring (0-100) to prioritize opportunities
- Overlap detection between clusters
- Detailed cluster metrics

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
import logging
from collections import defaultdict
from itertools import combinations

import pandas as pd
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

start_time = time.time()

# ----------------
# Configuration Variables
# ----------------

COMMON_URLS = 4  # minimum number of common URLs to match on. Default = 4
CLUSTERING_STRATEGY = 'all'  # Options: 'connected', 'cliques', 'core', 'all'
CORE_THRESHOLD = 0.7  # For core clustering: minimum connectivity percentage
FOLDER_LOCATION = os.path.join(os.getcwd(), 'valueserp_exports')  # folder containing exported CSV files
FILE_PREFIX = "/Batch_Results_*.csv"  # file prefix for ValueSERP exports
EXPORT_CSV_FILE_PATH = os.path.join(os.getcwd(), 'serp_cluster_results.csv')  # output file path


# ----------------
# Read and Clean Data
# ----------------

def validate_folder_and_files(folder_location, file_prefix):
    """Validates folder exists and contains required files."""
    if not os.path.exists(folder_location):
        raise ValueError(f"Folder not found: {folder_location}")

    file_pattern = f"{folder_location}{file_prefix}"
    files = glob.glob(file_pattern)

    if not files:
        raise ValueError(
            f"No CSV files found matching pattern '{file_prefix}' in folder: {folder_location}\n"
            f"Please ensure CSV files are present and match the pattern: Batch_Results_*.csv"
        )

    logger.info(f"Found {len(files)} CSV files to process")
    return files


def read_csv_files(file_paths):
    """Reads and concatenates multiple CSV files."""
    dataframes = []

    for file in tqdm(file_paths, desc="Reading CSV files"):
        try:
            df = pd.read_csv(file, usecols=["search.q", "result.organic_results.link"],
                             dtype="str", index_col=None, header=0)
            if not df.empty:
                dataframes.append(df)
            else:
                logger.warning(f"Empty CSV file: {file}")
        except Exception as e:
            logger.error(f"Error reading file {file}: {str(e)}")
            continue

    if not dataframes:
        raise ValueError("No valid data found in any of the CSV files")

    return pd.concat(dataframes, axis=0, ignore_index=True)


def prepare_data(df):
    """Prepares and cleans the data for clustering."""
    # Rename columns
    df = df.rename(columns={"search.q": "query", "result.organic_results.link": "link"})

    # Convert queries to lowercase
    df['query'] = df['query'].str.lower()

    # Remove duplicates but keep all queries even if they don't share links
    return df.drop_duplicates(subset=["query", "link"])


def create_query_map(df):
    """Creates a mapping of queries to their sets of URLs."""
    return df.groupby('query')['link'].apply(set).to_dict()


# ----------------
# Clustering Strategies
# ----------------

def build_similarity_matrix(query_map, common_urls_threshold):
    """Build similarity matrix between queries based on shared URLs."""
    similarity_matrix = defaultdict(dict)
    queries = list(query_map.keys())

    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            query1, query2 = queries[i], queries[j]
            shared_urls = len(query_map[query1] & query_map[query2])
            if shared_urls >= common_urls_threshold:
                similarity_matrix[query1][query2] = shared_urls
                similarity_matrix[query2][query1] = shared_urls

    return similarity_matrix, queries


def find_connected_components(similarity_matrix, queries):
    """Strategy 1: Find connected components (non-overlapping base clusters)."""
    visited = set()
    components = []

    def dfs(query, component):
        if query in visited:
            return
        visited.add(query)
        component.add(query)
        for neighbor in similarity_matrix[query]:
            dfs(neighbor, component)

    for query in queries:
        if query not in visited and query in similarity_matrix:
            component = set()
            dfs(query, component)
            if len(component) > 1:
                components.append(component)

    return components


def find_cliques(similarity_matrix, queries, min_clique_size=2):
    """Strategy 2: Find cliques (all queries must be connected to each other)."""
    cliques = []

    def is_clique(candidate_set):
        candidate_list = list(candidate_set)
        for i in range(len(candidate_list)):
            for j in range(i + 1, len(candidate_list)):
                if candidate_list[j] not in similarity_matrix[candidate_list[i]]:
                    return False
        return True

    # Find maximal cliques using a simplified approach
    for query in queries:
        if query not in similarity_matrix:
            continue

        candidates = {query}
        candidates.update(similarity_matrix[query].keys())

        # Try to build maximal clique
        for size in range(len(candidates), min_clique_size - 1, -1):
            for subset in combinations(candidates, size):
                if is_clique(set(subset)):
                    clique = set(subset)
                    # Check if this clique is already found or is a subset
                    is_new = True
                    for existing_clique in cliques:
                        if clique.issubset(existing_clique):
                            is_new = False
                            break
                    if is_new:
                        cliques.append(clique)
                    break

    # Remove subsets
    final_cliques = []
    for clique in cliques:
        is_subset = False
        for other_clique in cliques:
            if clique != other_clique and clique.issubset(other_clique):
                is_subset = True
                break
        if not is_subset:
            final_cliques.append(clique)

    return final_cliques


def find_core_clusters(similarity_matrix, queries, core_threshold=0.7):
    """Strategy 3: Core-based clustering (queries must share URLs with core set)."""
    core_clusters = []

    for seed_query in queries:
        if seed_query not in similarity_matrix:
            continue

        cluster = {seed_query}
        candidates = set(similarity_matrix[seed_query].keys())

        for candidate in candidates:
            # Check if candidate is connected to enough existing cluster members
            connections = sum(1 for member in cluster if candidate in similarity_matrix[member])
            if connections >= len(cluster) * core_threshold:
                cluster.add(candidate)

        if len(cluster) > 1 and cluster not in core_clusters:
            core_clusters.append(cluster)

    return core_clusters


# ----------------
# Cluster Analysis
# ----------------

def get_shortest_query_in_cluster(queries):
    """Returns the shortest query from a list of queries."""
    return min(queries, key=len)


def analyze_cluster_details(cluster_queries, query_map, similarity_matrix):
    """Analyze detailed metrics for a cluster."""
    queries = list(cluster_queries)

    # Find URLs shared by all queries
    if queries:
        shared_urls = set(query_map[queries[0]])
        for query in queries[1:]:
            shared_urls &= query_map[query]
    else:
        shared_urls = set()

    # Calculate pairwise metrics
    metrics = {
        'min_shared_urls': float('inf'),
        'max_shared_urls': 0,
        'avg_shared_urls': 0,
        'total_comparisons': 0,
        'connectivity_score': 0
    }

    possible_connections = len(queries) * (len(queries) - 1) / 2
    actual_connections = 0

    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            if queries[j] in similarity_matrix[queries[i]]:
                shared = similarity_matrix[queries[i]][queries[j]]
                metrics['min_shared_urls'] = min(metrics['min_shared_urls'], shared)
                metrics['max_shared_urls'] = max(metrics['max_shared_urls'], shared)
                metrics['avg_shared_urls'] += shared
                metrics['total_comparisons'] += 1
                actual_connections += 1

    if metrics['total_comparisons'] > 0:
        metrics['avg_shared_urls'] /= metrics['total_comparisons']
    else:
        metrics['min_shared_urls'] = 0

    if possible_connections > 0:
        metrics['connectivity_score'] = actual_connections / possible_connections

    return {
        'queries': queries,
        'shared_urls': list(shared_urls),
        'shared_url_count': len(shared_urls),
        'cluster_metrics': metrics,
        'cluster_size': len(queries)
    }


def calculate_consolidation_score(metrics, cluster_size, overlapping_count):
    """
    Calculate a score (0-100) indicating how strong the consolidation opportunity is.
    Higher scores mean stronger consolidation candidates.
    """
    # Base score from average shared URLs (normalize to 0-40 range)
    base_score = min(40, metrics['avg_shared_urls'] * 4)

    # Connectivity bonus (0-30 range)
    connectivity_bonus = metrics['connectivity_score'] * 30

    # Cluster size bonus (0-20 range)
    size_bonus = min(20, (cluster_size - 2) * 5)

    # Overlap penalty (0-10 range)
    overlap_penalty = min(10, overlapping_count * 5)

    total_score = base_score + connectivity_bonus + size_bonus - overlap_penalty

    return max(0, min(100, round(total_score)))


def get_consolidation_recommendation(score):
    """Get a recommendation based on the consolidation score."""
    if score >= 80:
        return "Strong consolidation candidate"
    elif score >= 60:
        return "Good consolidation candidate"
    elif score >= 40:
        return "Possible consolidation"
    elif score >= 20:
        return "Weak consolidation candidate"
    else:
        return "Keep separate"


# ----------------
# Main Clustering Function
# ----------------

def find_consolidation_clusters(query_map, common_urls_threshold, strategy='all', core_threshold=0.7):
    """
    Main clustering function supporting multiple strategies.
    Returns clusters with metadata and consolidation scores.
    """
    similarity_matrix, queries = build_similarity_matrix(query_map, common_urls_threshold)

    all_clusters = []

    # Get clusters based on selected strategy
    if strategy in ['connected', 'all']:
        components = find_connected_components(similarity_matrix, queries)
        for comp in components:
            cluster_data = analyze_cluster_details(comp, query_map, similarity_matrix)
            cluster_data['cluster_type'] = 'connected_component'
            all_clusters.append(cluster_data)

    if strategy in ['cliques', 'all']:
        cliques = find_cliques(similarity_matrix, queries)
        for clique in cliques:
            cluster_data = analyze_cluster_details(clique, query_map, similarity_matrix)
            cluster_data['cluster_type'] = 'clique'
            all_clusters.append(cluster_data)

    if strategy in ['core', 'all']:
        core_clusters = find_core_clusters(similarity_matrix, queries, core_threshold)
        for core_cluster in core_clusters:
            cluster_data = analyze_cluster_details(core_cluster, query_map, similarity_matrix)
            cluster_data['cluster_type'] = 'core_cluster'
            all_clusters.append(cluster_data)

    # Mark which queries appear in multiple clusters
    query_cluster_count = defaultdict(int)
    for cluster in all_clusters:
        for query in cluster['queries']:
            query_cluster_count[query] += 1

    # Add overlap information
    for cluster in all_clusters:
        cluster['overlapping_queries'] = [q for q in cluster['queries'] if query_cluster_count[q] > 1]

    return all_clusters, similarity_matrix


def create_cluster_dataframe(clusters, query_map):
    """
    Creates a dataframe with cluster results including consolidation scores.
    """
    rows = []
    processed_queries = set()

    # Process each cluster
    for cluster_idx, cluster in enumerate(clusters):
        cluster_name = get_shortest_query_in_cluster(cluster['queries'])

        for query in cluster['queries']:
            processed_queries.add(query)

            # Calculate consolidation score
            consolidation_score = calculate_consolidation_score(
                cluster['cluster_metrics'],
                cluster['cluster_size'],
                len(cluster['overlapping_queries'])
            )

            rows.append({
                'serp_cluster': cluster_name,
                'cluster_type': cluster['cluster_type'],
                'query': query,
                'total_urls': len(query_map[query]),
                'shared_url_count': cluster['shared_url_count'],
                'shared_urls': ', '.join(cluster['shared_urls'][:5]),
                'min_shared_urls': cluster['cluster_metrics']['min_shared_urls'],
                'max_shared_urls': cluster['cluster_metrics']['max_shared_urls'],
                'avg_shared_urls': round(cluster['cluster_metrics']['avg_shared_urls'], 2),
                'connectivity_score': round(cluster['cluster_metrics']['connectivity_score'], 2),
                '#_kws_in_cluster': cluster['cluster_size'],
                'is_in_multiple_clusters': query in cluster['overlapping_queries'],
                'consolidation_score': consolidation_score,
                'consolidation_recommendation': get_consolidation_recommendation(consolidation_score)
            })

    # Add unclustered queries
    for query in query_map.keys():
        if query not in processed_queries:
            rows.append({
                'serp_cluster': 'NO_CLUSTER',
                'cluster_type': 'none',
                'query': query,
                'total_urls': len(query_map[query]),
                'shared_url_count': 0,
                'shared_urls': '',
                'min_shared_urls': 0,
                'max_shared_urls': 0,
                'avg_shared_urls': 0,
                'connectivity_score': 0,
                '#_kws_in_cluster': 1,
                'is_in_multiple_clusters': False,
                'consolidation_score': 0,
                'consolidation_recommendation': 'Keep separate'
            })

    return pd.DataFrame(rows)


def export_results(df, file_path):
    """Exports the clustering results to a CSV file."""
    # Sort by consolidation score (descending), then by cluster and query
    df = df.sort_values(['consolidation_score', 'serp_cluster', 'query'],
                        ascending=[False, True, True])
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    logger.info(f"Results exported successfully to {file_path}")


# ----------------
# Main Processing Function
# ----------------

def process_serps():
    """Main function to process Search Engine Result Pages and identify consolidation opportunities."""
    try:
        # Validate folder and get file paths
        logger.info(f"Looking for CSV files in: {FOLDER_LOCATION}")
        file_paths = validate_folder_and_files(FOLDER_LOCATION, FILE_PREFIX)

        # Read CSV files
        df = read_csv_files(file_paths)
        logger.info(f"Successfully read {len(df)} rows from CSV files")

        # Prepare data
        df = prepare_data(df)
        logger.info(f"After preparation: {len(df)} unique query-URL pairs")

        # Create query to URL mapping
        query_map = create_query_map(df)
        logger.info(f"Processing {len(query_map)} unique queries")

        # Find consolidation clusters
        clusters, similarity_matrix = find_consolidation_clusters(
            query_map,
            COMMON_URLS,
            strategy=CLUSTERING_STRATEGY,
            core_threshold=CORE_THRESHOLD
        )

        # Create results DataFrame
        results_df = create_cluster_dataframe(clusters, query_map)

        # Export results
        export_results(results_df, EXPORT_CSV_FILE_PATH)

        # Calculate statistics
        total_queries = len(query_map)
        clustered_queries = len(results_df[results_df['serp_cluster'] != 'NO_CLUSTER']['query'].unique())
        clique_count = len([c for c in clusters if c['cluster_type'] == 'clique'])
        component_count = len([c for c in clusters if c['cluster_type'] == 'connected_component'])
        core_count = len([c for c in clusters if c['cluster_type'] == 'core_cluster'])

        logger.info(f'Script completed in {time.time() - start_time:.2f} seconds')
        logger.info(f'Found {len(clusters)} total clusters:')
        logger.info(f'  - {component_count} connected components')
        logger.info(f'  - {clique_count} cliques')
        logger.info(f'  - {core_count} core clusters')
        logger.info(f'{clustered_queries} out of {total_queries} queries are in clusters')
        logger.info(f'Results exported to: {EXPORT_CSV_FILE_PATH}')

        # Show top consolidation opportunities
        top_opportunities = results_df[results_df['consolidation_score'] >= 60].groupby('serp_cluster').first()
        if not top_opportunities.empty:
            logger.info(f"\nTop consolidation opportunities:")
            for idx, row in top_opportunities.head(5).iterrows():
                logger.info(f"  - {idx}: Score {row['consolidation_score']} ({row['#_kws_in_cluster']} keywords)")

        return results_df

    except Exception as e:
        logger.error(f"Error processing SERPs: {str(e)}")
        raise


# ----------------
# Execute the Main Function
# ----------------

if __name__ == "__main__":
    process_serps()
    print(f'The script took {time.time() - start_time:.2f} seconds!')
