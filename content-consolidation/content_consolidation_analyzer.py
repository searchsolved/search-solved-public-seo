####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://leefoot.co.uk                                               #
# Contact  : https://leefoot.co.uk/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                                              #
####################################################################################

"""Content Consolidation Analyzer

Identifies content consolidation opportunities by clustering pages that share
significant SERP overlap. Helps find cannibalizing pages competing for the same keywords.

Usage:
    python content_consolidation_analyzer.py --input data/*.csv --output results.csv
    python content_consolidation_analyzer.py --input serp_data.csv --min-urls 3 --query-col keyword --url-col url
"""

import argparse
import glob
import time
import pandas as pd
from tqdm import tqdm
import logging
import os
from collections import defaultdict
from itertools import combinations

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_files(file_pattern):
    """Validates files exist matching the pattern."""
    files = glob.glob(file_pattern)

    if not files:
        raise ValueError(f"No CSV files found matching pattern: {file_pattern}")

    logger.info(f"Found {len(files)} CSV files to process")
    return files


def read_csv_files(file_paths, query_col, url_col):
    """Reads and concatenates multiple CSV files."""
    dataframes = []

    for file in tqdm(file_paths, desc="Reading CSV files"):
        try:
            df = pd.read_csv(file, usecols=[query_col, url_col], dtype="str", header=0)
            if not df.empty:
                dataframes.append(df)
            else:
                logger.warning(f"Empty CSV file: {file}")
        except Exception as e:
            logger.error(f"Error reading file {file}: {str(e)}")
            continue

    if not dataframes:
        raise ValueError("No valid data found in any of the CSV files")

    combined = pd.concat(dataframes, axis=0, ignore_index=True)
    # Standardize column names
    combined = combined.rename(columns={query_col: "query", url_col: "link"})
    return combined


def prepare_data(df):
    """Prepares and cleans the data for clustering."""
    df['query'] = df['query'].str.lower()
    return df.drop_duplicates(subset=["query", "link"])


def create_query_map(df):
    """Creates a mapping of queries to their sets of URLs."""
    return df.groupby('query')['link'].apply(set).to_dict()


def get_shortest_query_in_cluster(queries):
    """Returns the shortest query from a list of queries."""
    return min(queries, key=len)


def find_consolidation_clusters(query_map, common_urls_threshold):
    """
    Clustering that allows overlapping clusters using multiple strategies.
    Returns clusters where pages share significant SERP overlap.
    """
    similarity_matrix = defaultdict(dict)
    queries = list(query_map.keys())

    # Calculate pairwise similarities
    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            query1, query2 = queries[i], queries[j]
            shared_urls = len(query_map[query1] & query_map[query2])
            if shared_urls >= common_urls_threshold:
                similarity_matrix[query1][query2] = shared_urls
                similarity_matrix[query2][query1] = shared_urls

    # Find connected components
    def find_connected_components():
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

    # Find cliques
    def find_cliques(min_clique_size=2):
        cliques = []

        def is_clique(candidate_set):
            candidate_list = list(candidate_set)
            for i in range(len(candidate_list)):
                for j in range(i + 1, len(candidate_list)):
                    if candidate_list[j] not in similarity_matrix[candidate_list[i]]:
                        return False
            return True

        for query in queries:
            if query not in similarity_matrix:
                continue

            candidates = {query}
            candidates.update(similarity_matrix[query].keys())

            for size in range(len(candidates), min_clique_size - 1, -1):
                for subset in combinations(candidates, size):
                    if is_clique(set(subset)):
                        clique = set(subset)
                        is_new = True
                        for existing_clique in cliques:
                            if clique.issubset(existing_clique):
                                is_new = False
                                break
                        if is_new:
                            cliques.append(clique)
                        break

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

    components = find_connected_components()
    cliques = find_cliques()

    all_clusters = []

    for comp in components:
        cluster_data = analyze_cluster_details(comp, query_map, similarity_matrix)
        cluster_data['cluster_type'] = 'connected_component'
        all_clusters.append(cluster_data)

    for clique in cliques:
        cluster_data = analyze_cluster_details(clique, query_map, similarity_matrix)
        cluster_data['cluster_type'] = 'clique'
        all_clusters.append(cluster_data)

    # Mark overlapping queries
    query_cluster_count = defaultdict(int)
    for cluster in all_clusters:
        for query in cluster['queries']:
            query_cluster_count[query] += 1

    for cluster in all_clusters:
        cluster['overlapping_queries'] = [q for q in cluster['queries'] if query_cluster_count[q] > 1]

    return all_clusters, similarity_matrix


def analyze_cluster_details(cluster_queries, query_map, similarity_matrix):
    """Analyze detailed metrics for a cluster."""
    queries = list(cluster_queries)

    if queries:
        shared_urls = set(query_map[queries[0]])
        for query in queries[1:]:
            shared_urls &= query_map[query]
    else:
        shared_urls = set()

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
    """Calculate a score (0-100) indicating consolidation strength."""
    base_score = min(40, metrics['avg_shared_urls'] * 4)
    connectivity_bonus = metrics['connectivity_score'] * 30
    size_bonus = min(20, (cluster_size - 2) * 5)
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


def create_cluster_dataframe(clusters, query_map):
    """Creates a dataframe with consolidation recommendations."""
    rows = []
    processed_queries = set()

    for cluster in clusters:
        cluster_name = get_shortest_query_in_cluster(cluster['queries'])

        for query in cluster['queries']:
            processed_queries.add(query)

            consolidation_score = calculate_consolidation_score(
                cluster['cluster_metrics'],
                cluster['cluster_size'],
                len(cluster['overlapping_queries'])
            )

            rows.append({
                'cluster_id': cluster_name,
                'cluster_type': cluster['cluster_type'],
                'query': query,
                'total_urls': len(query_map[query]),
                'shared_url_count': cluster['shared_url_count'],
                'shared_urls': ', '.join(cluster['shared_urls'][:5]),
                'min_shared_urls': cluster['cluster_metrics']['min_shared_urls'],
                'max_shared_urls': cluster['cluster_metrics']['max_shared_urls'],
                'avg_shared_urls': round(cluster['cluster_metrics']['avg_shared_urls'], 2),
                'connectivity_score': round(cluster['cluster_metrics']['connectivity_score'], 2),
                'cluster_size': cluster['cluster_size'],
                'is_in_multiple_clusters': query in cluster['overlapping_queries'],
                'consolidation_score': consolidation_score,
                'consolidation_recommendation': get_consolidation_recommendation(consolidation_score)
            })

    # Add unclustered queries
    for query in query_map.keys():
        if query not in processed_queries:
            rows.append({
                'cluster_id': 'NO_CLUSTER',
                'cluster_type': 'none',
                'query': query,
                'total_urls': len(query_map[query]),
                'shared_url_count': 0,
                'shared_urls': '',
                'min_shared_urls': 0,
                'max_shared_urls': 0,
                'avg_shared_urls': 0,
                'connectivity_score': 0,
                'cluster_size': 1,
                'is_in_multiple_clusters': False,
                'consolidation_score': 0,
                'consolidation_recommendation': 'Keep separate'
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Content Consolidation Analyzer - Find cannibalizing pages via SERP overlap"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input CSV file pattern (e.g., 'data/*.csv' or 'serp_data.csv')"
    )
    parser.add_argument(
        "--output", "-o",
        default="consolidation_results.csv",
        help="Output CSV file path (default: consolidation_results.csv)"
    )
    parser.add_argument(
        "--min-urls",
        type=int,
        default=4,
        help="Minimum shared URLs to cluster (default: 4)"
    )
    parser.add_argument(
        "--query-col",
        default="search.q",
        help="Column name for queries (default: search.q)"
    )
    parser.add_argument(
        "--url-col",
        default="result.organic_results.link",
        help="Column name for URLs (default: result.organic_results.link)"
    )

    args = parser.parse_args()

    start_time = time.time()

    try:
        logger.info(f"Looking for CSV files matching: {args.input}")
        file_paths = validate_files(args.input)

        df = read_csv_files(file_paths, args.query_col, args.url_col)
        logger.info(f"Successfully read {len(df)} rows from CSV files")

        df = prepare_data(df)
        logger.info(f"After preparation: {len(df)} unique query-URL pairs")

        query_map = create_query_map(df)
        logger.info(f"Processing {len(query_map)} unique queries")

        clusters, _ = find_consolidation_clusters(query_map, args.min_urls)

        results_df = create_cluster_dataframe(clusters, query_map)

        # Sort by score and save
        results_df = results_df.sort_values(
            ['consolidation_score', 'cluster_id', 'query'],
            ascending=[False, True, True]
        )
        results_df.to_csv(args.output, index=False, encoding='utf-8-sig')

        # Statistics
        total_queries = len(query_map)
        clustered_queries = len(results_df[results_df['cluster_id'] != 'NO_CLUSTER']['query'].unique())

        logger.info(f'Script completed in {time.time() - start_time:.2f} seconds')
        logger.info(f'Found {len(clusters)} clusters')
        logger.info(f'{clustered_queries} out of {total_queries} queries are in clusters')
        logger.info(f'Results exported to: {args.output}')

        # Show top opportunities
        top_opportunities = results_df[results_df['consolidation_score'] >= 60].groupby('cluster_id').first()
        if not top_opportunities.empty:
            logger.info(f"\nTop consolidation opportunities:")
            for idx, row in top_opportunities.head(5).iterrows():
                logger.info(f"  - {idx}: Score {row['consolidation_score']} ({row['cluster_size']} pages)")

        return results_df

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
