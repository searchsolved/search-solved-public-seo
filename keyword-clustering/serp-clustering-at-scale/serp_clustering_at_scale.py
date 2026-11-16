# Author: Lee Foot
# Website: https://leefoot.com

####################################################################################
#                                                                                  #
#  SERP Clustering at Scale                                                        #
#                                                                                  #
#  Clusters keywords based on common SERP URLs.                                    #
#  Supports CSV imports (SERP API etc.) and the DataForSEO SERP API (Standard    #
#  queue: send the searches now, collect the results later).                      #
#  Multiple clustering strategies with consolidation scoring.                      #
#                                                                                  #
####################################################################################
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                      #
####################################################################################

"""
SERP Clustering Script
Version: 4.0

A script to identify content consolidation opportunities based on shared URLs in
search results. Supports overlapping clusters and multiple clustering strategies.

Features:
- Multi-strategy clustering (connected components, cliques, core-based)
- Consolidation scoring (0-100) to prioritise opportunities
- Overlap detection between clusters
- Detailed cluster metrics
- SERP data from the DataForSEO SERP API (Standard queue)
- CSV import from SERP API or similar tools

Usage (CSV mode):
    1. Export SERP data from SERP API (or similar) in CSV format
    2. Place CSV files in a folder
    3. Run: python serp_clustering_at_scale.py

Usage (DataForSEO mode, two steps):
    Google now routes result links through google.com/goto. DataForSEO's Standard queue
    turns those back into real URLs, but a search can take 25 minutes or more to finish,
    so you send the searches, keep the job file, and collect the results later.

    1. Create a text file with one keyword per line
    2. Send: python serp_clustering_at_scale.py send --keywords-file keywords.txt --job job.csv
    3. About 30 minutes later, collect and cluster:
       python serp_clustering_at_scale.py collect --job job.csv --output clusters.csv
       (add --wait to keep checking every 60 seconds until every search has finished)
    Credentials via --login/--password or DATAFORSEO_LOGIN/DATAFORSEO_PASSWORD env vars.
    Cost: keywords x $0.0006 x (results per keyword / 10)

Requirements:
    pip install pandas tqdm requests
"""

import argparse
import glob
import os
import sys
import time
import logging
from collections import defaultdict
from itertools import combinations

import pandas as pd
from tqdm import tqdm

from dataforseo_queue import (
    QueueError,
    collect_results,
    estimate_cost,
    is_unresolved,
    job_from_csv,
    job_to_csv,
    organic_items,
    send_searches,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------
# DataForSEO Configuration
# ----------------

LANGUAGE_CODE = "en"
JOB_FILE_PATH = "serp_clustering_job.csv"  # job file written by send
WAIT_CHECK_SECONDS = 60  # collect --wait: seconds between checks
WAIT_LIMIT_MINUTES = 90  # collect --wait: stop waiting after this long

LOCATION_CODES = {
    "United Kingdom": 2826,
    "United States": 2840,
    "Australia": 2036,
    "Canada": 2124,
    "Germany": 2276,
    "France": 2250,
    "Spain": 2724,
    "Italy": 2380,
    "Netherlands": 2528,
    "India": 2356,
}

# ----------------
# Configuration Variables (CSV mode defaults)
# ----------------

COMMON_URLS = 4  # minimum number of common URLs to match on. Default = 4
CLUSTERING_STRATEGY = 'all'  # Options: 'connected', 'cliques', 'core', 'all'
CORE_THRESHOLD = 0.7  # For core clustering: minimum connectivity percentage
FOLDER_LOCATION = os.path.join(os.getcwd(), 'serp_exports')  # folder containing exported CSV files
FILE_PREFIX = "/Batch_Results_*.csv"  # file prefix for SERP export CSVs
EXPORT_CSV_FILE_PATH = os.path.join(os.getcwd(), 'serp_cluster_results.csv')  # output file path


# ----------------
# DataForSEO Standard queue results
# ----------------

def serps_to_dataframe(ready):
    """
    Turn collected DataForSEO results ({keyword: SERP items}) into a DataFrame with
    columns 'query' and 'link', matching the format produced by CSV mode.

    Organic URLs left as unresolved google.com/goto links are skipped and counted.
    Returns (df, unresolved).
    """
    rows = []
    unresolved = 0
    for keyword, items in ready.items():
        for item in organic_items(items):
            url = item.get("url", "")
            if is_unresolved(url):
                unresolved += 1
            elif url:
                rows.append({"query": keyword, "link": url})
    return pd.DataFrame(rows, columns=["query", "link"]), unresolved


# ----------------
# Read and Clean Data (CSV mode)
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
    # Rename columns (only if CSV mode columns are present)
    if "search.q" in df.columns and "result.organic_results.link" in df.columns:
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
    """Analyse detailed metrics for a cluster."""
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
    # Base score from average shared URLs (normalise to 0-40 range)
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
# CLI Argument Parsing
# ----------------

def add_clustering_args(parser):
    """Clustering and output options, used by CSV mode and by collect."""
    parser.add_argument(
        "--common-urls",
        type=int,
        default=COMMON_URLS,
        help=f"Minimum shared URLs to cluster keywords (default: {COMMON_URLS}).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=CLUSTERING_STRATEGY,
        choices=["connected", "cliques", "core", "all"],
        help=f"Clustering strategy (default: {CLUSTERING_STRATEGY}).",
    )
    parser.add_argument(
        "--core-threshold",
        type=float,
        default=CORE_THRESHOLD,
        help=f"Core clustering connectivity threshold (default: {CORE_THRESHOLD}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=EXPORT_CSV_FILE_PATH,
        help=f"Output CSV file path (default: {EXPORT_CSV_FILE_PATH}).",
    )


def add_credential_args(parser):
    """DataForSEO credentials, used by send and collect."""
    parser.add_argument(
        "--login",
        type=str,
        default=None,
        help="DataForSEO login (or set DATAFORSEO_LOGIN env var).",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="DataForSEO password (or set DATAFORSEO_PASSWORD env var).",
    )


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SERP Clustering at Scale - Cluster keywords by shared SERP URLs.",
        epilog="CSV mode (no subcommand): reads SERP API exports from a folder. "
               "DataForSEO mode: 'send' queues the searches and writes a job file; "
               "'collect' reads the job file, fetches the finished searches and clusters them. "
               "Searches can take 25 minutes or more to finish.",
    )

    # CSV mode options
    parser.add_argument(
        "--folder",
        type=str,
        default=FOLDER_LOCATION,
        help=f"Folder containing SERP CSV exports (default: {FOLDER_LOCATION}).",
    )
    parser.add_argument(
        "--file-prefix",
        type=str,
        default=FILE_PREFIX,
        help=f"File prefix pattern for CSV files (default: {FILE_PREFIX}).",
    )
    add_clustering_args(parser)

    sub = parser.add_subparsers(dest="command")

    # Step 1: send the searches
    p_send = sub.add_parser("send", help="Queue the searches at DataForSEO and write a job file")
    p_send.add_argument(
        "--keywords-file",
        "--input",
        dest="keywords_file",
        type=str,
        required=True,
        help="Path to a text file with keywords (one per line).",
    )
    p_send.add_argument(
        "--job",
        type=str,
        default=JOB_FILE_PATH,
        help=f"Job file to write (default: {JOB_FILE_PATH}).",
    )
    p_send.add_argument(
        "--location",
        type=str,
        default="United Kingdom",
        choices=list(LOCATION_CODES.keys()),
        help="Location for SERP results (default: United Kingdom).",
    )
    p_send.add_argument(
        "--device",
        type=str,
        default="desktop",
        choices=["desktop", "mobile"],
        help="Device type for SERP results (default: desktop).",
    )
    p_send.add_argument(
        "--depth",
        type=int,
        default=10,
        help="Results to fetch per keyword, 10 to 100, billed per 10 (default: 10).",
    )
    p_send.add_argument("--yes", action="store_true", help="Skip the cost confirmation prompt.")
    add_credential_args(p_send)

    # Step 2: collect the results and cluster them
    p_collect = sub.add_parser("collect", help="Collect finished searches, cluster them and write the CSV")
    p_collect.add_argument("--job", type=str, required=True, help="Job file written by send.")
    p_collect.add_argument(
        "--wait",
        action="store_true",
        help=f"Check every {WAIT_CHECK_SECONDS}s until all searches finish (up to {WAIT_LIMIT_MINUTES} minutes).",
    )
    add_credential_args(p_collect)
    add_clustering_args(p_collect)

    return parser.parse_args(argv)


# ----------------
# Main Processing Functions
# ----------------

def cluster_and_export(df, common_urls, strategy, core_threshold, output_path, start_time):
    """Cluster a query/link DataFrame, export the CSV and log a summary."""
    # Prepare data
    df = prepare_data(df)
    logger.info(f"After preparation: {len(df)} unique query-URL pairs")

    # Create query to URL mapping
    query_map = create_query_map(df)
    logger.info(f"Processing {len(query_map)} unique queries")

    # Find consolidation clusters
    clusters, similarity_matrix = find_consolidation_clusters(
        query_map,
        common_urls,
        strategy=strategy,
        core_threshold=core_threshold,
    )

    # Create results DataFrame
    results_df = create_cluster_dataframe(clusters, query_map)

    # Export results
    export_results(results_df, output_path)

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
    logger.info(f'Results exported to: {output_path}')

    # Show top consolidation opportunities
    top_opportunities = results_df[results_df['consolidation_score'] >= 60].groupby('serp_cluster').first()
    if not top_opportunities.empty:
        logger.info("\nTop consolidation opportunities:")
        for idx, row in top_opportunities.head(5).iterrows():
            logger.info(f"  - {idx}: Score {row['consolidation_score']} ({row['#_kws_in_cluster']} keywords)")

    return results_df


def process_serps(args=None):
    """CSV mode: read SERP exports from a folder and identify consolidation opportunities."""
    start_time = time.time()

    try:
        logger.info("Mode: CSV import")
        folder = args.folder if args else FOLDER_LOCATION
        prefix = args.file_prefix if args else FILE_PREFIX

        logger.info(f"Looking for CSV files in: {folder}")
        file_paths = validate_folder_and_files(folder, prefix)

        df = read_csv_files(file_paths)
        logger.info(f"Successfully read {len(df)} rows from CSV files")

        return cluster_and_export(
            df,
            args.common_urls if args else COMMON_URLS,
            args.strategy if args else CLUSTERING_STRATEGY,
            args.core_threshold if args else CORE_THRESHOLD,
            args.output if args else EXPORT_CSV_FILE_PATH,
            start_time,
        )

    except Exception as e:
        logger.error(f"Error processing SERPs: {str(e)}")
        raise


def resolve_credentials(args):
    """DataForSEO credentials from the arguments or environment variables."""
    login = args.login or os.environ.get("DATAFORSEO_LOGIN")
    password = args.password or os.environ.get("DATAFORSEO_PASSWORD")
    if not login or not password:
        logger.error(
            "DataForSEO credentials required. Provide --login and --password "
            "or set DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD environment variables."
        )
        sys.exit(1)
    return login, password


def send(args):
    """Step 1: queue one DataForSEO search per keyword and write the job file."""
    login, password = resolve_credentials(args)

    if not 10 <= args.depth <= 100:
        logger.error("--depth must be between 10 and 100.")
        sys.exit(1)

    if not os.path.exists(args.keywords_file):
        logger.error(f"Keywords file not found: {args.keywords_file}")
        sys.exit(1)

    with open(args.keywords_file, "r", encoding="utf-8") as f:
        # Exact duplicates are sent once
        keywords = list(dict.fromkeys(line.strip() for line in f if line.strip()))

    if not keywords:
        logger.error("No keywords found in the file.")
        sys.exit(1)

    location_code = LOCATION_CODES[args.location]
    logger.info(f"Keywords: {len(keywords)}")
    logger.info(f"Location: {args.location} (code: {location_code})")
    logger.info(f"Device: {args.device}")
    logger.info(f"Results per keyword: {args.depth}")
    logger.info(f"Estimated cost: ${estimate_cost(len(keywords), args.depth):.4f}")

    if not args.yes:
        if not sys.stdin.isatty():
            logger.error("Run with --yes to confirm the cost when not running interactively.")
            sys.exit(1)
        if input("Send the searches? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Cancelled.")
            sys.exit(0)

    try:
        job_rows, warnings, cost = send_searches(
            keywords, login, password, location_code, LANGUAGE_CODE,
            device=args.device, depth=args.depth,
        )
    except QueueError as e:
        logger.error(str(e))
        sys.exit(1)

    for warning in warnings:
        logger.warning(warning)
    if not job_rows:
        logger.error("No searches were queued.")
        sys.exit(1)

    with open(args.job, "w", encoding="utf-8", newline="") as f:
        f.write(job_to_csv(job_rows))
    logger.info(f"Queued {len(job_rows)} search(es). Cost reported by DataForSEO: ${cost:.4f}")
    logger.info(f"Job file saved to: {args.job}")
    logger.info("DataForSEO can take 25 minutes or more to finish. Then run:")
    logger.info(f"  python {os.path.basename(__file__)} collect --job {args.job}")


def collect(args):
    """Step 2: fetch the finished searches in a job file, cluster them and export the CSV."""
    login, password = resolve_credentials(args)

    if not os.path.exists(args.job):
        logger.error(f"Job file not found: {args.job}")
        sys.exit(1)
    try:
        with open(args.job, "r", encoding="utf-8") as f:
            job_rows = job_from_csv(f.read())
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    start_time = time.time()
    while True:
        try:
            with tqdm(total=len(job_rows), desc="Collecting SERPs from DataForSEO") as bar:
                result = collect_results(
                    job_rows, login, password,
                    progress_callback=lambda done, total, keyword: bar.update(done - bar.n),
                )
        except QueueError as e:
            logger.error(str(e))
            sys.exit(1)
        logger.info(f"{len(result['ready'])} of {len(job_rows)} search(es) finished, "
                    f"{len(result['pending'])} still running, {len(result['errors'])} failed.")
        if not result['pending'] or not args.wait:
            break
        if time.time() - start_time > WAIT_LIMIT_MINUTES * 60:
            logger.warning(f"Stopped waiting after {WAIT_LIMIT_MINUTES} minutes. Run collect again later.")
            break
        time.sleep(WAIT_CHECK_SECONDS)

    for error in result['errors']:
        logger.warning(error)
    if result['pending']:
        logger.warning(f"{len(result['pending'])} search(es) are still running, so the clusters below leave "
                       "them out. Run collect again later for the full set.")
    if not result['ready']:
        logger.info("No finished searches to cluster yet.")
        sys.exit(0)

    df, unresolved = serps_to_dataframe(result['ready'])
    if unresolved:
        logger.warning(f"{unresolved} result(s) came back as unresolved google.com/goto links "
                       "and were left out of the clustering.")
    if df.empty:
        logger.error("No SERP URLs to cluster in the finished searches.")
        sys.exit(1)
    logger.info(f"Retrieved {len(df)} query-URL pairs for {df['query'].nunique()} keywords")

    return cluster_and_export(df, args.common_urls, args.strategy, args.core_threshold, args.output, start_time)


# ----------------
# Execute the Main Function
# ----------------

if __name__ == "__main__":
    args = parse_args()
    start = time.time()
    if args.command == "send":
        send(args)
    elif args.command == "collect":
        collect(args)
    else:
        process_serps(args)
    print(f'The script took {time.time() - start:.2f} seconds!')
