####################################################################################
#                                                                                  #
#  SERP Clustering API                                                             #
#                                                                                  #
#  FastAPI service for clustering keywords based on common SERP URLs.              #
#                                                                                  #
####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                                              #
####################################################################################

"""
SERP Clustering API

A FastAPI service that clusters keywords based on common SERP URLs.
Upload a CSV file with ValueSERP export data and receive clustered results.

Usage:
    uvicorn app:app --reload

Endpoint:
    POST /cluster-serps - Upload CSV file, returns JSON clusters

Requirements:
    pip install fastapi uvicorn pandas python-multipart
"""

import pandas as pd
from collections import defaultdict
from fastapi import FastAPI, File, UploadFile
from typing import List, Dict, Any

app = FastAPI(
    title="SERP Clustering API",
    description="Cluster keywords based on common SERP URLs from ValueSERP exports",
    version="1.0.0",
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "SERP Clustering API",
        "version": "1.0.0",
        "author": "Lee Foot",
        "website": "https://www.leefoot.com",
        "endpoints": {
            "/cluster-serps": "POST - Upload ValueSERP CSV for clustering"
        }
    }


@app.post("/cluster-serps")
async def cluster_serps(
    file: UploadFile = File(...),
    common_urls: int = 4
) -> Dict[str, Any]:
    """
    Cluster keywords based on common SERP URLs.

    Args:
        file: CSV file with ValueSERP export columns:
              - search.q (query)
              - result.organic_results.link (ranking URL)
        common_urls: Minimum number of common URLs to form a cluster (default: 4)

    Returns:
        JSON object with clustered keywords
    """
    # Load the CSV data from the uploaded file
    df = pd.read_csv(
        file.file,
        usecols=["search.q", "result.organic_results.link"],
        dtype=str
    )

    # Clean and deduplicate
    df = df.drop_duplicates(subset=["search.q", "result.organic_results.link"])
    df["search.q"] = df["search.q"].str.lower()

    # Build link and query maps
    link_map = defaultdict(set)
    query_map = defaultdict(set)
    for query, link in df.values:
        link_map[link].add(query)
        query_map[query].add(link)

    # Find query pairs with common URLs
    result = []
    processed_queries = set()

    for query1, query_links1 in query_map.items():
        if query1 in processed_queries:
            continue

        for query2, query_links2 in query_map.items():
            if query1 >= query2 or query2 in processed_queries:
                continue

            common_links = query_links1.intersection(query_links2)

            if len(common_links) >= common_urls:
                result.append((
                    query1,
                    query2,
                    len(common_links),
                    list(common_links)
                ))
                processed_queries.add(query1)
                processed_queries.add(query2)

    # Build result DataFrame
    df_result = pd.DataFrame(
        result,
        columns=["query", "similar_query", "common_urls_count", "common_urls"]
    )

    if df_result.empty:
        return {
            "serp_clusters": [],
            "message": "No clusters found with the specified common_urls threshold"
        }

    # Sort and group
    df_result = df_result.sort_values(
        by=["query", "common_urls_count"],
        ascending=[True, False]
    )
    df_result["cluster"] = (df_result["query"] != df_result["query"].shift()).cumsum()

    # Aggregate clusters
    df_result = df_result.groupby(["cluster", "similar_query"]).agg({
        "query": lambda x: list(x),
        "common_urls_count": lambda x: list(x),
        "common_urls": lambda x: [y for z in x for y in z]
    }).reset_index()

    df_result = df_result.rename(columns={
        "query": "queries",
        "common_urls_count": "common_urls_counts",
        "common_urls": "common_urls"
    })

    clusters = df_result.to_dict(orient="records")

    return {
        "serp_clusters": clusters,
        "total_clusters": len(clusters),
        "common_urls_threshold": common_urls
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
