# Author: Lee Foot
# Website: https://leefoot.com

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
Upload a CSV with one row per keyword and ranking URL (for example a
DataForSEO SERP export) and receive clustered results.

Usage:
    uvicorn app:app --reload

Endpoint:
    POST /cluster-serps - Upload CSV file, returns JSON clusters
        file           CSV upload (required)
        keyword_column form field, name of the keyword column (default: keyword)
        url_column     form field, name of the ranking URL column (default: url)
        common_urls    query parameter, minimum shared URLs to form a cluster (default: 4)

Requirements:
    pip install fastapi uvicorn pandas python-multipart
"""

import pandas as pd
from collections import defaultdict
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from typing import List, Dict, Any

app = FastAPI(
    title="SERP Clustering API",
    description="Cluster keywords based on common SERP URLs from a keyword and URL CSV export",
    version="1.1.0",
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "SERP Clustering API",
        "version": "1.1.0",
        "author": "Lee Foot",
        "website": "https://www.leefoot.com",
        "endpoints": {
            "/cluster-serps": "POST - Upload a keyword and URL CSV for clustering"
        }
    }


def cluster_keywords(df: pd.DataFrame, common_urls: int = 4) -> Dict[str, Any]:
    """
    Cluster keywords that share ranking URLs.

    Args:
        df: DataFrame with exactly two columns, keyword first and ranking URL second
        common_urls: Minimum number of common URLs to form a cluster

    Returns:
        JSON-ready dict with clustered keywords
    """
    query_col, link_col = df.columns[0], df.columns[1]

    # Clean and deduplicate
    df = df.dropna(subset=[query_col, link_col])
    df = df.drop_duplicates(subset=[query_col, link_col])
    df[query_col] = df[query_col].str.lower()

    # Build link and query maps
    link_map = defaultdict(set)
    query_map = defaultdict(set)
    for query, link in df[[query_col, link_col]].values:
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


@app.post("/cluster-serps")
async def cluster_serps(
    file: UploadFile = File(...),
    keyword_column: str = Form("keyword"),
    url_column: str = Form("url"),
    common_urls: int = 4
) -> Dict[str, Any]:
    """
    Cluster keywords based on common SERP URLs.

    Args:
        file: CSV file with one row per keyword and ranking URL
        keyword_column: Name of the keyword column (form field, default: keyword)
        url_column: Name of the ranking URL column (form field, default: url)
        common_urls: Minimum number of common URLs to form a cluster (default: 4)

    Returns:
        JSON object with clustered keywords
    """
    if keyword_column == url_column:
        raise HTTPException(status_code=400, detail="keyword_column and url_column must be different columns.")

    # Load the CSV data from the uploaded file
    try:
        df = pd.read_csv(file.file, dtype=str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read the CSV file: {e}")

    missing = [col for col in (keyword_column, url_column) if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(f"Column(s) not found: {', '.join(missing)}. "
                    f"Columns in the file: {', '.join(map(str, df.columns))}. "
                    "Set keyword_column and url_column to match your file.")
        )

    return cluster_keywords(df[[keyword_column, url_column]].copy(), common_urls)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
