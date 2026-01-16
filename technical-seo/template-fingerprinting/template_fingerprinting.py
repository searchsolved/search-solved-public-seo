####################################################################################
#                                                                                  #
#  Template Fingerprinting Tool                                                    #
#                                                                                  #
#  Automatically classify pages into template types using HTML structure analysis. #
#                                                                                  #
####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://leefoot.com                                               #
# Contact  : https://leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                                              #
####################################################################################

"""
Template Fingerprinting Tool

Analyzes the HTML structure of pages to automatically identify and group pages
by template type. Uses TF-IDF vectorization and K-Means clustering to identify
common page templates based on:
- HTML tag structure and counts
- CSS class names
- ID attributes
- Meta tags

Usage:
    1. Export URLs from Screaming Frog (internal_html.csv)
    2. Update INPUT_FILE path
    3. Run the script

Requirements:
    pip install pandas requests beautifulsoup4 scikit-learn tqdm
"""

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from tqdm import tqdm

# Configuration
INPUT_FILE = os.path.join(os.getcwd(), 'urls.csv')  # CSV with 'Address' column
OUTPUT_FILE = os.path.join(os.getcwd(), 'classified_urls.csv')
N_CLUSTERS = 5  # Number of template types to identify
TIMEOUT = 10  # Request timeout in seconds


def fetch_html(url):
    """
    Fetches HTML content from a URL.

    Args:
        url (str): The URL to fetch

    Returns:
        str or None: HTML content or None if request fails
    """
    try:
        response = requests.get(url, timeout=TIMEOUT)
        return response.text
    except:
        return None


def extract_features(html):
    """
    Extracts structural features from HTML for fingerprinting.

    Features include:
    - Tag counts (div:15, span:8, etc.)
    - CSS class names
    - ID attributes
    - Meta tag names/properties

    Args:
        html (str): Raw HTML content

    Returns:
        str: Space-separated feature string for vectorization
    """
    if html is None:
        return ""

    soup = BeautifulSoup(html, 'html.parser')

    features = []

    # Extract tag counts
    tag_counts = Counter(tag.name for tag in soup.find_all())
    features.extend([f"{tag}:{count}" for tag, count in tag_counts.items()])

    # Extract class names
    class_counts = Counter(cls for tag in soup.find_all() for cls in tag.get('class', []))
    features.extend([f"class:{cls}" for cls in class_counts])

    # Extract id attributes
    id_counts = Counter(tag.get('id') for tag in soup.find_all() if tag.get('id'))
    features.extend([f"id:{id}" for id in id_counts])

    # Extract meta tags
    meta_tags = soup.find_all('meta')
    features.extend([f"meta:{tag.get('name', tag.get('property', ''))}" for tag in meta_tags])

    return " ".join(features)


def classify_pages(input_file, output_file, n_clusters=5):
    """
    Main function to classify pages by template type.

    Args:
        input_file (str): Path to CSV file with 'Address' column containing URLs
        output_file (str): Path for output CSV with classifications
        n_clusters (int): Number of template types to identify

    Returns:
        pandas.DataFrame: DataFrame with URL classifications
    """
    print(f"Starting classification process...")

    # Read URLs from CSV
    print(f"Reading URLs from {input_file}...")
    df = pd.read_csv(input_file)
    urls = df['Address'].tolist()
    print(f"Found {len(urls)} URLs to process.")

    # Extract features
    print("Extracting features...")
    features = []
    for url in tqdm(urls, desc="Processing URLs"):
        html = fetch_html(url)
        features.append(extract_features(html))

    # Vectorize features
    print("Vectorizing features...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(features)

    # Perform clustering
    print(f"Performing clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Add cluster labels to the dataframe
    df['Cluster'] = clusters
    df['Page Type'] = [f"Type {i}" for i in clusters]

    # Print cluster information
    print("\nCluster information:")
    for i in range(n_clusters):
        cluster_features = vectorizer.inverse_transform(X[clusters == i])
        top_features = Counter([feature for page in cluster_features for feature in page]).most_common(5)
        print(f"Cluster {i}: {top_features}")

    # Save results to CSV
    print(f"Saving results to {output_file}...")
    df.to_csv(output_file, index=False)

    print(f"Classification complete. Results saved to {output_file}")
    print("\nCluster distribution:")
    print(df['Page Type'].value_counts())

    return df


if __name__ == "__main__":
    result_df = classify_pages(INPUT_FILE, OUTPUT_FILE, n_clusters=N_CLUSTERS)
