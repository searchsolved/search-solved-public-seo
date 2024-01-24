"""
This script clusters keywords using the PolyFuzz library, which leverages the SentenceTransformers library for semantic similarity. The script reads a CSV file, clusters keywords based on their semantic similarity, and outputs a sunburst chart using Plotly and a CSV file with the results.

The script accepts several parameters:
- FILE_PATH: The path to the input CSV file. The file should be in UTF-8 or UTF-16 encoding.
- COLUMN_NAME: The name of the column in the CSV file that contains the keywords to be clustered.
- OUTPUT_PATH: The path where the output CSV file will be saved.
- DEVICE: The device to be used by the SentenceTransformer model. This can be 'cpu' or 'cuda' if a GPU is available.
- MODEL_NAME: The name of the SentenceTransformer model to be used.
- MIN_SIMILARITY: The minimum similarity for two keywords to be considered in the same cluster.
- REMOVE_DUPES: Whether to remove duplicate keywords.

To use the script, make sure the required libraries are installed (`pip install sentence_transformers polyfuzz nltk plotly pandas chardet`), adjust the parameters as needed, and run the script. The progress of the script will be printed to the console.
"""

import time
import chardet
import pandas as pd
import os
from collections import Counter
from sentence_transformers import SentenceTransformer
from polyfuzz import PolyFuzz
from polyfuzz.models import SentenceEmbeddings
import plotly.express as px
import plotly.io as pio
from rich import print

# File paths and column name specified outside of the function
FILE_PATH = "/python_scripts/waw_keywords.csv"
COLUMN_NAME = "Keyword"
OUTPUT_PATH = "/python_scripts/your_keywords_clustered.csv"
CHART_TYPE = "treemap"  # change this to "treemap" for a treemap
DEVICE = "cpu"

# MODEL_NAME = "all-MiniLM-L6-v2"  # best balance between speed and semantic similarity
MODEL_NAME = "paraphrase-MiniLM-L3-v2"  # fastest, lowest semantic matching
# MODEL_NAME = "paraphrase-MiniLM-L3-v2"  # slowest, highest semantic matching score


MIN_SIMILARITY = 0.85
REMOVE_DUPES = True


def create_unigram(cluster: str):
    """Create unigram from the cluster and return the most common word."""
    words = cluster.split()
    most_common_word = Counter(words).most_common(1)[0][0]
    return most_common_word


def get_model(model_name: str):
    """Create and return a SentenceTransformer model based on the given model name."""

    print("[bold green]Loading the SentenceTransformer model...[/bold green]")
    model = SentenceTransformer(model_name)
    print("[bold green]Model loaded.[/bold green]")
    return model


def load_file(file_path: str):
    """Load a CSV file and return a DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    print("[bold green]Loading the CSV file...[/bold green]")
    result = chardet.detect(open(file_path, 'rb').read())
    encoding_value = result["encoding"]
    white_space = False if encoding_value != "UTF-16" else True

    df = pd.read_csv(
        file_path,
        encoding=encoding_value,
        delim_whitespace=white_space,
        on_bad_lines='skip',
    )
    print("[bold green]CSV file loaded.[/bold green]")
    return df


def create_chart(df, chart_type):
    """Create a sunburst chart or a treemap."""
    if chart_type == "sunburst":
        fig = px.sunburst(df, path=['hub', 'spoke'], values='cluster_size',
                          color_discrete_sequence=px.colors.qualitative.Pastel2)
    elif chart_type == "treemap":
        fig = px.treemap(df, path=['hub', 'spoke'], values='cluster_size',
                         color_discrete_sequence=px.colors.qualitative.Pastel2)
    else:
        print(f"Invalid chart type: {chart_type}. Valid options are 'sunburst' and 'treemap'.")
        return

    fig.show()

    # Save the chart in the same directory as the final CSV.
    chart_file_path = os.path.join(os.path.dirname(OUTPUT_PATH), f"{chart_type}.html")
    pio.write_html(fig, chart_file_path)


def main():
    """Main function to run the keyword clustering."""
    try:
        model = get_model(MODEL_NAME)
    except ValueError as e:
        print(e)
        return

    try:
        df = load_file(FILE_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    if COLUMN_NAME not in df.columns:
        print(f"The column name {COLUMN_NAME} is not in the DataFrame.")
        return

    df.rename(columns={COLUMN_NAME: 'keyword', "spoke": "spoke Old"}, inplace=True)

    if REMOVE_DUPES:
        df.drop_duplicates(subset='keyword', inplace=True)

    df = df[df["keyword"].notna()]
    df['keyword'] = df['keyword'].astype(str)
    from_list = df['keyword'].to_list()

    embedding_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    distance_model = SentenceEmbeddings(embedding_model)

    startTime = time.time()
    print("[bold green]Starting to cluster keywords...[/bold green]")

    model = PolyFuzz(distance_model)
    model = model.fit(from_list)
    model.group(link_min_similarity=MIN_SIMILARITY)

    df_cluster = model.get_matches()
    df_cluster.rename(columns={"From": "keyword", "Similarity": "similarity", "Group": "spoke"}, inplace=True)
    df = pd.merge(df, df_cluster[['keyword', 'spoke']], on='keyword', how='left')

    df['cluster_size'] = df['spoke'].map(df.groupby('spoke')['spoke'].count())
    df.loc[df["cluster_size"] == 1, "spoke"] = "no_cluster"
    df.insert(0, 'spoke', df.pop('spoke'))
    df['spoke'] = df['spoke'].str.encode('ascii', 'ignore').str.decode('ascii')

    df['keyword_len'] = df['keyword'].astype(str).apply(len)
    df = df.sort_values(by="keyword_len", ascending=True)

    df.insert(0, 'hub', df['spoke'].apply(create_unigram))

    df = df[
        ['hub', 'spoke', 'cluster_size'] + [col for col in df.columns if col not in ['hub', 'spoke', 'cluster_size']]]

    df.sort_values(["spoke", "cluster_size"], ascending=[True, False], inplace=True)

    df['spoke'] = (df['spoke'].str.split()).str.join(' ')

    print(f"All keywords clustered successfully. Took {time.time() - startTime} seconds!")

    create_chart(df, CHART_TYPE)

    df.drop(columns=['cluster_size', 'keyword_len'], inplace=True)

    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
