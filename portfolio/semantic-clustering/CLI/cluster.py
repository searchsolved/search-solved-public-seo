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
from nltk import ngrams
import typer

app = typer.Typer()


def create_unigram(cluster: str):
    """Create unigram from the cluster and return the most common word."""
    words = cluster.split()
    most_common_word = Counter(words).most_common(1)[0][0]
    return most_common_word


def get_model(model_name: str):
    """Create and return a SentenceTransformer model based on the given model name."""
    valid_model_names = ["all-MiniLM-L6-v2", "multi-qa-mpnet-base-dot-v1", "paraphrase-multilingual-MiniLM-L12-v2"]
    if model_name not in valid_model_names:
        raise ValueError(f"The model name {model_name} is not valid. Choose from {valid_model_names}.")

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
    chart_file_path = os.path.join(os.path.dirname(output_path), f"{chart_type}.html")
    pio.write_html(fig, chart_file_path)


@app.command()
def main(
    file_path: str = typer.Option("/python_scripts/urls_with_images.csv", help="The path to your input CSV file."),
    column_name: str = typer.Option("Address", help="The name of the column to be processed."),
    output_path: str = typer.Option("/python_scripts/your_keywords_clustered.csv", help="The path to the output CSV file."),
    chart_type: str = typer.Option("treemap", help="The type of chart to generate. Possible values are 'sunburst' and 'treemap'."),
    device: str = typer.Option("cpu", help="The device to be used by the SentenceTransformer. Possible values are 'cpu' and 'cuda'."),
    model_name: str = typer.Option("all-MiniLM-L6-v2", help="The name of the SentenceTransformer model to use."),
    min_similarity: float = typer.Option(0.85, help="The minimum similarity for clustering."),
    remove_dupes: bool = typer.Option(True, help="Whether to remove duplicates from the dataset."),
):
    """
    A command line application for keyword clustering.

    file_path: Path to your CSV file.
    column_name: Name of the column in your CSV to be processed.
    output_path: Path where the output CSV will be saved.
    chart_type: Type of chart to generate. 'sunburst' or 'treemap'.
    device: Device to be used by SentenceTransformer. 'cpu' or 'cuda'.
    model_name: Name of the SentenceTransformer model to use.
    min_similarity: Minimum similarity for clustering.
    remove_dupes: Whether to remove duplicates from the dataset.
    """
    try:
        model = get_model(model_name)
    except ValueError as e:
        print(e)
        return

    try:
        df = load_file(file_path)
    except FileNotFoundError as e:
        print(e)
        return

    if column_name not in df.columns:
        print(f"The column name {column_name} is not in the DataFrame.")
        return

    df.rename(columns={column_name: 'keyword', "spoke": "spoke Old"}, inplace=True)

    if remove_dupes:
        df.drop_duplicates(subset='keyword', inplace=True)

    df = df[df["keyword"].notna()]
    df['keyword'] = df['keyword'].astype(str)
    from_list = df['keyword'].to_list()

    embedding_model = SentenceTransformer(model_name, device=device)
    distance_model = SentenceEmbeddings(embedding_model)

    startTime = time.time()
    print("[bold green]Starting to cluster keywords...[/bold green]")

    model = PolyFuzz(distance_model)
    model = model.fit(from_list)
    model.group(link_min_similarity=min_similarity)

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

    create_chart(df, chart_type)

    df.drop(columns=['cluster_size', 'keyword_len'], inplace=True)

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    app()
