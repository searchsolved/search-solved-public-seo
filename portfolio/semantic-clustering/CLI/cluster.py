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

COMMON_COLUMN_NAMES = [
    "Keyword", "Keywords", "keyword", "keywords",
    "Search Terms", "Search terms", "Search term", "Search Term"
]


def create_unigram(cluster: str):
    """Create unigram from the cluster and return the most common word."""
    words = cluster.split()
    most_common_word = Counter(words).most_common(1)[0][0]
    return most_common_word


def get_model(model_name: str):
    """Create and return a SentenceTransformer model based on the given model name."""
    print(f"[bold green]Loading the SentenceTransformer model '{model_name}'...[/bold green]")
    model = SentenceTransformer(model_name)
    print("[bold green]Model loaded.[/bold green]")
    return model


def load_file(file_path: str):
    """Load a CSV file and return a DataFrame."""
    print(f"[bold green]Loading the CSV file from '{file_path}'...[/bold green]")
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


def create_chart(df, chart_type, output_path):
    """Create a sunburst chart or a treemap."""
    if chart_type == "sunburst":
        fig = px.sunburst(df, path=['hub', 'spoke'], values='cluster_size',
                          color_discrete_sequence=px.colors.qualitative.Pastel2)
    elif chart_type == "treemap":
        fig = px.treemap(df, path=['hub', 'spoke'], values='cluster_size',
                         color_discrete_sequence=px.colors.qualitative.Pastel2)
    else:
        print(f"[bold red]Invalid chart type: {chart_type}. Valid options are 'sunburst' and 'treemap'.[/bold red]")
        return

    fig.show()

    # Save the chart in the same directory as the final CSV.
    chart_file_path = os.path.join(os.path.dirname(output_path), f"{chart_type}.html")
    pio.write_html(fig, chart_file_path)


@app.command()
def main(
        file_path: str = typer.Argument(..., help='Path to your CSV file.'),
        column_name: str = typer.Option(None, help='Name of the column in your CSV to be processed.'),
        output_path: str = typer.Option(None, help='Path where the output CSV will be saved.'),
        chart_type: str = typer.Option("treemap", help="Type of chart to generate. 'sunburst' or 'treemap'."),
        device: str = typer.Option("cpu", help="Device to be used by SentenceTransformer. 'cpu' or 'cuda'."),
        model_name: str = typer.Option("all-MiniLM-L6-v2",
                                       help="Name of the SentenceTransformer model to use. For available models, refer to https://www.sbert.net/docs/pretrained_models.html"),
        min_similarity: float = typer.Option(0.85, help="Minimum similarity for clustering."),
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
    if device not in ["cpu", "cuda"]:
        print("[bold red]Invalid device. Valid options are 'cpu' and 'cuda'.[/bold red]")
        return

    try:
        model = get_model(model_name)
    except Exception as e:
        print(f"[bold red]Failed to load the SentenceTransformer model: {e}[/bold red]")
        return

    try:
        df = load_file(file_path)
    except FileNotFoundError as e:
        print(f"[bold red]The file {file_path} does not exist.[/bold red]")
        return

    if column_name is None:
        print("[bold green]Searching for a column from the list of default column names...[/bold green]")
        for common_name in COMMON_COLUMN_NAMES:
            if common_name in df.columns:
                column_name = common_name
                print(f"[bold green]Found column '{column_name}'.[/bold green]")
                break
        else:
            print(f"[bold red]Could not find a suitable column for processing. Please specify the column name with the --column option.[/bold red]")
            return

    if column_name not in df.columns:
        print(f"[bold red]The column name {column_name} is not in the DataFrame.[/bold red]")
        return

    print(f"[bold green]Using the following options:[/bold green]\n"
          f"File path: {file_path}\n"
          f"Column name: {column_name}\n"
          f"Output path: {output_path}\n"
          f"Chart type: {chart_type}\n"
          f"Device: {device}\n"
          f"SentenceTransformer model: {model_name}\n"
          f"Minimum similarity: {min_similarity}\n"
          f"Remove duplicates: {remove_dupes}\n")

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

    if output_path is None:
        output_path = os.path.splitext(file_path)[0] + '_output.csv'

    create_chart(df, chart_type, output_path)

    df.drop(columns=['cluster_size', 'keyword_len'], inplace=True)

    df.to_csv(output_path, index=False)

    print(f"[bold green]Results saved to '{output_path}'.[/bold green]")


if __name__ == "__main__":
    app()
