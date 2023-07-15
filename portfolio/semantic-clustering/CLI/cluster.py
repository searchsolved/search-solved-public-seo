#!/usr/bin/env python3
import os
import platform
import string
import time
from collections import Counter

import chardet
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import typer
import win32com.client as win32
from polyfuzz import PolyFuzz
from polyfuzz.models import SentenceEmbeddings
from rich import print
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from sentence_transformers import SentenceTransformer

win32c = win32.constants

app = typer.Typer()

live = Live(auto_refresh=False)  # Initialize Live with auto_refresh set to False
live.start()  # Start the Live context manager

startTime = time.time()  # start timing the script

COMMON_COLUMN_NAMES = [
    "Keyword", "Keywords", "keyword", "keywords",
    "Search Terms", "Search terms", "Search term", "Search Term"
]

def print_messages(message):
    panel = Panel.fit(message, title="[b]Clustering Progess[/b]", style="cyan", border_style="black")
    live.update(panel)
    live.refresh()  # Manually refresh the Live display

def stem_and_remove_punctuation(text: str, stem: bool):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Stem the text if the stem flag is True
    if stem:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

def create_unigram(cluster: str, stem: bool):
    """Create unigram from the cluster and return the most common word."""
    words = cluster.split()
    word_counts = Counter(words)

    # Filter out number-only words
    word_counts = Counter({word: count for word, count in word_counts.items() if not word.isdigit()})

    if word_counts:
        # If there are any words left after filtering, return the most common one
        most_common_word = word_counts.most_common(1)[0][0]
    else:
        # If all words were number-only and thus filtered out, return 'no_keyword'
        most_common_word = 'no_keyword'

    return stem_and_remove_punctuation(most_common_word, stem)

def get_model(model_name: str):
    """Create and return a SentenceTransformer model based on the given model name."""
    model = SentenceTransformer(model_name)
    return model

def load_file(file_path: str):
    """Load a CSV file and return a DataFrame."""
    result = chardet.detect(open(file_path, 'rb').read())
    encoding_value = result["encoding"]
    white_space = False if encoding_value != "UTF-16" else True

    df = pd.read_csv(
        file_path,
        encoding=encoding_value,
        delim_whitespace=white_space,
        on_bad_lines='skip',
    )
    return df

def create_chart(df, chart_type, output_path, volume):
    """Create a sunburst chart or a treemap."""
    if volume is not None:
        chart_df = df.groupby(['hub', 'spoke'])[volume].sum().reset_index(name='cluster_size')
    else:
        chart_df = df.groupby(['hub', 'spoke']).size().reset_index(name='cluster_size')

    if chart_type == "sunburst":
        fig = px.sunburst(chart_df, path=['hub', 'spoke'], values='cluster_size',
                          color_discrete_sequence=px.colors.qualitative.Pastel2)
    elif chart_type == "treemap":
        fig = px.treemap(chart_df, path=['hub', 'spoke'], values='cluster_size',
                         color_discrete_sequence=px.colors.qualitative.Pastel2)
    else:
        print(f"[bold magenta]Invalid chart type: {chart_type}. Valid options are 'sunburst' and 'treemap'.[/bold magenta]")
        return

    fig.show()

    # Save the chart in the same directory as the final CSV.
    chart_file_path = os.path.join(os.path.dirname(output_path), f"{chart_type}.html")
    pio.write_html(fig, chart_file_path)

    # Create a message panel for the file saved location
    file_saved_message = f"[bold]Chart saved to:[/bold] [magenta]{chart_file_path}[/magenta]"

    # Print the file saved location message in the same box as other messages
    print_messages(file_saved_message)

    # Print message when saving the CSV
    csv_saved_message = f"[bold]CSV saved to:[/bold] [magenta]{output_path}[/magenta]"
    print_messages(csv_saved_message)


@app.command()
def main(
        chart_type: str = typer.Option("treemap", help="Type of chart to generate. 'sunburst' or 'treemap'."),
        column_name: str = typer.Option(None, help='Name of the column in your CSV to be processed.'),
        device: str = typer.Option("cpu", help="Device to be used by SentenceTransformer. 'cpu' or 'cuda'."),
        excel_pivot: bool = typer.Option(False, help="Whether to save the output as an Excel pivot table."),
        file_path: str = typer.Argument(..., help='Path to your CSV file.'),
        min_similarity: float = typer.Option(0.80, help="Minimum similarity for clustering."),
        model_name: str = typer.Option("all-MiniLM-L6-v2",
                                       help="Name of the SentenceTransformer model to use. For available models, refer to https://www.sbert.net/docs/pretrained_models.html"),
        output_path: str = typer.Option(None, help='Path where the output CSV will be saved.'),
        remove_dupes: bool = typer.Option(True, help="Whether to remove duplicates from the dataset."),
        stem: bool = typer.Option(False, "--stem", help="Whether to perform stemming on the 'hub' column.", show_default=False),
        volume: str = typer.Option(None, help='Name of the column containing numerical values. If --volume is used, the keyword with the largest volume will be used as the name of the cluster. If not, the shortest word will be used.')
):
    # Clear the screen
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')

    # Print welcome message
    console = Console()
    welcome_message = "[bold cyan]Keyword Clustering CLI Tool to find Semantic Relationships Between Keywords[/bold cyan]"
    panel = Panel(welcome_message, style="bold magenta", border_style="black",
                  title="[b]SBERT Clustering - V1.0 - www.LeeFoot.co.uk[/b]")
    console.print(panel)

    if device not in ["cpu", "cuda"]:
        print("[bold magenta]Invalid device. Valid options are 'cpu' and 'cuda'.[/bold magenta]")
        return

    try:
        model = get_model(model_name)
    except Exception as e:
        print(f"[bold magenta]Failed to load the SentenceTransformer model: {e}[/bold magenta]")
        return

    try:
        df = load_file(file_path)
    except FileNotFoundError as e:
        print(f"[bold magenta]The file {file_path} does not exist.[/bold magenta]")
        return

    if column_name is None:
        for common_name in COMMON_COLUMN_NAMES:
            if common_name in df.columns:
                column_name = common_name
                break
        else:
            print(f"[bold magenta]Could not find a suitable column for processing. Please specify the column name with the --column option.[/bold magenta]")
            return

    if column_name not in df.columns:
        print(f"[bold magenta]The column name {column_name} is not in the DataFrame.[/bold magenta]")
        return

    if volume is not None and volume not in df.columns:
        print(f"[bold magenta]The column name {volume} is not in the DataFrame.[/bold magenta]")
        return

        # Print options
    options_message = (
        f"[cyan]File path:[/cyan] [bold magenta]{file_path}[/bold magenta]\n"
        f"[cyan]Column name:[/cyan] [bold magenta]{column_name}[/bold magenta]\n"
        f"[cyan]Output path:[/cyan] [bold magenta]{output_path}[/bold magenta]\n"
        f"[cyan]Chart type:[/cyan] [bold magenta]{chart_type}[/bold magenta]\n"
        f"[cyan]Device:[/cyan] [bold magenta]{device}[/bold magenta]\n"
        f"[cyan]SentenceTransformer model:[/cyan] [bold magenta]{model_name}[/bold magenta]\n"
        f"[cyan]Minimum similarity:[/cyan] [bold magenta]{min_similarity}[/bold magenta]\n"
        f"[cyan]Remove duplicates:[/cyan] [bold magenta]{remove_dupes}[/bold magenta]\n"
        f"[cyan]Excel Pivot:[/cyan] [bold magenta]{excel_pivot}[/bold magenta]\n"
        f"[cyan]Volume column:[/cyan] [bold magenta]{volume}[/bold magenta]\n"
        f"[cyan]Stemming enabled:[/cyan] [bold magenta]{stem}[/bold magenta]"
    )
    panel = Panel.fit(options_message, title="[b]Using The Following Options[/b]", style="magenta", border_style="black")
    console.print(panel)

    df.rename(columns={column_name: 'keyword', "spoke": "spoke Old"}, inplace=True)

    if remove_dupes:
        df.drop_duplicates(subset='keyword', inplace=True)

    df = df[df["keyword"].notna()]
    df['keyword'] = df['keyword'].astype(str)
    from_list = df['keyword'].to_list()

    embedding_model = SentenceTransformer(model_name, device=device)
    distance_model = SentenceEmbeddings(embedding_model)

    # clustering started message
    message = "Clustering keywords, this can take a while!"
    print_messages(message)


    model = PolyFuzz(distance_model)
    model = model.fit(from_list)
    model.group(link_min_similarity=min_similarity)

    df_cluster = model.get_matches()
    df_cluster["Group"] = df_cluster.apply(lambda row: "no_cluster" if row["Similarity"] < min_similarity else row["Group"], axis=1)

    # this logic moves exact matches back into the right group. Sometimes they can stray when they have an identical
    # match score with a different group. for example 2 socks vs two socks with score the same.

    data_dict = df_cluster.groupby('Group')['From'].apply(list).to_dict()

    # Check for each Group if it exists in its corresponding From values
    for group, from_values in data_dict.items():
        if group not in from_values:
            # If it doesn't exist, add it to the From values of that Group
            from_values.append(group)

    # Convert the updated dictionary back to a DataFrame
    df_missing = pd.DataFrame([(k, v) for k, vs in data_dict.items() for v in vs], columns=['Group', 'From'])

    df_missing = pd.concat([df_cluster, df_missing], ignore_index=True)
    df_missing['Match'] = df_missing['From'] == df_missing['Group']
    df_missing = df_missing[df_missing["Match"].isin([True])]
    df_missing = df_missing.drop('Match', axis=1)
    df_missing.drop_duplicates(subset=["Group", "From"], keep="first", inplace=True)
    df_missing['Similarity'] = 1
    df_cluster = pd.concat([df_cluster, df_missing])
    df_cluster = df_cluster.sort_values(by="Similarity", ascending=False)
    df_cluster = df_cluster[df_cluster.duplicated(subset=['From'], keep='first') == False]  # drop first duplicate only

    df_cluster.rename(columns={"From": "keyword", "Similarity": "similarity", "Group": "spoke"}, inplace=True)
    df = pd.merge(df, df_cluster[['keyword', 'spoke']], on='keyword', how='left')

    df['cluster_size'] = df['spoke'].map(df.groupby('spoke')['spoke'].count())
    df.loc[df["cluster_size"] == 1, "spoke"] = "no_cluster"
    df.insert(0, 'spoke', df.pop('spoke'))
    df['keyword_len'] = df['keyword'].astype(str).apply(len)
    if volume is not None:
        df[volume] = df[volume].astype(str).replace({'': '0', 'nan': '0'}).str.replace('\D', '', regex=True).astype(int)

        df = df.sort_values(by=volume, ascending=False)
    else:
        df = df.sort_values(by="keyword_len", ascending=True)

    df.insert(0, 'hub', df['spoke'].apply(lambda x: create_unigram(x, stem)))

    df['hub'] = df['hub'].apply(lambda x: stem_and_remove_punctuation(x, stem))

    df = df[
        ['hub', 'spoke', 'cluster_size'] + [col for col in df.columns if col not in ['hub', 'spoke', 'cluster_size']]]

    # If volume is used, sort by volume. Otherwise, sort by keyword length.
    if volume is not None:
        df[volume] = df[volume].replace({'': 0, np.nan: 0}).astype(int)
        df = df.sort_values(by=volume, ascending=False)
    else:
        df['keyword_len'] = df['keyword'].astype(str).apply(len)
        df = df.sort_values(by="keyword_len", ascending=True)

    # Use the first keyword in each sorted group as the cluster name.
    df['spoke'] = df.groupby('spoke')['keyword'].transform('first')

    df.sort_values(["spoke", "cluster_size"], ascending=[True, False], inplace=True)

    df['spoke'] = (df['spoke'].str.split()).str.join(' ')

    message += f"\nAll keywords clustered successfully. Took {round(time.time() - startTime, 2)} seconds!"
    print_messages(message)


    if output_path is None:
        output_path = os.path.splitext(file_path)[0] + '_output.csv'

    # drop unused columns
    df.drop(columns=['cluster_size', 'keyword_len'], inplace=True)

    # logic to fix no_cluster hub and spoke names
    df["hub"] = df["hub"].apply(lambda x: "no_cluster" if x == "noclust" else x)
    df["hub"] = df["hub"].apply(lambda x: "no_cluster" if x == "nocluster" else x)
    df.loc[df["hub"] == "no_cluster", "spoke"] = "no_cluster"

    create_chart(df, chart_type, output_path, volume)

    output_dir = os.getcwd()
    output_path = os.path.join(output_dir, output_path+ '_output.xlsx')

    if excel_pivot:
        try:
            # Save the DataFrame to an Excel file
            df.to_excel(output_path, index=False)

            # Start an instance of Excel
            excel = win32.gencache.EnsureDispatch('Excel.Application')
            excel.Visible = False

            # Open the workbook in Excel
            wb = excel.Workbooks.Open(output_path)

            # Get the active worksheet and rename it
            ws1 = wb.ActiveSheet
            ws1.Name = "Clustered Keywords"

            # Freeze the top row in ws1
            ws1.Activate()
            ws1.Range("A2").Select()
            ws1.Application.ActiveWindow.FreezePanes = True

            # Create a new worksheet for the pivot table
            ws2 = wb.Sheets.Add()
            ws2.Name = "PivotTable"

            # Freeze the top row in ws2
            ws2.Activate()
            ws2.Range("A2").Select()
            ws2.Application.ActiveWindow.FreezePanes = True

            # Set up the pivot table source data
            source_range = ws1.UsedRange

            # Get the range address
            source_range_address = source_range.GetAddress()

            # Create the pivot cache
            pc = wb.PivotCaches().Create(SourceType=win32c.xlDatabase, SourceData=ws1.UsedRange)

            # Define the pivot table range
            pivot_range = ws2.Range(f"A1:C{df.shape[0] + 1}")

            # Create the pivot table
            pivot_table = pc.CreatePivotTable(TableDestination=pivot_range, TableName="PivotTable")

            # Set up the row fields
            pivot_table.PivotFields("hub").Orientation = win32c.xlRowField
            pivot_table.PivotFields("hub").Position = 1
            pivot_table.PivotFields("spoke").Orientation = win32c.xlRowField
            pivot_table.PivotFields("spoke").Position = 2
            pivot_table.PivotFields("keyword").Orientation = win32c.xlRowField

            if volume is not None:
                pivot_table.PivotFields(volume).Orientation = win32c.xlDataField

                # Save and close
            wb.Save()
            excel.Application.Quit()

            message += f"\nResults saved to '{output_path}'."
        except Exception as e:
            print(
                f"[bold magenta]Failed to create an Excel pivot table: {e}. Creating a pandas pivot table instead.[/bold magenta]")
            pandas_pivot = True  # set pandas_pivot to True as a fallback

    else:
        df_indexed = df.set_index(['hub', 'spoke', 'keyword'])
        # Save the DataFrame to an Excel file
        with pd.ExcelWriter(output_path) as writer:
            df_indexed.to_excel(writer, sheet_name='PivotTable')
            df.to_excel(writer, sheet_name='Clustered Keywords', index=False)

        message += f"\nResults saved to '{output_path}'."

    # message += f"\nResults saved to '{output_path}'."
    print_messages(message)

    live.stop()  # Stop the Live context manager


if __name__ == "__main__":
    app()
