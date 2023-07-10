# Semantic Keyword Clustering Tool - Lee Foot SEO 10/07/2023

This tool allows you to perform keyword clustering using a SentenceTransformer model. The script reads a CSV file, processes a specified column, and saves the result in a new CSV file. It also generates a sunburst chart or a treemap for visualization.

## Requirements

This tool requires Python 3.7 or above.

## Installation

Clone this repository to your local machine.

Navigate to the project directory in your terminal.

Install the required packages using pip:

pip install -r requirements.txt

## Usage

The script takes three mandatory arguments: the path to your input CSV file, the name of the column to be processed, and the path to the output CSV file.

python script_name.py /path/to/your/input.csv YourColumnName /path/to/output.csv

Replace script_name.py with the actual name of the Python script. Replace /path/to/your/input.csv with the path to your input CSV file. Replace YourColumnName with the name of the column you want to process. Replace /path/to/output.csv with the path where you want to save the output CSV file.

## The script also supports several options:

--device (default: cpu): The device to be used by the SentenceTransformer. Possible values are cpu and cuda.

--model_name (default: all-MiniLM-L6-v2): The name of the SentenceTransformer model to use. For a list of available models, visit this page.

--min_similarity (default: 0.85): The minimum similarity for clustering.

--remove_dupes (default: True): Whether to remove duplicates from the dataset.

--chart_type (default: sunburst): The type of chart to generate. Possible values are sunburst and treemap.

To use any of these options, add them to the command line arguments. For example:

python script_name.py /path/to/your/input.csv YourColumnName /path/to/output.csv --device cuda --model_name multi-qa-mpnet-base-dot-v1 --min_similarity 0.9 --remove_dupes False --chart_type treemap

## Support

If you encounter any problems or have any questions, please open an issue on this GitHub repository.