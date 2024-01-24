
# Semantic Keyword Clustering with Python

This collection of Python scripts harnesses the power of the SentenceTransformers library and HDBScan to semantically cluster keywords. It employs sentence embeddings to group semantically similar keywords, streamlining the analysis of extensive keyword lists. This suite offers both CLI versions and an HDBScan version to accommodate various clustering needs and dataset sizes.

## General Overview

The scripts in this repository offer a robust solution for semantic keyword clustering. By generating sentence embeddings and performing clustering, these tools efficiently group similar keywords, aiding in the organization and analysis of keyword data. The output is a CSV or Excel file with a Pivot Table, alongside insightful visualisations such as treemaps or sunburst charts.

## Versions

### 1. Semantic Keyword Clustering - CLI Versions

#### Core Functionality
- Load a CSV file containing keywords.
- Generate sentence embeddings using SentenceTransformers.
- Cluster the embeddings to group similar keywords.
- Save the results in a CSV file, associating each keyword with its cluster.
- Generate visualizations like treemaps or sunburst charts to illustrate the clustering.

#### How to Use
Run the script from the command line, specifying your parameters. Here's an example command:

```bash
python cluster.py mycsv.csv --column_name "Keyword" --output_path "output.csv" --chart_type "sunburst" --device "cpu" --model_name "all-MiniLM-L6-v2" --min_similarity 0.80 --remove_dupes True --volume "Volume" --stem True
```

For a minimal setup, use:

```bash
python cluster.py mycsv.csv
```

### 2. Semantic Keyword Clustering with HDBScan (Recommended for Very Large Clustering Jobs)

#### Core Functionality
- Utilizes HDBScan for efficient clustering of large datasets.
- Follows the same core functionality as the CLI versions.

#### How to Use
Similar to the CLI version, but optimized for handling larger datasets with the power of HDBScan.

## Common Options

- `--file-path:` Path to your CSV file.
- `--column-name:` Name of the column in your CSV to process.
- `--output-path:` Path for saving the output CSV.
- `--chart-type:` Type of chart to generate: "sunburst" or "treemap".
- `--device:` Device for SentenceTransformer: "cpu" or "cuda".
- `--model-name:` Name of the SentenceTransformer model. Refer to SentenceTransformers documentation for models.
- `--min-similarity:` Minimum similarity for clustering (0-1 scale).
- `--remove-dupes:` Option to remove duplicates from the dataset.
- `--volume:` Column name with numerical values for volume analysis.
- `--stem:` Option to perform stemming on the 'hub' column.

## Dependencies

Ensure these libraries are installed before running the scripts:

```bash
pip install chardet numpy pandas plotly typer pywin32 polyfuzz rich sentence_transformers
```

## Additional Notes

- SentenceTransformers uses PyTorch; a machine with ample RAM is recommended.
- For GPU usage (--device "cuda"), ensure a CUDA-compatible GPU and the correct PyTorch version is installed.
- For large datasets, consider setting a large page file or using a lighter sentence transformer to manage system resources efficiently.

For specific usage instructions, refer to the individual `README.md` files in each version's directory.
