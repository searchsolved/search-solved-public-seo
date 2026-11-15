# Content Consolidation Analyzer

Find pages cannibalizing each other by clustering URLs that share significant SERP overlap.

## What It Does

- Analyzes SERP data to find queries that rank similar pages
- Clusters pages using graph-based algorithms (connected components + cliques)
- Calculates consolidation scores (0-100) for each cluster
- Outputs actionable recommendations for content consolidation

## Use Cases

- Identify content cannibalization issues
- Find pages competing for the same keywords
- Make data-driven decisions about merging or consolidating content
- Improve internal linking between related pages

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python content_consolidation_analyzer.py --input serp_data.csv --output results.csv
```

### With Multiple Files

```bash
python content_consolidation_analyzer.py --input "data/*.csv" --output results.csv
```

### Custom Column Names

```bash
python content_consolidation_analyzer.py \
    --input serp_data.csv \
    --query-col "keyword" \
    --url-col "url" \
    --min-urls 3
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input`, `-i` | Input CSV file or pattern | Required |
| `--output`, `-o` | Output CSV file | `consolidation_results.csv` |
| `--min-urls` | Minimum shared URLs to cluster | `4` |
| `--query-col` | Column name for queries | `search.q` |
| `--url-col` | Column name for URLs | `result.organic_results.link` |

## Input Format

CSV file with columns for queries and URLs. Example:

```csv
search.q,result.organic_results.link
best running shoes,https://example.com/shoes
best running shoes,https://example.com/trainers
running shoe reviews,https://example.com/shoes
```

## Output

The tool outputs a CSV with:

- `cluster_id`: The shortest query in the cluster. Unique within a `cluster_type`; a repeat is numbered, for example `running socks (2)`
- `cluster_type`: `connected_component` or `clique`. The same group of queries can appear once as each type
- `query`: The search query
- `consolidation_score`: 0-100 score indicating consolidation strength
- `consolidation_recommendation`: Human-readable recommendation
- `cluster_size`: Number of queries in the cluster
- `is_in_multiple_clusters`: True when the query sits in more than one cluster of the same type
- `shared_urls`: URLs that appear for multiple queries
- And more metrics...

## Consolidation Scores

The score is built from four parts:

| Part | Points | How it is measured |
|------|--------|--------------------|
| Shared URLs | up to 50 | 5 points per average shared URL, capped |
| Connectivity | up to 30 | Share of query pairs in the cluster that overlap |
| Cluster size | up to 20 | 5 points per query beyond two |
| Overlap penalty | up to -10 | 5 points per query that also sits in another cluster of the same type |

| Score | Recommendation |
|-------|----------------|
| 80-100 | Strong consolidation candidate |
| 60-79 | Good consolidation candidate |
| 40-59 | Possible consolidation |
| 20-39 | Weak consolidation candidate |
| 0-19 | Keep separate |

## Author

Lee Foot - [leefoot.com](https://www.leefoot.com)
