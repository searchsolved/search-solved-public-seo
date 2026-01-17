# SERP Clustering at Scale

Cluster keywords based on common SERP URLs. This script processes batch exports from ValueSERP and groups keywords that share common ranking URLs.

## Features

- Process multiple CSV files from ValueSERP batch exports
- Cluster keywords based on configurable number of common URLs
- Automatic cluster naming based on shortest keyword
- Progress bars for long-running operations
- Export results to CSV

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

1. Export SERP data from [ValueSERP](https://www.valueserp.com/) in CSV format
2. Place all CSV files in a folder (default: `valueserp_exports` in the script directory)
3. Update the configuration variables if needed:
   - `FOLDER_LOCATION`: Path to your CSV files
   - `COMMON_URLS`: Minimum number of common URLs to form a cluster (default: 4)
   - `EXPORT_CSV_FILE_PATH`: Output file path
4. Run the script:

```bash
python serp_clustering_at_scale.py
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CONSOLIDATE_QUERIES` | `True` | Consolidate overlapping clusters |
| `COMMON_URLS` | `4` | Minimum common URLs to match keywords |
| `FOLDER_LOCATION` | `./valueserp_exports` | Folder containing ValueSERP exports |
| `FILE_PREFIX` | `/Batch_Results_*.csv` | File pattern to match |

## Output

The script generates a CSV file with the following columns:
- `serp_cluster`: The cluster name (shortest keyword in cluster)
- `query`: The keyword
- `#_kws_in_cluster`: Number of keywords in the cluster
- `#_common_urls`: Number of common URLs
- `common_urls`: List of common URLs

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)