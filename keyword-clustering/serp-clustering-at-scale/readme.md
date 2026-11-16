# SERP Clustering at Scale

Cluster keywords based on common SERP URLs to find content consolidation opportunities. Keywords that share enough ranking URLs are grouped, and each cluster gets a consolidation score (0 to 100) and a recommendation.

Two ways to get SERP data:

1. **Upload SERP CSVs**: exports you already have (e.g. from SERP API), one row per keyword and result URL
2. **DataForSEO SERP API**: send your keywords to DataForSEO, then collect the results and cluster them

Both are available in the Streamlit app (`serp_clustering_at_scale_app.py`) and the script (`serp_clustering_at_scale.py`).

## Why DataForSEO takes two steps

Google now routes result links through google.com/goto. DataForSEO's Standard queue turns those back into real URLs, but a search can take 25 minutes or more to finish. So you send the searches, keep the job file, and collect the results later.

Any organic result that still comes back as an unresolved google.com/goto link is left out of the clustering and counted in a warning.

**Cost:** keywords x $0.0006 x (results per keyword / 10). For example, 1,000 keywords at 10 results each costs about $0.60. The estimate is shown before anything is sent.

## Requirements

```bash
pip install -r requirements.txt
```

`dataforseo_queue.py` must sit next to the app and the script. It is a copy of `shared/dataforseo_queue.py` in this repository; do not edit the copy.

## Streamlit app

```bash
streamlit run serp_clustering_at_scale_app.py
```

- **Upload SERP CSVs:** upload one or more CSVs. The first column is the search query and the second is the result URL. Click "Run Clustering".
- **1. Send searches:** enter your DataForSEO login and password in the sidebar, choose the location, device and results per keyword, paste your keywords (one per line), check the estimated cost and click "Send searches". Download the job file.
- **2. Collect results:** about 30 minutes later, upload the job file (or use the job sent from the same page) and click "Collect and Cluster". If some searches are still running, the clusters cover the finished ones only; try again in a few minutes.

The "Minimum Shared URLs" setting in the sidebar applies to both methods. After collecting, you can change it and the collected results are clustered again without collecting them a second time.

## Script

### CSV mode

Reads SERP API batch exports (columns `search.q` and `result.organic_results.link`) from a folder.

```bash
python serp_clustering_at_scale.py
python serp_clustering_at_scale.py --folder ./serp_exports --file-prefix "/Batch_Results_*.csv" --output clusters.csv
```

### DataForSEO mode

Credentials come from `--login` and `--password`, or the `DATAFORSEO_LOGIN` and `DATAFORSEO_PASSWORD` environment variables.

Step 1, send the searches and write a job file:

```bash
python serp_clustering_at_scale.py send --keywords-file keywords.txt --job job.csv --location "United Kingdom" --device desktop --depth 10
```

The estimated cost is shown and you are asked to confirm; add `--yes` to skip the prompt (needed when not running interactively).

Step 2, about 30 minutes later, collect the finished searches and cluster them:

```bash
python serp_clustering_at_scale.py collect --job job.csv --output clusters.csv
python serp_clustering_at_scale.py collect --job job.csv --output clusters.csv --wait
```

`--wait` checks every 60 seconds until every search has finished, for up to 90 minutes. Without it, collect clusters the searches that have finished and tells you how many are still running.

## Options

| Option | Default | Used by | Description |
|--------|---------|---------|-------------|
| `--folder` | `./serp_exports` | CSV mode | Folder containing SERP CSV exports |
| `--file-prefix` | `/Batch_Results_*.csv` | CSV mode | File pattern to match |
| `--keywords-file` | | send | Text file with one keyword per line |
| `--job` | `serp_clustering_job.csv` | send, collect | Job file written by send and read by collect |
| `--location` | `United Kingdom` | send | Search location |
| `--device` | `desktop` | send | `desktop` or `mobile` |
| `--depth` | `10` | send | Results per keyword, 10 to 100, billed per 10 |
| `--yes` | off | send | Skip the cost confirmation prompt |
| `--wait` | off | collect | Keep checking until every search has finished |
| `--common-urls` | `4` | CSV mode, collect | Minimum shared URLs to cluster keywords |
| `--strategy` | `all` | CSV mode, collect | `connected`, `cliques`, `core` or `all` |
| `--core-threshold` | `0.7` | CSV mode, collect | Core clustering connectivity threshold |
| `--output` | `./serp_cluster_results.csv` | CSV mode, collect | Output CSV path |

Put the collect options after the word `collect`.

## Output

The script writes a CSV sorted by consolidation score with these columns: `serp_cluster` (the shortest keyword in the cluster), `cluster_type`, `query`, `total_urls`, `shared_url_count`, `shared_urls`, `min_shared_urls`, `max_shared_urls`, `avg_shared_urls`, `connectivity_score`, `#_kws_in_cluster`, `is_in_multiple_clusters`, `consolidation_score` and `consolidation_recommendation`. Keywords that share too few URLs with any other keyword are listed with the cluster `NO_CLUSTER`.

The app shows the results in tabs (all results, top opportunities, statistics, individual clusters) and offers them as a CSV download.

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
