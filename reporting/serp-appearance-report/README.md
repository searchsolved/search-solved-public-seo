# SERP Appearance Report

## Features

Parse ValueSERP batch JSON output and report every organic appearance of your domain, with the query, position, title and snippet for each result.

- Streamlit web interface + CLI version
- Upload one or more ValueSERP batch JSON files (CLI also accepts a directory)
- Filter organic results by any domain
- Handles files in a range of encodings
- Export findings to CSV

## Getting ValueSERP Batch JSON Exports

1. Log in to [ValueSERP](https://www.valueserp.com/) and open **Batches**
2. Create a batch and add your searches (for example, your top queries by clicks from Google Search Console)
3. Set the destination output format to **JSON** and run the batch
4. When the batch finishes, download the JSON result set file(s)

Each result set is a JSON array where every item contains a `result` object with `search_parameters` (including the query `q`) and an `organic_results` array.

## Usage

### Streamlit App

```bash
pip install -r requirements.txt
streamlit run serp_appearance_report.py
```

Upload your JSON file(s), enter your domain (e.g. `example.com`), then review the results table and download the CSV.

### CLI

```bash
# Single file
python serp_appearance_cli.py --input results.json --domain example.com

# Directory of batch exports
python serp_appearance_cli.py --input ./batch_exports/ --domain example.com --output report.csv
```

Arguments:

- `--input` - path to a ValueSERP batch JSON file, or a directory of `.json` files
- `--domain` - domain to filter organic results by (e.g. `example.com`)
- `--output` - output CSV path (default: `serp_appearance_report.csv`)

## Output

One row per organic appearance of the domain:

| Column | Description |
| --- | --- |
| query | The search query from the batch |
| position | Organic position of the result |
| link | Result URL |
| title | Result title as shown in the SERP |
| snippet | Result snippet as shown in the SERP |

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
