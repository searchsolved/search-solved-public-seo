# SERP Appearance Report

## Features

Get the Google results for a list of keywords from the DataForSEO SERP API and report every organic appearance of your domain, with the query, position, title and snippet for each result.

- Streamlit web interface + CLI version
- Two steps: send the searches, then collect the results about 30 minutes later (see below)
- Keyword input: paste one per line, or upload a CSV (a Google Search Console queries export works as is)
- Location (10 markets) and device (desktop or mobile) selectable
- Reports organic results on your domain and its subdomains
- Estimated cost shown before sending
- Export findings to CSV

## Why two steps

Google now routes result links through google.com/goto. DataForSEO's Live requests return many of those links unresolved, so the real URL and domain are missing. Its Standard queue resolves them, but a search can take 25 minutes or more to finish. So the tool sends the searches and gives you a job file; you collect the results from that file later. DataForSEO keeps finished searches for 30 days.

## Requirements

- DataForSEO account (sign up at [dataforseo.com](https://dataforseo.com/))
- Streamlit app: enter your login and password in the sidebar (prefilled from `DATAFORSEO_LOGIN` and `DATAFORSEO_PASSWORD` when set)
- CLI: set the `DATAFORSEO_LOGIN` and `DATAFORSEO_PASSWORD` environment variables

```bash
pip install -r requirements.txt
```

## Cost

Each keyword is one search in the DataForSEO Standard queue (Google Organic, Advanced). DataForSEO bills per 10 results, so the estimate is:

`keywords x $0.0006 x (results per keyword / 10)`

For example, 100 keywords at 10 results each is estimated at $0.06, or $0.60 at 100 results each. Collecting results costs nothing.

## Usage

### Streamlit App

```bash
streamlit run serp_appearance_report.py
```

1. Enter your DataForSEO login and password in the sidebar
2. **1. Send searches:** choose the location, device and results per keyword, paste your keywords or upload a CSV, check the estimated cost and click **Send searches**. Download the job file.
3. **2. Collect results** (about 30 minutes later): upload the job file (or, if the page is still open, use the job it kept), enter your domain (e.g. `example.com`) and click **Collect results**. If some searches are still running, the table covers the finished ones; collect again later for the rest.
4. Download the CSV

### CLI

```bash
export DATAFORSEO_LOGIN="your_login"
export DATAFORSEO_PASSWORD="your_password"

# Step 1: queue the searches (text file, one keyword per line, or a CSV with a keyword column)
python serp_appearance_cli.py send --input keywords.txt --job job.csv

# Step 2: collect and report (add --wait to keep checking until every search is done)
python serp_appearance_cli.py collect --job job.csv --domain example.com --output report.csv
```

`send` arguments:

- `--input` - keyword file: `.txt` with one keyword per line, or `.csv` with a column named `keyword`, `keywords`, `query`, `queries`, `search term` or `top queries` (otherwise the first column is used)
- `--job` - job file to write (default: `serp_appearance_job.csv`)
- `--location` - search location (default: `United Kingdom`)
- `--device` - `desktop` or `mobile` (default: `desktop`)
- `--depth` - results to fetch per keyword, 10 to 100 (default: `10`)
- `--yes` - skip the cost confirmation prompt (needed when running non-interactively)

`collect` arguments:

- `--job` - job file written by `send`
- `--domain` - domain to report (e.g. `example.com`); subdomains are included. Add a path (e.g. `example.com/blog`) to report one section only
- `--output` - output CSV path (default: `serp_appearance_report.csv`)
- `--wait` - check every 60 seconds until all searches finish (up to 90 minutes)

## Output

One row per organic appearance of the domain:

| Column | Description |
| --- | --- |
| query | The keyword searched |
| position | Organic position of the result (organic results only, ads and SERP features are not counted) |
| link | Result URL |
| title | Result title as shown in the SERP |
| snippet | Result snippet as shown in the SERP |

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
