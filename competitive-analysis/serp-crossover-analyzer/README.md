# SERP Crossover Analyzer

## Features

Analyse SERP results across multiple keywords to identify URL crossover patterns. Great for finding keyword cannibalisation and topic overlap opportunities.

- Streamlit web interface + CLI version
- Gets the Google results for multiple keywords from the DataForSEO SERP API (Standard queue)
- Calculates URL crossover (Jaccard similarity) between keyword SERPs
- Identifies URLs ranking for multiple queries
- Detects keyword cannibalisation patterns
- Export analysis to CSV and Excel

## Two steps: send, then collect

Google now routes result links through google.com/goto. DataForSEO's Standard queue turns those back into real URLs, but a search can take 25 minutes or more to finish. So the tool works in two steps:

1. **Send searches:** choose the location, device and results per SERP, enter 2 to 20 keywords and send them. You get a job file (`serp_crossover_job.csv`).
2. **Collect results:** upload the job file (or, in the app, use the job from the same session) and collect the finished searches. Searches still running are listed, and the analysis covers the finished ones; collect again later for the rest.

Any organic result that still comes back as an unresolved google.com/goto link is left out of the crossover and counted in a warning, as its real URL is unknown.

## Requirements

- DataForSEO account (sign up at [dataforseo.com](https://dataforseo.com/))
- Streamlit app: enter your login and password in the sidebar (prefilled from `DATAFORSEO_LOGIN` and `DATAFORSEO_PASSWORD` when set)
- CLI: pass `--login` and `--password`, or set the `DATAFORSEO_LOGIN` and `DATAFORSEO_PASSWORD` environment variables

```bash
pip install -r requirements.txt
```

## Cost

Each keyword is one search in the DataForSEO Standard queue (Google Organic). DataForSEO bills per 10 results, so the estimate is:

`keywords x $0.0006 x (results per keyword / 10)`

Results per keyword is rounded up to the next 10. For example, 20 keywords at 10 results each is estimated at $0.012. The app and CLI show the estimate before sending and the cost DataForSEO reports after. Collecting results does not add to the cost.

## Usage

### Streamlit App

```bash
streamlit run serp_crossover_analyzer.py
```

1. Enter your DataForSEO login and password in the sidebar
2. **1. Send searches:** choose the location, device and results per SERP, enter your keywords (one per line), check the estimated cost, click **Send searches** and download the job file
3. **2. Collect results** (about 30 minutes later): upload the job file and click **Collect results**
4. Review the crossover matrix and overlapping URLs, then download the CSV or Excel report

### CLI

```bash
export DATAFORSEO_LOGIN="your_login"
export DATAFORSEO_PASSWORD="your_password"

# Step 1: send the searches and write a job file
python serp_crossover_analyzer_cli.py send --keywords "seo tools" "keyword research" "content marketing" \
    --job job.csv --location "United States" --device mobile --num-results 20

# Step 2: collect the finished searches and write the crossover matrix
python serp_crossover_analyzer_cli.py collect --job job.csv --output serp_crossover.csv

# Or keep checking every 60 seconds (up to 90 minutes) until every search has finished
python serp_crossover_analyzer_cli.py collect --job job.csv --wait
```

`send` arguments:

- `--keywords` - keywords to compare (at least 2)
- `--job` - job file to write (default: `serp_crossover_job.csv`)
- `--location` - search location name (default: `United Kingdom`)
- `--location-code` - DataForSEO location code (overrides `--location`)
- `--device` - `desktop` or `mobile` (default: `desktop`)
- `--num-results` - results per SERP, billed per 10 (default: `10`)
- `--login`, `--password` - DataForSEO credentials (or use the environment variables)

`collect` arguments:

- `--job` - job file written by `send`
- `--output` - crossover matrix CSV path (default: `serp_crossover.csv`); overlapping URLs are saved next to it as `<name>_overlaps.csv`
- `--wait` - keep checking until every search has finished
- `--login`, `--password` - DataForSEO credentials (or use the environment variables)

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
