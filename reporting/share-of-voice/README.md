# Share of Voice

## Features

Calculate estimated organic traffic share using ranking positions and industry-standard CTR curves (Sistrix, AWR, etc.).

- Streamlit web interface + CLI version
- Two inputs in the app: upload your own ranking CSV, or get the Google results for your keywords from the DataForSEO SERP API
- Multiple CTR curve models (Sistrix, Advanced Web Ranking, Backlinko, Conservative, or custom values)
- Position-based calculations (positions 1 to 10)
- Competitor comparison, grouped by domain or by category
- Market share estimation
- Export analysis to Excel (app) or CSV (CLI)

## Requirements

```bash
pip install -r requirements.txt
```

For the DataForSEO option: a DataForSEO account (sign up at [dataforseo.com](https://dataforseo.com/)). Enter your login and password in the sidebar (prefilled from `DATAFORSEO_LOGIN` and `DATAFORSEO_PASSWORD` when set).

## Usage

### Streamlit App

```bash
streamlit run share_of_voice.py
```

**Upload CSV:** upload a CSV with `keyword`, `volume`, `position` and `domain` columns (and `category` if you group by category), check the column mapping and click **Calculate Share of Voice**.

**Fetch SERPs (DataForSEO):** this works in two steps. Google now routes result links through google.com/goto, and DataForSEO's Standard queue turns those back into real URLs, but a search can take 25 minutes or more to finish. So you send the searches, keep the job file, and collect the results later.

1. **Send searches:** choose the location and device, enter one `keyword, volume` pair per line, check the estimated cost and click **Send searches**. Download the job file (`share_of_voice_job.csv`); it holds each search's task ID and the volume you entered.
2. **Collect results** (about 30 minutes later): upload the job file (or use the job sent from the same page) and click **Collect results**. Finished searches are used straight away; if some are still running, the app says so and you can collect again in a few minutes. The CTR curve and Top N settings in the sidebar apply at this step, so you can change them without sending new searches.

Organic results that come back as unresolved google.com/goto links are counted, shown as a warning, and left out of the calculation, as they cannot be credited to a domain.

### CLI

The CLI works from a ranking CSV only; it does not call DataForSEO.

```bash
python share_of_voice_cli.py --input rankings.csv --output share_of_voice.csv --ctr-curve sistrix
```

Arguments:

- `--input` - ranking CSV (required)
- `--output` - output CSV path (default: `share_of_voice.csv`)
- `--ctr-curve` - `sistrix`, `awr`, `backlinko` or `conservative` (default: `sistrix`)
- `--keyword-col`, `--volume-col`, `--position-col`, `--domain-col`, `--category-col` - column names, when they cannot be found automatically
- `--top-n` - top N domains to show (default: `20`)

## Cost (DataForSEO option)

Each keyword is one search in the DataForSEO SERP API Standard queue (Google Organic), fetching the top 10 results. DataForSEO bills per 10 results, so the estimate is:

`keywords x $0.0006 x (results per keyword / 10)`

The app always fetches 10 results per keyword, so 100 keywords are estimated at $0.06. The app shows the estimate before sending and the cost DataForSEO reports after. Collecting results is normally not charged.

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
