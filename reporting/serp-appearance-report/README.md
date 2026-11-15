# SERP Appearance Report

## Features

Fetch the live Google results for a list of keywords from the DataForSEO SERP API and report every organic appearance of your domain, with the query, position, title and snippet for each result.

- Streamlit web interface + CLI version
- Keyword input: paste one per line, or upload a CSV (a Google Search Console queries export works as is)
- Location (10 markets) and device (desktop or mobile) selectable
- Reports organic results on your domain and its subdomains
- Estimated cost shown before every run; requests are sent one at a time with a pause between them
- Export findings to CSV

## Requirements

- DataForSEO account (sign up at [dataforseo.com](https://dataforseo.com/))
- Streamlit app: enter your login and password in the sidebar (prefilled from `DATAFORSEO_LOGIN` and `DATAFORSEO_PASSWORD` when set)
- CLI: set the `DATAFORSEO_LOGIN` and `DATAFORSEO_PASSWORD` environment variables

```bash
pip install -r requirements.txt
```

## Cost

Each keyword is one call to the DataForSEO SERP API (Google Organic, Live Advanced endpoint). DataForSEO bills per 10 results, so the estimate is:

`keywords x $0.002 x (results per keyword / 10)`

For example, 100 keywords at 10 results each is estimated at $0.20, or $2.00 at 100 results each. The app and CLI show the estimate before running and the cost DataForSEO reports after.

## Usage

### Streamlit App

```bash
streamlit run serp_appearance_report.py
```

1. Enter your DataForSEO login and password in the sidebar
2. Choose the location, device, results per keyword and the pause between requests
3. Enter your domain (e.g. `example.com`)
4. Paste your keywords or upload a CSV
5. Check the estimated cost, click **Fetch SERPs**, then download the CSV

### CLI

```bash
export DATAFORSEO_LOGIN="your_login"
export DATAFORSEO_PASSWORD="your_password"

# Text file, one keyword per line
python serp_appearance_cli.py --input keywords.txt --domain example.com

# CSV with a keyword column, US mobile results, top 20
python serp_appearance_cli.py --input keywords.csv --domain example.com --output report.csv \
    --location "United States" --device mobile --depth 20
```

Arguments:

- `--input` - keyword file: `.txt` with one keyword per line, or `.csv` with a column named `keyword`, `keywords`, `query`, `queries`, `search term` or `top queries` (otherwise the first column is used)
- `--domain` - domain to report (e.g. `example.com`); subdomains are included. Add a path (e.g. `example.com/blog`) to report one section only
- `--output` - output CSV path (default: `serp_appearance_report.csv`)
- `--location` - search location (default: `United Kingdom`)
- `--device` - `desktop` or `mobile` (default: `desktop`)
- `--depth` - results to fetch per keyword, 10 to 100 (default: `10`)
- `--delay` - seconds between requests (default: `2.0`)
- `--yes` - skip the cost confirmation prompt (needed when running non-interactively)

The CLI prints the estimated cost and asks you to confirm before it sends any request.

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
