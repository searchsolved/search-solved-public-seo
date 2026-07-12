# Title Keyword Gap Finder v2

Two analysis modes for page title optimisation, available as a Streamlit app and a CLI tool.

## Modes

### 1. Keyword Gap (original)

Compares Google Search Console queries against page titles to find keywords that drive impressions but are missing from the title. Useful for quick-win title optimisation.

### 2. Title Segment Analysis (new in v2)

Splits page titles by a configurable delimiter (default `|`), treats each segment as a potential keyword, and cross-references against GSC query data for the same page. Surfaces:

- **Wasted segments** - title segments with zero search impressions (occupying title space without earning traffic).
- **Missing opportunities** - high-performing GSC keywords that are not represented in any title segment.

Output is a highlighted Excel file: green rows indicate keywords matched in the title, yellow rows indicate gaps.

## Data Requirements

| File | Source | Required Columns |
|------|--------|-----------------|
| Crawl CSV | Screaming Frog (or similar) | `Address`, `Title 1` |
| GSC CSV | Search Console export or API | `page`, `query`, `clicks`, `impressions` |

Column names are matched case-insensitively with common aliases (e.g. `url`, `landing page`, `keyword`).

## Streamlit App

```bash
pip install -r requirements.txt
streamlit run title_keyword_gap.py
```

The app presents two tabs. Upload your crawl and GSC CSVs, configure brand exclusions and delimiter, then download highlighted Excel or CSV results.

## CLI

```bash
# Keyword Gap mode
python title_keyword_gap_cli.py keyword-gap \
    --crawl crawl.csv \
    --gsc gsc_queries.csv \
    --brand "example store" \
    --delimiter "|" \
    --excel

# Segment Analysis mode
python title_keyword_gap_cli.py segment \
    --crawl crawl.csv \
    --gsc gsc_queries.csv \
    --brand "example store" \
    --delimiter "|" \
    --url-filter "/category/" \
    --excel
```

### CLI Options (both modes)

| Flag | Default | Description |
|------|---------|-------------|
| `--crawl` | (required) | Path to crawl CSV |
| `--gsc` | (required) | Path to GSC query CSV |
| `--output` | `title_keyword_gaps.csv` | Output file path |
| `--delimiter` | `\|` | Character to split title segments |
| `--brand` | (none) | Brand terms to exclude (comma-separated) |
| `--url-filter` | (none) | Only analyse URLs containing this text |
| `--max-keywords` | 10 | Max GSC keywords per page |
| `--min-impressions` | 0 | Minimum impression threshold |
| `--excel` | off | Export as Excel with row highlighting |

## Example

Given a page with title `Industrial Widgets | Plumbing Tools | Example Store` and GSC data showing the query `copper fittings` drives 200 clicks to that page:

- **Segment Analysis** will flag `Example Store` as a branded segment (excluded), note that `Plumbing Tools` has zero impressions for this page, and surface `copper fittings` as a high-performing keyword absent from the title.
- **Keyword Gap** will simply flag `copper fittings` as missing from the title.

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
