# Topical Map Visualiser

## Features

Turn a tagged keyword CSV into an interactive, zoomable D3.js circle packing chart. Each parent topic becomes a large circle, each subtopic a circle inside it, and each keyword a leaf circle sized by the metric you choose. Click a circle to zoom in, click the background to zoom back out.

- Streamlit web interface + CLI version
- Zoomable D3.js circle packing chart, saved as a standalone HTML file
- Circle sizes driven by keyword count, impressions, clicks, first page keywords, or top 3 keywords
- Column mapping flags so it works with any CSV layout
- Pastel colour coding by parent topic

## Pairs with the Topical Map Generator

This tool is designed to visualise the output of the [Topical Map Generator](../topical-map-generator/) in this repository. Generate a topical map CSV with that tool, then point this tool at it with the column mapping flags:

```
python topical_map_visualiser_cli.py \
  --input topical_map.csv \
  --parent-col "Parent Topic" \
  --child-col "Niche Topic 1" \
  --keyword-col "Keyword" \
  --metric count
```

## Expected input columns

A CSV with one row per keyword and a two-level topic hierarchy:

| Column (default name) | Required | Description |
|---|---|---|
| `Parent` | Yes | Top-level topic, e.g. "Garden Furniture" |
| `Child` | Yes | Subtopic, e.g. "Garden Benches" |
| `query` | Yes | The keyword, e.g. "wooden garden bench" |
| `impressions` | For the `impressions` metric | Numeric, e.g. from Google Search Console |
| `clicks` | For the `clicks` metric | Numeric, e.g. from Google Search Console |
| `position` | For `first_page_count` and `top_3_count` | Average position |

All column names can be remapped with the CLI flags or the Streamlit sidebar, so a CSV with `Parent Topic` / `Niche Topic 1` / `Keyword` columns works fine.

## Metrics

- `count` (default): each keyword counts as 1, so circle size reflects how many keywords sit under each topic. Use this when you have no performance data, e.g. straight from the Topical Map Generator.
- `impressions`: sums impressions per topic.
- `clicks`: sums clicks per topic.
- `first_page_count`: counts keywords ranking in positions 1 to 10.
- `top_3_count`: counts keywords ranking in positions 1 to 3.

## Usage

### CLI

```
pip install -r requirements.txt
python topical_map_visualiser_cli.py --input tagged_keywords.csv --output topical_map.html --metric impressions
```

Then open `topical_map.html` in a browser. The chart loads D3.js from the official CDN, so it needs an internet connection to render.

Optional flags: `--title`, `--parent-col`, `--child-col`, `--keyword-col`, `--position-col`, `--impressions-col`, `--clicks-col`.

### Streamlit

```
pip install -r requirements.txt
streamlit run topical_map_visualiser.py
```

Upload your CSV, map the columns in the sidebar, choose a metric, and generate the chart. You can view it in the app and download it as a standalone HTML file.

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
