# AI vs Classic Search Volume

Compare AI search volume (AI Overviews/ChatGPT) against traditional Google search volume per keyword to identify which keywords are migrating to AI platforms.

## Features

- Streamlit web interface + CLI version
- CSV upload or paste keywords (up to 1000 per run)
- Batched API calls to DataForSEO
- AI share percentage calculation (AI volume / total volume)
- Horizontal bar chart showing top 20 keywords
- Summary statistics (average AI share, AI-dominant count, classic-dominant count)
- CSV and Excel download
- Cost estimate shown before execution

## Usage

### Streamlit App

```bash
pip install -r requirements.txt
streamlit run ai_vs_classic_volume.py
```

### CLI

```bash
python ai_vs_classic_volume_cli.py \
    --login your@email.com \
    --password your_api_password \
    --keywords "best crm software" "how to train a puppy" "weather tomorrow"

# Or from a file (one keyword per line):
python ai_vs_classic_volume_cli.py \
    --login your@email.com \
    --password your_api_password \
    --keywords-file keywords.txt \
    --output results.csv
```

#### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--login` | (required) | DataForSEO login email |
| `--password` | (required) | DataForSEO API password |
| `--keywords` | | Keywords (space-separated, quote multi-word) |
| `--keywords-file` | | Text file with one keyword per line |
| `--output` | `ai_vs_classic_volume.csv` | Output CSV path |
| `--location-code` | `2826` (UK) | DataForSEO location code |
| `--language-code` | `en` | Language code |

## Output Columns

| Column | Description |
|--------|-------------|
| `keyword` | The input keyword |
| `ai_search_volume` | Monthly searches via AI platforms (AI Overviews, ChatGPT) |
| `classic_search_volume` | Traditional Google Ads monthly search volume |
| `ai_share_pct` | AI volume as a percentage of total: ai / (ai + classic) * 100 |
| `delta` | Difference: ai_search_volume minus classic_search_volume |

## Pricing

- AI Keyword Search Volume: ~$0.01 per batch (up to 1000 keywords)
- Google Ads Search Volume: ~$0.05 per batch (up to 1000 keywords)
- A single run of 1000 keywords costs approximately $0.06

## Requirements

- Python 3.9+
- DataForSEO API credentials ([dataforseo.com](https://dataforseo.com))

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
