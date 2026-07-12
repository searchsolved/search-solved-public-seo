# AI Visibility Checker

See who gets cited in **Google AI Overviews** and **ChatGPT** answers for a given domain's topic space.

Uses the [DataForSEO AI Optimization API](https://dataforseo.com/apis/ai-optimization-api) (LLM Mentions Search endpoint) to retrieve all questions where a domain or brand entity appears in AI-generated answers.

## Features

- Check up to 10 domains/entities per query
- Platform selector: Google AI Overviews, ChatGPT, or Both
- Location and language selectors (defaults to UK/English)
- Results table with AI search volume, source count, fan-out queries, and dates
- Expandable detail per mention: full AI answer (markdown), cited sources with domain/position/title, fan-out queries
- Summary stats: total mentions, total AI search volume, top cited domains
- CSV and Excel download
- CLI version with environment variable auth

## Requirements

- Python 3.9+
- DataForSEO API credentials ([get them here](https://dataforseo.com))

## Installation

```bash
pip install -r requirements.txt
```

## Usage: Streamlit App

```bash
streamlit run ai_visibility_checker.py
```

Enter your DataForSEO login and password in the sidebar, type your domain(s), and click "Check AI Visibility".

## Usage: CLI

Set environment variables:

```bash
export DATAFORSEO_LOGIN=your@email.com
export DATAFORSEO_PASSWORD=yourpassword
```

Run:

```bash
python ai_visibility_checker_cli.py --entities example.com --platform google --limit 50
```

Options:

| Flag | Description | Default |
|------|-------------|---------|
| `--entities` | Domains or brand names (space-separated, max 10) | Required |
| `--platform` | `google`, `chat_gpt`, or `both` | `google` |
| `--location` | Location name | `United Kingdom` |
| `--language` | Language name | `English` |
| `--limit` | Max results (1-1000) | `50` |
| `--output` | Output CSV path | `ai_visibility_results.csv` |
| `--show-domains` | Print top cited domains to stdout | Off |
| `--login` | DataForSEO login (overrides env var) | `DATAFORSEO_LOGIN` |
| `--password` | DataForSEO password (overrides env var) | `DATAFORSEO_PASSWORD` |

## Pricing

Each API call costs approximately $0.10 and returns up to 1000 mentions. Selecting "Both" platforms makes two calls (~$0.20).

## Author

**Lee Foot** - [leefoot.com](https://www.leefoot.com)
