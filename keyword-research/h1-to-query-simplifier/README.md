# H1 to Query Simplifier

## Features

Convert marketing-heavy H1 headings into clean, natural search queries using the Anthropic API. Useful as a pre-processing step before SERP clustering or keyword matching, where promotional H1s make poor queries.

- Streamlit web interface + CLI version
- Claude-powered query simplification (defaults to Haiku for speed and cost)
- Works with Screaming Frog exports (`H1-1` column) or any CSV
- Resume support: completed rows are skipped on re-runs
- Periodic progress saves so interrupted runs lose nothing
- Export results to CSV

## Usage

### Streamlit app

```bash
pip install -r requirements.txt
streamlit run h1_to_query_simplifier.py
```

Enter your Anthropic API key in the sidebar, upload a CSV, pick the H1 column and click "Simplify H1s". To resume an interrupted run, download the partial results and re-upload them later; rows with an existing `H1_Simplified` value are skipped.

### CLI

```bash
export ANTHROPIC_API_KEY=your-key-here
python h1_to_query_simplifier_cli.py --input h1s.csv --output h1s_simplified.csv
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--input` | required | Path to the input CSV |
| `--output` | required | Path to the output CSV (also used to resume) |
| `--column` | `H1-1` | Column containing the H1 text |
| `--model` | `claude-haiku-4-5` | Anthropic model to use |
| `--save-frequency` | `10` | Save progress every N processed rows |
| `--delay` | `0.1` | Delay in seconds between API requests |

If the output file already exists, the run resumes from where it left off.

## Example

| H1 | Simplified query |
|---|---|
| 10 Proven Ways to Boost Your Rankings Fast! | how to improve search rankings |
| The Ultimate Guide to Hiring a Plumber (2024 Edition) | how to hire a plumber |
| Why Thousands Trust Us for Garden Maintenance | garden maintenance services |

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
