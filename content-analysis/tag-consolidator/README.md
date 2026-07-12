# Tag Consolidator

Consolidate granular tags into broader generic categories using OpenAI GPT. Useful when a tagging exercise (for example, mining customer feedback or reviews) produces hundreds of near-duplicate secondary tags that need rolling up into a manageable set of categories.

## Features

- Streamlit web interface + CLI version
- Groups rows by primary tag and consolidates each group's secondary tags in a single API call
- AI-powered grouping of similar tags into broader generic categories
- Adds a 'Generic Tag' column mapping every secondary tag to its category
- Incremental checkpoint saving in the CLI, so progress is not lost mid-run
- Export results to CSV

## Input Format

A CSV with a primary tag column and a secondary tag column (default names: `Primary Tag` and `Secondary Tag`).

| Primary Tag | Secondary Tag |
|---|---|
| Delivery | arrived late |
| Delivery | left in wrong place |
| Sizing | runs small |
| Sizing | inconsistent measurements |
| Build Quality | flimsy material |

The output is the same CSV with a new `Generic Tag` column, for example "arrived late" mapped to "Late Delivery".

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Streamlit App

```bash
streamlit run tag_consolidator_app.py
```

1. Enter your OpenAI API key in the sidebar
2. Choose a model (gpt-4o-mini is the default and is recommended for cost)
3. Upload your CSV and select the primary and secondary tag columns
4. Click "Consolidate Tags" and download the results

### CLI

The CLI reads your API key from the `OPENAI_API_KEY` environment variable.

```bash
export OPENAI_API_KEY=your-key
python tag_consolidator_cli.py --input tags.csv --output consolidated_tags.csv
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--input` | (required) | Input CSV with tags |
| `--output` | `consolidated_tags.csv` | Output CSV path |
| `--model` | `gpt-4o-mini` | OpenAI model |
| `--primary-column` | `Primary Tag` | Name of the primary tag column |
| `--secondary-column` | `Secondary Tag` | Name of the secondary tag column |

## Notes

- Each primary tag group is one API call, so cost scales with the number of groups, not the number of rows.
- Rows whose secondary tag was not returned by the model are left blank in the `Generic Tag` column and flagged in the output summary; review these manually.

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
