# Review Tagger

Tag customer reviews with a one or two-word descriptive label using OpenAI GPT. Upload a CSV of review text and the model assigns a short topic tag to each review (e.g. "Delivery", "Build Quality", "Sizing", "Value"). Useful as a first-pass classification step before deeper analysis.

## Features

- Streamlit web interface + CLI version
- Batch processing to keep API calls efficient
- Configurable batch size, model, and review column
- Retry logic with exponential back-off on API failures
- Progress tracking in both interfaces
- Export results to CSV

## Input Format

A CSV with at least one column containing review text (default column name: `Review`).

| Review |
|---|
| Delivery was late and the box was damaged |
| Great build quality, very sturdy |
| Sizing was off, had to return it |
| Easy to install, took five minutes |
| Good value for money overall |

The output is the same CSV with a new `Tag` column:

| Review | Tag |
|---|---|
| Delivery was late and the box was damaged | Delivery |
| Great build quality, very sturdy | Build Quality |
| Sizing was off, had to return it | Sizing |
| Easy to install, took five minutes | Installation |
| Good value for money overall | Value |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Streamlit App

```bash
streamlit run review_tagger_app.py
```

1. Enter your OpenAI API key in the sidebar
2. Choose a model (gpt-4o-mini is the default and recommended for cost)
3. Upload your CSV and select the column containing review text
4. Adjust the batch size if needed (default: 25 reviews per API call)
5. Click "Tag Reviews" and download the results

### CLI

The CLI reads your API key from the `OPENAI_API_KEY` environment variable.

```bash
export OPENAI_API_KEY=your-key
python review_tagger_cli.py --input reviews.csv --output tagged_reviews.csv
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--input` | (required) | Input CSV with reviews |
| `--output` | `tagged_reviews.csv` | Output CSV path |
| `--column` | `Review` | Name of the review text column |
| `--model` | `gpt-4o-mini` | OpenAI model |
| `--batch-size` | `25` | Number of reviews per API call |

## Pairing with Tag Consolidator

This tool assigns granular, review-level tags. For grouping those tags into broader categories (e.g. rolling "Late Delivery", "Damaged Box", "Wrong Address" into a single "Delivery Issues" category), use the **Tag Consolidator** tool in [`content-analysis/tag-consolidator/`](../tag-consolidator/).

A typical workflow:
1. Run **Review Tagger** to assign a tag to each review
2. Run **Tag Consolidator** to consolidate those tags into higher-level categories

## Notes

- Cost scales with the number of batches (total reviews divided by batch size).
- Reviews that the model could not tag are left blank in the `Tag` column and flagged in the output summary; review these manually.
- The tool uses JSON mode (`response_format: json_object`) so the model always returns parseable output.

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
