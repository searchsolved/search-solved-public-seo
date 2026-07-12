# Representative Keyword Normaliser

## Features

Suggest a cleaner, more descriptive representative keyword for each keyword in a CSV using an LLM. Useful as a pre-processing step before SERP clustering or semantic clustering.

- Streamlit web interface and CLI
- Works with any OpenAI-compatible endpoint (local Ollama by default, or the OpenAI API)
- Structured JSON schema responses for reliable parsing
- Multi-language support (output stays in the same language as the source keyword)
- Automatic retry on failed or unparseable responses
- Export results to CSV

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Streamlit app

```bash
streamlit run app.py
```

Set the base URL, model and (optionally) API key in the sidebar, upload a CSV, choose the keyword column and download the results.

### CLI

```bash
python cli.py --input keywords.csv --output normalised.csv
```

Options:

| Flag | Description | Default |
| --- | --- | --- |
| `-i`, `--input` | Input CSV file path | required |
| `-o`, `--output` | Output CSV file path | required |
| `-c`, `--column` | Keyword column name | first column |
| `--base-url` | OpenAI-compatible API base URL | `http://127.0.0.1:11434/v1` |
| `--model` | Model name | `qwen2.5:7b` |
| `--api-key` | API key | `OPENAI_API_KEY` env var, or none for local servers |
| `-v`, `--verbose` | Increase output verbosity | off |

### Local models (Ollama)

Start the Ollama with a model loaded and the defaults will work as-is. No API key is required.

### OpenAI

```bash
export OPENAI_API_KEY=your-key
python cli.py --input keywords.csv --output normalised.csv \
    --base-url https://api.openai.com/v1 --model gpt-4o-mini
```

## Output

The output CSV is a copy of the input with a `Suggested Keyword` column appended.

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
