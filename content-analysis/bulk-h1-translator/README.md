# Bulk H1 Translator

## Features

Translate H1 headings to English in bulk using any OpenAI-compatible API. Reads a CSV with H1 and Language columns (Screaming Frog export format by default, with mappable column names) and appends a `translated_h1` column while preserving all original columns.

- Streamlit web interface + CLI version
- Works with a local LLM (Ollama) or the OpenAI API
- Mappable H1 and Language columns (defaults: `H1-1`, `Language`)
- Structured JSON schema responses for reliable output
- Retry handling with per-row error reporting
- Export results to CSV (UTF-8 with BOM, Excel friendly)

## Requirements

The endpoint must support the chat completions API with JSON schema structured responses. Ollama and the OpenAI API both do.

```bash
pip install -r requirements.txt
```

## Usage with a Local LLM (Ollama)

1. Start Ollama with an instruction-tuned model (e.g. `ollama run qwen2.5:7b`) (default: `http://localhost:11434/v1`).
2. No API key is required; any placeholder value works.

**Streamlit:**
```bash
streamlit run bulk_h1_translator.py
```
The default settings in the sidebar already point at Ollama.

**CLI:**
```bash
python bulk_h1_translator_cli.py --input crawl.csv --output translated.csv
```

## Usage with OpenAI

**Streamlit:** set the base URL to `https://api.openai.com/v1`, enter your API key and a model name such as `gpt-4o-mini` in the sidebar.

**CLI:**
```bash
export OPENAI_API_KEY=your-key
python bulk_h1_translator_cli.py \
    --input crawl.csv \
    --output translated.csv \
    --base-url https://api.openai.com/v1 \
    --model gpt-4o-mini
```

Any other OpenAI-compatible provider works the same way: set `--base-url`, `--model` and `OPENAI_API_KEY` accordingly.

## CLI Options

| Option | Default | Description |
| --- | --- | --- |
| `--input` | required | Input CSV with H1 and Language columns |
| `--output` | `translated_h1s.csv` | Output CSV path |
| `--h1-column` | `H1-1` | Name of the H1 column |
| `--language-column` | `Language` | Name of the language column |
| `--base-url` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `--model` | `local-model` | Model name (Ollama uses whichever model is loaded) |
| `--retries` | `3` | Attempts per H1 before recording an error |

## Input Format

A Screaming Frog internal HTML export works out of the box. Minimum example:

| H1-1 | Language |
| --- | --- |
| Guía para principiantes | Spanish |
| Anleitung für Anfänger | German |

H1s that are already in English (or empty) are returned unchanged. Rows that fail after all retries are marked with an `Error:` value in `translated_h1`.

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
