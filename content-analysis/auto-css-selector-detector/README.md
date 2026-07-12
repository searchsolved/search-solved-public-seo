# Auto CSS Selector Detector

## Features

Automatically identify the best CSS selector for a web page's main content area using an LLM, then extract and convert the content to Markdown.

- Streamlit web interface + CLI version
- Sends a structural summary of the page to an LLM to detect the content selector
- Extracts content using BeautifulSoup and converts to Markdown via html2text
- Deduplicates near-identical paragraphs
- Extracts internal links from the content
- Supports any OpenAI-compatible API (local LLMs, OpenRouter, etc.)
- Model selector (default: gpt-4o-mini)

## Usage

### Streamlit App

```bash
pip install -r requirements.txt
streamlit run auto_css_selector_detector_app.py
```

Enter your API key in the sidebar, paste a URL, and click **Detect & Extract**.

### CLI

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python auto_css_selector_detector_cli.py --url https://example.com/page
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--url` | (required) | URL to analyse |
| `--model` | `gpt-4o-mini` | Model name |
| `--base-url` | `https://api.openai.com/v1` | API base URL (change for local LLMs) |

## How It Works

1. Fetches the page HTML and removes non-content elements (nav, footer, sidebar, etc.)
2. Generates structural summaries of the top-level content containers
3. Sends the summaries to the LLM, which returns the best CSS selector as JSON
4. Refines the selector to target the most content-rich descendant
5. Extracts the content, converts to Markdown, and deduplicates

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
