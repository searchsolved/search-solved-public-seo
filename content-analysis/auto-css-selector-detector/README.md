# Auto CSS Selector Detector

## Features

Automatically identify the best CSS selector for a web page's main content area using an LLM, then extract and convert the content to Markdown.

- Streamlit web interface + CLI version
- Sends a structural summary of the page to an LLM to detect the content selector
- Extracts content using BeautifulSoup and converts to Markdown via html2text
- Deduplicates near-identical paragraphs
- Extracts internal links from the content
- Supports any OpenAI-compatible API (local LLMs, OpenRouter, etc.)
- Model selector (default: gpt-6-luna; OpenAI models are gpt-6-luna and gpt-6.1-terra)

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
| `--model` | `gpt-6-luna` | Model name (`gpt-6-luna` or `gpt-6.1-terra` on OpenAI, or a local model name) |
| `--base-url` | `https://api.openai.com/v1` | API base URL (change for local LLMs) |

## How It Works

1. Fetches the page HTML (a 4xx or 5xx response stops the run)
2. Summarises the top-level content containers, leaving out navigation, headers and footers, with a selector for each that is valid on the live page (`#id` where the id is unique, otherwise an `nth-of-type` path from `body`)
3. Sends the summaries to the LLM, which returns the best CSS selector as JSON
4. Narrows that selector: steps into a child container while it holds at least 80% of the text, and never past the element that contains the content's H1
5. Extracts the content from the narrowed selector, converts it to plain text and removes near-duplicate paragraphs. If neither selector matches content, a scoring heuristic picks the element, and `extraction_source` in the result says which was used

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
