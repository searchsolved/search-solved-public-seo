# Product Attribute Extractor

## Features

Extract structured product attributes from titles and descriptions using any OpenAI-compatible LLM. Attributes are iteratively discovered across the catalogue, so the model reuses consistent names (brand, colour, voltage, material, etc.) rather than inventing new ones for each product.

- Streamlit web interface + CLI version
- Iterative attribute discovery (new attributes are learnt from earlier products)
- Works with OpenAI, Azure OpenAI, or any local LLM with an OpenAI-compatible API
- Configurable model and base URL (defaults to gpt-4o-mini)
- Outputs enriched CSV with one column per discovered attribute
- Columns sorted by population frequency

## Usage

### Streamlit App

```bash
pip install -r requirements.txt
streamlit run product_attribute_extractor_app.py
```

1. Enter your API key in the sidebar
2. Upload a CSV containing product titles or descriptions
3. Select the text column and click Extract
4. Download the enriched CSV

### CLI

```bash
export OPENAI_API_KEY="your-key-here"
python product_attribute_extractor_cli.py --input products.csv --column "title" --output enriched.csv
```

For a local LLM (e.g. LM Studio, Ollama):

```bash
python product_attribute_extractor_cli.py \
    --input products.csv \
    --column "H1" \
    --base-url http://localhost:1234/v1 \
    --model local-model
```

## How It Works

1. Each product's text is sent to the LLM with a system prompt containing few-shot examples and the list of attributes discovered so far.
2. The model extracts attributes as a JSON object, reusing known attribute names where possible.
3. Newly discovered attributes are added to the known set and used in subsequent prompts.
4. The output DataFrame is sorted so the most commonly populated attributes appear first.

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
