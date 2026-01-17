# Content Hub Classification

Classify article content into content hub categories using OpenAI's GPT models. Automatically extracts primary topics, subtopics, and product recommendations.

## Features

- AI-powered content classification using GPT-4o-mini
- Extracts primary topic and content hub category
- Identifies key subtopics within the content
- Suggests recommended products based on content
- Structured JSON output with strict schema validation
- CSV export for easy integration with other tools

## Requirements

```bash
pip install -r requirements.txt
```

## Setup

### Set your OpenAI API Key

**Linux/Mac:**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

**Windows:**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**In Python:**
```python
import os
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
```

### Get an API Key

1. Sign up at [OpenAI](https://platform.openai.com/)
2. Navigate to API Keys section
3. Create a new secret key
4. Store it securely (never commit to version control!)

## Usage

```bash
python content_hub_classification.py
```

Or use in your own script:

```python
from content_hub_classification import analyze_article, save_analysis_to_dataframe

article_text = "Your article content here..."
result = analyze_article(article_text)

if result:
    print(result)
    save_analysis_to_dataframe(result, 'output.csv')
```

## Output Format

The tool returns structured JSON:

```json
{
  "content_analysis": {
    "primary_topic": "Sensor Connectors",
    "content_hub_category": "Industrial Electronics",
    "key_subtopics": [
      "M8 connectors",
      "M12 connectors",
      "Sensor cables"
    ],
    "recommended_products": [
      "Proximity sensors",
      "Photoelectric sensors"
    ]
  }
}
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `gpt-4o-mini` | OpenAI model to use |
| `SAVE_PATH` | `./content_analysis_output.csv` | Output CSV path |

## Cost

Uses GPT-4o-mini which is cost-effective for classification tasks. Check [OpenAI pricing](https://openai.com/pricing) for current rates.

## Security

**Never hardcode API keys!** Always use environment variables or secure credential management.

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
