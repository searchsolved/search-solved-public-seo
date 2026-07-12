# Fan-Out Query Explorer

Surfaces the sub-questions (fan-out queries) that AI platforms generate when answering queries in your topic space. Use this to discover content gaps and plan content that gets cited in AI responses.

## What are fan-out queries?

When an AI system (Google AI Overview, ChatGPT) answers a complex question, it internally breaks it into sub-questions to research. These are "fan-out queries". If your content answers those sub-questions, you are more likely to be cited in the AI's response.

This tool reveals those sub-questions so you can build content around them.

## How it works

Uses the DataForSEO LLM Mentions Search endpoint (`/v3/ai_optimization/llm_mentions/search/live`) to retrieve mentions for a keyword or domain, then extracts and deduplicates all fan-out queries, ranking them by frequency.

## Requirements

- Python 3.9+
- DataForSEO API credentials ([dataforseo.com](https://dataforseo.com))
- Estimated cost: ~$0.10 per request

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Streamlit App

```bash
streamlit run fan_out_query_explorer.py
```

### CLI

```bash
# By keyword
python fan_out_query_explorer_cli.py \
    --login your@email.com \
    --password your_api_password \
    --keyword "welding helmets"

# By domain
python fan_out_query_explorer_cli.py \
    --login your@email.com \
    --password your_api_password \
    --domain example.com

# With options
python fan_out_query_explorer_cli.py \
    --login your@email.com \
    --password your_api_password \
    --keyword "seo tools" \
    --platform chat_gpt \
    --location us \
    --language en \
    --limit 200 \
    --output my_fan_outs.csv
```

Environment variables `DATAFORSEO_LOGIN` and `DATAFORSEO_PASSWORD` are used as fallbacks if flags are not provided.

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--keyword` | - | Keyword or topic to explore (mutually exclusive with --domain) |
| `--domain` | - | Domain to explore mentions for |
| `--platform` | google | AI platform: `google` or `chat_gpt` |
| `--location` | uk | Location: uk, us, au, ca, de, fr, es, it, nl, in, br, jp |
| `--language` | en | Language: en, de, fr, es, it, nl, pt, ja |
| `--limit` | 100 | Max mention items to retrieve |
| `--include-subdomains` | True | Include subdomains (domain mode) |
| `--output` | fan_out_queries.csv | Output path for fan-out queries |
| `--output-parents` | parent_questions.csv | Output path for parent questions |

## Output

### Primary: Fan-Out Queries CSV

| Column | Description |
|--------|-------------|
| fan_out_query | The sub-question text |
| frequency | How many parent questions generated this sub-question |
| parent_questions | Semicolon-separated list of parent questions |

### Secondary: Parent Questions CSV

| Column | Description |
|--------|-------------|
| parent_question | The original question asked of the AI |
| fan_out_count | Number of fan-out queries it generated |
| ai_search_volume | AI search volume metric |
| source_count | Number of sources cited in the response |

## Difference from AI Visibility Checker

Both tools use the same DataForSEO endpoint, but with a different lens:

- **AI Visibility Checker**: focuses on *who* is cited (brand/domain visibility)
- **Fan-Out Query Explorer**: focuses on *what questions* the AI generates (content planning)

## Author

Lee Foot - [leefoot.com](https://www.leefoot.com)
