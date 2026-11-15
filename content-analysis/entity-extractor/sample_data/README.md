# Sample data: Entity Extractor (spaCy version)

Demo data. The input pages are made up, for a fictional running blog, though the people, places, events and organisations named in them are real. The output is what the tool returned, unedited.

| File | What it is |
|---|---|
| `make_demo_input.py` | Writes the input file |
| `demo_articles.csv` | Input: 6 short pages (url, content) |
| `entity_counts.csv` | Output: 51 unique entities with Entity, Label and Count |

To reproduce:

    pip install spacy
    python -m spacy download en_core_web_sm
    python make_demo_input.py
    python ../entity_extractor_cli.py --input demo_articles.csv --id-col url --output entity_counts.csv

The output shows what the small model gets right and wrong. It finds the brands, people and places, but labels some of them incorrectly: Strava as a person, New Balance and Polar as places, Scafell Pike as an organisation. The larger models (`en_core_web_md`, `en_core_web_lg`) are more accurate and slower.

Output produced on 2026-11-15 with spaCy 3.8.18 and `en_core_web_sm`, on branch fix/content-tools-wave0. This sample is for the spaCy version of the tool (`entity_extractor.py` and `entity_extractor_cli.py`), not the Dandelion app in the same folder.
