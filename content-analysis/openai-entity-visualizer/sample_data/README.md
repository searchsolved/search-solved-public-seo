# Sample data: AI Entity Visualiser

Demo data. The text is made up, for a fictional running blog, though the people, places, events and organisations named in it are real. The output is what the tool returned, unedited.

| File | What it is |
|---|---|
| `demo_text.txt` | Input: 328 words. The six demo pages from the Entity Extractor sample (`../../entity-extractor/sample_data/demo_articles.csv`) joined into one text |
| `extracted_entities.csv` | Output: 49 entities with entity, spacy_label, descriptive_tag, wikipedia_url, context and count, as the app's Download CSV button saves them |
| `circle_packing_chart.html` | Output: the circle chart, as the app's Download D3 Visualization button saves it. It loads D3 from d3js.org, so it needs a connection to display |

To reproduce: open the app, paste the contents of `demo_text.txt` and press Process Text. The run used the app's default settings: gpt-4o-mini, 1000 tokens per batch.

What the output shows:

- 49 entities in six groups: 18 GPE, 17 ORG, 6 EVENT, 6 PERSON, 1 AWARD and 1 PROGRAM. The model labels brands correctly that the small spaCy model gets wrong on the same text (New Balance, Polar and Strava are all ORG here).
- The label column is not limited to spaCy's label set: AWARD and PROGRAM are the model's own, and mountains and a national park are labelled GPE.
- Counts are matched without regard to case, so a short name that is also an ordinary word is over-counted. "On" (the shoe brand) is counted 3 times; it appears once, and the other two are the word "on".
- The Wikipedia URLs are written by the model and are not checked by the tool. Check them before use.
- The context column holds a list of every place the entity appears, in one cell per entity.

These files were produced on 2026-11-15 by calling the app's own processing and chart functions on the text, outside the Streamlit interface, with gpt-4o-mini (called through OpenRouter), in 1 API call, on branch fix/content-tools-wave0. A different run may find a slightly different set of entities.
