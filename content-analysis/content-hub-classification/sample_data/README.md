# Sample data: Content Hub Classifier

Demo data. The articles are made up (a fictional running blog). The output is what the tool returned for them, unedited.

| File | What it is |
|---|---|
| `make_demo_input.py` | Writes the input file |
| `demo_articles.csv` | Input: 10 short articles (url, content) on four subjects: shoes, training, food and drink, injuries |
| `content_hub_results.csv` | Output: Article (first 200 characters), Primary Topic, Content Hub Category, Key Subtopics, Recommended Products, Status |

To reproduce: run `python make_demo_input.py`, open the app (`content_hub_classification_app.py`), choose "Batch Upload (CSV)", upload `demo_articles.csv` and choose the `content` column.

All 10 articles were classified. One result shows a limit of the tool: the 10 articles came back with 10 different categories, although they cover four subjects. The three shoe articles, for example, became "Footwear Guide", "Footwear Care and Maintenance" and "Running Footwear". Each article is classified on its own, with no list of categories to choose from and an instruction to avoid broad ones, so related articles are not given the same name. To build hubs from the output, group the categories afterwards (the Tag Consolidator does this kind of grouping).

The output does not carry the url column from the input. Rows stay in input order.

These files were produced on 2026-11-15 by calling the app's own batch function on the input, outside the Streamlit interface, with gpt-4o-mini (called through OpenRouter), in 10 API calls, on branch fix/content-tools-wave0. A different run may choose different names.
