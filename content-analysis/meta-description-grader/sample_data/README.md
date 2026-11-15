# Sample data: Meta Description Grader

Demo data. The pages and descriptions are made up (a fictional running and outdoor shop). The output is what the tool returned for them, unedited.

| File | What it is |
|---|---|
| `make_demo_input.py` | Writes the input file |
| `demo_meta_descriptions.csv` | Input: 10 pages (url, title, h1, meta_description). The descriptions vary on purpose: hype, too short, too long, keyword lists and sound ones |
| `meta_description_scores.csv` | Output: url, meta_description, four scores from 0 to 10 and total_score out of 40 |

To reproduce: run `python make_demo_input.py`, open the app, upload `demo_meta_descriptions.csv` on the Bulk Analysis tab and map the URL and Meta Description columns.

All 10 descriptions were scored, with totals from 11 to 33 out of 40. One result is worth knowing before relying on the scores: the highest total (33) went to the description written as hype ("BEST mens running shoes!!! Shop NOW for AMAZING deals..."). The four criteria (emotional hook, benefit, active voice, urgency) reward that style, and nothing in them penalises capitals, exclamation marks or unsupported claims. The two-word and title-only descriptions scored lowest, as expected.

These files were produced on 2026-11-15 by calling the app's own scoring function on each row, outside the Streamlit interface, with gpt-4o-mini (called through OpenRouter), in 10 API calls, on branch fix/content-tools-wave0. A different run will give slightly different scores.
