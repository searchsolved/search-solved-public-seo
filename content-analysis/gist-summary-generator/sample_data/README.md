# Sample data: Gist Summary Generator

Demo data. The article is made up. The output is what the tool returned for it, unedited.

| File | What it is |
|---|---|
| `demo_article.txt` | Input: one made-up article of 388 words, "How to look after your running shoes", for a fictional running shop's blog |
| `summary.md` | Output: the "At a glance" summary, as the app's Download Summary button saves it |

To reproduce: open the app, paste the contents of `demo_article.txt` on the Single Content tab and generate a summary. The run used the app's default settings: 3 to 5 bullet points, "Concise (15-20 words)".

The summary has five bullet points of 16 or 17 words each, inside the range asked for, and each states a point from the article and does not describe the article.

These files were produced on 2026-11-15 by calling the app's own function, outside the Streamlit interface, with gpt-4o-mini (called through OpenRouter), in 1 API call, on branch fix/content-tools-wave0. A different run will give different wording.
