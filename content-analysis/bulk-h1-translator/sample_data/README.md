# Sample data: Bulk H1 Translator

Demo data. The pages and headings are made up (regional sites of a fictional running and outdoor shop). The output is what the tool returned for them, unedited.

| File | What it is |
|---|---|
| `make_demo_input.py` | Writes the input file |
| `demo_h1s.csv` | Input: 10 rows (Address, H1-1, Language): two H1s each in French, German, Spanish and Italian, one already in English and one page with no H1 |
| `translated_h1s.csv` | Output: the same rows with a translated_h1 column |

To reproduce:

    python make_demo_input.py
    export OPENAI_API_KEY=your-key
    python ../bulk_h1_translator_cli.py --input demo_h1s.csv --output translated_h1s.csv --base-url https://api.openai.com/v1 --model gpt-4o-mini

All 10 rows came back without an error. Three things to note in the output:

- Capitalisation is not consistent: some translations are in sentence case ("Running shoes for women") and some in title case ("Trail Running Shoes").
- One translation is debatable: "Giacche antivento da corsa" became "Racing windbreakers", where "windproof running jackets" is closer.
- The H1 that was already in English came back with an apostrophe added ("Men's Running Shoes"), so "unchanged" is not exact. The page with no H1 came back blank.

Output produced on 2026-11-15 with gpt-4o-mini (called through OpenRouter), in 10 API calls, on branch fix/content-tools-wave0. A different run may word the translations differently.
