# Sample data: Review Tagger

Demo data. The reviews are made up (a fictional running and outdoor shop). The output is what the tool returned for them, unedited.

| File | What it is |
|---|---|
| `make_demo_input.py` | Writes the input file. It uses a fixed random seed, so it produces the same file each time |
| `demo_reviews.csv` | Input: 40 reviews (order_id, Review) |
| `tagged_reviews.csv` | Output: the same rows with a Tag column |

To reproduce:

    python make_demo_input.py
    export OPENAI_API_KEY=your-key
    python ../review_tagger_cli.py --input demo_reviews.csv --output tagged_reviews.csv

All 40 reviews were tagged, with 11 different tags. Because the tags are free-form, some mean nearly the same thing (Sizing and Fit, Refund and Returns). The Tag Consolidator can group them.

Output produced on 2026-11-15 with gpt-4o-mini (called through OpenRouter), in 2 API calls, on branch fix/content-tools-wave0. A different run will give slightly different tags.
