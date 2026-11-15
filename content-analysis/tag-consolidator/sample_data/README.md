# Sample data: Tag Consolidator

Demo data. The tags are made up (product tags for a fictional running and outdoor shop). The output is what the tool returned for them, unedited.

| File | What it is |
|---|---|
| `make_demo_input.py` | Writes the input file |
| `demo_tags.csv` | Input: 48 rows (Primary Tag, Secondary Tag) across three product groups |
| `consolidated_tags.csv` | Output: the same rows with a Generic Tag column |

To reproduce:

    python make_demo_input.py
    export OPENAI_API_KEY=your-key
    python ../tag_consolidator_cli.py --input demo_tags.csv --output consolidated_tags.csv

The 48 detailed tags were grouped into 18 generic categories, and every row was mapped. Categories are chosen separately for each primary tag: Running Shoes has Cushioning, Fit, Performance, Support, Terrain and Traction, while Socks has Features, Length, Material and Packaging.

Output produced on 2026-11-15 with gpt-4o-mini (called through OpenRouter), in 3 API calls, on branch fix/content-tools-wave0. A different run may choose different category names.
