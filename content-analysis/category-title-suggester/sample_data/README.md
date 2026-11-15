# Sample data: Category Title Suggester

Demo data. The input is made up (category pages and Search Console rows for a fictional running and outdoor shop). The output is what the tool returned for it, unedited.

| File | What it is |
|---|---|
| `make_demo_input.py` | Writes the input files. It uses a fixed random seed, so it produces the same files each time |
| `demo_pages.csv` | Input: 9 pages with their titles (Address, Title 1) |
| `demo_gsc.csv` | Input: 64 Search Console rows (page, query, clicks) |
| `title_suggestions.csv` | Output: 47 rows, one per page and suggested query, for 8 category pages |

To reproduce:

    python make_demo_input.py
    python ../category_title_suggester_cli.py --crawl demo_pages.csv --gsc demo_gsc.csv --brand "your store" --output title_suggestions.csv

The command line version writes four columns (page, query, clicks, current_title). The Streamlit app adds a kw_source column.

Output produced on 2026-11-14 on branch fix/content-tools-wave0.
