# Sample data: Competitor Content Gap Finder

Demo data. The input is made up (product titles for a fictional running and outdoor shop and two competitors). The output is what the tool returned for it, unedited.

| File | What it is |
|---|---|
| `make_demo_input.py` | Writes the input files. It uses a fixed random seed, so it produces the same files each time |
| `source_products.csv` | Input: 24 of your product titles (mpn, h1, url) |
| `competitor_a.csv`, `competitor_b.csv` | Input: 17 and 12 competitor titles for the same MPNs |
| `content_gaps.csv` | Output: 22 rows, one per product that at least one competitor also lists |

To reproduce:

    python make_demo_input.py
    python ../competitor_gap_finder.py --source source_products.csv --compare "competitor_*.csv" --output content_gaps.csv

Output produced on 2026-11-14 on branch fix/content-tools-wave0.
