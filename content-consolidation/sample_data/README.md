# Sample data: Content Consolidation Analyser

Demo data. The input is made up (search results for a fictional running and outdoor shop). The output is what the tool returned for it, unedited.

| File | What it is |
|---|---|
| `make_demo_input.py` | Writes the input file. It uses a fixed random seed, so it produces the same file each time |
| `demo_serp_rows.csv` | Input: 30 queries, 300 rows, one row per query and ranking URL |
| `consolidation_results.csv` | Output: 49 rows, one per query per cluster, with 12 unclustered queries listed under NO_CLUSTER |

To reproduce:

    python make_demo_input.py
    python ../content_consolidation_analyzer.py --input demo_serp_rows.csv --output consolidation_results.csv

Output produced on 2026-11-14 on branch fix/content-tools-wave0.
