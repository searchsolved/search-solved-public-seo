# Sample data: Content Duplication Finder

Demo data. The input is made up (pages for a fictional running and outdoor shop). The output is what the tool's matching code returned for it, unedited.

| File | What it is |
|---|---|
| `make_demo_input.py` | Writes the input file. It uses a fixed random seed, so it produces the same file each time |
| `demo_crawl_with_copy.csv` | Input: 15 pages with Address, H1-1 and Copy 1. One page has no copy and is skipped |
| `content_duplication_results.csv` | Output with Minimum Similarity Score set to 0.5: 8 rows in two clusters (five store pages built from one template, three near-identical guides) |
| `content_duplication_results_default.csv` | Output at the default setting of 0.9: 2 rows, the two guides that are almost word for word the same |

To reproduce: run `python make_demo_input.py`, start the app with `streamlit run ../content_duplication_finder.py`, upload `demo_crawl_with_copy.csv`, set the Minimum Similarity Score and press Find Duplicates.

These files were produced on 2026-11-15 by running the app's matching code directly on the input, outside the Streamlit interface, on branch fix/content-tools-wave0.
