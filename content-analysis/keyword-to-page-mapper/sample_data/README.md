# Sample data: Keyword-to-Page Mapper

Demo data. The input is made up (pages and keywords for a fictional running and outdoor shop). The output is what the tool returned for it, unedited.

| File | What it is |
|---|---|
| `make_demo_input.py` | Writes the input files. It uses a fixed random seed for the volumes, so it produces the same files each time |
| `demo_pages.csv` | Input: 14 pages (Address, H1-1) |
| `demo_keywords.csv` | Input: 32 keywords with a volume. 22 have a page that suits them, 10 have none |
| `keyword_mapping.csv` | Output: 18 keywords mapped to a page (Keyword, Matched H1, Similarity, URL, Volume) |
| `keyword_mapping_gaps.csv` | Output: 14 keywords with no match at or above the threshold (Keyword, Volume) |

To reproduce:

    pip install polyfuzz sentence-transformers
    python make_demo_input.py
    python ../keyword_to_page_mapper_cli.py --pages demo_pages.csv --keywords demo_keywords.csv --output keyword_mapping.csv

What the output shows, at the default threshold of 0.75 with the `all-MiniLM-L6-v2` model:

- Most keywords land on the right page ("ladies running shoes" on Womens Running Shoes at 0.95, "hiking boots" on Walking Boots at 0.85).
- Some matches are wrong or debatable: "cycling shorts" is mapped to Running Shorts (0.81) and "kids walking boots" to Walking Boots (0.77), although the shop has no page for either.
- Six keywords that do have a suitable page fall just under the threshold and are listed as gaps, for example "waterproof walking boots" and "hiking backpack". Lower the threshold to catch them, and review the matches either way.

The command line version writes five columns. The Streamlit app adds Opportunity Size, the number of keywords mapped to each URL.

Output produced on 2026-11-15 with PolyFuzz 0.4.3, on branch fix/content-tools-wave0.
