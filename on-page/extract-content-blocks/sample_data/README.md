# Sample data: Content Block Extractor

Demo data. The pages are five of the made-up running blog pages from the Content Extractor sample (`../../../content-analysis/content-extractor/sample_data/demo_site/`), served locally. All five use the same template. The output is what the tool returned, unedited.

| File | What it is |
|---|---|
| `demo_urls.txt` | Input: 5 URLs on `http://localhost:8765`, one per line |
| `blocks/content_blocks_20261115_005805_d2543d4c.csv` | Output: the batch file, one row per block per page (url, name, xpath, notes) |
| `blocks/combined_output_with_frequency.csv` | Output: the same 17 rows with a frequency column, the number of pages in this run that share the XPath |

To reproduce (run the server in a second terminal and stop it afterwards):

    python -m http.server 8765 --directory ../../../content-analysis/content-extractor/sample_data/demo_site
    export ANTHROPIC_API_KEY=your-key
    python ../extract_content_blocks.py --input demo_urls.txt --output-dir blocks

All five pages were processed, giving 17 blocks: the article, its title, the body paragraphs and the related guides sidebar. Header, navigation and footer are removed before the page is sent to the model, so they are never listed.

One result shows a limit of the frequency count. The five pages share one template, but the model wrote the same block two ways: `//article/h1` on three pages and `//main/article/h1` on two. Both are correct. Because frequency counts exact XPath text, no XPath reaches 5 of 5. Read the frequency column as a lower bound and group equivalent XPaths by hand before choosing one for a crawler.

Output produced on 2026-11-15 with Claude Haiku 4.5 (called through OpenRouter), in 5 API calls, on branch fix/content-tools-wave0. A different run may name the blocks differently.
