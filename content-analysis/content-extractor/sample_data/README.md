# Sample data: Content Extractor

Demo data. The pages are made up (a fictional running blog) and served from this folder, so the run needs no outside site. The output is what the tool returned for them, unedited.

| File | What it is |
|---|---|
| `make_demo_site.py` | Writes the demo site and the URL list |
| `demo_site/` | Input: 10 HTML pages. Each has a script, a style block, a header with navigation, the article, a sidebar of related links and a footer |
| `demo_urls.csv` | Input: the 10 page URLs on `http://localhost:8765` (url column) |
| `extracted_content.csv` | Output: URL, Title, H1, Content, Content_Length, Status, Error |

To reproduce:

    python make_demo_site.py
    python -m http.server 8765 --directory demo_site
    python ../content_extractor_cli.py --input demo_urls.csv --output extracted_content.csv

(Run the server in a second terminal and stop it afterwards.)

All 10 pages were fetched, with 441 to 539 characters of content each. What the output shows about the extraction:

- The script, style block, header, navigation and footer are removed.
- The page title is not removed, so every Content value starts with the title tag text, followed by the H1.
- The sidebar is kept: each Content value ends with "Related guides" and the three link texts. The tool removes `script`, `style`, `nav`, `footer` and `header` elements only, so anything else on the page (sidebars, cookie notices, related links) stays in the text.
- Rows are written in the order the pages finish, not the order of the input file (three pages are fetched at a time by default).

Output produced on 2026-11-15 on branch fix/content-tools-wave0.
