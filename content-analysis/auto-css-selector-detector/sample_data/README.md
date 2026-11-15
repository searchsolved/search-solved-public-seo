# Sample data: Auto CSS Selector Detector

Demo data. The page is one of the made-up running blog pages from the Content Extractor sample (`../../content-extractor/sample_data/demo_site/`), served locally. The output is what the tool printed, unedited.

| File | What it is |
|---|---|
| `detected_selector.txt` | Output: the URL, H1, the selector the model chose, the narrowed selector, the model's reasoning, where the text was extracted from, the extracted content and the internal links |

To reproduce (run the server in a second terminal and stop it afterwards):

    python -m http.server 8765 --directory ../../content-extractor/sample_data/demo_site
    export OPENAI_API_KEY=your-key
    python ../auto_css_selector_detector_cli.py --url http://localhost:8765/how-to-choose-running-shoes.html

The page has a header with navigation, an article, a sidebar of related links and a footer. What the output shows:

- The model chose `body > main:nth-of-type(1)`, which holds the article and the sidebar.
- The tool then narrowed that to `body > main:nth-of-type(1) > article:nth-of-type(1)` and extracted from there, so the sidebar text is left out. The Content Extractor sample keeps the sidebar on the same page.
- The H1 is printed in capitals in the extracted content.
- The internal links list is empty because the article itself contains no links.

Output produced on 2026-11-15 with gpt-4o-mini (called through OpenRouter), in 1 API call, on branch fix/content-tools-wave0. A different run may word the reasoning differently.
