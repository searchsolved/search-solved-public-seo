# Sample data: Keyword Extractor

Demo data. The article is made up. The output is what the tool returned for it, unedited.

| File | What it is |
|---|---|
| `demo_article.txt` | Input: one made-up article of 388 words, "How to look after your running shoes", for a fictional running shop's blog; the same article used for the Gist Summary Generator sample |
| `extracted_keywords.csv` | Output: two columns, internal_link_opportunities and new_page_ideas, as the app's Download Keywords button saves them |

To reproduce: open the app, paste the contents of `demo_article.txt` on the Single Content tab, enter the URL `https://www.yourstore.example/blog/how-to-look-after-running-shoes` and the H1 "How to look after your running shoes", and extract. The run used the default settings: 2 words minimum, up to 20 keywords per category, no anchor texts excluded.

The tool returned 6 internal link opportunities and 2 new page ideas. The two page ideas are sensible ("gait analysis at a running shop", "proper running socks"). The six link opportunities are the article's own subheadings ("dry them properly", "rotate two pairs"), which are poor anchor text for a link to another page. The tool keeps only phrases that appear word for word in the content, and this model chose headings to meet that rule. Review the list before using it; a larger model may choose better phrases.

These files were produced on 2026-11-15 by calling the app's own function, outside the Streamlit interface, with gpt-4o-mini (called through OpenRouter), in 1 API call, on branch fix/content-tools-wave0. A different run may return different phrases.
