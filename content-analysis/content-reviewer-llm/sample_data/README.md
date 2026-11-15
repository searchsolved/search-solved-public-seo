# Sample data: Content Reviewer

Demo data. The article is made up. The output is what the tool returned for it, unedited.

| File | What it is |
|---|---|
| `demo_article.txt` | Input: one made-up article of 388 words, "How to look after your running shoes", for a fictional running shop's blog; the same article used for the Gist Summary Generator sample |
| `review_pasted_content.md` | Output: the article with review notes added, then a summary, as the app's Download Review button saves it |

To reproduce: open the app, open "Or paste content directly" on the Single URL Review tab, paste the contents of `demo_article.txt` with the page title "How to look after your running shoes", and press Review Pasted Content. The run used the default review focus, SEO Content Quality.

What the output shows:

- The article comes back in full with seven notes placed after the paragraphs they refer to: 2 STRENGTH, 2 IMPROVE, 2 SEO and 1 MISSING. This run produced no ISSUE note.
- The notes are HTML comments (`<!-- IMPROVE: ... -->`), so they are visible in the downloaded file as text and hidden by anything that renders the Markdown.
- The tool turned the article's plain subheadings into Markdown headings, and its summary then recommends using H2 and H3 tags. Treat structural advice with care when the input is pasted text with no markup.
- The summary lists five priority improvements and three quick wins.

These files were produced on 2026-11-15 by calling the app's own function, outside the Streamlit interface, with gpt-4o-mini (called through OpenRouter), in 1 API call, on branch fix/content-tools-wave0. A different run will give different notes.
