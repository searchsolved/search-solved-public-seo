# Sample data: Content Repurposer

Demo data. The article is made up. The output is what the tool returned for it, unedited.

| File | What it is |
|---|---|
| `demo_article.txt` | Input: one made-up article of 388 words, "How to look after your running shoes", for a fictional running shop's blog; the same article used for the Gist Summary Generator sample |
| `twitter_thread.txt` | Output: a thread of 7 posts, 120 to 253 characters each |
| `linkedin_post.txt` | Output: a post of 225 words |
| `instagram_carousel.txt` | Output: 10 slides, each with a headline and body |
| `reddit_post.txt` | Output: a title, three suggested subreddits and a post (348 words in all) |

To reproduce: open the app, choose Paste Content, enter the title "How to look after your running shoes", paste the contents of `demo_article.txt`, press Social Only (which selects these four formats) and repurpose. The run used the default tone, Professional. The app saves each format as `<format>.txt` and offers all of them in one ZIP.

All four formats were produced and each follows its layout. Two things to check before publishing anything like this: the model adds emoji and sales phrasing that the article does not have, and the Reddit output suggests subreddits by name without checking that they exist.

These files were produced on 2026-11-15 by calling the app's own function, outside the Streamlit interface, with gpt-4o-mini (called through OpenRouter), in 4 API calls, one per format, on branch fix/content-tools-wave0. A different run will give different wording.
