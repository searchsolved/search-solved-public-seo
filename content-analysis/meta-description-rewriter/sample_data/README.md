# Sample data: Meta Description Rewriter

Demo data. The pages and descriptions are made up (a fictional running and outdoor shop); they are the same 10 pages used for the Meta Description Grader sample. The output is what the tool returned for them, unedited.

| File | What it is |
|---|---|
| `demo_meta_descriptions.csv` | Input: 10 pages (url, title, h1, meta_description). Written by `../../meta-description-grader/sample_data/make_demo_input.py` |
| `rewritten_meta_descriptions.csv` | Output: url, original_meta, original_length, rewritten_desktop, desktop_length, rewritten_mobile, mobile_length |

To reproduce: open the app, upload `demo_meta_descriptions.csv` on the Bulk Rewrite tab and map the URL, Meta Description, H1 and Title columns. The run used the app's default settings: Professional tone, no exclamation marks, no hyperbole, 160 characters for desktop and 130 for mobile.

What the output shows:

- All 10 descriptions were rewritten. The desktop versions are 127 to 158 characters, all within the 160 limit, and none contains an exclamation mark.
- The mobile version is not a second rewrite. It is the desktop text cut at a word boundary with "..." added, so 9 of the 10 mobile versions end in "...". The tenth was already under 130 characters.
- The rewrites are safe but similar to each other (most open with "Explore" or "Discover"), and some drop a specific detail from the original. Review them before use.

These files were produced on 2026-11-15 by calling the app's own rewrite and length functions on each row, outside the Streamlit interface, with gpt-4o-mini (called through OpenRouter), in 10 API calls, on branch fix/content-tools-wave0. A different run will give different wording.
