# Sample data: Reading Score Analyser

The input is a list of ten public pages on python.org, the same list used for the Page Intent Classifier sample. The output is what the tool returned for them, unedited. It holds scores only, not the page text.

| File | What it is |
|---|---|
| `demo_urls.txt` | Input: 10 URLs, one per line |
| `reading_scores.csv` | Output: url and 11 measures per page (flesch_reading_ease, flesch_kincaid_grade, gunning_fog, smog_index, automated_readability_index, coleman_liau_index, word_count, sentence_count, avg_sentence_length, difficult_words, reading_time_mins) |

To reproduce:

    python ../reading_score_analyzer_cli.py --urls demo_urls.txt --output reading_scores.csv --delay 2

All 10 pages were scored. The output shows where the scores can be trusted and where they cannot:

- Pages of ordinary prose score in a believable range: Flesch Reading Ease from 43 to 68 for seven of the ten pages.
- Pages that are mostly lists score nonsense. The jobs page scored -161.9 with a grade level of 82.3, and the events page -11.21. A list has no full stops, so the text is read as 2 or 3 very long sentences (187 words per sentence on the jobs page). Check sentence_count and avg_sentence_length before acting on a low score, and leave listing pages out.
- Flesch Reading Ease is usually described as a 0 to 100 scale. The formula has no lower limit, as the two pages above show.

The command line version writes the 12 columns above. The readability_category column exists only in `reading_score_analyzer.py`.

Output produced on 2026-11-15 on branch fix/content-tools-wave0, with a 2 second pause between pages. The pages may have changed since.
