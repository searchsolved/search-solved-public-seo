# Sample data: Review Sentiment Extractor

Demo data. The reviews are made up (a fictional running and outdoor shop); they are the same 40 reviews used for the Review Tagger sample. The output is what the tool returned for them, unedited.

| File | What it is |
|---|---|
| `demo_reviews.csv` | Input: 40 reviews (order_id, Review). Written by `../../review-tagger/sample_data/make_demo_input.py` |
| `review_sentiments.csv` | Output: the same rows with id, positive, negative and sentiment |

To reproduce:

    python ../review_sentiment_extractor_cli.py --api-key your-key --input demo_reviews.csv --output review_sentiments.csv --review-column Review --id-column order_id --context "running and outdoor clothing and footwear"

The run classed 20 reviews as negative, 14 as positive and 6 as mixed. The positive and negative columns hold the points the model pulled from each review, and are empty where a review had none.

Output produced on 2026-11-15 with gpt-4o-mini (called through OpenRouter), in 8 API calls of 5 reviews each, on branch fix/content-tools-wave0. A different run will give slightly different wording.
