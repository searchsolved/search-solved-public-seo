# N-gram SERP Extractor

## Features

Extract page titles and content n-grams from the Google results for a seed keyword, using the DataForSEO SERP API, for title optimisation and content coverage analysis.

- Streamlit web interface
- DataForSEO SERP API (Google Organic, Standard queue)
- Region (10 markets), device (desktop or mobile) and number of result pages selectable
- Fetches each ranking page with trafilatura and extracts its title and main content
- Most common content bigrams (stop words removed)
- Words appearing in more than one page title
- Export each table to CSV

## How it works: two steps

Google now routes result links through google.com/goto. DataForSEO's Standard queue turns those back into real URLs, but a search can take 25 minutes or more to finish. So the tool works in two steps:

1. **Send search:** enter a seed keyword, choose the region, device and number of pages, check the estimated cost and click **Send search**. Download the job file (`ngram_serp_job.csv`).
2. **Collect results** (about 30 minutes later): upload the job file (or use the job kept in the open page) and click **Collect results**. The tool fetches the ranking pages and shows the bigrams, title keywords and extracted titles. If the search is still running, it says so; try again in a few minutes.

Any result DataForSEO still returns as an unresolved google.com/goto link is skipped (never fetched) and counted in a warning.

## Requirements

- DataForSEO account (sign up at [dataforseo.com](https://dataforseo.com/))
- Enter your login and password in the sidebar (prefilled from `DATAFORSEO_LOGIN` and `DATAFORSEO_PASSWORD` when set)

```bash
pip install -r requirements.txt
streamlit run ngram_serp_extractor.py
```

## Cost

DataForSEO bills the Standard queue per 10 results, so the estimate is:

`keywords x $0.0006 x (results per keyword / 10)`

This tool sends one keyword per search, with 10 results per page, so one page costs about $0.0006 and ten pages about $0.006. Collecting the results is normally free. The app shows the estimate before sending and the cost DataForSEO reports after.

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
