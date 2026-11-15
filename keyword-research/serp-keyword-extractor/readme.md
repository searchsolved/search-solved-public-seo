# SERP Keyword Extractor

## Features

Find related keywords by splitting the titles of the pages that rank on Google for a seed keyword, using the DataForSEO SERP API.

- Streamlit web interface
- Fetches the live Google organic results for your seed keyword (region and device selectable)
- Splits each ranking title into phrases and counts how many titles use each phrase
- Scores each phrase for similarity to the seed keyword
- Bar chart and CSV download
- Great for content expansion and dual optimisation

## Requirements

- DataForSEO account (sign up at [dataforseo.com](https://dataforseo.com/))
- Enter your DataForSEO login and password in the sidebar, or set the `DATAFORSEO_LOGIN` and `DATAFORSEO_PASSWORD` environment variables to prefill them

## Usage

```bash
pip install -r requirements.txt
streamlit run serp_keyword_extractor.py
```

1. Enter your DataForSEO login and password in the sidebar
2. Choose the region, device (desktop or mobile) and the number of results to analyse
3. Enter a seed keyword and click Submit

## Cost

Each search is one call to the DataForSEO SERP API (Google Organic, Live Advanced). DataForSEO bills per 10 results, so the default of 20 results is estimated at $0.004 per search. The estimate is shown in the sidebar before you run, and the cost DataForSEO reports is shown after each search.

## Output

| Column | Description |
| --- | --- |
| query | Your seed keyword |
| extracted_keywords | A phrase taken from the ranking titles |
| serp_frequency | Number of times the phrase appears across the ranking titles |
| similarity | Fuzzy partial match score (0 to 100) between the phrase and the seed keyword |

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
