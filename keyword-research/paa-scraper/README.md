# PAA Scraper

## Features

Collect the "People Also Ask" questions Google shows for your keywords, with the answer snippet and its source, using the DataForSEO SERP API (Google Organic, Live Advanced).

- Streamlit web interface + CLI version
- PAA click depth 1 to 4: DataForSEO clicks the questions to load more of them
- Location, language and device (desktop or mobile) selectable
- Export to CSV or Excel (the CLI also writes JSON)

## Output columns

| Column | Description |
| --- | --- |
| original_query | Your keyword |
| level | Depth of the question |
| parent_query | The query or question that produced this one |
| question | The PAA question |
| answer_snippet | The answer text shown under the question |
| source_url | URL of the answer's source page |
| source_title | Title of the answer's source page |

Google now routes some result links through google.com/goto. When DataForSEO's Live request cannot resolve one, `source_url` reads `(unresolved google.com/goto link)`. The question and answer text are not affected.

## Cost

About $0.002 per keyword, plus $0.00015 for each PAA click DataForSEO makes (refunded when it makes fewer clicks than the depth you set). The app and CLI estimate the base cost only.

## Requirements

- DataForSEO account (sign up at [dataforseo.com](https://dataforseo.com/))

```bash
pip install -r requirements.txt
streamlit run paa_scraper.py
python paa_scraper_cli.py --help
```

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
