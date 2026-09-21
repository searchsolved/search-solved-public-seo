# PPC vs Organic Overlap

Compare a Google Ads search terms report with a Google Search Console queries export to see where paid and organic overlap, and how much ad spend sits on terms the site already ranks well for organically.

## Features

- Venn diagram of paid vs organic keyword overlap
- Savings opportunities view: ad spend on terms ranking at or above a configurable organic position, sorted by cost
- Side-by-side paid and organic stats for every overlapping term (clicks, impressions, CTR, cost, conversions, conversion value, campaigns)
- Paid-only view (landing page and content gaps) and organic-only view (paid expansion candidates)
- Brand term tagging with brand / non-brand filtering
- Handles raw exports as-is: report preamble rows, Total footer rows, formatted numbers, UTF-8 and UTF-16 files, excluded search terms
- CSV download for every view
- Bundled synthetic sample dataset (fictional hotel) for a quick demo

## Inputs

- **GSC queries export**: Search Console > Performance > Export > Queries tab (or an API pull with a query column)
- **Google Ads search terms report**: Google Ads > Insights and reports > Search terms > Download > CSV

Use the same property, market, and date range for both exports so the comparison is like for like.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Switch on "Use sample data" in the sidebar to explore the app without uploads.

## Reading the results

The savings opportunities tab is the headline: terms where you pay for ads and already hold a strong organic position. Brand terms ranking number one organically are usually the first candidates for a spend reduction test. Reducing spend does not guarantee organic recovers every paid click, so validate with an incrementality or geo holdout test before making permanent changes.

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
