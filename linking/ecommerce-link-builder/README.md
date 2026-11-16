# Automatically Find Distributer Sites to Link to You

## Features

Easily identify 'Where to buy' type links on autopilot. Great for B2B sites.

For every brand in `brands.txt` (one per line), the script searches Google for "<brand> Stockists" through the DataForSEO SERP API and keeps the position 1 organic result, leaving out homepages and duplicate URLs.

## How it works

The searches go through the DataForSEO Standard queue, because it returns the real result URLs (faster requests now return many as unresolved google.com/goto links). A search can take 25 minutes or more to finish, so the script works in two steps:

1. `send` queues one search per brand and writes a job file.
2. `collect` reads the job file, fetches the finished searches and writes `brand_links_output.csv` (query, url, title, description). Searches still running are left out; run `collect` again later. Add `--wait` to check every 60 seconds until all are done (up to 90 minutes).

Any result that still comes back as a google.com/goto link is left out and counted in a warning.

## Usage

Set `DATAFORSEO_LOGIN` and `DATAFORSEO_PASSWORD` as environment variables (or edit the constants at the top of the script). Location (default United Kingdom, code 2826) and language (`en`) are also constants there. Run from the folder holding `brands.txt`:

```bash
python ecommerce_link_builder.py send                  # asks to confirm the estimated cost; add --yes to skip
python ecommerce_link_builder.py collect               # writes brand_links_output.csv
python ecommerce_link_builder.py collect --wait        # keep checking until every search is done
```

Options: `send --job FILE`, `collect --job FILE --output FILE` (job file default `brand_links_job.csv`).

## Cost

Estimated cost = keywords x $0.0006 x (results per keyword / 10). The script fetches 10 results per brand, so each brand costs about $0.0006. Collecting results is free.

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
