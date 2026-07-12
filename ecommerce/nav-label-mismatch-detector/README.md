# Navigation Label Mismatch Detector

## Features

Find navigation links whose anchor text does not match the destination page's H1 or the primary keyword from its page title, using two Screaming Frog exports.

- Streamlit web interface + CLI version
- Identifies navigation links (rows where the alt text matches the anchor text)
- Compares each navigation label against the destination page's H1
- Compares each navigation label against the page title's primary keyword (the part before the separator, e.g. "Widgets | Example Store" on example.com becomes "Widgets")
- Flags mismatched labels for review
- Export results to CSV

## Required Screaming Frog Exports

Both exports should come from the same crawl.

**1. All Inlinks** (Bulk Export > Links > All Inlinks)

Required columns:

- `Source`
- `Destination`
- `Alt Text`
- `Anchor`

**2. Internal HTML** (Internal tab > filter HTML > Export)

Required columns:

- `Address`
- `H1-1`
- `Title 1`

## Usage

**Streamlit app:**

```bash
pip install -r requirements.txt
streamlit run nav_label_mismatch_detector.py
```

Upload both exports, check the column mapping and click "Find Mismatches". The Streamlit app also lets you change the title separator and remap columns if your export headers differ.

**CLI:**

```bash
python nav_label_mismatch_detector_cli.py --inlinks all_inlinks.csv --internal internal_html.csv --output results.csv
```

The CLI expects the standard Screaming Frog column names listed above and uses `|` as the title separator.

## How It Works

1. Navigation links are identified as inlink rows where the alt text matches the anchor text, which is typical of image-based navigation and menu templates.
2. Each navigation link is joined to its destination page's H1 and title from the Internal HTML export.
3. Duplicate anchor/H1/title combinations are removed.
4. The primary keyword is taken from the page title (everything before the separator).
5. Labels are compared case-insensitively against the H1 and the title keyword, and any label that fails either comparison is flagged as a mismatch.

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
