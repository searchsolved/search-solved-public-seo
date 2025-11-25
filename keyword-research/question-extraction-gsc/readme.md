# Question Extraction from Google Search Console

Extract question-type keywords from your Google Search Console data to identify informational content opportunities.

## Features

- Connects directly to Google Search Console API
- Pattern matching for question-type queries
- Identifies "how to", "what is", "best", "vs" and similar query patterns
- Exports to CSV sorted by impressions

## Requirements

```bash
pip install pandas
pip install git+https://github.com/joshcarty/google-searchconsole
```

## Setup

### 1. Create Google Cloud Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable the Search Console API
4. Create OAuth 2.0 credentials (Desktop app)
5. Download credentials as `client_secrets.json`

### 2. Configure the Script

Update these variables in the script:

```python
DOMAIN = "https://yoursite.com/"  # Your GSC property
DAYS = 360  # Lookback period
CLIENT_SECRETS_PATH = "path/to/client_secrets.json"
```

### 3. Run

```bash
python extract_questions_from_gsc.py
```

On first run, you'll be prompted to authenticate via browser.

## Pattern Matching

The script uses regex patterns to identify question-type queries:

**Detected patterns include:**
- Question words: who, what, when, where, why, how
- Action queries: installing, fitting, measuring, comparing
- Comparison queries: vs, alternatives, difference
- Intent signals: best, cost, price, types

## Output

CSV file with columns:
- `query`: The search query
- `clicks`: Number of clicks
- `impressions`: Number of impressions
- `ctr`: Click-through rate
- `position`: Average position

## Customization

Edit `QUESTION_PATTERN` regex to customize which queries are matched. A looser pattern is commented in the code for reference.

## Author

**Lee Foot** - eCommerce SEO Consultant

- Website: [leefoot.com](https://www.leefoot.com)
- Twitter/X: [@LeeFootSEO](https://x.com/LeeFootSEO)
- LinkedIn: [lee-foot](https://www.linkedin.com/in/lee-foot/)
- Contact: [Get in touch](https://www.leefoot.com/contact)
