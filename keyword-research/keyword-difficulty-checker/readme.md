# Keyword Difficulty Checker

A Streamlit app to check keyword difficulty using multiple metrics: allintitle results, phrase match results, and SERP clustering.

## Features

- **Allintitle Check**: Find pages with exact keyword in title
- **Phrase Match**: Find pages with exact phrase match
- **SERP Clustering**: Group keywords by common ranking URLs
- **Question Filter**: Focus on question-type keywords
- **Multi-threaded**: Process keywords quickly with concurrent requests

## Requirements

```bash
pip install -r requirements.txt
```

## Setup

### Get a ValueSERP API Key

1. Sign up at [ValueSERP](https://www.valueserp.com/)
2. Get your API key from the dashboard
3. PAYG pricing with no expiration

## Usage

1. **Start the app:**
   ```bash
   streamlit run keyword_difficulty_checker.py
   ```

2. **Enter your ValueSERP API key** in the sidebar

3. **Upload a keyword file** (CSV with keyword column)

4. **Select the keyword column** and click Submit

5. **Download Excel results** with multiple sheets

## Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| Device | Mobile, Desktop, or Tablet | Desktop |
| Location | Country for search results | United States |
| Threads | Concurrent request threads | 10 |
| Common URLs | Min URLs to form cluster | 3 |
| Max Difficulty | Filter keywords by difficulty | 10 |
| Question Filter | Only process question keywords | Off |

## Output

Excel file with two sheets:

### Competitive Analysis
- Serp Cluster
- Keyword
- Search Results (total indexed pages)
- Quoted Results (phrase match)
- Allintitle Results
- Any original columns from your file

### Questions Only
- Filtered view of question-type keywords

## How It Works

1. **Search Types**: For each keyword, the app searches:
   - `keyword` - regular search
   - `"keyword"` - phrase match
   - `allintitle: keyword` - title match

2. **SERP Clustering**: Keywords are grouped if they share common ranking URLs

3. **Result Interpretation**:
   - Low allintitle = easier to rank in title
   - Low phrase match = less exact competition
   - Same cluster = can target with single page

## API Credits

Each keyword uses 3 API credits (regular + phrase + allintitle).

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)