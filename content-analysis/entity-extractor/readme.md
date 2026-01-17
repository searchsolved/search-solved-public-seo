# Entity Extraction App

A multi-page Streamlit app for extracting named entities from various sources using the Dandelion.eu API.

## Features

Three extraction modes:

### 1. SERPs
- Search Google for a keyword
- Extract content from top-ranking pages
- Identify common entities across the SERP
- Visualize entity frequency

### 2. CSV
- Upload keyword/URL exports
- Extract entities from grouped keywords
- Batch process multiple URLs

### 3. YouTube
- Paste a YouTube URL
- Extract entities from video transcript
- Identify key topics discussed

## Requirements

```bash
pip install -r requirements.txt
```

## API Keys Required

### Dandelion.eu (Required for all pages)
- Free tier: 1,000 requests/day
- Sign up at [dandelion.eu](https://dandelion.eu/)

### ValueSERP (Required for SERPs page only)
- PAYG pricing, no expiration
- Sign up at [valueserp.com](https://www.valueserp.com/)

## Usage

1. **Start the app:**
   ```bash
   streamlit run Home.py
   ```

2. **Enter your API keys** in the sidebar

3. **Select a page** from the sidebar:
   - SERPs - Search and extract from Google results
   - CSV - Upload a keyword file
   - YouTube - Paste a video URL

4. **Download results** as CSV

## Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| Entity Accuracy | Confidence threshold (0-100%) | 80% |
| Number of Pages | SERP pages to analyze | 1 |
| Top Entities | Entities to display in chart | 15 |

## Output

CSV file with columns:
- entity - The extracted entity
- confidence - Confidence score (0-1)
- category - Entity type/category
- wiki_url - Wikipedia link
- most_frequent / # of mentions - Frequency count

## Use Cases

- **Content Optimization**: Find entities competitors mention
- **Keyword Research**: Discover related topics
- **Content Gaps**: Find entities you're not covering
- **Video Research**: Extract topics from YouTube content

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
