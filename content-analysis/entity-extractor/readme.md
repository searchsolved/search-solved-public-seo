# Entity Extraction App

A multi-page Streamlit app for extracting named entities from various sources using the Dandelion.eu API.

## Features

Three extraction modes:

### 1. SERPs
Works in two steps. Google now routes result links through google.com/goto; DataForSEO's Standard queue turns those back into real URLs, but a search can take 25 minutes or more to finish, so you send the search now and collect the results later.
- **1. Send search:** enter a seed keyword, choose the region, device (desktop or mobile) and number of results pages, check the estimated cost and send it. Download the job file.
- **2. Collect results:** upload the job file (or use the job sent from the same session), set the entity accuracy and number of entities to display, and collect. If the search is still running, try again in a few minutes.
- Extract content from top-ranking pages (any result still returned as an unresolved google.com/goto link is skipped and counted in a warning)
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

### DataForSEO (Required for SERPs page only)
- Pay as you go; the SERPs page uses the Google Organic Standard queue (task_post, then task_get/advanced). Cost: `keywords x $0.0006 x (results per keyword / 10)`, so one keyword at 1 page (10 results) is about $0.0006 and at 10 pages (100 results) about $0.006. The estimate is shown before you send.
- Sign up at [dataforseo.com](https://dataforseo.com/)
- Enter your login and password in the sidebar, or set `DATAFORSEO_LOGIN` and `DATAFORSEO_PASSWORD` to prefill them

## Usage

1. **Start the app:**
   ```bash
   streamlit run Home.py
   ```

2. **Enter your API keys** in the sidebar (DataForSEO login and password for the SERPs page, Dandelion.eu key for every page)

3. **Select a page** from the sidebar:
   - SERPs - Send a Google search (step 1), then collect the results and extract entities (step 2)
   - CSV - Upload a keyword file
   - YouTube - Paste a video URL

4. **Download results** as CSV

## Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| Number of Pages | SERP pages to analyze (step 1, SERPs page) | 1 |
| Entity Accuracy | Confidence threshold (0-100%) (step 2, SERPs page) | 80% |
| Top Entities | Entities to display in chart (step 2, SERPs page) | 15 |

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
