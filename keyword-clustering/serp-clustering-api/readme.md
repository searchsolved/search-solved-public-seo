# SERP Clustering API

A FastAPI service for clustering keywords based on common SERP URLs. Upload a CSV with one row per keyword and ranking URL (for example from the DataForSEO SERP API) and receive JSON clusters.

## Features

- REST API for SERP clustering
- Configurable common URL threshold
- Configurable keyword and URL column names (defaults: `keyword` and `url`)
- JSON response format
- Built-in API documentation (Swagger UI)

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### Start the server

```bash
uvicorn app:app --reload
```

Or run directly:

```bash
python app.py
```

Server starts at `http://localhost:8000`

### API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Endpoints

#### GET /
Returns API information.

#### POST /cluster-serps
Upload a CSV file for clustering.

**Parameters:**
- `file`: CSV file (required, multipart upload) - one row per keyword and ranking URL
- `keyword_column`: string (optional form field, default: `keyword`) - name of the keyword column
- `url_column`: string (optional form field, default: `url`) - name of the ranking URL column
- `common_urls`: int (optional query parameter, default: 4) - Minimum common URLs to form cluster

**CSV Format:**
One row per keyword and ranking URL, for example:

```csv
keyword,url
best running shoes,https://example.com/running-shoes
best running shoes,https://example.org/reviews/running
top running shoes,https://example.com/running-shoes
```

Other columns are ignored. If your file uses different names, pass them as `keyword_column` and `url_column`. A missing column returns HTTP 400 with the list of columns found in the file.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/cluster-serps?common_urls=4" \
     -F "file=@serp_results.csv"

# Custom column names
curl -X POST "http://localhost:8000/cluster-serps?common_urls=4" \
     -F "file=@serp_results.csv" \
     -F "keyword_column=Keyword" \
     -F "url_column=URL"
```

### Producing the CSV

The file is the top organic results for each keyword, one row per ranking URL. The top 10 is usually enough for clustering. Ways to make it:

- **DataForSEO SERP API (script).** Call the Google Organic Live Advanced endpoint for each keyword and keep the `url` of each item where `type` is `organic`. Costs about $0.002 per keyword for the top 10. A minimal script:

  ```python
  import csv, os, time, requests

  keywords = ["best running shoes", "top running shoes"]
  auth = (os.environ["DATAFORSEO_LOGIN"], os.environ["DATAFORSEO_PASSWORD"])

  with open("serp_results.csv", "w", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerow(["keyword", "url"])
      for kw in keywords:
          resp = requests.post(
              "https://api.dataforseo.com/v3/serp/google/organic/live/advanced",
              auth=auth,
              json=[{"keyword": kw, "location_code": 2826, "language_code": "en",
                     "device": "desktop", "depth": 10}],
              timeout=60,
          ).json()
          items = (resp["tasks"][0].get("result") or [{}])[0].get("items") or []
          for item in items:
              if item.get("type") == "organic":
                  writer.writerow([kw, item["url"]])
          time.sleep(2)
  ```

- **Share of Voice tool** (`reporting/share-of-voice`, live DataForSEO mode). The Excel download's **Raw Data** sheet has `keyword` and `url` columns (top 10 results). Save that sheet as CSV and upload it with the default column names.
- **SERP Crossover Analyzer** (`competitive-analysis/serp-crossover-analyzer`). The Excel report's **Raw SERP Data** sheet has `Keyword` and `URL` columns. Save it as CSV and pass `keyword_column=Keyword` and `url_column=URL`.
- Any rank tracker export with one row per keyword and ranking URL, with the column names passed as above.

**Example Response:**
```json
{
  "serp_clusters": [
    {
      "cluster": 1,
      "similar_query": "running shoes",
      "queries": ["best running shoes", "top running shoes"],
      "common_urls_counts": [5, 4],
      "common_urls": ["https://example.com/shoes", "..."]
    }
  ],
  "total_clusters": 1,
  "common_urls_threshold": 4
}
```

## Integration

### Python
```python
import requests

files = {'file': open('serp_results.csv', 'rb')}
response = requests.post(
    'http://localhost:8000/cluster-serps',
    files=files,
    data={'keyword_column': 'keyword', 'url_column': 'url'},
    params={'common_urls': 4}
)
clusters = response.json()
```

### JavaScript
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('keyword_column', 'keyword');
formData.append('url_column', 'url');

const response = await fetch('http://localhost:8000/cluster-serps?common_urls=4', {
    method: 'POST',
    body: formData
});
const clusters = await response.json();
```

## Deployment

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production
For production, use gunicorn with uvicorn workers:
```bash
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)