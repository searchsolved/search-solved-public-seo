# SERP Clustering API

A FastAPI service for clustering keywords based on common SERP URLs. Upload ValueSERP export data and receive JSON clusters.

## Features

- REST API for SERP clustering
- Configurable common URL threshold
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
- `file`: CSV file (required) - ValueSERP export
- `common_urls`: int (optional, default: 4) - Minimum common URLs to form cluster

**CSV Format:**
Must contain columns:
- `search.q` - The search query
- `result.organic_results.link` - The ranking URL

**Example Request:**
```bash
curl -X POST "http://localhost:8000/cluster-serps?common_urls=4" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@Batch_Results.csv"
```

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

files = {'file': open('Batch_Results.csv', 'rb')}
response = requests.post(
    'http://localhost:8000/cluster-serps',
    files=files,
    params={'common_urls': 4}
)
clusters = response.json()
```

### JavaScript
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

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

- Website: [leefoot.com](https://www.leefoot.com)
- Twitter/X: [@LeeFootSEO](https://x.com/LeeFootSEO)
- LinkedIn: [lee-foot](https://www.linkedin.com/in/lee-foot/)
- Contact: [Get in touch](https://www.leefoot.com/contact)
