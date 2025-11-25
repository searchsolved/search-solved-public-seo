# Template Fingerprinting Tool

Automatically identify and classify pages by their template type using HTML structure analysis and machine learning clustering.

## Features

- Analyzes HTML structure (tags, classes, IDs, meta tags)
- Uses TF-IDF vectorization for feature extraction
- K-Means clustering to identify template patterns
- Identifies common structural patterns per cluster
- Exports results with page type classifications

## Use Cases

- Identify different page templates on a website (PDP, PLP, blog, etc.)
- Audit template usage across large sites
- Find pages with unusual/broken templates
- Group pages for template-specific SEO recommendations

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

1. Export URLs from Screaming Frog (or create a CSV with an `Address` column)
2. Update configuration variables in the script:
   - `INPUT_FILE`: Path to your CSV file
   - `OUTPUT_FILE`: Where to save results
   - `N_CLUSTERS`: Number of template types to identify
3. Run the script:

```bash
python template_fingerprinting.py
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_FILE` | `./urls.csv` | CSV file with 'Address' column |
| `OUTPUT_FILE` | `./classified_urls.csv` | Output file path |
| `N_CLUSTERS` | `5` | Number of template types to detect |
| `TIMEOUT` | `10` | HTTP request timeout in seconds |

## Output

The script generates a CSV file with:
- Original URL data
- `Cluster`: Numeric cluster ID
- `Page Type`: Human-readable type label (Type 0, Type 1, etc.)

Console output includes top features for each cluster to help identify what each template type represents.

## How It Works

1. **Feature Extraction**: For each URL, the script fetches HTML and extracts:
   - Tag counts (e.g., `div:15`, `article:1`)
   - CSS class names
   - ID attributes
   - Meta tag properties

2. **Vectorization**: Features are converted to TF-IDF vectors

3. **Clustering**: K-Means groups similar page structures

4. **Analysis**: Top features per cluster help identify template types

## Author

**Lee Foot** - eCommerce SEO Consultant

- Website: [leefoot.com](https://www.leefoot.com)
- Twitter/X: [@LeeFootSEO](https://x.com/LeeFootSEO)
- LinkedIn: [lee-foot](https://www.linkedin.com/in/lee-foot/)
- Contact: [Get in touch](https://www.leefoot.com/contact)
