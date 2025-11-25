# Automatic Category Page Suggester

A Streamlit app that analyzes your crawl data to suggest new category pages based on your product inventory and search demand.

## Features

- Analyzes product H1s to generate category suggestions via n-grams
- Matches suggestions to existing categories using fuzzy matching
- Optional Keywords Everywhere integration for search volume data
- Filters suggestions based on similarity to existing pages
- Export results to CSV

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

1. **Start the app:**
   ```bash
   streamlit run category_generator.py
   ```

2. **Upload files:**
   - `inlinks.csv` - Internal links export from Screaming Frog
   - `internal_html.csv` - HTML export from Screaming Frog

3. **Map columns:**
   - Select which column identifies product pages
   - Select which column identifies category pages

4. **Configure settings:**
   - Minimum product matches
   - Similarity threshold
   - Search volume filters (if using Keywords Everywhere)

5. **Download results**

## Input Files

### inlinks.csv
Export from Screaming Frog: Bulk Export > Links > All Inlinks

### internal_html.csv
Export from Screaming Frog: Bulk Export > All > Internal HTML

## Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| Min Product Match (Exact) | Minimum products a keyword must match | 3 |
| Min Product Match (Fuzzy) | Minimum fuzzy matches required | 3 |
| Min Similarity | Max similarity to existing category (lower = more different) | 96 |
| Min CPC | Minimum cost-per-click filter | 0 |
| Min Search Volume | Minimum monthly search volume | 100 |

## Keywords Everywhere Integration

Optionally add your Keywords Everywhere API key to get search volume and CPC data. Get your key at [keywordseverywhere.com](https://keywordseverywhere.com/).

## Output

CSV file with columns:
- Parent Category
- Keyword (suggested new category)
- Search Volume (if KWE enabled)
- CPC (if KWE enabled)
- Matching Products
- Matched Category (similar existing)
- Similarity score

## Author

**Lee Foot** - eCommerce SEO Consultant

- Website: [leefoot.com](https://www.leefoot.com)
- Twitter/X: [@LeeFootSEO](https://x.com/LeeFootSEO)
- LinkedIn: [lee-foot](https://www.linkedin.com/in/lee-foot/)
- Contact: [Get in touch](https://www.leefoot.com/contact)
