# Automatic Category Page Suggester

A Streamlit app that analyzes your crawl data to suggest new category pages based on your product inventory and real search demand.

**Originally presented at Brighton SEO.**

[![Open App](https://img.shields.io/badge/Open_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://category-generator.streamlit.app/)

## How It Works

1. **N-Gram Generation**: Extracts 2-7 word n-grams from product H1 tags to generate thousands of potential category keywords
2. **Product Matching**: Filters suggestions to only those matching a minimum number of products (exact and fuzzy matching)
3. **Duplicate Detection**: Uses PolyFuzz TF-IDF matching to identify suggestions too similar to existing categories
4. **Search Validation**: Keywords Everywhere API validates suggestions against real search volume data, keeping only legitimate keywords with actual demand
5. **Fragment Removal**: Optionally keeps only the longest keyword variant, removing shorter fragments

## Features

- Generates thousands of keyword variations from product titles using n-grams
- Matches suggestions to existing categories using fuzzy matching (PolyFuzz)
- Keywords Everywhere API integration filters to only real search terms
- Configurable similarity threshold to avoid duplicate categories
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

2. **Upload files from Screaming Frog:**
   - `inlinks.csv` - Export inlinks to your product pages (Bulk Export > Links > All Inlinks)
   - `internal_html.csv` - HTML export (Bulk Export > All > Internal HTML)

3. **Map columns:**
   - Select which custom extraction column identifies product pages
   - Select which custom extraction column identifies category pages

4. **Configure settings** in the sidebar

5. **Download results** as CSV

## Input Files

### inlinks.csv
Export from Screaming Frog: **Bulk Export > Links > All Inlinks**

Required columns:
- `From` / `Source` - The linking page
- `To` / `Destination` - The linked page

### internal_html.csv
Export from Screaming Frog: **Bulk Export > All > Internal HTML**

Required columns:
- `Address` - Page URL
- `Indexability` - Indexability status
- `H1-1` - Primary H1 tag
- `Title 1` - Page title
- Custom extraction columns for product/category identification

## Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| Min Product Match (Exact) | Minimum products a keyword must exactly match | 3 |
| Min Product Match (Fuzzy) | Minimum fuzzy matches required | 3 |
| Min Similarity | Max similarity to existing category (lower = more unique) | 96% |
| Min CPC | Minimum cost-per-click filter | $0 |
| Min Search Volume | Minimum monthly search volume | 100 |
| Keep Longest Word | Remove shorter keyword fragments | Enabled |
| Fuzzy Product Match | Enable slower but more thorough matching | Disabled |

## Keywords Everywhere Integration

Add your [Keywords Everywhere](https://keywordseverywhere.com/) API key to validate suggestions against real search data. This filters out n-gram combinations that nobody actually searches for.

- Validates keyword suggestions have real search volume
- Provides CPC data for commercial intent analysis
- PAYG pricing with no expiration

## Output

CSV file with columns:
- **Parent Category** - The category the products belong to
- **Keyword** - Suggested new category name
- **Search Volume** - Monthly searches (if KWE enabled)
- **CPC** - Cost per click (if KWE enabled)
- **Matching Products (Exact)** - Products containing exact keyword
- **Matching Products (Fuzzy)** - Products matching all words
- **Matched Category** - Most similar existing category
- **Similarity** - How similar to existing (lower = more unique)

## Use Cases

- **Expand category structure** based on actual product inventory
- **Align taxonomy with search demand** using real keyword data
- **Identify gaps** between what you sell and how users search
- **Prioritize new categories** by search volume and product coverage

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)