# Competitor Content Gap Finder

See exactly which words competitors use in their product titles that you're missing.

## What It Does

- Matches your products to competitors by SKU/MPN/product ID
- Compares title content using NLP word extraction
- Identifies missing descriptive words
- Calculates title length gaps
- Finds the most verbose competitor title for reference

## Use Cases

- Improve thin product descriptions
- Identify missing product attributes
- Find content opportunities vs competitors
- Prioritize title optimization efforts

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python competitor_gap_finder.py --source my_products.csv --compare competitor_products.csv
```

### Multiple Competitor Files

```bash
python competitor_gap_finder.py --source my_crawl.csv --compare "competitors/*.csv" --output gaps.csv
```

### Custom Column Names

```bash
python competitor_gap_finder.py \
    --source my_crawl.csv \
    --compare competitors/ \
    --match-col sku \
    --title-col product_name \
    --output gaps.csv
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--source`, `-s` | Your product CSV file | Required |
| `--compare`, `-c` | Competitor CSV file(s) - supports wildcards | Required |
| `--output`, `-o` | Output CSV file | `content_gaps.csv` |
| `--match-col` | Column for matching key (mpn, sku, etc.) | `mpn` |
| `--title-col` | Column for product title | `h1` |
| `--url-col` | Column for URL (optional) | `url` |

## Input Format

### Source File (your products)
```csv
mpn,h1,url
ABC123,Widget Pro 500,https://mysite.com/widget-pro
DEF456,Gadget Basic,https://mysite.com/gadget
```

### Competitor File(s)
```csv
mpn,h1,url
ABC123,Widget Pro 500 Premium Edition with Extended Warranty,https://competitor.com/widget
DEF456,Gadget Basic Starter Kit Complete,https://competitor.com/gadget
```

## Output

The tool outputs a CSV with:

- `match_key`: The matching identifier (SKU/MPN)
- `source_title`: Your current title
- `missing_words`: Words competitors use that you don't
- `missing_word_count`: Number of unique missing words
- `competitor_matches`: Number of competitors with this product
- `most_verbose_title`: Longest competitor title (for reference)
- `length_difference`: Word count gap (positive = competitors are more detailed)

## How It Works

1. Loads your product data and competitor data
2. Matches products by the specified key (MPN, SKU, etc.)
3. Processes titles to extract meaningful words (removes stopwords, punctuation)
4. Compares word sets to find gaps
5. Calculates statistics and outputs prioritized results

## Tips

- Use Screaming Frog exports for both your site and competitors
- Include a custom extraction for MPN/SKU if not in standard fields
- Results are sorted by biggest gaps first for easy prioritization

## Author

Lee Foot - [leefoot.com](https://www.leefoot.com)
