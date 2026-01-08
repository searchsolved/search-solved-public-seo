# Product Title Optimizer

LLM-powered title restructuring with data integrity checks.

## What It Does

- Restructures messy product titles to optimal word order
- Groups by category for consistent formatting
- Creates category-specific templates automatically
- Validates no data is lost (numbers, model codes, etc.)
- Works with OpenAI API or local LLMs (LM Studio, Ollama, etc.)

## Use Cases

- Standardize inconsistent product naming conventions
- Improve title readability for better SEO and UX
- Apply consistent brand positioning (Brand first vs Product first)
- Batch process thousands of titles efficiently

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### With OpenAI API

```bash
# Using environment variable
export OPENAI_API_KEY=your-api-key
python product_title_optimizer.py --input products.csv --output optimized.csv

# Or pass key directly
python product_title_optimizer.py --input products.csv --api-key sk-xxx --output optimized.csv
```

### With Local LLM (LM Studio)

```bash
python product_title_optimizer.py \
    --input products.csv \
    --base-url http://localhost:1234/v1 \
    --model local-model \
    --output optimized.csv
```

### Custom Column Names

```bash
python product_title_optimizer.py \
    --input products.csv \
    --title-col "Product Name" \
    --category-col "Category" \
    --output optimized.csv
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input`, `-i` | Input CSV file | Required |
| `--output`, `-o` | Output CSV file | `optimized_titles.csv` |
| `--title-col` | Column for product titles | `Name` |
| `--category-col` | Column for categories | `Categories` |
| `--api-key` | OpenAI API key | `OPENAI_API_KEY` env var |
| `--base-url` | Custom API URL (for local LLMs) | None |
| `--model` | Model to use | `gpt-4o` |
| `--batch-size` | Titles per API call | `20` |
| `--limit` | Limit products (0 = no limit) | `0` |

## Input Format

```csv
Name,Categories
3M Speedglas 100 Welding Helmet Xterminator Design,Safety Equipment
DeWalt DCD796N 18V XR Brushless Compact Combi Drill Body Only,Power Tools
```

## Output

Adds columns to your CSV:

- `Optimized Title`: The restructured title
- `Is Same`: Whether the title was changed
- `Missing Words`: Any words from original not in optimized (for QA)

## How It Works

1. Groups products by category
2. Creates a category-specific template using LLM analysis
3. Optimizes each title using the template
4. Validates data integrity:
   - All numbers preserved
   - Word overlap check (80%+ required)
   - Logs any potential issues

## Title Structure

Follows adapted Basic English word order:

```
Brand + Product Type + Model/Series + Key Features + Specifications + Additional Info
```

Example:
- Before: `3M Speedglas 100 Welding Helmet Xterminator Design with Welding Filter`
- After: `3M Speedglas Welding Helmet - 100 Series - Xterminator Design - with Welding Filter`

## Data Integrity

The tool ensures:
- All numbers and measurements are preserved
- All model codes are kept
- No invented information is added
- Titles that fail validation keep their original

## Using Local LLMs

Works with any OpenAI-compatible API:

- **LM Studio**: `--base-url http://localhost:1234/v1`
- **Ollama**: `--base-url http://localhost:11434/v1`
- **Text Generation WebUI**: `--base-url http://localhost:5000/v1`

## Author

Lee Foot - [leefoot.com](https://www.leefoot.com)
