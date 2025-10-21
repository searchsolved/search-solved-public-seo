# Web Content Block Extractor

**By Lee Foot | 21st October 2025** [LeeFoot.com](https://LeeFoot.com)

A Python script that uses Claude AI to automatically identify and extract major content blocks from web pages, generating XPath selectors for each element.

## Overview

This tool crawls a list of URLs, extracts the HTML content, and uses Claude Haiku to intelligently identify major content sections (hero sections, feature blocks, carousels, etc.) along with robust XPath expressions to select them. Perfect for web scraping, content analysis, or building automated testing selectors.

## Features

- **AI-Powered Extraction**: Uses Claude Haiku to intelligently identify content blocks
- **Frequency Analysis**: Automatically counts how often each XPath appears across all pages
- **Incremental Saving**: Saves progress every N rows to prevent data loss
- **Batch Processing**: Process multiple URLs with automatic rate limiting
- **Post-Processing**: Combines all results and standardizes naming by XPath
- **Debug Mode**: Test with a small subset of URLs before full run

## Prerequisites

- Python 3.7+
- Anthropic API key (Claude)

## Installation

1. Download the script `extract_content_blocks.py`

2. Install required packages:
```bash
pip install requests beautifulsoup4 anthropic pandas
```

3. Get your Anthropic API key from [https://console.anthropic.com/](https://console.anthropic.com/)

4. Open the script and add your API key:
```python
# Your API key
api_key = "YOUR CLAUDE KEY HERE"  # Replace with your actual API key
```

## Project Structure

```
zb_extract_content_blocks/
│
├── input/
│   └── urls.txt                          # Your list of URLs to process
│
├── output/
│   ├── content_blocks_*.csv              # Individual batch files
│   └── combined_output_with_frequency.csv # Final processed results
│
└── extract_content_blocks.py             # Main script
```

## Usage

### 1. Prepare Your URL List

Create a file at `input/urls.txt` with one URL per line:

```
https://example.com
https://example.com/products
https://example.com/about
https://anothersite.com
```

### 2. Configure the Script (Optional)

Edit these settings at the top of `extract_content_blocks.py`:

```python
# DEBUG MODE: Set to True to process only 2 URLs for testing
DEBUG_MODE = False

# INCREMENTAL SAVE: Save progress every N rows
SAVE_EVERY_N_ROWS = 50

# Update paths if needed
INPUT_FILE = r"C:\python_scripts\zb_extract_content_blocks\input\urls.txt"
OUTPUT_DIR = r"C:\python_scripts\zb_extract_content_blocks\output"
```

### 3. Run the Script

```bash
python extract_content_blocks.py
```

## Output

### Individual CSV Files
The script creates timestamped CSV files as it processes URLs:
- `content_blocks_20240121_143022_a1b2c3d4.csv`
- `content_blocks_20240121_143145_e5f6g7h8.csv`

Each contains:
| Column | Description |
|--------|-------------|
| `url` | Source URL |
| `name` | Descriptive name of the content block |
| `xpath` | XPath selector for the element |
| `notes` | Brief description |

### Final Combined Output
After processing all URLs, the script generates:

**`combined_output_with_frequency.csv`**

Contains all results with additional columns:
| Column | Description |
|--------|-------------|
| `url` | Source URL |
| `name` | Standardized name (consistent for same XPath) |
| `xpath` | XPath selector |
| `notes` | Description |
| `frequency` | How many times this XPath appears across all pages |

Sorted by frequency (most common XPaths first).

## Example Output

```
Top 10 XPaths by frequency:
================================================================================
  45x - Hero Section
        //div[@class='hero-banner']
  38x - Navigation Menu
        //nav[@id='main-navigation']
  35x - Footer
        //footer[@class='site-footer']
  28x - Feature Grid
        //section[@class='features']
```

## Debug Mode

For testing, enable debug mode to process only the first 2 URLs:

```python
DEBUG_MODE = True
```

This is useful for:
- Testing your API key setup
- Verifying URL format
- Checking output structure
- Estimating API costs

## Rate Limiting

The script includes a 1-second delay between requests to be respectful to web servers:

```python
time.sleep(1)  # In fetch_webpage()
```

Adjust as needed based on the target site's policies.

## Token Usage

The script uses prompt caching to reduce costs:
- HTML content is cached
- System prompt is cached
- Subsequent similar requests use cached tokens at ~90% discount

Console output shows token usage for each request:
```
Tokens - Input: 1234, Cache creation: 5678, Cache read: 9012, Output: 345
```

## Error Handling

- Invalid URLs are skipped with error messages
- Network errors are caught and logged
- JSON parsing errors are handled gracefully
- Progress is saved incrementally to prevent data loss

## Cost Estimation

With Claude Haiku 4.5 (current pricing):
- Input: $1 per million tokens
- Output: $5 per million tokens
- Cached input: ~$0.10 per million tokens (90% discount)

Average cost per URL: $0.002 - $0.015 depending on page size

The script uses prompt caching to significantly reduce costs on repeated similar requests.

## Troubleshooting

**"No module named 'anthropic'"**
```bash
pip install anthropic
```

**"401 Unauthorized" or API authentication errors**
- Ensure you've replaced `"YOUR CLAUDE KEY HERE"` with your actual API key
- Verify your API key is valid at [https://console.anthropic.com/](https://console.anthropic.com/)

**"403 Forbidden" errors when fetching URLs**
- Check if the website allows scraping (check robots.txt)
- Some sites block automated requests

**Large HTML pages timing out**
- Increase timeout in `fetch_webpage()`: `timeout=60`
- Consider filtering more aggressively in `filter_html()`

## Best Practices

1. **Always test with DEBUG_MODE=True first**
2. **Keep your API key secure** - don't share your script with the key in it
3. **Respect robots.txt and terms of service**
4. **Use appropriate rate limiting for target sites**
5. **Monitor API usage and costs in the Anthropic console**

## License

MIT License - feel free to use and modify as needed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built with [Anthropic's Claude API](https://www.anthropic.com/)
- Uses BeautifulSoup for HTML parsing
- Powered by Python

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review [Anthropic's documentation](https://docs.anthropic.com/)
3. Open an issue on GitHub

---

**Note**: This tool is for educational and legitimate web scraping purposes only. Always ensure you have permission to scrape websites and comply with their terms of service and robots.txt files.
