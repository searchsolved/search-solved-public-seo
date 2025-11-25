# Keyword Research Tools

Powerful automation tools for discovering, analyzing, and organizing keywords from various sources. These scripts help expand keyword lists, extract competitive intelligence, and streamline keyword research workflows.

## Tools Overview

### 🏷️ **Bulk Keyword Tagger**
Automatically tags and categorizes large keyword lists based on customizable criteria and patterns.
- **Use Case**: Organize and categorize extensive keyword datasets
- **Input**: CSV file with keywords
- **Output**: Tagged keywords with categories and classifications
- **Features**: Pattern matching, regex support, custom tagging rules

### 🛒 **eBay Related Searches**
Scrapes and extracts related search suggestions from eBay to discover commercial keyword opportunities.
- **Use Case**: eCommerce keyword research and competitive analysis
- **Input**: Seed keywords or product terms
- **Output**: Related search terms with commercial intent
- **Features**: Automated scraping, bulk processing, commercial focus

### 📊 **SERP Keyword Extractor**
Extracts keywords and insights from Search Engine Results Pages (SERPs) using various APIs and scraping methods.
- **Use Case**: Competitive keyword analysis and SERP feature extraction
- **Input**: Target keywords or URLs
- **Output**: SERP-based keyword data and competitor insights
- **Features**: API integration, SERP feature analysis, competitor tracking

## Use Cases

### 🎯 **Competitive Research**
- Extract competitor keywords from SERPs
- Analyze related searches and suggestions
- Identify keyword gaps and opportunities

### 📈 **Keyword Expansion**
- Generate related keyword variations
- Discover long-tail opportunities
- Build comprehensive keyword lists

### 🏪 **eCommerce Research**
- Find product-specific keywords
- Analyze commercial search patterns
- Identify buying intent keywords

### 📋 **Data Organization**
- Categorize keywords by intent or topic
- Tag keywords for campaign organization
- Prepare keywords for content planning

## Quick Start

### Bulk Keyword Tagger
```bash
cd bulk-keyword-tagger
pip install -r requirements.txt
# Configure tagging rules in the notebook
jupyter notebook bulk_keyword_tagger_v2.ipynb
```

### eBay Related Searches
```bash
cd ebay-related-searches
pip install -r requirements.txt
python ebay_related_searches.py
```

### SERP Keyword Extractor
```bash
cd serp-keyword-extractor
pip install -r requirements.txt
python serp_keyword_extractor.py
```

## Input Requirements

### **General Requirements**
- Keywords or seed terms (CSV format preferred)
- API keys for relevant services (when applicable)
- Target websites or competitor domains

### **Specific Tool Requirements**

#### **Bulk Keyword Tagger**
- CSV file with keyword column
- Tagging rules and categories defined
- Optional: volume or metrics data

#### **eBay Related Searches**
- Seed keywords or product categories
- Target number of results per keyword
- Optional: geographic targeting

#### **SERP Keyword Extractor**
- Target keywords for SERP analysis
- API credentials (varies by service)
- Geographic and language settings

## Output Formats

- **CSV files** with processed keyword data
- **Excel workbooks** with multiple data sheets
- **JSON data** for API integrations
- **Visualization charts** for analysis

## Technical Specifications

### **Dependencies**
```bash
# Common dependencies
pip install pandas requests beautifulsoup4 streamlit

# Tool-specific requirements
pip install selenium webdriver-manager  # For scraping tools
pip install google-api-python-client    # For Google APIs
pip install plotly matplotlib          # For visualizations
```

### **Rate Limiting & Ethics**
- Built-in delays for responsible scraping
- Respect robots.txt and terms of service
- API rate limit handling
- User-agent rotation for scraping tools

### **Performance Considerations**
- Batch processing for large keyword lists
- Parallel processing where appropriate
- Progress tracking for long-running tasks
- Error handling and retry logic

## Configuration

### **API Setup**
Many tools require API access:
- Google Custom Search Engine API
- SEMrush API (optional)
- Ahrefs API (optional)
- SerpAPI (optional)

### **Scraping Configuration**
- User agent settings
- Delay configurations
- Proxy support (where applicable)
- Geographic targeting options

## Best Practices

### 🔍 **Research Strategy**
1. Start with broad seed keywords
2. Use multiple tools for comprehensive coverage
3. Validate keyword data across sources
4. Consider search intent in categorization

### 📊 **Data Management**
1. Regular backups of keyword data
2. Version control for keyword lists
3. Standardized naming conventions
4. Quality checks for scraped data

### ⚖️ **Ethical Considerations**
1. Respect website terms of service
2. Use reasonable request delays
3. Don't overload target servers
4. Follow platform-specific guidelines

## Troubleshooting

### **Common Issues**
- API rate limit exceeded: Implement delays and retries
- Scraping blocked: Update user agents and headers
- Large datasets: Use batch processing
- Memory issues: Process data in chunks

### **Error Handling**
- Comprehensive logging for debugging
- Graceful degradation when APIs fail
- Data validation and cleaning
- Resume functionality for interrupted tasks

## Support & Documentation

Each tool includes detailed setup instructions and configuration guides in its respective directory. For advanced implementations or custom requirements, visit [LeeFoot.com](https://www.leefoot.com).

## Author

**Lee Foot** - SEO Consultant specializing in keyword research automation and competitive analysis.

- 🌐 [Website](https://www.leefoot.com)
- 🐦 [Twitter/X](https://x.com/LeeFootSEO)
- 💼 [LinkedIn](https://www.linkedin.com/in/lee-foot/)
- ✉️ [Contact](https://www.leefoot.com/contact)

---

*Part of the Search Solved Public SEO toolkit - Comprehensive keyword research automation.*