# Technical SEO Tools

Comprehensive toolkit for technical SEO audits, schema markup, internationalization, and site infrastructure analysis. Includes AI-powered tools for generating regex patterns, schema markup, and sitemap structures.

## Tools Overview

### AI-Powered Technical Tools

#### 🤖 **LLM Sitemap Creator**
Use GPT to generate hierarchical sitemap structures from keywords.
- **Use Case**: Information architecture planning, new site structure
- **Input**: Keyword list with search volumes, OpenAI API key
- **Output**: Hierarchical sitemap structure with categories
- **Features**: GPT-powered organization, volume-aware hierarchy, validation

#### 🔣 **Regex Generator for SEO**
Generate regex patterns from plain English descriptions for SEO tasks.
- **Use Case**: GSC filters, .htaccess redirects, Screaming Frog extraction
- **Input**: Plain English description, test strings
- **Output**: Regex patterns with platform-specific syntax
- **Features**: Claude and GPT-4 support, live testing, preset patterns

#### 📋 **Schema Markup Generator**
Generate structured data markup for various schema types.
- **Use Case**: Rich results optimization, structured data implementation
- **Input**: Page content and type selection
- **Output**: JSON-LD schema markup
- **Features**: Multiple schema types, validation, copy-ready output

### Internationalization Tools

#### 🌍 **Hreflang Checker**
Validate hreflang implementations across your international site.
- **Use Case**: International SEO audits, hreflang debugging
- **Input**: URLs or Screaming Frog crawl data
- **Output**: Validation report with errors and warnings
- **Features**: Self-reference checking, return link validation, x-default checking

#### 🌐 **Hreflang Generator**
Generate hreflang tags for international page variants.
- **Use Case**: International site setup, hreflang implementation
- **Input**: URL mappings across languages/regions
- **Output**: Hreflang tag sets for implementation
- **Features**: Multiple output formats, bulk processing

### Sitemap & Crawling Tools

#### 🗺️ **Sitemap URL Extractor**
Extract all URLs from XML sitemaps including nested sitemap indexes.
- **Use Case**: Site audits, migration preparation, coverage analysis
- **Input**: Sitemap URL or sitemap index URL
- **Output**: Complete URL list from all sitemaps
- **Features**: Handles sitemap indexes, recursive extraction, deduplication

#### 🔥 **Firecrawl Markdown Scraper**
Scrape URLs and convert them to clean markdown format.
- **Use Case**: LLM training data, content migration, documentation
- **Input**: URLs to scrape, Firecrawl API key
- **Output**: Clean markdown files with metadata
- **Features**: Main content extraction, bulk processing, metadata headers

### Redirect & Migration Tools

#### ↔️ **Redirect Validator**
Validate implemented redirects against your mapping specification.
- **Use Case**: Migration QA, redirect audits
- **Input**: Redirect mapping specification, crawled redirect data
- **Output**: Mismatch report, missing redirects, extra redirects
- **Features**: Bulk validation, detailed reporting, chain detection

### Schema & Structured Data Tools

#### ❓ **Q&A Schema Extractor**
Extract Question/Answer pairs from JSON-LD schema markup.
- **Use Case**: Content auditing, competitive analysis, schema migration
- **Input**: URLs with Q&A schema or crawl data
- **Output**: Extracted question-answer pairs with source URLs
- **Features**: JSON-LD parsing, bulk extraction

### Site Analysis Tools

#### 🏗️ **Template Fingerprinting**
Identify page templates using HTML structure analysis and ML clustering.
- **Use Case**: Site audits, template-based optimization
- **Input**: Screaming Frog crawl data
- **Output**: Pages grouped by template type
- **Features**: HTML structure analysis, ML clustering, template naming

#### 📊 **OnCrawl Extractor**
Extract and process data from OnCrawl exports.
- **Use Case**: Technical SEO audits using OnCrawl data
- **Input**: OnCrawl export files
- **Output**: Processed analysis data
- **Features**: Multiple report types, data transformation

## Use Cases

### 🌍 **International SEO**
- Validate hreflang implementations
- Generate hreflang tags for new pages
- Audit international site structure

### 📋 **Structured Data**
- Generate schema markup for rich results
- Extract and audit existing schema
- Validate Q&A schema implementation

### 🔄 **Migrations**
- Validate redirect implementations
- Extract URLs from sitemaps
- Convert content to markdown format

### 🔍 **Technical Audits**
- Identify page templates
- Generate regex for filtering and extraction
- Plan site architecture with AI assistance

## Quick Start

### LLM Sitemap Creator
```bash
cd llm-sitemap-creator
pip install -r requirements.txt
streamlit run llm_sitemap_creator.py
```

### Regex Generator
```bash
cd regex-generator
pip install -r requirements.txt
streamlit run regex_generator.py
```

### Hreflang Checker
```bash
cd hreflang-checker
pip install -r requirements.txt
streamlit run hreflang_checker.py
```

### Sitemap URL Extractor
```bash
cd sitemap-url-extractor
pip install -r requirements.txt
streamlit run sitemap_url_extractor.py
```

## Input Requirements

### **API Keys**
- OpenAI or Anthropic API key (for AI-powered tools)
- Firecrawl API key (for markdown scraper)

### **Common Input Formats**
- Screaming Frog crawl exports
- XML sitemap URLs
- CSV files with URL data
- OnCrawl export files

## Output Formats

- **CSV files** with extracted/validated data
- **JSON-LD** schema markup (copy-ready)
- **Markdown files** for content migration
- **Excel reports** with multiple sheets

## Technical Specifications

### **Dependencies**
```bash
# Common dependencies
pip install pandas requests streamlit lxml

# AI tools
pip install openai anthropic

# Scraping tools
pip install beautifulsoup4 trafilatura
```

### **Performance**
- Batch processing for large sites
- Progress tracking for long operations
- Error handling and retry logic

## Best Practices

### 🌍 **Hreflang Implementation**
1. Always include self-referencing hreflang tags
2. Ensure bidirectional linking between all variants
3. Use x-default for language/region selectors
4. Validate after every change

### 📋 **Schema Markup**
1. Test markup with Google's Rich Results Test
2. Start with most impactful schema types
3. Keep markup accurate and up-to-date
4. Monitor rich result performance in GSC

### ↔️ **Redirects**
1. Validate redirects before and after go-live
2. Check for redirect chains and loops
3. Monitor 404s post-migration
4. Keep redirect mapping documentation updated

## Support & Documentation

Each tool includes detailed setup instructions in its respective directory. For advanced implementations or custom requirements, visit [leefoot.com](https://www.leefoot.com).

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
---

*Part of the Search Solved Public SEO toolkit - Technical SEO automation and analysis.*
