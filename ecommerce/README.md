# eCommerce SEO Tools

A comprehensive collection of Python scripts designed to automate and enhance eCommerce SEO workflows. These tools help optimize product pages, improve image quality, and streamline various eCommerce-specific SEO tasks.

## Tools Overview

### 🏷️ **Automatic Category Suggester**
Analyzes crawl data to suggest new category pages based on product inventory and search demand patterns.
- **Use Case**: Identify content gaps and optimization opportunities
- **Input**: Screaming Frog crawl data (Internal HTML + Inlinks)
- **Output**: CSV with category suggestions and similarity scores

### 🎯 **Best Selling Products to XML Sitemap**
Generates XML sitemaps prioritizing your best-selling products for better search engine discovery.
- **Use Case**: Improve crawl budget allocation for high-value pages
- **Input**: Product performance data
- **Output**: Optimized XML sitemap

### 🖼️ **Image Centering Tool**
Automatically centers product images for consistent eCommerce displays using computer vision.
- **Use Case**: Standardize product image presentation
- **Input**: Product images (various formats)
- **Output**: Centered, consistently-sized images

### 🔍 **Google Vision Higher Resolution Images**
Leverages Google Vision API to find and source higher resolution versions of product images.
- **Use Case**: Enhance image quality without manual searching
- **Input**: Current product images
- **Output**: Higher resolution alternatives with size comparisons

### 📄 **PDF Branding Injection**
Adds branded elements to PDF files for consistent brand presentation.
- **Use Case**: Maintain brand consistency across all downloadable content
- **Input**: PDF files
- **Output**: Branded PDF files

### 🔗 **Internal Search Mapper**
Maps internal site searches to relevant landing pages to improve user experience and reduce bounce rates.
- **Use Case**: Connect user intent with existing content
- **Input**: Internal search query data
- **Output**: Search-to-page mapping recommendations

### 📊 **Low Links vs High Transactions Analysis**
Identifies high-performing products with insufficient internal linking for optimization opportunities.
- **Use Case**: Optimize internal linking for revenue-generating pages
- **Input**: Analytics data + crawl data
- **Output**: Link opportunity analysis

### ⚡ **WooCommerce Product Relevancy Sorter**
Automatically sorts WooCommerce products by relevancy scores to improve user experience.
- **Use Case**: Enhance product discovery and conversion rates
- **Input**: WooCommerce product data
- **Output**: Relevancy-sorted product listings

## Quick Start

1. **Choose your tool** based on your specific needs
2. **Navigate** to the tool's directory
3. **Read** the tool-specific README for detailed instructions
4. **Install** dependencies: `pip install -r requirements.txt`
5. **Configure** input paths and settings
6. **Run** the script with your data

## Prerequisites

- Python 3.7+
- Various APIs depending on tools (Google Vision, WooCommerce, etc.)
- Input data (usually CSV exports from SEO tools)

## Support & Documentation

Each tool includes detailed documentation in its respective directory. For additional support or custom implementations, visit [LeeFoot.com](https://www.leefoot.com).

## Author

**Lee Foot** - eCommerce SEO Consultant specializing in technical SEO automation.

- 🌐 [Website](https://www.leefoot.com)
- 🐦 [Twitter/X](https://x.com/LeeFootSEO)
- 🦋 [Bluesky](https://bsky.app/profile/leefootseo.bsky.social)
- 💼 [LinkedIn](https://www.linkedin.com/in/lee-foot/)
- ✉️ [Contact](https://www.leefoot.com/contact)

---

*Part of the Search Solved Public SEO toolkit - Practical automation for technical SEO professionals.*