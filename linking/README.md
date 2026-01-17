# Linking & Internal Optimization Tools

Advanced tools for analyzing, optimizing, and automating internal linking strategies. These scripts help improve site architecture, recover broken links, and enhance internal link equity distribution.

## Tools Overview

### 🧠 **BERT Interlinker**
Uses BERT embeddings to intelligently suggest internal links based on semantic content similarity.
- **Use Case**: Automated internal linking based on content relevance
- **Technology**: BERT neural language model
- **Input**: Website content or page text
- **Output**: Semantic link suggestions with relevance scores

### 🏪 **eCommerce Link Builder**
Specialized internal linking tool designed for eCommerce websites to improve product and category page connections.
- **Use Case**: Optimize product discovery and category navigation
- **Input**: Product catalog and site structure data
- **Output**: Strategic internal link recommendations
- **Features**: Revenue-based prioritization, category optimization

### 🕰️ **Wayback Machine URL Mapper**
Maps and recovers broken links using Internet Archive data to maintain link equity and user experience.
- **Use Case**: Broken link recovery and historical URL mapping
- **Input**: Broken URLs or site crawl data
- **Output**: Working replacement URLs from Internet Archive
- **Formats**: Both Google Colab and Python script versions

### 📚 **Wikipedia Citation Finder**
Discovers Wikipedia citation opportunities for link building and authority building campaigns.
- **Use Case**: White-hat link building and citation discovery
- **Input**: Website URLs or topic keywords
- **Output**: Relevant Wikipedia pages with citation opportunities
- **Features**: Streamlit interface, automated discovery

## Technical Approaches

### 🔗 **Semantic Linking (BERT)**
- Content analysis using transformer models
- Semantic similarity scoring
- Context-aware link suggestions
- Natural language understanding

### 📊 **Data-Driven Linking (eCommerce)**
- Revenue and conversion data integration
- Product relationship analysis
- Category hierarchy optimization
- Performance-based prioritization

### 🔍 **Archive-Based Recovery**
- Internet Archive API integration
- URL pattern matching
- Historical data analysis
- Automated redirect mapping

### 🌐 **Authority Link Discovery**
- Wikipedia API utilization
- Citation gap analysis
- Content relevance scoring
- Opportunity prioritization

## Use Cases

### 🎯 **Internal SEO Optimization**
- Improve crawlability and indexation
- Distribute PageRank effectively
- Enhance topical authority
- Reduce orphaned pages

### 💰 **eCommerce Revenue Optimization**
- Guide users to high-converting products
- Improve product discovery
- Optimize category navigation
- Increase average order value

### 🔧 **Technical SEO Maintenance**
- Recover broken link equity
- Maintain historical URL value
- Improve site architecture
- Fix user experience issues

### 📈 **Link Building & PR**
- Discover citation opportunities
- Build authoritative backlinks
- Enhance brand mentions
- Improve domain authority

## Quick Start

### BERT Interlinker
```bash
cd bert-interlinker
pip install -r requirements.txt
# Setup instructions in directory README
python bert_interlinker.py
```

### eCommerce Link Builder
```bash
cd ecommerce-link-builder
pip install -r requirements.txt
python ecommerce_link_builder.py
```

### Wayback Machine Mapper
```bash
# Python version
cd map-urls-wayback-machine/python-source/map-links-from-wayback-machine
pip install -r requirements.txt
python archive_org_broken_link_mapper.py

# Or use Google Colab version
# Open the .ipynb file in Google Colab
```

### Wikipedia Citation Finder
```bash
cd wikipedia-citation-finder
pip install -r requirements.txt
streamlit run wikipedia_citation_finder_streamlit_source.py
```

## Input Requirements

### **BERT Interlinker**
- Website content or page text files
- Site structure data (optional)
- Target pages for link suggestions

### **eCommerce Link Builder**
- Product catalog data
- Site architecture information
- Performance metrics (conversions, revenue)
- Existing internal link structure

### **Wayback Machine Mapper**
- List of broken URLs
- Site crawl data from SEO tools
- Target domain information

### **Wikipedia Citation Finder**
- Website URLs or brand keywords
- Target topics or industries
- Geographic targeting (optional)

## Output Formats

- **CSV files** with link recommendations
- **Excel spreadsheets** with prioritization data
- **JSON data** for API integrations
- **Interactive reports** with visualizations
- **Implementation guides** for development teams

## Technical Specifications

### **Machine Learning Requirements**
```bash
# BERT Interlinker dependencies
pip install transformers torch sentence-transformers
pip install numpy pandas scikit-learn
```

### **Web Scraping & APIs**
```bash
# General requirements
pip install requests beautifulsoup4 selenium
pip install streamlit plotly pandas
```

### **Hardware Recommendations**
- **CPU**: 4+ cores for BERT processing
- **RAM**: 8GB minimum, 16GB+ for large sites
- **GPU**: Optional for faster BERT inference
- **Storage**: Varies by site size and archive data

## Implementation Guidelines

### 🔗 **Internal Linking Best Practices**
1. **Relevance First**: Prioritize semantic relevance over quantity
2. **User Experience**: Ensure links add value for users
3. **Anchor Text**: Use descriptive, natural anchor text
4. **Link Equity**: Consider PageRank flow and distribution

### 📊 **Data Quality**
1. **Clean Input Data**: Validate URLs and content quality
2. **Regular Updates**: Refresh link suggestions periodically
3. **Performance Monitoring**: Track link performance metrics
4. **A/B Testing**: Test link placements for optimization

### ⚡ **Performance Optimization**
1. **Batch Processing**: Handle large sites efficiently
2. **Caching**: Store processed results for faster iterations
3. **Incremental Updates**: Process only changed content
4. **Parallel Processing**: Utilize multi-threading where possible

## Advanced Features

### **BERT Interlinker**
- Custom model fine-tuning
- Domain-specific vocabulary
- Content freshness scoring
- Link context analysis

### **eCommerce Tools**
- Seasonal trend integration
- Inventory-based suggestions
- Price-point optimization
- Cross-selling opportunities

### **Archive Tools**
- Bulk URL processing
- Historical trend analysis
- Redirect chain optimization
- 404 error prioritization

## Support & Documentation

Each tool includes comprehensive documentation in its respective directory. For custom implementations or enterprise solutions, visit [LeeFoot.com](https://www.leefoot.com).

## Author

**Lee Foot** - Technical SEO Consultant specializing in internal linking optimization and site architecture.

- 🌐 [Website](https://www.leefoot.com)
- 🐦 [Twitter/X](https://x.com/LeeFootSEO)
- 🦋 [Bluesky](https://bsky.app/profile/leefootseo.bsky.social)
- 💼 [LinkedIn](https://www.linkedin.com/in/lee-foot/)
- ✉️ [Contact](https://www.leefoot.com/contact)

---

*Part of the Search Solved Public SEO toolkit - Advanced linking solutions for modern websites.*