# SEO Tools & Scripts Repository

[![Tools](https://img.shields.io/badge/Tools-100+-9cf.svg)](#tools-by-category)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-54_Apps-FF4B4B.svg)](https://streamlit.io/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebooks-F9AB00.svg)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

The largest open-source collection of **SEO tools** on GitHub — **100+ scripts, apps, and APIs** for eCommerce SEO, keyword research, content analysis, link building, technical audits, and reporting.

> Used by SEO professionals worldwide. Featured in **Search Engine Journal**. Free and open source.

---

## Author

**Lee Foot** — eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)

---

## As Featured In

<a href="https://www.searchenginejournal.com/"><img src="https://img.shields.io/badge/Search_Engine_Journal-Featured-FF5722?style=for-the-badge&logoColor=white" alt="Search Engine Journal"></a>

Tools and methodologies published in [Search Engine Journal](https://www.searchenginejournal.com/):

| Publication | Tool | Description |
|-------------|------|-------------|
| [Semantic Keyword Clustering](./search-engine-journal) | Semantic Clustering Tool | AI-powered keyword clustering using sentence transformers |
| [Top Traffic Pages Analysis](./search-engine-journal) | Top Traffic Pages | Identify top-performing pages via the Search Console API |

---

## Live Apps

Try these tools directly in your browser — no installation required:

| App | Description | Link |
|-----|-------------|------|
| **BERTlinker** | AI-powered internal linking at scale | [![Visit Site](https://img.shields.io/badge/Visit_Site-007EC6?style=for-the-badge&logoColor=white)](https://bertlinker.com) |
| **Website Migration Mapper** | Auto-map URLs from old site to new | [![Open App](https://img.shields.io/badge/Open_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://migration.streamlit.app/) |
| **GSC Data Downloader** | Download your Google Search Console data | [![Open App](https://img.shields.io/badge/Open_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://gscdata.streamlit.app/) |
| **Wikipedia Citation Finder** | Find Wikipedia pages needing citations for link opportunities | [![Open App](https://img.shields.io/badge/Open_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://wikicite.streamlit.app/) |
| **Image Centering Tool** | Batch center and resize product images | [![Open App](https://img.shields.io/badge/Open_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://imagewiz.streamlit.app/) |
| **Wayback URL Extractor** | Extract historical URLs from archive.org | [![Open App](https://img.shields.io/badge/Open_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://wayback.streamlit.app/) |
| **Category Generator** | Auto-suggest new category pages from product inventory | [![Open App](https://img.shields.io/badge/Open_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://category-generator.streamlit.app/) |

---

## Recently Added

| Tool | Category | Description |
|------|----------|-------------|
| [Firecrawl Markdown Scraper](./technical-seo/firecrawl-markdown-scraper) | Technical SEO | Scrape URLs and convert to clean markdown via Firecrawl API |
| [Content Repurposer](./content-analysis/content-repurposer) | Content | Transform blog posts into social posts, email content, and video scripts using AI |
| [Regex Generator](./technical-seo/regex-generator) | Technical SEO | Generate regex patterns from plain English using Claude or GPT |
| [Schema Markup Generator](./technical-seo/schema-markup-generator) | Technical SEO | Generate valid JSON-LD structured data for FAQPage, Product, and more |
| [Gist Summary Generator](./content-analysis/gist-summary-generator) | Content | Create "At a glance" summaries from articles using AI — perfect for featured snippets |
| [LLM Sitemap Creator](./technical-seo/llm-sitemap-creator) | Technical SEO | Generate hierarchical sitemap structures from keyword lists using GPT |

---

## Tools by Category

### Content Analysis (17 tools)

| Tool | Description | Type |
|------|-------------|------|
| [Category Title Suggester](./content-analysis/category-title-suggester) | Suggest high-performing keywords for category page titles using GSC data | Streamlit |
| [Content Duplication Finder](./content-analysis/content-duplication-finder) | Find duplicate content clusters using PolyFuzz fuzzy matching | Streamlit |
| [Content Extractor](./content-analysis/content-extractor) | Extract main text content and H1 headings from URLs | Streamlit |
| [Content Hub Classification](./content-analysis/content-hub-classification) | Classify articles into content hub categories using OpenAI GPT | Streamlit |
| [Content Repurposer](./content-analysis/content-repurposer) | Transform blog posts into social posts, emails, and video scripts using AI | Streamlit |
| [Content Reviewer (LLM)](./content-analysis/content-reviewer-llm) | Review and annotate web content using AI for quality improvements | Streamlit |
| [Entity Extractor](./content-analysis/entity-extractor) | Extract entities from SERPs, CSV files, or YouTube transcripts | Streamlit |
| [Gist Summary Generator](./content-analysis/gist-summary-generator) | Create "At a glance" bullet summaries from articles using Claude or GPT | Streamlit |
| [Keyword Extractor (LLM)](./content-analysis/keyword-extractor-llm) | Extract and categorize keywords from content using OpenAI or Anthropic APIs | Streamlit |
| [Keyword to Page Mapper](./content-analysis/keyword-to-page-mapper) | Semantically match competitor keywords to existing pages | Streamlit |
| [Meta Description Grader](./content-analysis/meta-description-grader) | Score meta descriptions on SEO criteria using GPT-4 analysis | Streamlit |
| [Meta Description Rewriter](./content-analysis/meta-description-rewriter) | Rewrite meta descriptions using AI with length optimization | Streamlit |
| [N-gram SERP Extractor](./content-analysis/ngram-serp-extractor) | Extract page title and content n-grams from SERP results via ValueSERP | Streamlit |
| [OpenAI Entity Visualizer](./content-analysis/openai-entity-visualizer) | Interactive entity extraction visualization using OpenAI | Streamlit |
| [Page Intent Classifier](./content-analysis/page-intent-classifier) | Classify page intent (informational, transactional, navigational) using GPT | Streamlit |
| [Reading Score Analyzer](./content-analysis/reading-score-analyzer) | Calculate Flesch readability metrics for web pages from sitemaps or CSV | Streamlit |
| [Review Sentiment Extractor](./content-analysis/review-sentiment-extractor) | Analyze product reviews to extract sentiment, pain points, and praise themes | Streamlit |

### Keyword Research (16 tools)

| Tool | Description | Type |
|------|-------------|------|
| [Bulk Keyword Tagger](./keyword-research/bulk-keyword-tagger) | Tag thousands of keywords with custom categories | Colab |
| [Category Keyword Finder](./keyword-research/category-keyword-finder) | Extract n-gram keyword opportunities from product titles by category | Streamlit |
| [DataForSEO Suggestions](./keyword-research/dataforseo-suggestions) | Get keyword suggestions with search volumes from the DataForSEO API | Streamlit |
| [eBay Related Searches](./keyword-research/ebay-related-searches) | Scrape related search keywords from eBay with tree visualization | Streamlit |
| [Keyword Deduplication Tool](./keyword-research/keyword-deduplication-tool) | Remove close keyword variations using RapidFuzz fuzzy matching | Streamlit |
| [Keyword Difficulty Checker](./keyword-research/keyword-difficulty-checker) | Check difficulty with allintitle, phrase match, and SERP clustering | Streamlit |
| [Keyword Grouper](./keyword-research/keyword-grouper) | Group similar keywords using TF-IDF similarity via PolyFuzz | Streamlit |
| [Keyword to Questions](./keyword-research/keyword-to-questions) | Convert keyword phrases into natural FAQ questions using Claude or GPT | Streamlit |
| [Keyword Topic Classifier](./keyword-research/keyword-topic-classifier) | Classify keywords into hierarchical themes using Claude or GPT | Streamlit |
| [Keyword Trends Analyzer](./keyword-research/keyword-trends-analyzer) | Analyze Google Trends data with YoY trend slope calculations | Streamlit |
| [Keywords Everywhere API](./keyword-research/keywords-everywhere-api) | Keyword research with search volume, CPC, and competition data | Streamlit |
| [Micro-Moments Classifier](./keyword-research/micro-moments-classifier) | Classify keywords into Google's 4 micro-moments using OpenAI | Streamlit |
| [PAA Scraper](./keyword-research/paa-scraper) | Recursively extract "People Also Ask" questions with tree visualization | Streamlit |
| [Question Extraction GSC](./keyword-research/question-extraction-gsc) | Extract question-type keywords from Google Search Console data | Python |
| [Related Searches Tree](./keyword-research/related-searches-tree) | Build hierarchical trees of related searches from Google | Streamlit |
| [SERP Keyword Extractor](./keyword-research/serp-keyword-extractor) | Extract PAA questions and related searches via ValueSERP API | Streamlit |
| [Topical Map Generator](./keyword-research/topical-map-generator) | Organize keywords into hierarchical topical maps using GPT-4o | Streamlit |

### eCommerce SEO (15 tools)

| Tool | Description | Type |
|------|-------------|------|
| [Automatic Category Suggester](./ecommerce/automatic-category-suggester) | Auto-suggest new category pages from product inventory using n-grams | Streamlit |
| [Best Selling Products to XML Sitemap](./ecommerce/best-selling-products-to-xml-sitemap) | Create dedicated sitemaps for top-performing products | Colab |
| [Breadcrumb Relevancy Checker](./ecommerce/breadcrumb-relevancy-checker) | Check if products are in relevant categories using TF-IDF matching | Streamlit |
| [eCommerce Image Centering Tool](./ecommerce/ecommerce-image-centering-tool) | Batch center and resize product images with white background | Streamlit |
| [E-commerce Page Title Optimizer](./ecommerce/ecom-page-title-optimizer) | Optimize page titles using GSC keyword data for high-impact improvements | Streamlit |
| [Google Vision Higher Res Images](./ecommerce/google-vision-find-higher-resolution-images) | Find higher resolution product images using Google Vision API | Python |
| [Inject Branding into PDFs](./ecommerce/inject-branding-into-pdf-files) | Add custom text branding to PDF files in batch | Python |
| [Internal Search Mapper](./ecommerce/internal-search-mapper) | Map GA site search queries to landing pages using fuzzy matching | Python / Colab |
| [Low Links vs High Transactions](./ecommerce/low-links-vs-high-transactions) | Find high-converting pages that need more internal links | Python |
| [Non-White Background Detector](./ecommerce/non-white-background-detector) | Detect product images with non-white backgrounds for QA | Streamlit |
| [Product Q&A Extractor](./ecommerce/product-qa-extractor) | Extract product reviews, ratings, and Q&A content from e-commerce pages | Streamlit |
| [Product Spec Extractor](./ecommerce/product-spec-extractor) | Scrape product specifications with configurable CSS selectors | Streamlit |
| [Product Title Gap](./ecommerce/product-title-gap) | Compare product titles with competitors to identify missing keywords | Streamlit |
| [SERP Title Generator](./ecommerce/serp-title-generator) | Generate optimized product titles using n-gram and phrase analysis | Streamlit |
| [WooCommerce Product Relevancy](./ecommerce/woocommerce-sort-products-by-relevancy) | Sort WooCommerce products by category relevancy score | Python |

### Technical SEO (11 tools)

| Tool | Description | Type |
|------|-------------|------|
| [Firecrawl Markdown Scraper](./technical-seo/firecrawl-markdown-scraper) | Scrape URLs and convert to clean markdown using Firecrawl API | Streamlit |
| [Hreflang Checker](./technical-seo/hreflang-checker) | Extract and validate hreflang tags from web pages | Streamlit |
| [Hreflang Generator](./technical-seo/hreflang-generator) | Generate hreflang XML tags from Screaming Frog crawl data | Streamlit |
| [LLM Sitemap Creator](./technical-seo/llm-sitemap-creator) | Generate hierarchical sitemap structures from keyword lists using GPT | Streamlit |
| [OnCrawl Extractor](./technical-seo/oncrawl-extractor) | Comprehensive OnCrawl API client for data extraction and crawl management | Streamlit |
| [QA Schema Extractor](./technical-seo/qa-schema-extractor) | Extract FAQ and Q&A structured data (JSON-LD) from websites | Streamlit |
| [Redirect Validator](./technical-seo/redirect-validator) | Validate redirects against redirect mapping specifications | Streamlit |
| [Regex Generator](./technical-seo/regex-generator) | Generate regex patterns from plain English using Claude or GPT | Streamlit |
| [Schema Markup Generator](./technical-seo/schema-markup-generator) | Generate valid JSON-LD for FAQPage, Product, Organization, and more | Streamlit |
| [Sitemap URL Extractor](./technical-seo/sitemap-url-extractor) | Extract all URLs from XML sitemap indexes and child sitemaps | Streamlit |
| [Template Fingerprinting](./technical-seo/template-fingerprinting) | Identify page templates using HTML structure analysis and ML clustering | Python |

### Reporting & Analytics (12 tools)

| Tool | Description | Type |
|------|-------------|------|
| [BCG Matrix from GA](./reporting/create-bcg-matrix-from-ga-landing-page-report) | Create BCG growth-share matrix from GA landing page data | Colab |
| [Content Decay Analyzer](./reporting/content-decay-analyzer) | Find pages losing traffic by comparing peak vs current GSC performance | Streamlit |
| [Core Update Analyser](./reporting/core-update-analyser) | Group Ahrefs ranking changes by URL folder to show update impact | Streamlit |
| [Delta Audit](./reporting/delta-audit) | Detect weeks with significant GSC traffic changes automatically | Streamlit |
| [Google Algorithm Tracker](./reporting/google-algorithm-tracker) | Scrape Google's Search Status page for algorithm update history | Streamlit |
| [Google Trends Forecasting](./reporting/forecasting-google-trends-single-keyword) | Forecast search trends using NeuralProphet time-series ML | Streamlit |
| [Batch Trends Forecasting](./reporting/forecasting-google-trends-crawl-file) | Forecast Google Trends for multiple keywords from a crawl file | Streamlit |
| [Resolution Screenshot Tool](./reporting/pyppeteer-render-pages-by-most-common-resolutions-in-ga) | Screenshot pages at your visitors' most common screen resolutions | Python |
| [Share of Voice](./reporting/share-of-voice) | Calculate estimated organic traffic share using CTR curves | Streamlit |
| [Top Traffic Pages (GSC)](./reporting/top-traffic-pages-search-console-sej) | Identify highest-traffic pages via the Search Console API | Colab |
| [Visualise Internal Links](./reporting/visualise-links-screaming_frog) | Interactive treemap visualization of internal link structure | Colab |
| [Visualise GSC Coverage](./reporting/visualise-search-console-coverage-reports) | Treemap and sunburst charts to visualize indexing issues by folder | Colab |

### Link Building (5 tools)

| Tool | Description | Type |
|------|-------------|------|
| [Backlink Intersector](./link-building/backlink-intersector) | Find opportunities by intersecting competitor backlink profiles | Streamlit |
| [eCommerce Link Builder](./linking/ecommerce-link-builder) | Find "Where to Buy" and distributor link opportunities | Python |
| [Link Quality Analyzer](./link-building/link-quality-analyzer) | Check link status codes and calculate reading metrics | Streamlit |
| [Wayback Machine Link Mapper](./linking/map-urls-wayback-machine) | Recover broken backlinks using archive.org historical data | Python / Colab |
| [Wikipedia Citation Finder](./linking/wikipedia-citation-finder) | Find Wikipedia pages with "citation needed" tags for link opportunities | Streamlit |

### Internal Linking (2 tools)

| Tool | Description | Type |
|------|-------------|------|
| [Anchor Text Interlinker](./internal-linking/anchor-text-interlinker) | Find internal linking opportunities by matching keywords to page content | Streamlit |
| [Anchor Text Relevance Checker](./internal-linking/anchor-text-relevance-checker) | Assess anchor text relevance using AI evaluation | Streamlit |

### Search Console (5 tools)

| Tool | Description | Type |
|------|-------------|------|
| [GSC Chart Visualizer](./search-console/gsc-chart-visualizer) | GSC data visualization and charting with custom dimension analysis | Streamlit |
| [GSC Folder Analyzer](./search-console/gsc-folder-analyzer) | Aggregate GSC data by URL folder to analyze site section performance | Streamlit |
| [Keyword Cannibalization](./search-console/keyword-cannibalization) | Find keywords where multiple pages compete for the same query | Streamlit |
| [Simple GSC Connector](./search-console/streamlit-simple-gsc-connector) | Minimal Streamlit boilerplate for Google Search Console OAuth | Streamlit |
| [Title Keyword Gap](./search-console/title-keyword-gap) | Find keywords driving impressions but missing from page titles | Streamlit |

### Keyword Clustering (3 tools)

| Tool | Description | Type |
|------|-------------|------|
| [Semantic Clustering Tool](./keyword-clustering/semantic-clustering) | Group keywords into topical clusters using sentence transformers and ML | CLI / Colab |
| [SERP Clustering at Scale](./keyword-clustering/serp-clustering-at-scale) | Cluster keywords based on common SERP URLs from ValueSERP exports | Python |
| [SERP Clustering API](./keyword-clustering/serp-clustering-api) | FastAPI service for clustering keywords via REST endpoint | FastAPI |

### On-Page SEO (4 tools)

| Tool | Description | Type |
|------|-------------|------|
| [Extract Content Blocks](./on-page/extract-content-blocks) | Extract and categorize page content blocks using Claude AI | Python |
| [Striking Distance (CSV)](./on-page/striking-distance-csv) | Find keywords ranking 4-20 from pre-exported Search Console data | Python |
| [Striking Distance V1](./on-page/striking-distance-v1) | Find keywords ranking positions 4-20 and check title/H1/copy presence | Python / Colab |
| [Striking Distance V2](./on-page/striking-distance-v2) | Enhanced striking distance with title/H1/copy presence checking | Streamlit |

### Competitive Analysis (3 tools)

| Tool | Description | Type |
|------|-------------|------|
| [Competitor Gap Finder](./competitor-gap-finder) | Find gaps in product titles by comparing against competitors | Python |
| [Keyword Gap Analyzer](./competitive-analysis/keyword-gap-analyzer) | Compare keyword lists against competitors using TF-IDF matching | Streamlit |
| [SERP Crossover Analyzer](./competitive-analysis/serp-crossover-analyzer) | Analyze SERP URL crossover patterns to detect cannibalization | Streamlit |

### Website Migration (1 tool)

| Tool | Description | Type |
|------|-------------|------|
| [Website Migration Mapper](./website-migration) | Auto-map URLs from old site to new using multi-algorithm matching | Streamlit / Python / Colab |

### Wayback Machine (1 tool)

| Tool | Description | Type |
|------|-------------|------|
| [Wayback URL Extractor](./wayback-url-tool) | Bulk extract historical URLs from archive.org with folder visualization | Streamlit |

### Site Search (1 tool)

| Tool | Description | Type |
|------|-------------|------|
| [Map Site Searches to Pages](./site-search/map-site-searches-to-landing-pages) | Match internal site searches to best landing pages using TF-IDF | Colab |

### PPC (1 tool)

| Tool | Description | Type |
|------|-------------|------|
| [AdWords Tools](./ppc/adwords-tools) | MPN Extractor and Bid Calculator for Google Ads campaigns | Streamlit |

### Other (2 tools)

| Tool | Description | Type |
|------|-------------|------|
| [Content Consolidation Analyzer](./content-consolidation) | Identify consolidation opportunities from SERP overlap and cannibalization | Python |
| [Product Title Optimizer](./product-title-optimizer) | LLM-powered title restructuring with data integrity checks | Python |

---

## Tool Types

| Type | Count | Run Command |
|------|-------|-------------|
| **Streamlit** | 54 | `streamlit run app.py` |
| **Python** | 24 | `python script.py` |
| **CLI** | 14 | `python script.py --args` |
| **Colab** | 9 | Click "Open in Colab" badge |
| **FastAPI** | 1 | `uvicorn app:app --reload` |

---

## Getting Started

### Prerequisites

```bash
# Most tools require Python 3.8+
python --version

# Install dependencies (each folder has its own requirements.txt)
pip install -r requirements.txt
```

### Run a Streamlit App

```bash
cd tool-folder
pip install -r requirements.txt
streamlit run app.py
```

### Run a Google Colab Notebook

1. Navigate to the `.ipynb` file on GitHub
2. Click the **"Open in Colab"** badge at the top
3. Run all cells (Runtime → Run all)

### Run a Python Script

```bash
cd tool-folder
pip install -r requirements.txt
python script.py
```

---

## Star History

<a href="https://star-history.com/#searchsolved/search-solved-public-seo&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=searchsolved/search-solved-public-seo&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=searchsolved/search-solved-public-seo&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=searchsolved/search-solved-public-seo&type=Date" />
 </picture>
</a>

---

## Contributing

Contributions, suggestions, and feedback are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- **Star** this repo if you find it useful
- **Open an issue** for bugs or feature requests
- **Submit a PR** with improvements

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <strong>Built for the SEO community</strong><br><br>
  <a href="https://www.leefoot.com">leefoot.com</a> &middot;
  <a href="https://www.linkedin.com/in/lee-foot/">LinkedIn</a> &middot;
  <a href="https://bsky.app/profile/leefootseo.bsky.social">Bluesky</a>
</p>
