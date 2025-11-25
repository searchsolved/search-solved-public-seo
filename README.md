# SEO Tools & Scripts Repository

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Apps-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebooks-F9AB00.svg)](https://colab.research.google.com/)

A comprehensive collection of **30+ SEO tools**, Streamlit applications, and Python scripts for eCommerce SEO, keyword research, link building, and data analysis.

---

## Author

**Lee Foot** — eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/Website-leefoot.com-blue)](https://www.leefoot.com)
[![X](https://img.shields.io/badge/@LeeFootSEO-000000?logo=x&logoColor=white)](https://x.com/LeeFootSEO)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/)
[![Contact](https://img.shields.io/badge/Contact-Get_in_touch-green)](https://www.leefoot.com/contact)

---

## Live Apps

Try these tools directly in your browser — no installation required:

| App | Description | Link |
|-----|-------------|------|
| **Image Centering Tool** | Batch center and resize product images | [![Open App](https://img.shields.io/badge/Open_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://imagewiz.streamlit.app/) |
| **Wikipedia Citation Finder** | Find Wikipedia pages needing citations for link opportunities | [![Open App](https://img.shields.io/badge/Open_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://wikicite.streamlit.app/) |
| **GSC Data Downloader** | Download your Google Search Console data | [![Open App](https://img.shields.io/badge/Open_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://gscdata.streamlit.app/) |
| **Website Migration Mapper** | Auto-map URLs from old site to new | [![Open App](https://img.shields.io/badge/Open_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://migration.streamlit.app/) |
| **BERT Semantic Interlinker** | Find internal linking opportunities using BERT | [![Open App](https://img.shields.io/badge/Open_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://bertlinker.streamlit.app/) |
| **Wayback URL Extractor** | Extract historical URLs from archive.org | [![Open App](https://img.shields.io/badge/Open_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://wayback.streamlit.app/) |

---

## Featured in Search Engine Journal

Tools and methodologies featured in [Search Engine Journal](https://www.searchenginejournal.com/) publications:

| Tool | Description | Type |
|------|-------------|------|
| [Semantic Clustering Tool](./search_engine_journal) | AI-powered keyword clustering using sentence transformers | Python / Colab |
| [Top Traffic Pages Analysis](./search_engine_journal) | Identify top-performing pages via Search Console API | Colab |

See the [search_engine_journal](./search_engine_journal) folder for full publication details.

---

## Tools by Category

### eCommerce SEO

| Tool | Description | Type |
|------|-------------|------|
| [Automatic Category Suggester](./ecommerce/automatic-category-suggester) | Auto-suggest website categories based on existing products using NLP | Python |
| [Best Selling Products to XML Sitemap](./ecommerce/best-selling-products-to-xml-sitemap) | Create dedicated sitemaps for top-performing products | Colab |
| [eCommerce Image Centering Tool](./ecommerce/ecommerce-image-centering-tool) | Batch center and resize product images with white background | Streamlit |
| [Google Vision Higher Res Images](./ecommerce/google-vision-find-higher-resolution-images) | Find higher resolution versions of product images using Google Vision API | Python |
| [Inject Branding into PDFs](./ecommerce/inject-branding-into-pdf-files) | Add custom text branding to PDF files in batch | Python |
| [Internal Search Mapper](./ecommerce/internal-search-mapper) | Map GA site search queries to landing pages using fuzzy matching | Python / Colab |
| [Low Links vs High Transactions](./ecommerce/low-links-vs-high-transactions) | Find high-converting pages that need more internal links | Python |
| [WooCommerce Product Relevancy](./ecommerce/woocommerce-sort-products-by-relevancy) | Sort WooCommerce products by category relevancy score | Python |

### Keyword Research

| Tool | Description | Type |
|------|-------------|------|
| [eBay Related Searches](./keyword-research/ebay-related-searches) | Scrape related search keywords from eBay with tree visualization | Streamlit |
| [Bulk Keyword Tagger](./keyword-research/bulk-keyword-tagger) | Tag thousands of keywords with custom categories | Colab |
| [SERP Keyword Extractor](./keyword-research/serp-keyword-extractor) | Extract PAA questions and related searches from Google via ValueSERP API | Streamlit |

### Keyword Clustering

| Tool | Description | Type |
|------|-------------|------|
| [Semantic Clustering Tool](./keyword-clustering/semantic-clustering) | Group keywords into topical clusters using sentence transformers and ML | CLI / Colab |

### Link Building

| Tool | Description | Type |
|------|-------------|------|
| [eCommerce Link Builder](./linking/ecommerce-link-builder) | Find "Where to Buy" and distributor link opportunities | Python |
| [Wayback Machine Link Mapper](./linking/map-urls-wayback-machine) | Recover broken backlinks using archive.org historical data | Python / Colab |
| [Wikipedia Citation Finder](./linking/wikipedia-citation-finder) | Find Wikipedia pages with "citation needed" tags for link opportunities | Streamlit |

### On-Page SEO

| Tool | Description | Type |
|------|-------------|------|
| [Extract Content Blocks](./on-page/extract-content-blocks) | Extract and categorize page content blocks using Claude AI | Python |
| [Striking Distance Keywords](./on-page/striking-distance-keywords) | Find keywords ranking positions 4-20 and check if they're in title/H1/copy | Python / Colab |

### Reporting & Analytics

| Tool | Description | Type |
|------|-------------|------|
| [BCG Matrix from GA](./reporting/create-bcg-matrix-from-ga-landing-page-report) | Create BCG growth-share matrix from GA landing page data | Colab |
| [Google Trends Forecasting](./reporting/forecasting-google-trends-single-keyword) | Forecast search trends using NeuralProphet time-series ML | Streamlit |
| [Batch Trends Forecasting](./reporting/forecasting-google-trends-crawl-file) | Forecast Google Trends for multiple keywords from a crawl file | Streamlit |
| [Resolution Screenshot Tool](./reporting/pyppeteer-render-pages-by-most-common-resolutions-in-ga) | Screenshot pages at your visitors' most common screen resolutions | Python |
| [Top Traffic Pages (GSC)](./reporting/top-traffic-pages-search-console-sej) | Identify highest-traffic pages via Search Console API | Colab |
| [Visualise Internal Links](./reporting/visualise-links-screaming_frog) | Interactive treemap visualization of internal link structure | Colab |
| [Visualise GSC Coverage](./reporting/visualise-search-console-coverage-reports) | Treemap and sunburst charts to visualize indexing issues by folder | Colab |

### Search Console

| Tool | Description | Type |
|------|-------------|------|
| [Simple GSC Connector](./search-console/streamlit-simple-gsc-connector) | Minimal Streamlit boilerplate for Google Search Console OAuth | Streamlit |

### Site Search

| Tool | Description | Type |
|------|-------------|------|
| [Map Site Searches to Pages](./site-search/map-site-searches-to-landing-pages) | Match internal site searches to best landing pages using TF-IDF | Colab |

### Website Migration

| Tool | Description | Type |
|------|-------------|------|
| [Website Migration Mapper](./website-migration) | Auto-map URLs from old site to new using PolyFuzz grid search | Streamlit / Python / Colab |

### Wayback Machine

| Tool | Description | Type |
|------|-------------|------|
| [Wayback URL Extractor](./wayback-url-tool) | Bulk extract historical URLs from archive.org for any domain | Streamlit |

---

## Tool Types

| Type | Description | Run Command |
|------|-------------|-------------|
| **Streamlit** | Interactive web apps (6 live) | `streamlit run app.py` |
| **Colab** | Google Colab notebooks | Click "Open in Colab" badge |
| **Python** | Standalone scripts | `python script.py` |
| **CLI** | Command-line tools | `python script.py --args` |

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

## Contributing

Contributions, suggestions, and feedback are welcome!

- **Star** this repo if you find it useful
- **Open an issue** for bugs or feature requests
- **Submit a PR** with improvements

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <strong>Built for the SEO community</strong><br><br>
  <a href="https://www.leefoot.com">leefoot.com</a> ·
  <a href="https://x.com/LeeFootSEO">𝕏 @LeeFootSEO</a> ·
  <a href="https://www.linkedin.com/in/lee-foot/">LinkedIn</a>
</p>
