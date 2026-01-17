# Content Analysis Tools

Advanced AI-powered tools for analyzing, classifying, and optimizing content at scale. This collection features LLM-powered tools using Claude and GPT models for intelligent content processing, entity extraction, and automated content optimization.

## Tools Overview

### AI & LLM-Powered Tools

#### 🤖 **AI Entity Visualizer**
Extract and visualize named entities using GPT with beautiful D3.js circle packing visualization.
- **Use Case**: Content entity analysis and visualization
- **Input**: Text content or URLs
- **Output**: Interactive D3.js circle packing visualization with entity labels
- **Features**: GPT-3.5/GPT-4 extraction, SpaCy-style labels, Wikipedia URLs

#### 📝 **Content Hub Classifier**
Categorize articles and content into topic hubs using GPT.
- **Use Case**: Organize content into topical clusters
- **Input**: URLs or content text, category definitions
- **Output**: Classified content with confidence scores
- **Features**: GPT classification, custom categories, hub identification

#### 🔍 **Content Reviewer (LLM)**
Review web content and get AI-powered annotations with improvement suggestions.
- **Use Case**: Content audits, SEO optimization suggestions
- **Input**: URLs, Firecrawl API key, Claude/OpenAI API key
- **Output**: Annotated reviews with inline suggestions
- **Features**: Firecrawl integration, Claude or GPT-4 review, SEO/UX/conversion tips

#### 🔑 **Keyword Extractor (LLM)**
Extract and categorize keywords from content for internal linking opportunities.
- **Use Case**: Internal linking strategy, content gap analysis
- **Input**: Content text, existing site URLs
- **Output**: Keywords categorized as link opportunities or new page ideas
- **Features**: GPT or Claude extraction, link opportunity identification

#### ✨ **Gist Summary Generator**
Create "At a glance" bullet point summaries from articles using AI.
- **Use Case**: Featured snippets, newsletters, content previews
- **Input**: Article URLs or text
- **Output**: Concise bullet point summaries
- **Features**: Claude or GPT-4 summaries, configurable bullet count

#### 🔄 **Content Repurposer**
Transform blog posts into social media, email, video scripts, and more.
- **Use Case**: Multi-channel content distribution
- **Input**: Blog post URL or content
- **Output**: 10 different content formats (Twitter threads, LinkedIn posts, etc.)
- **Features**: Claude and GPT-4 support, platform-specific optimization

#### 📊 **Meta Description Grader**
Score and compare meta descriptions using GPT-4 on key SEO criteria.
- **Use Case**: Meta description optimization, team training
- **Input**: Meta descriptions to analyze
- **Output**: Scores for emotional hook, benefit statement, active voice, urgency
- **Features**: GPT-4 powered grading, actionable feedback

#### ✏️ **Meta Description Rewriter**
Rewrite meta descriptions using AI with proper length optimization.
- **Use Case**: Bulk meta description optimization
- **Input**: Existing meta descriptions
- **Output**: Optimized meta descriptions for desktop and mobile
- **Features**: GPT-4 rewriting, length targets, SEO best practices

#### 🎯 **Page Intent Classifier**
Use OpenAI to classify page intent and expected user actions.
- **Use Case**: Content audits, information architecture planning
- **Input**: URLs to analyze
- **Output**: Intent classification (signup, purchase, browse, learn, etc.)
- **Features**: OpenAI GPT integration, action prediction

#### 💬 **Review Sentiment Extractor**
Use OpenAI to extract positive and negative sentiments from product reviews.
- **Use Case**: Product feedback analysis, content optimization
- **Input**: Customer reviews
- **Output**: Praise points and pain points extracted
- **Features**: Batch processing, sentiment categorization

### Content Processing Tools

#### 📑 **Category Title Suggester**
Analyze category pages and suggest high-performing keywords for titles.
- **Use Case**: Category page optimization
- **Input**: GSC data, category page titles
- **Output**: Keyword suggestions based on performance data
- **Features**: GSC performance analysis, title optimization

#### 🔍 **Content Duplication Finder**
Find duplicate and near-duplicate content across your site.
- **Use Case**: Content audits, consolidation planning
- **Input**: Screaming Frog crawl with custom extraction
- **Output**: Duplicate content pairs with similarity scores
- **Features**: TF-IDF similarity, cluster naming

#### 📄 **Content Extractor**
Extract main text content and H1 headings from URLs.
- **Use Case**: Content analysis, migration preparation
- **Input**: List of URLs
- **Output**: Extracted main content and headings
- **Features**: Trafilatura extraction, bulk processing

#### 🏢 **Entity Extractor**
Extract entities from SERPs, CSV files, or YouTube transcripts.
- **Use Case**: Content research, entity optimization
- **Input**: SERP data, CSV files, or YouTube URLs
- **Output**: Extracted entities with frequency analysis
- **Features**: Multiple input sources, SpaCy NER

#### 📊 **N-gram SERP Extractor**
Extract and analyze n-grams from SERP results.
- **Use Case**: Content gap analysis, keyword research
- **Input**: SERP data exports
- **Output**: N-gram frequency analysis
- **Features**: Configurable n-gram sizes, frequency filtering

#### 🔗 **Keyword to Page Mapper**
Map keywords to relevant pages on your site using semantic matching.
- **Use Case**: Content planning, keyword targeting
- **Input**: Keyword list, site pages
- **Output**: Keyword-to-page mapping with relevance scores
- **Features**: TF-IDF matching, bulk processing

#### 📖 **Reading Score Analyzer**
Calculate readability metrics including Flesch Reading Ease for your content.
- **Use Case**: Content accessibility, audience targeting
- **Input**: URLs or sitemap
- **Output**: Multiple readability scores per page
- **Features**: Trafilatura extraction, Flesch scores, bulk analysis

## Use Cases

### 🎯 **Content Optimization**
- Grade and rewrite meta descriptions at scale
- Classify page intent for better user experience
- Generate AI-powered content summaries

### 📈 **Content Strategy**
- Classify content into topical hubs
- Extract entities for content optimization
- Map keywords to existing content

### 🔄 **Content Repurposing**
- Transform blog posts into social media content
- Create newsletter summaries from articles
- Generate video scripts from written content

### 🔍 **Content Audits**
- Find duplicate content across your site
- Analyze content readability
- Get AI-powered improvement suggestions

## Quick Start

### AI Entity Visualizer
```bash
cd openai-entity-visualizer
pip install -r requirements.txt
streamlit run app.py
```

### Content Reviewer (LLM)
```bash
cd content-reviewer-llm
pip install -r requirements.txt
streamlit run content_reviewer_llm.py
```

### Meta Description Grader
```bash
cd meta-description-grader
pip install -r requirements.txt
streamlit run meta_description_grader.py
```

## Input Requirements

### **API Keys Required**
Most LLM-powered tools require:
- OpenAI API key (for GPT models)
- Anthropic API key (for Claude models)
- Firecrawl API key (for content fetching)

### **Common Input Formats**
- CSV files with URLs or content
- Direct text input
- Screaming Frog crawl exports

## Output Formats

- **CSV files** with analysis results
- **Excel workbooks** with multiple sheets
- **Interactive visualizations** (D3.js, charts)
- **Downloadable content** in multiple formats

## Technical Specifications

### **Dependencies**
```bash
# Common dependencies
pip install pandas openai anthropic streamlit

# Specific tool requirements
pip install trafilatura beautifulsoup4  # Content extraction
pip install spacy                        # Entity extraction
pip install scikit-learn                 # TF-IDF matching
```

### **LLM Configuration**
- Support for GPT-3.5, GPT-4, and GPT-4o
- Support for Claude 3 models
- Configurable temperature and token limits
- Batch processing for efficiency

## Best Practices

### 🤖 **Using LLM Tools**
1. Start with smaller batches to test output quality
2. Review and refine prompts for your specific use case
3. Validate AI outputs before bulk implementation
4. Monitor API costs for large-scale processing

### 📊 **Content Analysis**
1. Combine multiple tools for comprehensive audits
2. Use entity extraction to inform content strategy
3. Regular readability checks for audience alignment
4. Track meta description performance after optimization

## Support & Documentation

Each tool includes detailed setup instructions in its respective directory. For advanced implementations or custom requirements, visit [leefoot.com](https://www.leefoot.com).

## Author

**Lee Foot** - eCommerce SEO Consultant specializing in AI-powered content analysis and optimization.

- 🌐 [Website](https://www.leefoot.com)
- 🐦 [Twitter/X](https://x.com/LeeFootSEO)
- 🦋 [Bluesky](https://bsky.app/profile/leefootseo.bsky.social)
- 💼 [LinkedIn](https://www.linkedin.com/in/lee-foot/)
- ✉️ [Contact](https://www.leefoot.com/contact)

---

*Part of the Search Solved Public SEO toolkit - AI-powered content analysis and optimization.*
