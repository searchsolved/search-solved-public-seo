# Internal Linking Tools

AI-powered tools for optimizing internal link architecture and anchor text strategy. These tools help identify linking opportunities and assess anchor text relevance using advanced language models.

## Tools Overview

### 🔗 **Anchor Text Interlinker**
Automatically find internal linking opportunities based on anchor text matching.
- **Use Case**: Scale internal linking, find contextual link opportunities
- **Input**: Crawl data with page content
- **Output**: Link recommendations with source/target URLs and anchor text
- **Features**: Fuzzy matching, bulk processing, relevance scoring

### 🎯 **Anchor Text Relevance Checker**
Assess anchor text relevance for internal linking using AI (GPT-4).
- **Use Case**: Internal link audits, anchor text optimization
- **Input**: Anchor text and target page pairs, OpenAI API key
- **Output**: Relevance ratings (High/Medium/Fail/Typo) with explanations
- **Features**: GPT-4 relevance scoring, suggested improvements, batch processing

## Use Cases

### 🔍 **Internal Link Audits**
- Assess existing anchor text quality
- Identify weak or irrelevant anchors
- Get AI-powered improvement suggestions

### 📈 **Link Building Strategy**
- Find pages that should link to each other
- Optimize anchor text for target keywords
- Scale internal linking across large sites

### 👥 **Team Training**
- Train content teams on anchor best practices
- Establish consistent linking guidelines
- Review link implementations at scale

## Quick Start

### Anchor Text Interlinker
```bash
cd anchor-text-interlinker
pip install -r requirements.txt
streamlit run anchor_text_interlinker.py
```

### Anchor Text Relevance Checker
```bash
cd anchor-text-relevance-checker
pip install -r requirements.txt
streamlit run anchor_text_relevance_checker.py
```

## Input Requirements

### **Anchor Text Interlinker**
- Screaming Frog crawl export
- Page content data (H1s, body text)

### **Anchor Text Relevance Checker**
- CSV with anchor text and target URL pairs
- OpenAI API key for GPT-4 analysis

## Output Formats

- **CSV files** with link recommendations
- **Relevance scores** with explanations
- **Suggested anchor text** improvements

## Best Practices

### 🔗 **Anchor Text Optimization**
1. Use descriptive, relevant anchor text
2. Avoid generic anchors like "click here"
3. Match anchor text to target page topic
4. Vary anchor text naturally

### 📊 **Internal Linking Strategy**
1. Link from high-authority pages to important targets
2. Create topical clusters with hub pages
3. Ensure important pages are within 3 clicks
4. Regular audits to find broken or weak links

## Support & Documentation

Each tool includes detailed setup instructions in its respective directory. For advanced implementations or custom requirements, visit [leefoot.com](https://www.leefoot.com).

## Author

**Lee Foot** - eCommerce SEO Consultant specializing in internal linking and site architecture.

- 🌐 [Website](https://www.leefoot.com)
- 🐦 [Twitter/X](https://x.com/LeeFootSEO)
- 🦋 [Bluesky](https://bsky.app/profile/leefootseo.bsky.social)
- 💼 [LinkedIn](https://www.linkedin.com/in/lee-foot/)
- ✉️ [Contact](https://www.leefoot.com/contact)

---

*Part of the Search Solved Public SEO toolkit - Internal linking automation and optimization.*
