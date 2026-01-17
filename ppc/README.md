# PPC Tools

Tools for optimizing Google Ads campaigns, extracting insights from Search Term Reports, and calculating optimal bid adjustments.

## Tools Overview

### 💰 **AdWords Tool Set**
Two-in-one Google Ads toolkit for keyword extraction and bid optimization.
- **Use Case**: PPC campaign optimization, keyword mining, bid management
- **Input**: Google Ads Search Term Report CSV, Device/Location/Time reports
- **Output**: Extracted keyphrases, MPNs, brands, optimal bid adjustments
- **Features**: RaKUn2 NLP keyphrase detection, bid adjustment calculator

### Features

#### 📝 **MPN & Keyphrase Extraction**
Extract MPNs, brand names, and keyphrases from Search Term Reports using RaKUn2 NLP.
- Identifies product identifiers (MPNs, SKUs)
- Extracts brand mentions
- Discovers high-value keyphrases
- Bulk processing support

#### 📊 **Bid Adjustment Calculator**
Calculate optimal bid adjustments for device, location, and time of day.
- Device-based adjustments (mobile, desktop, tablet)
- Location-based bid modifiers
- Time-of-day optimization
- Data-driven recommendations

## Use Cases

### 🎯 **Keyword Mining**
- Extract valuable keywords from Search Term Reports
- Identify product-specific terms (MPNs, brands)
- Discover new keyword opportunities

### 💵 **Bid Optimization**
- Calculate device bid adjustments based on performance
- Optimize location targeting with data-driven bids
- Improve ROI with time-based bid modifiers

### 📈 **Campaign Analysis**
- Understand which products drive conversions
- Identify high-performing search patterns
- Optimize budget allocation

## Quick Start

```bash
cd adwords-tools
pip install -r requirements.txt
streamlit run Home.py
```

## Input Requirements

### **MPN & Keyphrase Extraction**
- Google Ads Search Term Report (CSV export)
- Campaign or ad group filters (optional)

### **Bid Adjustment Calculator**
- Device/Location/Time performance report
- Cost per conversion data
- Current bid modifiers

## Output Formats

- **CSV files** with extracted keyphrases and MPNs
- **Bid adjustment recommendations** with calculations
- **Analysis reports** with insights

## Technical Details

### **RaKUn2 NLP**
Uses RaKUn2 (Rapid Keyword Extraction using Unsupervised Learning) for intelligent keyphrase detection:
- Language-agnostic approach
- No training data required
- Fast processing of large datasets

### **Bid Calculations**
Data-driven bid adjustment formula based on:
- Conversion rate by segment
- Cost per conversion targets
- Statistical significance thresholds

## Best Practices

### 📝 **Keyword Extraction**
1. Export Search Term Reports regularly
2. Filter by conversion data for quality keywords
3. Review extracted terms before adding to campaigns
4. Use negative keywords for irrelevant terms

### 💰 **Bid Optimization**
1. Ensure sufficient data before adjusting bids
2. Make incremental changes (10-20% max)
3. Monitor performance after changes
4. Re-calculate adjustments monthly

## Support & Documentation

Detailed setup instructions are included in the tool directory. For advanced implementations or custom requirements, visit [leefoot.com](https://www.leefoot.com).

## Author

**Lee Foot** - eCommerce SEO Consultant with expertise in PPC optimization and data analysis.

- 🌐 [Website](https://www.leefoot.com)
- 🐦 [Twitter/X](https://x.com/LeeFootSEO)
- 🦋 [Bluesky](https://bsky.app/profile/leefootseo.bsky.social)
- 💼 [LinkedIn](https://www.linkedin.com/in/lee-foot/)
- ✉️ [Contact](https://www.leefoot.com/contact)

---

*Part of the Search Solved Public SEO toolkit - PPC optimization and keyword analysis.*
