# Keyword Clustering Tools

Advanced semantic clustering solutions for organizing and analyzing large keyword datasets. These tools use machine learning and natural language processing to group semantically similar keywords, enabling more effective content strategy and SEO planning.

## Tools Overview

### 🧠 **Semantic Clustering Suite**
A comprehensive collection of keyword clustering tools powered by SentenceTransformers and various clustering algorithms.

#### **Features**
- **Multiple Clustering Methods**: Standard clustering and HDBScan for large datasets
- **Semantic Understanding**: Uses sentence embeddings for true semantic similarity
- **CLI & Notebook Versions**: Choose your preferred interface
- **Visualization**: Generate treemaps and sunburst charts for cluster analysis
- **Configurable Parameters**: Adjust similarity thresholds, models, and output formats

#### **Available Versions**
- **CLI Application**: Command-line interface for production workflows
- **HDBScan CLI**: Optimized for very large keyword datasets
- **Python Script**: Direct integration into existing workflows
- **Jupyter Notebooks**: Interactive analysis and experimentation

#### **Clustering Algorithms Supported**
- Standard semantic clustering with similarity thresholds
- HDBScan for density-based clustering of large datasets
- Configurable minimum similarity parameters

## Quick Start

### Basic Usage
```bash
cd semantic-clustering/semantic-clustering-cli-app/CLI
python cluster.py keywords.csv --column_name "Keyword" --output_path "output.csv"
```

### Advanced Configuration
```bash
python cluster.py keywords.csv \
  --column_name "Keyword" \
  --output_path "clustered_keywords.csv" \
  --chart_type "sunburst" \
  --device "cpu" \
  --model_name "all-MiniLM-L6-v2" \
  --min_similarity 0.80 \
  --remove_dupes True \
  --volume "Volume" \
  --stem True
```

### For Large Datasets (10k+ keywords)
```bash
cd semantic-clustering/semantic-clustering-cli-app/CLI-HDBScan
python cluster-hdbscan.py large_keywords.csv
```

## Use Cases

### 🎯 **Content Strategy**
- Group keywords into topical clusters for content planning
- Identify content gaps and opportunities
- Plan content pillars and supporting pages

### 📊 **SEO Analysis**
- Analyze competitor keyword strategies
- Group keywords by search intent
- Optimize content for semantic keyword groups

### 🔍 **PPC Campaign Organization**
- Create logical ad groups based on semantic similarity
- Improve Quality Scores through better keyword grouping
- Reduce keyword cannibalization

### 📈 **Content Optimization**
- Identify related keywords for existing content
- Optimize for semantic search and entities
- Improve topical authority

## Input Requirements

- **CSV file** with keyword data
- **Keyword column** containing the terms to cluster
- **Optional volume column** for weighted analysis
- **Minimum 50+ keywords** for meaningful clustering

## Output Formats

- **CSV file** with cluster assignments
- **Excel file** with pivot tables
- **Interactive visualizations** (treemap/sunburst charts)
- **Cluster statistics** and similarity scores

## Technical Specifications

### **Models Supported**
- all-MiniLM-L6-v2 (default, balanced speed/accuracy)
- all-mpnet-base-v2 (higher accuracy, slower)
- distilbert-base-nli-stsb-mean-tokens
- Custom SentenceTransformer models

### **Hardware Requirements**
- **CPU**: 4+ cores recommended for large datasets
- **RAM**: 8GB minimum, 16GB+ for large datasets
- **GPU**: Optional CUDA support for faster processing
- **Storage**: Varies by dataset size

### **Dependencies**
```bash
pip install sentence-transformers pandas numpy plotly scikit-learn hdbscan
```

## Legacy Versions

The repository includes several legacy implementations for reference and backward compatibility:
- Google Colab versions for cloud processing
- Search Engine Journal optimized versions
- Historical clustering approaches

## Performance Guidelines

### **Dataset Size Recommendations**
- **< 1,000 keywords**: Standard CLI version
- **1,000 - 10,000 keywords**: CLI with optimized settings
- **10,000+ keywords**: HDBScan version
- **100,000+ keywords**: Contact for enterprise solutions

### **Processing Time Estimates**
- 1,000 keywords: ~2-5 minutes
- 10,000 keywords: ~10-30 minutes
- 50,000 keywords: ~1-3 hours (HDBScan)

## Support & Documentation

For detailed implementation guides and advanced configurations, visit the tool-specific directories. Each version includes comprehensive documentation and example usage.

## Author

**Lee Foot** - eCommerce SEO Consultant

[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)
---

*Part of the Search Solved Public SEO toolkit - Advanced clustering for modern SEO workflows.*