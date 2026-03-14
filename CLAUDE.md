# CLAUDE.md - Project Guidelines

## Overview

This is the public SEO tools repository containing 100+ Streamlit apps, Python scripts, and APIs for SEO automation. Tools are released publicly on GitHub and some are deployed to Streamlit Cloud.

## Repository Structure

```
/competitive-analysis/   - Competitor keyword and SERP analysis tools
/competitor-gap-finder/  - Product title gap analysis
/content-analysis/       - Content analysis tools (entity extraction, sentiment, readability)
/content-consolidation/  - Content consolidation opportunity finder
/ecommerce/              - E-commerce SEO tools (category suggester, title optimizer, specs)
/internal-linking/       - Internal linking tools (anchor text interlinker, relevance checker)
/keyword-clustering/     - Keyword clustering tools (SERP clustering, semantic clustering)
/keyword-research/       - Keyword research tools (trends, difficulty, grouping, PAA, topics)
/link-building/          - Link building tools (backlink intersector, link quality)
/linking/                - Link outreach tools (ecommerce links, Wayback, Wikipedia citations)
/on-page/                - On-page SEO tools (striking distance, content blocks)
/ppc/                    - PPC/AdWords tools
/product-title-optimizer/- LLM-powered title restructuring
/reporting/              - Reporting tools (core updates, SOV, trends, decay, algorithm tracker)
/search-console/         - GSC tools (data exporter, folder analyzer, cannibalization, charts)
/search-engine-journal/  - Tools featured in SEJ publications
/site-search/            - Site search analysis tools
/technical-seo/          - Technical SEO (hreflang, redirects, schema, sitemaps, regex, OnCrawl)
/wayback-url-tool/       - Wayback Machine URL extractor
/website-migration/      - Migration mapping tools
```

## Important Policies

### Private Code - DO NOT RELEASE

- **BERTlinker source code** — Only the SaaS at bertlinker.com is public. Source code is private.
- **Client-specific tools** — Must be fully sanitized before public release.

### When Releasing Tools

1. Remove ALL hardcoded file paths
2. Remove ALL client names and domains
3. Remove ALL API keys and credentials
4. Add Streamlit UI for file uploads and configuration
5. Add `requirements.txt`
6. Update `STREAMLIT_APPS.md`

### API Keys

Never commit API keys. If found, they must be revoked immediately. Use:
- Streamlit secrets for deployed apps
- Environment variables for local development
- User input fields for API keys in Streamlit UI

## Deployment

- Streamlit apps are tracked in `STREAMLIT_APPS.md`
- After creating a new tool, add it to `STREAMLIT_APPS.md`

## Tool Categories

Valid categories: `Internal Linking`, `Keyword Research`, `Content`, `Technical SEO`, `Reporting`, `E-commerce`, `Migration`, `Search Console`, `PPC`, `Competitive Analysis`, `Link Building`, `On-Page`

## Common Dependencies

Most Streamlit apps use:
- `streamlit>=1.28.0`
- `pandas>=2.0.0`
- `polyfuzz>=0.4.0` (for fuzzy matching)
- `sentence-transformers` (for semantic tools)
