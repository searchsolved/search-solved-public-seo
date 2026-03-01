# CLAUDE.md - Project Guidelines

## Overview

This is the public SEO tools repository containing Streamlit apps and Python scripts for SEO automation. Tools are released publicly on GitHub and some are deployed to Streamlit Cloud.

## Repository Structure

```
/content-analysis/       - Content analysis tools (entity extraction, duplication finder)
/ecommerce/              - E-commerce SEO tools (category suggester, title optimizer)
/internal-linking/       - Internal linking tools (anchor text interlinker)
/keyword-clustering/     - Keyword clustering tools (SERP clustering, semantic clustering)
/keyword-research/       - Keyword research tools (trends, difficulty, grouping)
/link-building/          - Link building tools (backlink intersector)
/on-page/                - On-page SEO tools (striking distance, content blocks)
/ppc/                    - PPC/AdWords tools
/reporting/              - Reporting tools (core update analyzer, SOV calculator)
/search-console/         - GSC tools (data exporter, folder analyzer, chart visualizer)
/technical-seo/          - Technical SEO tools (template fingerprinting, breadcrumb extractor)
/website-migration/      - Migration mapping tools
```

## Important Policies

### Private Code - DO NOT RELEASE

- **BERTlinker source code** - Only the SaaS at bertlinker.com is public. Source code remains private in the separate `seo/clients` repo.
- **Client-specific tools** - Tools in `/Users/leefoot/Documents/GitHub/seo/clients/` are private and should never be copied directly. They must be sanitized before public release.

### When Releasing Client Tools

1. Remove ALL hardcoded paths (e.g., `/python_scripts/`, `C:\python_scripts\`)
2. Remove ALL client names and domains
3. Remove ALL API keys and credentials
4. Add Streamlit UI for file uploads and configuration
5. Add requirements.txt
6. Update STREAMLIT_APPS.md
7. Add entry to tools.ts in website repo

### API Keys

Never commit API keys. If found, they must be revoked immediately. Use:
- Streamlit secrets for deployed apps
- Environment variables for local development
- User input fields for API keys in Streamlit UI

## Related Repositories

- **Website repo**: `/Users/leefoot/Documents/GitHub/lee-single-page-site/` - Contains tools.ts for the tools page
- **Private client repo**: `/Users/leefoot/Documents/GitHub/seo/clients/` - Private client work, never release directly

## Deployment

- Streamlit apps are tracked in `STREAMLIT_APPS.md`
- After creating a new tool, add it to:
  1. `STREAMLIT_APPS.md` (this repo)
  2. `src/data/tools.ts` (website repo)

## Tool Categories for tools.ts

Valid categories: `Internal Linking`, `Keyword Research`, `Content`, `Technical SEO`, `Reporting`, `E-commerce`, `Migration`, `Search Console`, `PPC`

## Common Dependencies

Most Streamlit apps use:
- `streamlit>=1.28.0`
- `pandas>=2.0.0`
- `polyfuzz>=0.4.0` (for fuzzy matching)
- `sentence-transformers` (for semantic tools)
