"""
Content Reviewer (LLM) - Streamlit App
Review and annotate web content using AI for quality improvements.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
from anthropic import Anthropic
from openai import OpenAI
import requests
import json
from io import BytesIO
from time import sleep

st.set_page_config(
    page_title="Content Reviewer (LLM)",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Content Reviewer (LLM)")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Reviews content quality using AI analysis
    - Checks for readability, SEO best practices
    - Provides actionable improvement suggestions

    **How to use:**
    1. Enter your OpenAI API key
    2. Paste content or upload URLs
    3. Select review criteria
    4. Generate detailed content review

    **Best for:**
    - Content quality assurance
    - Pre-publish content checks
    - Editorial workflow automation
    """)
st.markdown("Review web content and get AI-powered annotations with improvement suggestions.")

# Sidebar configuration
with st.sidebar:
    st.header("API Configuration")

    st.subheader("Content Scraping")
    firecrawl_key = st.text_input("Firecrawl API Key", type="password",
                                   help="Get from firecrawl.dev")

    st.subheader("AI Review")
    provider = st.selectbox("AI Provider", ["Anthropic (Claude)", "OpenAI (GPT)"])

    if provider == "Anthropic (Claude)":
        ai_key = st.text_input("Anthropic API Key", type="password")
        model = st.selectbox("Model", ["claude-sonnet-4-5-20250929", "claude-sonnet-4-20250514", "claude-haiku-4-5-20251015"])
    else:
        ai_key = st.text_input("OpenAI API Key", type="password")
        model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-4.1"])

    st.header("Review Settings")
    review_type = st.selectbox("Review Focus", [
        "SEO Content Quality",
        "Technical Accuracy",
        "User Experience",
        "Conversion Optimization",
        "Custom"
    ])

    if review_type == "Custom":
        custom_focus = st.text_area("Custom Review Focus",
                                     placeholder="Describe what aspects to review...")


def scrape_url(url, api_key):
    """Scrape URL using Firecrawl API."""
    api_url = "https://api.firecrawl.dev/v1/scrape"

    payload = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": True,
        "timeout": 30000,
        "blockAds": True,
        "removeBase64Images": True,
        "includeTags": ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table", "blockquote"],
        "waitFor": 2000
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()

        if data.get('success'):
            return {
                'content': data.get('data', {}).get('markdown', ''),
                'metadata': data.get('data', {}).get('metadata', {})
            }, None
        else:
            return None, data.get('error', 'Unknown error')
    except Exception as e:
        return None, str(e)


def get_review_prompt(review_type, custom_focus=None):
    """Get the review prompt based on focus type."""
    base_prompts = {
        "SEO Content Quality": """Focus on:
- Keyword usage and placement
- Header structure (H1, H2, H3)
- Content comprehensiveness
- Internal linking opportunities
- Meta description alignment
- Readability and engagement""",

        "Technical Accuracy": """Focus on:
- Factual correctness
- Technical terminology usage
- Data and statistics accuracy
- Source citations needed
- Outdated information
- Missing technical details""",

        "User Experience": """Focus on:
- Content flow and structure
- Clarity of explanations
- Call-to-action placement
- Visual content suggestions
- Mobile readability
- Scannability (bullets, headers)""",

        "Conversion Optimization": """Focus on:
- Value proposition clarity
- Trust signals and social proof
- Call-to-action effectiveness
- Objection handling
- Benefits vs features
- Urgency and scarcity elements"""
    }

    return base_prompts.get(review_type, custom_focus or "General content quality review")


def review_with_claude(client, model, content, metadata, review_focus):
    """Review content using Claude."""
    prompt = f"""Create an annotated version of the provided content with inline review comments and suggestions.

## Source Context
**URL**: {metadata.get('url', 'N/A')}
**Title**: {metadata.get('title', 'N/A')}

## Review Focus
{review_focus}

## Annotation Format
Use these annotation styles:

**For issues:**
<!-- ISSUE: [specific problem and suggested fix] -->

**For missing elements:**
<!-- MISSING: [what should be added here] -->

**For strengths:**
<!-- STRENGTH: [why this works well] -->

**For improvement suggestions:**
<!-- IMPROVE: [specific enhancement recommendation] -->

**For SEO opportunities:**
<!-- SEO: [optimization opportunity] -->

## Output Structure
1. Preserve all original headings, text, and formatting
2. Add annotations immediately after relevant sections
3. Add a summary section at the end with:
   - Overall assessment (2-3 sentences)
   - Top 5 priority improvements
   - Quick wins that can be implemented immediately

## Content to Review

{content}

Create the annotated version maintaining the original document structure while providing actionable insights."""

    try:
        message = client.messages.create(
            model=model,
            max_tokens=8000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text, None
    except Exception as e:
        return None, str(e)


def review_with_openai(client, model, content, metadata, review_focus):
    """Review content using OpenAI."""
    prompt = f"""Create an annotated version of the provided content with inline review comments and suggestions.

## Source Context
**URL**: {metadata.get('url', 'N/A')}
**Title**: {metadata.get('title', 'N/A')}

## Review Focus
{review_focus}

## Annotation Format
Use these annotation styles:

**For issues:**
<!-- ISSUE: [specific problem and suggested fix] -->

**For missing elements:**
<!-- MISSING: [what should be added here] -->

**For strengths:**
<!-- STRENGTH: [why this works well] -->

**For improvement suggestions:**
<!-- IMPROVE: [specific enhancement recommendation] -->

**For SEO opportunities:**
<!-- SEO: [optimization opportunity] -->

## Output Structure
1. Preserve all original headings, text, and formatting
2. Add annotations immediately after relevant sections
3. Add a summary section at the end with:
   - Overall assessment (2-3 sentences)
   - Top 5 priority improvements
   - Quick wins that can be implemented immediately

## Content to Review

{content}

Create the annotated version maintaining the original document structure while providing actionable insights."""

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert content reviewer providing detailed, actionable feedback."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=8000,
            temperature=0.1
        )
        return completion.choices[0].message.content, None
    except Exception as e:
        return None, str(e)


# Main interface
tab1, tab2 = st.tabs(["Single URL Review", "Bulk Review"])

with tab1:
    st.subheader("Review a Single Page")

    col1, col2 = st.columns([3, 1])
    with col1:
        url = st.text_input("Enter URL to review", placeholder="https://example.com/page")
    with col2:
        scrape_btn = st.button("Fetch & Review", type="primary",
                               disabled=not firecrawl_key or not ai_key or not url)

    # Option to paste content directly
    with st.expander("Or paste content directly"):
        pasted_content = st.text_area("Paste markdown or text content", height=300)
        pasted_title = st.text_input("Page title (optional)")
        review_pasted = st.button("Review Pasted Content",
                                   disabled=not ai_key or not pasted_content)

    if scrape_btn and url:
        with st.spinner("Fetching content..."):
            result, error = scrape_url(url, firecrawl_key)

        if error:
            st.error(f"Failed to fetch URL: {error}")
        elif result:
            content = result['content']
            metadata = result['metadata']
            metadata['url'] = url

            st.success(f"Fetched {len(content)} characters from: {metadata.get('title', url)}")

            with st.expander("View Raw Content"):
                st.markdown(content[:5000] + "..." if len(content) > 5000 else content)

            with st.spinner("Analyzing content..."):
                review_focus = get_review_prompt(review_type,
                                                  custom_focus if review_type == "Custom" else None)

                if provider == "Anthropic (Claude)":
                    client = Anthropic(api_key=ai_key)
                    review, error = review_with_claude(client, model, content, metadata, review_focus)
                else:
                    client = OpenAI(api_key=ai_key)
                    review, error = review_with_openai(client, model, content, metadata, review_focus)

            if error:
                st.error(f"Review failed: {error}")
            elif review:
                st.success("Review completed!")

                st.markdown("### Annotated Review")
                st.markdown(review)

                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("Download Review (Markdown)",
                                       review,
                                       f"review_{metadata.get('title', 'content')[:30]}.md",
                                       "text/markdown")
                with col2:
                    # Create combined report
                    full_report = f"""# Content Review Report

**URL**: {url}
**Title**: {metadata.get('title', 'N/A')}
**Review Focus**: {review_type}

---

## Original Content

{content}

---

## Annotated Review

{review}
"""
                    st.download_button("Download Full Report",
                                       full_report,
                                       f"full_review_{metadata.get('title', 'content')[:30]}.md",
                                       "text/markdown")

    if review_pasted and pasted_content:
        metadata = {'title': pasted_title or 'Pasted Content', 'url': 'N/A'}

        with st.spinner("Analyzing content..."):
            review_focus = get_review_prompt(review_type,
                                              custom_focus if review_type == "Custom" else None)

            if provider == "Anthropic (Claude)":
                client = Anthropic(api_key=ai_key)
                review, error = review_with_claude(client, model, pasted_content, metadata, review_focus)
            else:
                client = OpenAI(api_key=ai_key)
                review, error = review_with_openai(client, model, pasted_content, metadata, review_focus)

        if error:
            st.error(f"Review failed: {error}")
        elif review:
            st.success("Review completed!")
            st.markdown("### Annotated Review")
            st.markdown(review)

            st.download_button("Download Review (Markdown)",
                               review,
                               "review_pasted_content.md",
                               "text/markdown")

with tab2:
    st.subheader("Bulk URL Review")

    uploaded_file = st.file_uploader("Upload CSV/Excel with URLs", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write(f"Loaded {len(df)} rows")

            url_col = st.selectbox("URL Column", list(df.columns))

            max_urls = st.slider("Maximum URLs to process", 1, min(50, len(df)), min(10, len(df)))

            if st.button("Start Bulk Review", type="primary",
                        disabled=not firecrawl_key or not ai_key):

                urls = df[url_col].dropna().head(max_urls).tolist()
                results = []

                progress = st.progress(0)
                status = st.empty()

                for idx, url in enumerate(urls):
                    status.text(f"Processing {idx + 1}/{len(urls)}: {url[:50]}...")

                    # Fetch content
                    result, error = scrape_url(url, firecrawl_key)

                    if error:
                        results.append({
                            'url': url,
                            'status': 'Failed to fetch',
                            'error': error,
                            'review': ''
                        })
                    elif result:
                        content = result['content']
                        metadata = result['metadata']
                        metadata['url'] = url

                        # Review content
                        review_focus = get_review_prompt(review_type,
                                                          custom_focus if review_type == "Custom" else None)

                        if provider == "Anthropic (Claude)":
                            client = Anthropic(api_key=ai_key)
                            review, rev_error = review_with_claude(client, model, content, metadata, review_focus)
                        else:
                            client = OpenAI(api_key=ai_key)
                            review, rev_error = review_with_openai(client, model, content, metadata, review_focus)

                        results.append({
                            'url': url,
                            'title': metadata.get('title', ''),
                            'status': 'Success' if review else 'Review failed',
                            'content_length': len(content),
                            'review': review or '',
                            'error': rev_error or ''
                        })

                    progress.progress((idx + 1) / len(urls))
                    sleep(2)  # Rate limiting

                status.text("Complete!")

                # Display results
                results_df = pd.DataFrame(results)
                st.dataframe(results_df[['url', 'title', 'status', 'content_length']],
                            use_container_width=True)

                # Download
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("Download Results (CSV)",
                                       results_df.to_csv(index=False),
                                       "bulk_reviews.csv",
                                       "text/csv")
                with col2:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        results_df.to_excel(writer, index=False)
                    st.download_button("Download Results (Excel)",
                                       output.getvalue(),
                                       "bulk_reviews.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"Error loading file: {e}")

# Help section
with st.expander("How to Use"):
    st.markdown("""
    **Getting API Keys:**
    - **Firecrawl**: Sign up at [firecrawl.dev](https://firecrawl.dev) for web scraping
    - **Anthropic**: Get key from [console.anthropic.com](https://console.anthropic.com)
    - **OpenAI**: Get key from [platform.openai.com](https://platform.openai.com)

    **Review Focus Types:**
    - **SEO Content Quality**: Keyword usage, headers, comprehensiveness
    - **Technical Accuracy**: Facts, data, citations, terminology
    - **User Experience**: Flow, clarity, scannability
    - **Conversion Optimization**: CTAs, value props, trust signals

    **Output:**
    - Annotated content with inline comments
    - Priority improvements list
    - Quick wins for immediate action
    """)

# Footer
st.markdown("---")
