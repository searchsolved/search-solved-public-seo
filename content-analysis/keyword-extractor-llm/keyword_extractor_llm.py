"""
Keyword Extractor (LLM) - Streamlit App
Extract and categorize keywords from content using AI.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
from openai import OpenAI
from anthropic import Anthropic
import json
from io import BytesIO
from time import sleep

st.set_page_config(
    page_title="Keyword Extractor (LLM)",
    page_icon="🔑",
    layout="wide"
)

st.title("🔑 Keyword Extractor (LLM)")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Extracts relevant keywords from content using AI
    - Identifies primary, secondary, and LSI keywords
    - Suggests keyword optimization opportunities

    **How to use:**
    1. Enter your OpenAI API key
    2. Paste content or upload URLs
    3. Click "Extract Keywords"
    4. Review categorized keyword suggestions

    **Best for:**
    - On-page SEO optimization
    - Content keyword gap analysis
    - Semantic keyword research
    """)
st.markdown("Extract keywords from content and categorize them for internal linking or new page creation.")

# Sidebar configuration
with st.sidebar:
    st.header("API Configuration")

    provider = st.selectbox("AI Provider", ["OpenAI (GPT)", "Anthropic (Claude)"])

    if provider == "OpenAI (GPT)":
        api_key = st.text_input("OpenAI API Key", type="password")
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1"])
    else:
        api_key = st.text_input("Anthropic API Key", type="password")
        model = st.selectbox("Model", ["claude-sonnet-4-5-20250929", "claude-sonnet-4-20250514", "claude-haiku-4-5-20251015"])

    st.header("Extraction Settings")
    min_word_count = st.slider("Minimum words per keyword", 2, 5, 2)
    max_keywords = st.slider("Max keywords per category", 10, 50, 20)

    st.header("Output Categories")
    st.markdown("""
    - **Internal Links**: Keywords that could link to existing pages
    - **New Pages**: Keywords suggesting new content opportunities
    """)


def extract_keywords_openai(client, model, content, url="", h1="", existing_anchors=None, min_words=2, max_kw=20):
    """Extract keywords using OpenAI with structured output."""

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "keyword_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "internal_link_opportunities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Multi-word keywords that could be internal link anchors"
                    },
                    "new_page_ideas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keywords suggesting topics for new pages"
                    }
                },
                "required": ["internal_link_opportunities", "new_page_ideas"],
                "additionalProperties": False
            }
        }
    }

    system_prompt = f"""You are an SEO expert extracting keywords from content.

Rules:
1. Extract multi-word phrases ({min_words}+ words) that appear in the content
2. Keywords must be relevant and valuable for SEO
3. Limit to {max_kw} keywords per category
4. For internal links: phrases that could serve as anchor text to other pages
5. For new pages: topics mentioned that warrant dedicated content
6. Exclude generic phrases, brand names, and navigation text
7. All keywords must actually appear in the source content"""

    user_prompt = f"""Extract keywords from this content:

URL: {url}
H1: {h1}

Content:
{content[:8000]}

Return only keywords that appear verbatim in the content."""

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=response_format
        )
        result = json.loads(completion.choices[0].message.content)

        # Verify keywords exist in content
        content_lower = content.lower()
        result['internal_link_opportunities'] = [
            kw for kw in result['internal_link_opportunities']
            if kw.lower() in content_lower
        ]
        result['new_page_ideas'] = [
            kw for kw in result['new_page_ideas']
            if kw.lower() in content_lower
        ]

        # Filter against existing anchors
        if existing_anchors:
            anchors_lower = [a.lower() for a in existing_anchors]
            result['internal_link_opportunities'] = [
                kw for kw in result['internal_link_opportunities']
                if not any(kw.lower() in a or a in kw.lower() for a in anchors_lower)
            ]
            result['new_page_ideas'] = [
                kw for kw in result['new_page_ideas']
                if not any(kw.lower() in a or a in kw.lower() for a in anchors_lower)
            ]

        return result, None
    except Exception as e:
        return None, str(e)


def extract_keywords_claude(client, model, content, url="", h1="", existing_anchors=None, min_words=2, max_kw=20):
    """Extract keywords using Claude."""

    system_prompt = f"""You are an SEO expert extracting keywords from content.
Return ONLY valid JSON with no markdown or explanation.

Rules:
1. Extract multi-word phrases ({min_words}+ words) that appear in the content
2. Keywords must be relevant and valuable for SEO
3. Limit to {max_kw} keywords per category
4. All keywords must actually appear in the source content"""

    user_prompt = f"""Extract keywords from this content:

URL: {url}
H1: {h1}

Content:
{content[:8000]}

Return JSON only:
{{
    "internal_link_opportunities": ["keyword1", "keyword2"],
    "new_page_ideas": ["keyword1", "keyword2"]
}}

internal_link_opportunities = phrases good for anchor text to other pages
new_page_ideas = topics that warrant new dedicated content"""

    try:
        message = client.messages.create(
            model=model,
            max_tokens=2000,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        response_text = message.content[0].text.strip()

        # Clean response
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        if response_text.endswith("```"):
            response_text = response_text.rsplit("```", 1)[0]

        result = json.loads(response_text.strip())

        # Verify keywords exist in content
        content_lower = content.lower()
        result['internal_link_opportunities'] = [
            kw for kw in result.get('internal_link_opportunities', [])
            if kw.lower() in content_lower
        ]
        result['new_page_ideas'] = [
            kw for kw in result.get('new_page_ideas', [])
            if kw.lower() in content_lower
        ]

        # Filter against existing anchors
        if existing_anchors:
            anchors_lower = [a.lower() for a in existing_anchors]
            result['internal_link_opportunities'] = [
                kw for kw in result['internal_link_opportunities']
                if not any(kw.lower() in a or a in kw.lower() for a in anchors_lower)
            ]
            result['new_page_ideas'] = [
                kw for kw in result['new_page_ideas']
                if not any(kw.lower() in a or a in kw.lower() for a in anchors_lower)
            ]

        return result, None
    except Exception as e:
        return None, str(e)


# Main interface
tab1, tab2 = st.tabs(["Single Content", "Bulk Extraction"])

with tab1:
    st.subheader("Extract Keywords from Content")

    content_input = st.text_area("Paste content to analyze",
                                  height=300,
                                  placeholder="Paste your article or page content here...")

    col1, col2 = st.columns(2)
    with col1:
        url_input = st.text_input("Page URL (optional)", placeholder="https://example.com/page")
    with col2:
        h1_input = st.text_input("H1 / Title (optional)", placeholder="Page heading")

    existing_anchors_input = st.text_input("Existing anchor texts to exclude (comma-separated)",
                                            placeholder="anchor 1, anchor 2, anchor 3")

    if st.button("Extract Keywords", type="primary", disabled=not api_key or not content_input):
        existing_anchors = [a.strip() for a in existing_anchors_input.split(',') if a.strip()] if existing_anchors_input else None

        with st.spinner("Extracting keywords..."):
            if provider == "OpenAI (GPT)":
                client = OpenAI(api_key=api_key)
                result, error = extract_keywords_openai(
                    client, model, content_input, url_input, h1_input,
                    existing_anchors, min_word_count, max_keywords
                )
            else:
                client = Anthropic(api_key=api_key)
                result, error = extract_keywords_claude(
                    client, model, content_input, url_input, h1_input,
                    existing_anchors, min_word_count, max_keywords
                )

        if error:
            st.error(f"Extraction failed: {error}")
        elif result:
            st.success("Keywords extracted!")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Internal Link Opportunities")
                st.write(f"Found {len(result['internal_link_opportunities'])} keywords")
                for kw in result['internal_link_opportunities']:
                    st.markdown(f"- {kw}")

            with col2:
                st.markdown("### New Page Ideas")
                st.write(f"Found {len(result['new_page_ideas'])} keywords")
                for kw in result['new_page_ideas']:
                    st.markdown(f"- {kw}")

            # Create DataFrame for download
            max_len = max(len(result['internal_link_opportunities']), len(result['new_page_ideas']))
            df_data = {
                'internal_link_opportunities': result['internal_link_opportunities'] + [''] * (max_len - len(result['internal_link_opportunities'])),
                'new_page_ideas': result['new_page_ideas'] + [''] * (max_len - len(result['new_page_ideas']))
            }
            df = pd.DataFrame(df_data)

            st.download_button("Download Keywords (CSV)",
                               df.to_csv(index=False),
                               "extracted_keywords.csv",
                               "text/csv")

with tab2:
    st.subheader("Bulk Keyword Extraction")

    uploaded_file = st.file_uploader("Upload CSV/Excel with content", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write(f"Loaded {len(df)} rows")
            st.dataframe(df.head(), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                content_col = st.selectbox("Content Column", list(df.columns))
                url_col = st.selectbox("URL Column (optional)", ["(None)"] + list(df.columns))
            with col2:
                h1_col = st.selectbox("H1 Column (optional)", ["(None)"] + list(df.columns))
                anchors_col = st.selectbox("Existing Anchors Column (optional)", ["(None)"] + list(df.columns))

            if st.button("Extract from All Rows", type="primary", disabled=not api_key):
                if provider == "OpenAI (GPT)":
                    client = OpenAI(api_key=api_key)
                else:
                    client = Anthropic(api_key=api_key)

                results = []
                progress = st.progress(0)
                status = st.empty()

                for idx, row in df.iterrows():
                    status.text(f"Processing row {idx + 1}/{len(df)}...")

                    content = str(row.get(content_col, ''))
                    url = str(row.get(url_col, '')) if url_col != "(None)" else ""
                    h1 = str(row.get(h1_col, '')) if h1_col != "(None)" else ""

                    existing_anchors = None
                    if anchors_col != "(None)":
                        anchors_str = str(row.get(anchors_col, ''))
                        if anchors_str:
                            existing_anchors = [a.strip() for a in anchors_str.split(',') if a.strip()]

                    if provider == "OpenAI (GPT)":
                        result, error = extract_keywords_openai(
                            client, model, content, url, h1,
                            existing_anchors, min_word_count, max_keywords
                        )
                    else:
                        result, error = extract_keywords_claude(
                            client, model, content, url, h1,
                            existing_anchors, min_word_count, max_keywords
                        )

                    if result:
                        results.append({
                            'url': url,
                            'h1': h1,
                            'internal_link_opportunities': ', '.join(result['internal_link_opportunities']),
                            'new_page_ideas': ', '.join(result['new_page_ideas']),
                            'internal_count': len(result['internal_link_opportunities']),
                            'new_page_count': len(result['new_page_ideas'])
                        })
                    else:
                        results.append({
                            'url': url,
                            'h1': h1,
                            'internal_link_opportunities': '',
                            'new_page_ideas': '',
                            'internal_count': 0,
                            'new_page_count': 0,
                            'error': error or 'Unknown error'
                        })

                    progress.progress((idx + 1) / len(df))
                    sleep(1)  # Rate limiting

                status.text("Complete!")

                results_df = pd.DataFrame(results)
                st.success(f"Extracted keywords from {len(results_df)} rows")

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Internal Link Keywords", results_df['internal_count'].sum())
                col2.metric("Total New Page Ideas", results_df['new_page_count'].sum())
                col3.metric("Pages Processed", len(results_df))

                st.dataframe(results_df, use_container_width=True)

                # Download
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("Download CSV",
                                       results_df.to_csv(index=False),
                                       "bulk_keywords.csv",
                                       "text/csv")
                with col2:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        results_df.to_excel(writer, index=False)
                    st.download_button("Download Excel",
                                       output.getvalue(),
                                       "bulk_keywords.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"Error loading file: {e}")

# Help section
with st.expander("How to Use"):
    st.markdown("""
    **What This Tool Does:**
    - Extracts multi-word keywords from your content
    - Categorizes keywords for different purposes:
      - **Internal Link Opportunities**: Phrases that could become anchor text for internal links
      - **New Page Ideas**: Topics that could warrant dedicated new pages

    **Best Practices:**
    - Provide the page URL and H1 for context
    - List existing anchor texts to avoid duplicates
    - Use bulk extraction for large content audits

    **Use Cases:**
    - Internal linking audits
    - Content gap analysis
    - Keyword research from existing content
    - Finding anchor text opportunities
    """)

# Footer
st.markdown("---")
