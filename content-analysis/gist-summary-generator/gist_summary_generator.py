"""
Gist Summary Generator - Streamlit App
Create "At a glance" bullet point summaries from content using AI.

Author: Lee Foot
Website: https://leefoot.com
"""

import streamlit as st
import pandas as pd
from anthropic import Anthropic
from openai import OpenAI
import requests
from io import BytesIO
from time import sleep

st.set_page_config(
    page_title="Gist Summary Generator",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Gist Summary Generator")
st.markdown("Create concise 'At a glance' bullet point summaries from articles and content.")

# Sidebar configuration
with st.sidebar:
    st.header("API Configuration")

    provider = st.selectbox("AI Provider", ["Anthropic (Claude)", "OpenAI (GPT)"])

    if provider == "Anthropic (Claude)":
        api_key = st.text_input("Anthropic API Key", type="password")
        model = st.selectbox("Model", ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"])
    else:
        api_key = st.text_input("OpenAI API Key", type="password")
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"])

    st.header("Summary Settings")
    min_bullets = st.slider("Minimum bullet points", 2, 5, 3)
    max_bullets = st.slider("Maximum bullet points", 3, 7, 5)
    bullet_length = st.selectbox("Bullet point length", ["Concise (15-20 words)", "Standard (20-30 words)", "Detailed (30-40 words)"])

    st.header("Optional: Firecrawl")
    firecrawl_key = st.text_input("Firecrawl API Key (for URL fetching)", type="password")


SYSTEM_PROMPT = """You are an expert content summarizer. Your task is to create an "At a glance" summary with bullet points that capture the essential themes of an article.

Format requirements:
- Start with "**At a glance**" as a bold header
- Follow with bullet points using asterisks (*)
- Each bullet point should be {length} and capture a distinct key theme
- Cover different aspects: main topic, key findings, important details, and actionable insights
- Write in present tense and be factual and concise
- Use only as many bullet points as needed (minimum {min_bullets}, maximum {max_bullets})
- Use clear, accessible language

IMPORTANT: Focus on substance, not meta-descriptions. Avoid phrases like:
- "Article discusses..."
- "Guide provides..."
- "Content covers..."
- "This piece explores..."
Instead, state the actual substantive points directly.

Example output format:
**At a glance**
* First key point stated as a direct fact or finding
* Second important theme or conclusion from the content
* Third substantive takeaway for the reader
* Fourth actionable insight or recommendation (if applicable)"""


def get_length_guidance(bullet_length):
    """Get word count guidance based on selected length."""
    if "Concise" in bullet_length:
        return "15-20 words"
    elif "Standard" in bullet_length:
        return "20-30 words"
    else:
        return "30-40 words"


def scrape_url(url, api_key):
    """Scrape URL using Firecrawl API."""
    api_url = "https://api.firecrawl.dev/v1/scrape"

    payload = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": True,
        "timeout": 30000,
        "blockAds": True,
        "removeBase64Images": True
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
            return data.get('data', {}).get('markdown', ''), None
        else:
            return None, data.get('error', 'Unknown error')
    except Exception as e:
        return None, str(e)


def generate_summary_claude(client, model, content, system_prompt):
    """Generate summary using Claude."""
    try:
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": content}]
        )
        return message.content[0].text, None
    except Exception as e:
        return None, str(e)


def generate_summary_openai(client, model, content, system_prompt):
    """Generate summary using OpenAI."""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            max_tokens=1024,
            temperature=0.3
        )
        return completion.choices[0].message.content, None
    except Exception as e:
        return None, str(e)


# Main interface
tab1, tab2, tab3 = st.tabs(["Single Content", "URL Fetch", "Bulk Processing"])

with tab1:
    st.subheader("Summarize Pasted Content")

    content_input = st.text_area("Paste article or content to summarize",
                                  height=300,
                                  placeholder="Paste your article content here...")

    title_input = st.text_input("Article title (optional)", placeholder="Title for reference")

    if st.button("Generate Summary", type="primary", disabled=not api_key or not content_input):
        length_guidance = get_length_guidance(bullet_length)
        system = SYSTEM_PROMPT.format(length=length_guidance, min_bullets=min_bullets, max_bullets=max_bullets)

        with st.spinner("Generating summary..."):
            if provider == "Anthropic (Claude)":
                client = Anthropic(api_key=api_key)
                summary, error = generate_summary_claude(client, model, content_input, system)
            else:
                client = OpenAI(api_key=api_key)
                summary, error = generate_summary_openai(client, model, content_input, system)

        if error:
            st.error(f"Error: {error}")
        elif summary:
            st.success("Summary generated!")

            st.markdown("### Generated Summary")
            st.markdown(summary)

            # Download
            if title_input:
                filename = f"summary_{title_input[:30].replace(' ', '_')}.md"
            else:
                filename = "summary.md"

            st.download_button("Download Summary",
                               summary,
                               filename,
                               "text/markdown")

with tab2:
    st.subheader("Summarize from URL")

    url_input = st.text_input("Enter URL to summarize", placeholder="https://example.com/article")

    if st.button("Fetch & Summarize", type="primary",
                disabled=not api_key or not firecrawl_key or not url_input):

        with st.spinner("Fetching content..."):
            content, error = scrape_url(url_input, firecrawl_key)

        if error:
            st.error(f"Failed to fetch URL: {error}")
        elif content:
            st.success(f"Fetched {len(content)} characters")

            with st.expander("View fetched content"):
                st.markdown(content[:3000] + "..." if len(content) > 3000 else content)

            length_guidance = get_length_guidance(bullet_length)
            system = SYSTEM_PROMPT.format(length=length_guidance, min_bullets=min_bullets, max_bullets=max_bullets)

            with st.spinner("Generating summary..."):
                if provider == "Anthropic (Claude)":
                    client = Anthropic(api_key=api_key)
                    summary, error = generate_summary_claude(client, model, content, system)
                else:
                    client = OpenAI(api_key=api_key)
                    summary, error = generate_summary_openai(client, model, content, system)

            if error:
                st.error(f"Error: {error}")
            elif summary:
                st.success("Summary generated!")

                st.markdown("### Generated Summary")
                st.markdown(summary)

                st.download_button("Download Summary",
                                   summary,
                                   "url_summary.md",
                                   "text/markdown")

with tab3:
    st.subheader("Bulk Summarization")

    uploaded_file = st.file_uploader("Upload CSV/Excel with content", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write(f"Loaded {len(df)} rows")
            st.dataframe(df.head(), use_container_width=True)

            content_col = st.selectbox("Content Column", list(df.columns))
            title_col = st.selectbox("Title Column (optional)", ["(None)"] + list(df.columns))

            if st.button("Generate All Summaries", type="primary", disabled=not api_key):
                length_guidance = get_length_guidance(bullet_length)
                system = SYSTEM_PROMPT.format(length=length_guidance, min_bullets=min_bullets, max_bullets=max_bullets)

                if provider == "Anthropic (Claude)":
                    client = Anthropic(api_key=api_key)
                else:
                    client = OpenAI(api_key=api_key)

                results = []
                progress = st.progress(0)
                status = st.empty()

                for idx, row in df.iterrows():
                    content = str(row[content_col])
                    title = str(row[title_col]) if title_col != "(None)" else ""

                    status.text(f"Processing {idx + 1}/{len(df)}: {title[:50]}...")

                    if len(content.strip()) < 100:
                        results.append({
                            'title': title,
                            'summary': 'Content too short to summarize',
                            'status': 'Skipped'
                        })
                    else:
                        if provider == "Anthropic (Claude)":
                            summary, error = generate_summary_claude(client, model, content, system)
                        else:
                            summary, error = generate_summary_openai(client, model, content, system)

                        results.append({
                            'title': title,
                            'summary': summary or '',
                            'status': 'Success' if summary else f'Error: {error}'
                        })

                    progress.progress((idx + 1) / len(df))
                    sleep(1)  # Rate limiting

                status.text("Complete!")

                results_df = pd.DataFrame(results)

                # Metrics
                success_count = len(results_df[results_df['status'] == 'Success'])
                col1, col2 = st.columns(2)
                col1.metric("Successful", success_count)
                col2.metric("Failed/Skipped", len(results_df) - success_count)

                st.dataframe(results_df, use_container_width=True)

                # Download
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("Download CSV",
                                       results_df.to_csv(index=False),
                                       "bulk_summaries.csv",
                                       "text/csv")
                with col2:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        results_df.to_excel(writer, index=False)
                    st.download_button("Download Excel",
                                       output.getvalue(),
                                       "bulk_summaries.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"Error loading file: {e}")

# Help section
with st.expander("How to Use"):
    st.markdown("""
    **What This Tool Does:**
    Creates concise "At a glance" bullet point summaries that capture key themes from articles or content.

    **Best Use Cases:**
    - Blog post summaries for featured snippets
    - Article previews for newsletters
    - Executive summaries for reports
    - Content briefs for social media

    **Tips for Best Results:**
    - Provide complete articles (minimum 100+ words)
    - Content with clear structure summarizes better
    - Adjust bullet count based on content length
    - Use "Concise" length for short content, "Detailed" for comprehensive articles

    **Output Format:**
    ```
    **At a glance**
    * Key point one stated directly
    * Important finding or theme
    * Actionable insight or takeaway
    ```
    """)

# Footer
st.markdown("---")
st.markdown("Built by [Lee Foot](https://leefoot.com)")
