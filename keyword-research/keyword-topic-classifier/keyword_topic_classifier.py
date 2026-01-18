"""
Keyword Topic Classifier - Streamlit App
Classify keywords into hierarchical themes/subthemes using AI.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
from anthropic import Anthropic
from openai import OpenAI
import json
from time import sleep
from io import BytesIO
from collections import defaultdict

st.set_page_config(
    page_title="Keyword Topic Classifier",
    page_icon="🏷️",
    layout="wide"
)

st.title("🏷️ Keyword Topic Classifier")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Classifies keywords into topic categories
    - Groups keywords by search intent
    - Creates topical keyword maps

    **How to use:**
    1. Upload keyword list
    2. Configure classification settings
    3. Run topic classification
    4. Download categorized keywords

    **Best for:**
    - Content planning
    - Keyword organization
    - Topical authority mapping
    """)
st.markdown("Classify keywords into hierarchical themes and subthemes using AI.")

# Sidebar configuration
with st.sidebar:
    st.header("🔑 API Configuration")

    provider = st.selectbox("AI Provider", ["Anthropic (Claude)", "OpenAI (GPT)"])

    if provider == "Anthropic (Claude)":
        api_key = st.text_input("Anthropic API Key", type="password")
        model = st.selectbox("Model", ["claude-sonnet-4-5-20250929", "claude-sonnet-4-20250514", "claude-haiku-4-5-20251015"])
    else:
        api_key = st.text_input("OpenAI API Key", type="password")
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1"])

    st.header("⚙️ Settings")
    batch_size = st.slider("Keywords per batch", 10, 50, 30)

    st.header("📊 Output Format")
    st.markdown("""
    Each keyword will be assigned:
    - **Theme**: High-level category
    - **Subtheme**: Specific topic within theme
    - **Confidence**: 0-1 confidence score
    """)


def classify_keywords_claude(client, model, keywords):
    """Classify keywords using Claude."""

    system_prompt = """You are an expert at classifying keywords into hierarchical topics.
Analyze the provided keywords and classify them into relevant themes and subthemes.
You must respond with valid JSON only - no preamble, explanations or other text."""

    example = '''
Input keywords: sole proprietorship vs llc, llc vs sole proprietorship, advantages of llc

Output:
{
  "themes": [
    {
      "theme_name": "Comparison",
      "subthemes": [
        {
          "subtheme_name": "General Comparison",
          "keywords": ["sole proprietorship vs llc", "llc vs sole proprietorship"],
          "confidence": 0.95
        }
      ]
    },
    {
      "theme_name": "Benefits & Advantages",
      "subthemes": [
        {
          "subtheme_name": "Advantage Queries",
          "keywords": ["advantages of llc"],
          "confidence": 0.9
        }
      ]
    }
  ]
}
'''

    user_prompt = f"""Classify these keywords into hierarchical themes and subthemes:

{', '.join(keywords)}

Example output format:
{example}

Return ONLY a JSON object with this structure (no markdown, no explanation):
{{
    "themes": [
        {{
            "theme_name": "string",
            "subthemes": [
                {{
                    "subtheme_name": "string",
                    "keywords": ["string"],
                    "confidence": float
                }}
            ]
        }}
    ]
}}

Group similar keywords into appropriate themes. Be specific with theme names."""

    try:
        message = client.messages.create(
            model=model,
            system=system_prompt,
            max_tokens=4000,
            temperature=0,
            messages=[{"role": "user", "content": user_prompt}]
        )

        response_content = message.content[0].text.strip()

        # Clean response
        if response_content.startswith("```json"):
            response_content = response_content.split("```json")[1]
        if response_content.endswith("```"):
            response_content = response_content.rsplit("```", 1)[0]

        return json.loads(response_content.strip()), None
    except Exception as e:
        return None, str(e)


def classify_keywords_openai(client, model, keywords):
    """Classify keywords using OpenAI."""

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "keyword_classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "themes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "theme_name": {"type": "string"},
                                "subthemes": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "subtheme_name": {"type": "string"},
                                            "keywords": {"type": "array", "items": {"type": "string"}},
                                            "confidence": {"type": "number"}
                                        },
                                        "required": ["subtheme_name", "keywords", "confidence"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["theme_name", "subthemes"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["themes"],
                "additionalProperties": False
            }
        }
    }

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You classify keywords into hierarchical themes and subthemes. Be specific with theme names."
                },
                {
                    "role": "user",
                    "content": f"Classify these keywords into themes and subthemes:\n\n{', '.join(keywords)}"
                }
            ],
            response_format=response_format
        )

        return json.loads(completion.choices[0].message.content), None
    except Exception as e:
        return None, str(e)


def process_classification_result(result):
    """Convert classification result to flat DataFrame rows."""
    rows = []

    if result and 'themes' in result:
        for theme in result['themes']:
            theme_name = theme['theme_name']
            for subtheme in theme.get('subthemes', []):
                subtheme_name = subtheme['subtheme_name']
                confidence = subtheme.get('confidence', 0)
                for keyword in subtheme.get('keywords', []):
                    rows.append({
                        'keyword': keyword,
                        'theme': theme_name,
                        'subtheme': subtheme_name,
                        'confidence': confidence
                    })

    return rows


# Main interface
tab1, tab2 = st.tabs(["📝 Single Batch", "📊 Bulk Classification"])

with tab1:
    st.subheader("Classify a List of Keywords")

    keywords_input = st.text_area("Enter keywords (one per line or comma-separated)",
                                   height=200,
                                   placeholder="business formation\nllc vs corporation\nhow to start a business")

    if st.button("Classify Keywords", type="primary", disabled=not api_key or not keywords_input):
        # Parse keywords
        if ',' in keywords_input:
            keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
        else:
            keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]

        st.write(f"Classifying {len(keywords)} keywords...")

        with st.spinner("Classifying..."):
            if provider == "Anthropic (Claude)":
                client = Anthropic(api_key=api_key)
                result, error = classify_keywords_claude(client, model, keywords)
            else:
                client = OpenAI(api_key=api_key)
                result, error = classify_keywords_openai(client, model, keywords)

        if error:
            st.error(f"Error: {error}")
        elif result:
            rows = process_classification_result(result)

            if rows:
                df = pd.DataFrame(rows)
                st.success(f"Classified {len(df)} keywords into {df['theme'].nunique()} themes!")

                # Display by theme
                for theme in df['theme'].unique():
                    theme_df = df[df['theme'] == theme]
                    with st.expander(f"📁 {theme} ({len(theme_df)} keywords)"):
                        for subtheme in theme_df['subtheme'].unique():
                            subtheme_df = theme_df[theme_df['subtheme'] == subtheme]
                            st.markdown(f"**{subtheme}** (confidence: {subtheme_df['confidence'].mean():.2f})")
                            for kw in subtheme_df['keyword'].tolist():
                                st.write(f"  - {kw}")

                # Download
                st.download_button("Download CSV", df.to_csv(index=False),
                                   "classified_keywords.csv", "text/csv")
            else:
                st.warning("No classifications returned.")

            # Show raw result
            with st.expander("View Raw Result"):
                st.json(result)

with tab2:
    st.subheader("Bulk Classification from File")

    uploaded_file = st.file_uploader("Upload CSV or Excel with keywords", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write(f"Loaded {len(df)} rows")
            st.dataframe(df.head(), use_container_width=True)

            # Column mapping
            keyword_col = st.selectbox("Keyword Column", list(df.columns))

            # Optional grouping column
            group_col = st.selectbox("Group by Column (optional)", ["(None)"] + list(df.columns),
                                     help="Process keywords grouped by URL or page")

            if st.button("Classify All", type="primary", disabled=not api_key):
                if provider == "Anthropic (Claude)":
                    client = Anthropic(api_key=api_key)
                else:
                    client = OpenAI(api_key=api_key)

                all_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                if group_col != "(None)":
                    # Process by group
                    groups = df.groupby(group_col)[keyword_col].apply(list).to_dict()
                    total_groups = len(groups)

                    for idx, (group_key, keywords) in enumerate(groups.items()):
                        status_text.text(f"Processing group {idx + 1} of {total_groups}: {group_key[:50]}...")

                        # Process in batches within group
                        for batch_start in range(0, len(keywords), batch_size):
                            batch = keywords[batch_start:batch_start + batch_size]

                            if provider == "Anthropic (Claude)":
                                result, error = classify_keywords_claude(client, model, batch)
                            else:
                                result, error = classify_keywords_openai(client, model, batch)

                            if result:
                                rows = process_classification_result(result)
                                for row in rows:
                                    row['group'] = group_key
                                all_results.extend(rows)

                            sleep(1)

                        progress_bar.progress((idx + 1) / total_groups)
                else:
                    # Process all keywords
                    keywords = df[keyword_col].dropna().unique().tolist()
                    total_batches = (len(keywords) + batch_size - 1) // batch_size

                    for batch_idx in range(total_batches):
                        start = batch_idx * batch_size
                        end = min(start + batch_size, len(keywords))
                        batch = keywords[start:end]

                        status_text.text(f"Processing batch {batch_idx + 1} of {total_batches}...")

                        if provider == "Anthropic (Claude)":
                            result, error = classify_keywords_claude(client, model, batch)
                        else:
                            result, error = classify_keywords_openai(client, model, batch)

                        if result:
                            rows = process_classification_result(result)
                            all_results.extend(rows)

                        sleep(1)
                        progress_bar.progress((batch_idx + 1) / total_batches)

                status_text.text("Complete!")

                if all_results:
                    results_df = pd.DataFrame(all_results)
                    st.success(f"Classified {len(results_df)} keywords!")

                    # Summary
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Keywords Classified", len(results_df))
                    col2.metric("Themes Found", results_df['theme'].nunique())
                    col3.metric("Subthemes Found", results_df['subtheme'].nunique())

                    st.dataframe(results_df, use_container_width=True)

                    # Download
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button("Download CSV", results_df.to_csv(index=False),
                                           "bulk_classified_keywords.csv", "text/csv")
                    with col2:
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            results_df.to_excel(writer, index=False)
                        st.download_button("Download Excel", output.getvalue(),
                                           "bulk_classified_keywords.xlsx",
                                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else:
                    st.warning("No classifications returned.")

        except Exception as e:
            st.error(f"Error loading file: {e}")

# Footer
st.markdown("---")
