"""
Keyword to Questions Converter - Streamlit App
Convert keyword phrases into natural questions for FAQ content.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
from anthropic import Anthropic
from openai import OpenAI
import json
import re
from io import BytesIO
from time import sleep

st.set_page_config(
    page_title="Keyword to Questions",
    page_icon="❓",
    layout="wide"
)

st.title("❓ Keyword to Questions Converter")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Transforms keywords into question formats
    - Generates who/what/where/when/why/how variations
    - Creates FAQ-ready content ideas

    **How to use:**
    1. Upload or paste keywords
    2. Select question types
    3. Generate question variations
    4. Download question list

    **Best for:**
    - FAQ content creation
    - People Also Ask optimization
    - Voice search optimization
    """)
st.markdown("Transform keyword phrases into natural questions for FAQ pages and content.")

# Sidebar configuration
with st.sidebar:
    st.header("API Configuration")

    provider = st.selectbox("AI Provider", ["Anthropic (Claude)", "OpenAI (GPT)"])

    if provider == "Anthropic (Claude)":
        api_key = st.text_input("Anthropic API Key", type="password")
        model = st.selectbox("Model", ["claude-sonnet-4-5-20250929", "claude-sonnet-4-20250514", "claude-haiku-4-5-20251015"])
    else:
        api_key = st.text_input("OpenAI API Key", type="password")
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1"])

    st.header("Settings")
    consolidate = st.checkbox("Consolidate similar keywords", value=True,
                              help="Group keywords with same intent into one question")
    question_style = st.selectbox("Question Style", [
        "Natural conversational",
        "Direct and simple",
        "Formal/professional"
    ])


def convert_keywords_claude(client, model, keywords, topic="", consolidate=True, style="Natural conversational"):
    """Convert keywords to questions using Claude."""

    consolidate_instruction = """
Group similar keywords together and create a single natural-sounding question when the intent is the same.
For example, "llc formation cost", "how much to form llc", "llc filing fees" could become one question about LLC costs.""" if consolidate else """
Create one question per keyword without grouping."""

    style_instruction = {
        "Natural conversational": "Use a conversational tone as if someone is asking a friend or expert.",
        "Direct and simple": "Keep questions short and to the point.",
        "Formal/professional": "Use professional language suitable for business contexts."
    }.get(style, "")

    topic_context = f'These keywords are about "{topic}".' if topic else ""

    prompt = f"""Convert these keyword phrases into natural questions:

{', '.join(keywords)}

{topic_context}

{consolidate_instruction}

Style: {style_instruction}

Return ONLY a valid JSON object with this structure:
{{
  "questions": [
    {{
      "original_keywords": ["keyword1", "keyword2"],
      "question": "Natural question that covers these keywords?"
    }}
  ]
}}"""

    try:
        message = client.messages.create(
            model=model,
            max_tokens=2000,
            system="You convert keyword phrases into natural questions. Always respond with valid JSON only.",
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text.strip()

        # Clean response
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        if response_text.endswith("```"):
            response_text = response_text.rsplit("```", 1)[0]

        return json.loads(response_text.strip()), None
    except Exception as e:
        return None, str(e)


def convert_keywords_openai(client, model, keywords, topic="", consolidate=True, style="Natural conversational"):
    """Convert keywords to questions using OpenAI."""

    consolidate_instruction = "Group similar keywords with same intent into one question." if consolidate else "Create one question per keyword."

    style_instruction = {
        "Natural conversational": "conversational tone",
        "Direct and simple": "short and direct",
        "Formal/professional": "professional language"
    }.get(style, "conversational")

    topic_context = f'Topic: {topic}. ' if topic else ""

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "keyword_questions",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "original_keywords": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "question": {"type": "string"}
                            },
                            "required": ["original_keywords", "question"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["questions"],
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
                    "content": f"Convert keyword phrases into natural questions. {consolidate_instruction} Use {style_instruction}."
                },
                {
                    "role": "user",
                    "content": f"{topic_context}Keywords: {', '.join(keywords)}"
                }
            ],
            response_format=response_format
        )

        return json.loads(completion.choices[0].message.content), None
    except Exception as e:
        return None, str(e)


# Main interface
tab1, tab2 = st.tabs(["Convert Keywords", "Bulk Processing"])

with tab1:
    st.subheader("Convert Keywords to Questions")

    col1, col2 = st.columns([2, 1])

    with col1:
        keywords_input = st.text_area("Enter keywords (one per line or comma-separated)",
                                       height=200,
                                       placeholder="llc formation\nhow to start llc\nllc vs corporation\nbest state for llc")

    with col2:
        topic_input = st.text_input("Topic/Category (optional)",
                                     placeholder="e.g., LLC Formation",
                                     help="Provides context for better questions")

    if st.button("Convert to Questions", type="primary", disabled=not api_key or not keywords_input):
        # Parse keywords
        if ',' in keywords_input and '\n' not in keywords_input:
            keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
        else:
            keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]

        st.write(f"Processing {len(keywords)} keywords...")

        with st.spinner("Converting keywords to questions..."):
            if provider == "Anthropic (Claude)":
                client = Anthropic(api_key=api_key)
                result, error = convert_keywords_claude(client, model, keywords, topic_input, consolidate, question_style)
            else:
                client = OpenAI(api_key=api_key)
                result, error = convert_keywords_openai(client, model, keywords, topic_input, consolidate, question_style)

        if error:
            st.error(f"Error: {error}")
        elif result:
            questions = result.get('questions', [])
            st.success(f"Generated {len(questions)} questions from {len(keywords)} keywords!")

            # Display results
            for i, q in enumerate(questions, 1):
                with st.expander(f"Q{i}: {q['question']}", expanded=True):
                    st.markdown(f"**Question:** {q['question']}")
                    st.markdown(f"**Source keywords:** {', '.join(q['original_keywords'])}")

            # Create DataFrame for download
            rows = []
            for q in questions:
                rows.append({
                    'question': q['question'],
                    'original_keywords': ', '.join(q['original_keywords']),
                    'keyword_count': len(q['original_keywords'])
                })

            df = pd.DataFrame(rows)

            st.markdown("### Export")
            col1, col2 = st.columns(2)

            with col1:
                st.download_button("Download CSV",
                                   df.to_csv(index=False),
                                   "keyword_questions.csv",
                                   "text/csv")

            with col2:
                # Plain text format for FAQ
                faq_text = "\n\n".join([f"Q: {q['question']}\nA: [Your answer here]" for q in questions])
                st.download_button("Download FAQ Template",
                                   faq_text,
                                   "faq_template.txt",
                                   "text/plain")

with tab2:
    st.subheader("Bulk Keyword Conversion")

    uploaded_file = st.file_uploader("Upload CSV/Excel with keywords", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write(f"Loaded {len(df)} rows")
            st.dataframe(df.head(), use_container_width=True)

            keyword_col = st.selectbox("Keyword Column", list(df.columns))
            topic_col = st.selectbox("Topic/Category Column (optional)", ["(None)"] + list(df.columns))

            batch_size = st.slider("Keywords per batch", 10, 50, 25,
                                   help="Process keywords in batches for better results")

            if st.button("Convert All Keywords", type="primary", disabled=not api_key):
                if provider == "Anthropic (Claude)":
                    client = Anthropic(api_key=api_key)
                else:
                    client = OpenAI(api_key=api_key)

                all_results = []
                progress = st.progress(0)
                status = st.empty()

                # Group by topic if topic column is selected
                if topic_col != "(None)":
                    groups = df.groupby(topic_col)[keyword_col].apply(list).to_dict()
                else:
                    # Process all keywords together in batches
                    all_keywords = df[keyword_col].dropna().tolist()
                    groups = {"All Keywords": all_keywords}

                total_groups = len(groups)

                for idx, (topic, keywords) in enumerate(groups.items()):
                    status.text(f"Processing group {idx + 1}/{total_groups}: {topic}")

                    # Process in batches within each group
                    for batch_start in range(0, len(keywords), batch_size):
                        batch = keywords[batch_start:batch_start + batch_size]

                        if provider == "Anthropic (Claude)":
                            result, error = convert_keywords_claude(client, model, batch, topic if topic != "All Keywords" else "", consolidate, question_style)
                        else:
                            result, error = convert_keywords_openai(client, model, batch, topic if topic != "All Keywords" else "", consolidate, question_style)

                        if result:
                            for q in result.get('questions', []):
                                all_results.append({
                                    'topic': topic if topic != "All Keywords" else "",
                                    'question': q['question'],
                                    'original_keywords': ', '.join(q['original_keywords']),
                                    'keyword_count': len(q['original_keywords'])
                                })

                        sleep(1)  # Rate limiting

                    progress.progress((idx + 1) / total_groups)

                status.text("Complete!")

                if all_results:
                    results_df = pd.DataFrame(all_results)

                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Questions Generated", len(results_df))
                    col2.metric("Keywords Processed", results_df['keyword_count'].sum())
                    col3.metric("Topics/Groups", results_df['topic'].nunique() if results_df['topic'].any() else 1)

                    st.dataframe(results_df, use_container_width=True)

                    # Download
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button("Download CSV",
                                           results_df.to_csv(index=False),
                                           "bulk_questions.csv",
                                           "text/csv")
                    with col2:
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            results_df.to_excel(writer, index=False)
                        st.download_button("Download Excel",
                                           output.getvalue(),
                                           "bulk_questions.xlsx",
                                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else:
                    st.warning("No questions generated.")

        except Exception as e:
            st.error(f"Error loading file: {e}")

# Help section
with st.expander("How to Use"):
    st.markdown("""
    **What This Tool Does:**
    Converts keyword phrases into natural-sounding questions for FAQ pages, content briefs, and user-focused content.

    **Features:**
    - **Consolidation**: Groups similar keywords into single questions (e.g., "llc cost", "llc fees" → "How much does it cost to form an LLC?")
    - **Topic Context**: Provide a topic for more relevant question framing
    - **Style Options**: Choose conversational, direct, or professional tone

    **Best Use Cases:**
    - Creating FAQ sections from keyword research
    - Developing People Also Ask content
    - Building question-based content briefs
    - Generating featured snippet opportunities

    **Tips:**
    - Group related keywords together for better consolidation
    - Use topic context for more specific questions
    - Review and edit generated questions for your brand voice
    """)

# Footer
st.markdown("---")
