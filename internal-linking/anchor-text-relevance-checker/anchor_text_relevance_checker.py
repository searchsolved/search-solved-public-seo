"""
Anchor Text Relevance Checker - Streamlit App
Assess anchor text relevance for internal linking using AI.

Author: Lee Foot
Website: https://leefoot.com
"""

import streamlit as st
import pandas as pd
from openai import OpenAI
import json
from time import sleep
from io import BytesIO

st.set_page_config(
    page_title="Anchor Text Relevance Checker",
    page_icon="🔗",
    layout="wide"
)

st.title("🔗 Anchor Text Relevance Checker")
st.markdown("Assess whether anchor text is relevant, natural, and appropriate for target pages.")

# Sidebar configuration
with st.sidebar:
    st.header("🔑 API Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")

    st.header("⚙️ Settings")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"], index=0)
    batch_size = st.slider("Batch size", 1, 20, 10, help="Number of anchor texts to evaluate per API call")

    st.header("📊 Rating Scale")
    st.markdown("""
    - **High**: Excellent anchor text - relevant, natural, citable
    - **Medium**: Acceptable but not optimal
    - **Fail**: Poor anchor text - shouldn't be used
    - **Typo**: Contains spelling/grammatical errors
    """)

    st.header("🚫 Auto-Fail Criteria")
    st.markdown("""
    - Questions (e.g., "how do I start?")
    - Sentence fragments
    - Unnatural/SEO-manipulative text
    - Doesn't indicate link destination
    """)


def assess_relevance_batch(client, model, batch):
    """Assess relevance for a batch of anchor text/URL pairs."""

    system_message = """You are an SEO expert specializing in internal linking. Evaluate whether the proposed anchor text is appropriate for creating a link to the target page.

Rate each anchor text on:
1. RELEVANCE: Does the anchor text accurately describe the target page's content?
2. NATURALNESS: Would this anchor text fit naturally in a sentence?
3. CITATION QUALITY: Is this text something you'd actually cite/link to?

IMMEDIATELY FAIL any anchor text that:
- Is a question (e.g., "how do I start a business?")
- Is a sentence fragment (e.g., "a wellwritten business plan should contain")
- Sounds unnatural or SEO-manipulative
- Doesn't clearly indicate what the user will find after clicking

Rate as:
- "High" - Excellent anchor text that's relevant, natural, and citable
- "Medium" - Acceptable but not optimal anchor text
- "Fail" - Poor anchor text that shouldn't be used
- "Typo" - Contains spelling/grammatical errors

IMPORTANT: Your response must be a valid JSON with this exact structure:
{
  "assessments": [
    {
      "relevance_score": "High"
    }
  ]
}

DO NOT include explanations in your assessment."""

    user_content = "Please assess the relevance between these anchor texts and their target URL contexts:\n\n"
    for item in batch:
        user_content += f"Anchor Text: {item['anchor_text']}\n"
        user_content += f"Target URL: {item['target_url']}\n"
        if item.get('h1'):
            user_content += f"Target H1: {item['h1']}\n"
        if item.get('title'):
            user_content += f"Target Title: {item['title']}\n"
        user_content += "\n"

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "relevance_assessment",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "assessments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "relevance_score": {"type": "string"}
                            },
                            "required": ["relevance_score"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["assessments"],
                "additionalProperties": False
            }
        }
    }

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ],
            response_format=response_format
        )

        result = json.loads(completion.choices[0].message.content)
        return result["assessments"], None
    except Exception as e:
        return None, str(e)


def get_score_badge(score):
    """Return a colored badge for the score."""
    badges = {
        "High": "🟢 High",
        "Medium": "🟡 Medium",
        "Fail": "🔴 Fail",
        "Typo": "🟠 Typo"
    }
    return badges.get(score, f"⚪ {score}")


# Main interface
tab1, tab2 = st.tabs(["🔗 Single Check", "📊 Bulk Check"])

with tab1:
    st.subheader("Check Single Anchor Text")

    col1, col2 = st.columns(2)

    with col1:
        anchor_text = st.text_input("Anchor Text", placeholder="e.g., business formation services")
        target_url = st.text_input("Target URL", placeholder="https://example.com/services")

    with col2:
        target_h1 = st.text_input("Target Page H1 (optional)", placeholder="Business Formation Services")
        target_title = st.text_input("Target Page Title (optional)", placeholder="Business Formation | Example")

    if st.button("Check Relevance", type="primary", disabled=not api_key or not anchor_text or not target_url):
        with st.spinner("Assessing relevance..."):
            client = OpenAI(api_key=api_key)
            batch = [{
                'anchor_text': anchor_text,
                'target_url': target_url,
                'h1': target_h1,
                'title': target_title
            }]
            assessments, error = assess_relevance_batch(client, model, batch)

        if error:
            st.error(f"Error: {error}")
        elif assessments:
            score = assessments[0]['relevance_score']
            st.markdown(f"### Result: {get_score_badge(score)}")

            # Recommendations based on score
            if score == "High":
                st.success("This anchor text is appropriate for internal linking.")
            elif score == "Medium":
                st.warning("This anchor text is acceptable but could be improved for better relevance.")
            elif score == "Fail":
                st.error("This anchor text should not be used. Consider:")
                st.markdown("""
                - Make it more descriptive of the target page
                - Avoid questions or sentence fragments
                - Use natural, citable language
                """)
            elif score == "Typo":
                st.error("This anchor text contains errors. Please fix spelling/grammar before using.")

with tab2:
    st.subheader("Bulk Relevance Check")

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write(f"Loaded {len(df)} rows")
            st.dataframe(df.head(), use_container_width=True)

            # Column mapping
            st.markdown("**Map your columns:**")
            col1, col2 = st.columns(2)

            with col1:
                anchor_col = st.selectbox("Anchor Text Column", list(df.columns))
                url_col = st.selectbox("Target URL Column", list(df.columns))

            with col2:
                h1_col = st.selectbox("H1 Column (optional)", ["(None)"] + list(df.columns))
                title_col = st.selectbox("Title Column (optional)", ["(None)"] + list(df.columns))

            # Limit for testing
            max_rows = st.number_input("Max rows to process (0 = all)", 0, len(df), 0)

            if st.button("Check All", type="primary", disabled=not api_key):
                client = OpenAI(api_key=api_key)

                results = []
                process_df = df.head(max_rows) if max_rows > 0 else df

                progress_bar = st.progress(0)
                status_text = st.empty()

                # Process in batches
                total_batches = (len(process_df) + batch_size - 1) // batch_size

                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(process_df))

                    status_text.text(f"Processing batch {batch_idx + 1} of {total_batches}...")

                    # Prepare batch
                    batch = []
                    batch_rows = []
                    for idx in range(start_idx, end_idx):
                        row = process_df.iloc[idx]
                        anchor = str(row.get(anchor_col, "")) if pd.notna(row.get(anchor_col)) else ""
                        url = str(row.get(url_col, "")) if pd.notna(row.get(url_col)) else ""
                        h1 = str(row.get(h1_col, "")) if h1_col != "(None)" and pd.notna(row.get(h1_col)) else ""
                        title = str(row.get(title_col, "")) if title_col != "(None)" and pd.notna(row.get(title_col)) else ""

                        batch.append({
                            'anchor_text': anchor,
                            'target_url': url,
                            'h1': h1,
                            'title': title
                        })
                        batch_rows.append((anchor, url, h1, title))

                    # Get assessments
                    if batch:
                        assessments, error = assess_relevance_batch(client, model, batch)

                        if assessments:
                            for i, assessment in enumerate(assessments):
                                anchor, url, h1, title = batch_rows[i]
                                results.append({
                                    'anchor_text': anchor,
                                    'target_url': url,
                                    'target_h1': h1,
                                    'target_title': title,
                                    'relevance_score': assessment['relevance_score']
                                })
                        else:
                            for anchor, url, h1, title in batch_rows:
                                results.append({
                                    'anchor_text': anchor,
                                    'target_url': url,
                                    'target_h1': h1,
                                    'target_title': title,
                                    'relevance_score': 'Error',
                                    'error': error
                                })

                    sleep(1)  # Rate limiting
                    progress_bar.progress((batch_idx + 1) / total_batches)

                status_text.text("Complete!")

                # Display results
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)

                # Summary stats
                st.markdown("**Score Distribution:**")
                score_counts = results_df['relevance_score'].value_counts()

                cols = st.columns(4)
                for i, (score, count) in enumerate(score_counts.items()):
                    pct = (count / len(results_df)) * 100
                    cols[i % 4].metric(get_score_badge(score), f"{count} ({pct:.1f}%)")

                # Download
                col1, col2 = st.columns(2)
                with col1:
                    csv = results_df.to_csv(index=False)
                    st.download_button("Download CSV", csv, "anchor_text_relevance.csv", "text/csv")
                with col2:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        results_df.to_excel(writer, index=False)
                    st.download_button("Download Excel", output.getvalue(),
                                       "anchor_text_relevance.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"Error loading file: {e}")

# Footer
st.markdown("---")
st.markdown("Built by [Lee Foot](https://leefoot.com)")
