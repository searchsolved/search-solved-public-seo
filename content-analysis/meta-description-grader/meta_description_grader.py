"""
Meta Description Grader - Streamlit App
Score and compare meta descriptions using GPT-4 on key SEO criteria.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
from openai import OpenAI
import json
from time import sleep
from io import BytesIO

st.set_page_config(
    page_title="Meta Description Grader",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Meta Description Grader")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Scores meta descriptions on key SEO criteria using AI
    - Analyzes emotional hooks, benefits, active voice, and urgency
    - Provides detailed breakdown with improvement suggestions

    **How to use:**
    1. Enter your OpenAI API key in the sidebar
    2. Upload a CSV with URL and Meta Description columns
    3. Click "Analyze" to score your descriptions
    4. Review scores and recommendations

    **Scoring criteria (0-10 each):**
    - Emotional Hook: Power verb in opening
    - Benefit: Clear outcome stated
    - Active Voice: Uses active voice
    - Urgency: Creates interest
    """)
st.markdown("Score meta descriptions on key SEO criteria using AI analysis.")

# Sidebar configuration
with st.sidebar:
    st.header("🔑 API Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", help="Your OpenAI API key")

    st.header("⚙️ Settings")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"], index=0)

    st.header("📊 Scoring Criteria")
    st.markdown("""
    Each criterion scored 0-10:
    - **Emotional Hook**: Power verb/hook in opening
    - **Benefit**: Clear benefit/outcome stated
    - **Active Voice**: Uses active voice
    - **Urgency**: Creates interest/urgency

    **Total Score**: Sum of all criteria (max 40)
    """)


def analyze_meta_description(client, model, url, meta_description):
    """Analyze a meta description and return scores."""

    messages = [
        {
            "role": "system",
            "content": """You are an expert meta description analyzer. Score these components from 0-10:
            1. Emotional hook/power verb in opening
            2. Clear benefit/outcome
            3. Active voice usage
            4. Creates urgency/interest

            Be strict but fair. A score of 10 means exceptional, 7-8 is good, 5-6 is average, below 5 needs improvement."""
        },
        {
            "role": "user",
            "content": f"""Analyze this meta description for the four key components.

URL: {url or 'Not provided'}
Meta Description: {meta_description or 'Not provided'}

Score each component from 0-10."""
        }
    ]

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "meta_description_analysis",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "scores": {
                        "type": "object",
                        "properties": {
                            "emotional_hook": {"type": "integer"},
                            "benefit": {"type": "integer"},
                            "active_voice": {"type": "integer"},
                            "creates_urgency": {"type": "integer"}
                        },
                        "required": ["emotional_hook", "benefit", "active_voice", "creates_urgency"],
                        "additionalProperties": False
                    }
                },
                "required": ["scores"],
                "additionalProperties": False
            }
        }
    }

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format
        )
        result = json.loads(completion.choices[0].message.content)
        return result["scores"], None
    except Exception as e:
        return None, str(e)


def get_score_color(score, max_score=10):
    """Return color based on score percentage."""
    pct = score / max_score
    if pct >= 0.8:
        return "🟢"
    elif pct >= 0.6:
        return "🟡"
    elif pct >= 0.4:
        return "🟠"
    else:
        return "🔴"


# Main interface
tab1, tab2 = st.tabs(["📝 Single Analysis", "📊 Bulk Analysis"])

with tab1:
    st.subheader("Analyze a Single Meta Description")

    col1, col2 = st.columns(2)

    with col1:
        url = st.text_input("URL (optional)", placeholder="https://example.com/page")
        meta_desc = st.text_area("Meta Description", height=100,
                                  placeholder="Enter the meta description to analyze...")
        char_count = len(meta_desc) if meta_desc else 0

        if char_count > 0:
            if char_count < 120:
                st.warning(f"Length: {char_count} chars (too short, aim for 150-160)")
            elif char_count <= 160:
                st.success(f"Length: {char_count} chars (good)")
            else:
                st.error(f"Length: {char_count} chars (too long, may be truncated)")

    with col2:
        if st.button("Analyze", type="primary", disabled=not api_key or not meta_desc):
            with st.spinner("Analyzing meta description..."):
                client = OpenAI(api_key=api_key)
                scores, error = analyze_meta_description(client, model, url, meta_desc)

            if error:
                st.error(f"Error: {error}")
            elif scores:
                total = sum(scores.values())

                st.metric("Total Score", f"{total}/40", delta=f"{(total/40)*100:.0f}%")

                st.markdown("**Individual Scores:**")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"{get_score_color(scores['emotional_hook'])} Emotional Hook: **{scores['emotional_hook']}/10**")
                    st.write(f"{get_score_color(scores['benefit'])} Benefit: **{scores['benefit']}/10**")
                with col_b:
                    st.write(f"{get_score_color(scores['active_voice'])} Active Voice: **{scores['active_voice']}/10**")
                    st.write(f"{get_score_color(scores['creates_urgency'])} Urgency: **{scores['creates_urgency']}/10**")

                # Recommendations
                st.markdown("**Recommendations:**")
                if scores['emotional_hook'] < 6:
                    st.info("Consider starting with a power verb or emotional hook")
                if scores['benefit'] < 6:
                    st.info("Clearly state the benefit or value proposition")
                if scores['active_voice'] < 6:
                    st.info("Rewrite passive constructions in active voice")
                if scores['creates_urgency'] < 6:
                    st.info("Add elements that create interest or mild urgency")

with tab2:
    st.subheader("Bulk Analysis from CSV/Excel")

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
                url_col = st.selectbox("URL Column", ["(None)"] + list(df.columns))
                meta_col = st.selectbox("Meta Description Column", list(df.columns))

            with col2:
                # Option to compare multiple descriptions
                compare_mode = st.checkbox("Compare multiple descriptions per row")
                if compare_mode:
                    meta_cols = st.multiselect("Select description columns to compare",
                                               list(df.columns), max_selections=4)

            if st.button("Analyze All", type="primary", disabled=not api_key):
                client = OpenAI(api_key=api_key)

                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                if compare_mode and meta_cols:
                    # Compare multiple descriptions
                    for idx, row in df.iterrows():
                        status_text.text(f"Processing row {idx + 1} of {len(df)}...")

                        row_result = {
                            'url': row[url_col] if url_col != "(None)" else "",
                        }

                        best_score = 0
                        best_desc = ""

                        for meta_c in meta_cols:
                            meta_text = str(row.get(meta_c, "")) if pd.notna(row.get(meta_c)) else ""

                            if meta_text:
                                scores, error = analyze_meta_description(client, model,
                                    row_result['url'], meta_text)

                                if scores:
                                    total = sum(scores.values())
                                    row_result[f'{meta_c}_text'] = meta_text
                                    row_result[f'{meta_c}_emotional_hook'] = scores['emotional_hook']
                                    row_result[f'{meta_c}_benefit'] = scores['benefit']
                                    row_result[f'{meta_c}_active_voice'] = scores['active_voice']
                                    row_result[f'{meta_c}_urgency'] = scores['creates_urgency']
                                    row_result[f'{meta_c}_total'] = total

                                    if total > best_score:
                                        best_score = total
                                        best_desc = meta_c

                                sleep(0.5)  # Rate limiting

                        row_result['winning_description'] = best_desc
                        row_result['winning_score'] = best_score
                        results.append(row_result)

                        progress_bar.progress((idx + 1) / len(df))

                else:
                    # Single description per row
                    for idx, row in df.iterrows():
                        status_text.text(f"Processing row {idx + 1} of {len(df)}...")

                        url = row[url_col] if url_col != "(None)" else ""
                        meta_text = str(row.get(meta_col, "")) if pd.notna(row.get(meta_col)) else ""

                        scores, error = analyze_meta_description(client, model, url, meta_text)

                        if scores:
                            results.append({
                                'url': url,
                                'meta_description': meta_text,
                                'emotional_hook_score': scores['emotional_hook'],
                                'benefit_score': scores['benefit'],
                                'active_voice_score': scores['active_voice'],
                                'urgency_score': scores['creates_urgency'],
                                'total_score': sum(scores.values())
                            })
                        else:
                            results.append({
                                'url': url,
                                'meta_description': meta_text,
                                'error': error or "Unknown error"
                            })

                        sleep(0.5)  # Rate limiting
                        progress_bar.progress((idx + 1) / len(df))

                status_text.text("Complete!")

                # Display results
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)

                # Summary stats
                if 'total_score' in results_df.columns:
                    st.markdown("**Summary Statistics:**")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Average Score", f"{results_df['total_score'].mean():.1f}/40")
                    col2.metric("Max Score", f"{results_df['total_score'].max()}/40")
                    col3.metric("Min Score", f"{results_df['total_score'].min()}/40")
                    col4.metric("Rows Analyzed", len(results_df))

                # Download
                col1, col2 = st.columns(2)
                with col1:
                    csv = results_df.to_csv(index=False)
                    st.download_button("Download CSV", csv, "meta_description_scores.csv", "text/csv")
                with col2:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        results_df.to_excel(writer, index=False)
                    st.download_button("Download Excel", output.getvalue(),
                                       "meta_description_scores.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"Error loading file: {e}")

# Footer
st.markdown("---")
