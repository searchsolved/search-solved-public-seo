"""
Review Sentiment Extractor - Streamlit App

Use OpenAI to extract positive and negative sentiments from product reviews.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
import json
import time

try:
    from openai import OpenAI
except ImportError:
    st.error("Please install openai: pip install openai")
    st.stop()

st.set_page_config(
    page_title="Review Sentiment Extractor",
    page_icon="💭",
    layout="wide"
)

st.title("💭 Review Sentiment Extractor")
st.markdown("Extract positive and negative sentiments from product reviews using AI.")


def get_system_prompt(positive=True, negative=True, context=""):
    """Generate system prompt based on extraction options."""
    context_hint = f"\n\nContext: These reviews are for {context}." if context else ""

    if positive and negative:
        return f"""You are a review analyst. For each review, extract:
1. POSITIVE aspects (praise, what customers liked)
2. NEGATIVE aspects (complaints, pain points)

If a review has no positive aspects, use "N/A" for positive.
If a review has no negative aspects, use "N/A" for negative.
Keep summaries concise (1-2 sentences each).{context_hint}

Respond ONLY with valid JSON in this format:
{{
  "reviews": [
    {{"id": "1", "positive": "summary", "negative": "summary", "sentiment": "positive|negative|mixed|neutral"}}
  ]
}}"""
    elif positive:
        return f"""You are a review analyst focused on POSITIVE aspects only.
For each review, extract what customers liked, praised, or found valuable.
If a review has no positive aspects, use "N/A".{context_hint}

Respond ONLY with valid JSON:
{{"reviews": [{{"id": "1", "positive": "summary"}}]}}"""
    else:
        return f"""You are a review analyst focused on NEGATIVE aspects only.
For each review, extract complaints, pain points, and issues.
If a review has no negative aspects, use "N/A".{context_hint}

Respond ONLY with valid JSON:
{{"reviews": [{{"id": "1", "negative": "summary"}}]}}"""


def process_reviews(df, review_col, id_col, client, model, batch_size, delay,
                    extract_positive, extract_negative, context, progress_bar, status_text):
    """Process reviews in batches."""
    system_prompt = get_system_prompt(extract_positive, extract_negative, context)
    results = []
    total_batches = (len(df) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(df))

        progress_bar.progress((batch_idx + 1) / total_batches)
        status_text.text(f"Processing batch {batch_idx + 1}/{total_batches}...")

        batch_df = df.iloc[start_idx:end_idx]
        reviews_data = []

        for _, row in batch_df.iterrows():
            reviews_data.append({
                "id": str(row[id_col]),
                "review": str(row[review_col])[:1000]
            })

        user_content = json.dumps({"reviews": reviews_data})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            batch_results = json.loads(response.choices[0].message.content)

            for item in batch_results.get("reviews", []):
                result = {"id": item.get("id")}
                if extract_positive:
                    result["positive"] = item.get("positive", "N/A")
                if extract_negative:
                    result["negative"] = item.get("negative", "N/A")
                if extract_positive and extract_negative:
                    result["sentiment"] = item.get("sentiment", "unknown")
                results.append(result)

        except Exception as e:
            for _, row in batch_df.iterrows():
                results.append({"id": str(row[id_col]), "error": str(e)})

        time.sleep(delay)

    return pd.DataFrame(results)


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key"
    )

    model = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        help="Select the OpenAI model to use"
    )

    st.markdown("---")
    st.subheader("🎯 Extraction Settings")

    extract_positive = st.checkbox("Extract Positive", value=True)
    extract_negative = st.checkbox("Extract Negative", value=True)

    context = st.text_input(
        "Product Context",
        placeholder="e.g., curtains, electronics",
        help="Help the AI understand the product type"
    )

    st.markdown("---")
    st.subheader("⚡ Processing Settings")

    batch_size = st.slider(
        "Batch Size",
        min_value=1,
        max_value=20,
        value=5,
        help="Reviews per API call"
    )

    delay = st.slider(
        "Delay (seconds)",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.5,
        help="Delay between batches"
    )

# Main content
st.markdown("### 📤 Upload Reviews")

uploaded_file = st.file_uploader(
    "Upload CSV with reviews",
    type=["csv"],
    help="CSV should contain a column with review text"
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except:
        df = pd.read_csv(uploaded_file, encoding='latin-1')

    st.success(f"✅ Loaded {len(df):,} reviews")

    # Column selection
    col1, col2 = st.columns(2)

    with col1:
        review_col = st.selectbox(
            "Review Text Column",
            df.columns.tolist(),
            help="Select the column containing review text"
        )

    with col2:
        id_options = ['Auto-generate'] + df.columns.tolist()
        id_selection = st.selectbox(
            "ID Column (optional)",
            id_options,
            help="Select ID column or auto-generate"
        )

        if id_selection == 'Auto-generate':
            df['_id'] = range(1, len(df) + 1)
            id_col = '_id'
        else:
            id_col = id_selection

    # Preview
    with st.expander("Preview Reviews"):
        preview_cols = [id_col, review_col] if id_col in df.columns else [review_col]
        st.dataframe(df[preview_cols].head(10), use_container_width=True)

    # Limit reviews
    max_reviews = st.number_input(
        "Max Reviews to Process",
        min_value=1,
        max_value=len(df),
        value=min(100, len(df)),
        help="Limit reviews for testing or cost control"
    )

    df_to_process = df.head(max_reviews)

    if api_key:
        if st.button("💭 Extract Sentiments", type="primary", use_container_width=True):
            client = OpenAI(api_key=api_key)
            progress_bar = st.progress(0)
            status_text = st.empty()

            results_df = process_reviews(
                df_to_process, review_col, id_col, client, model, batch_size, delay,
                extract_positive, extract_negative, context, progress_bar, status_text
            )

            progress_bar.empty()
            status_text.empty()

            # Merge with original
            df_to_process[id_col] = df_to_process[id_col].astype(str)
            results_df['id'] = results_df['id'].astype(str)
            df_final = pd.merge(
                df_to_process, results_df,
                left_on=id_col, right_on='id',
                how='left'
            )

            st.success(f"✅ Processed {len(df_final)} reviews!")

            # Results tabs
            tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Analysis", "💡 Insights"])

            with tab1:
                display_cols = [review_col]
                if extract_positive:
                    display_cols.append('positive')
                if extract_negative:
                    display_cols.append('negative')
                if extract_positive and extract_negative:
                    display_cols.append('sentiment')

                st.dataframe(df_final[display_cols], use_container_width=True, height=400)

                csv = df_final.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "📥 Download Results CSV",
                    data=csv,
                    file_name="review_sentiments.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with tab2:
                if 'sentiment' in df_final.columns:
                    st.subheader("Sentiment Distribution")

                    sentiment_counts = df_final['sentiment'].value_counts()

                    col1, col2 = st.columns(2)

                    with col1:
                        import plotly.express as px
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title='Overall Sentiment'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        for sentiment, count in sentiment_counts.items():
                            pct = count / len(df_final) * 100
                            st.metric(sentiment.title(), f"{count} ({pct:.1f}%)")

                # Word frequency
                if extract_positive and 'positive' in df_final.columns:
                    st.subheader("Common Positive Themes")
                    positive_text = ' '.join(df_final['positive'].dropna().astype(str))
                    # Simple word frequency
                    words = positive_text.lower().split()
                    from collections import Counter
                    common_positive = Counter(w for w in words if len(w) > 4 and w != 'n/a').most_common(10)
                    if common_positive:
                        st.write([f"{w}: {c}" for w, c in common_positive])

            with tab3:
                st.subheader("💡 Key Insights")

                if 'sentiment' in df_final.columns:
                    positive_pct = (df_final['sentiment'] == 'positive').mean() * 100
                    negative_pct = (df_final['sentiment'] == 'negative').mean() * 100

                    if positive_pct > 70:
                        st.success(f"✅ Great sentiment! {positive_pct:.0f}% positive reviews")
                    elif positive_pct > 50:
                        st.info(f"👍 Good sentiment. {positive_pct:.0f}% positive")
                    else:
                        st.warning(f"⚠️ Mixed sentiment. Only {positive_pct:.0f}% positive")

                if extract_negative and 'negative' in df_final.columns:
                    st.markdown("### Common Complaints")
                    negatives = df_final[df_final['negative'] != 'N/A']['negative'].dropna()
                    for neg in negatives.head(5):
                        st.markdown(f"- {neg}")

                if extract_positive and 'positive' in df_final.columns:
                    st.markdown("### Common Praise")
                    positives = df_final[df_final['positive'] != 'N/A']['positive'].dropna()
                    for pos in positives.head(5):
                        st.markdown(f"- {pos}")

    else:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar")

else:
    st.info("👆 Upload a CSV file with reviews to get started")

    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        This tool uses **AI to analyze product reviews** and extract:

        - **Positive aspects**: What customers liked
        - **Negative aspects**: Complaints and pain points
        - **Overall sentiment**: positive, negative, mixed, or neutral

        **Use cases:**
        - Understand customer feedback at scale
        - Identify product improvements
        - Create marketing copy from positive feedback
        - Address common complaints
        """)

# Footer
st.markdown("---")
st.markdown(
    "Built by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · "
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · "
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)"
)
