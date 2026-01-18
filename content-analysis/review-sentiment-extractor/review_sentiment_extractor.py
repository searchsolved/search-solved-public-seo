####################################################################################
#                                                                                  #
#  Review Sentiment Extractor                                                      #
#                                                                                  #
#  Use OpenAI to extract positive and negative sentiments from reviews.            #
#                                                                                  #
####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                                   #
# Contact  : https://www.leefoot.com/contact                                           #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                       #
####################################################################################

"""
Review Sentiment Extractor

Uses OpenAI's GPT models to analyze product reviews and extract
positive and negative sentiments, pain points, and praise points.

Features:
- Upload CSV with review text
- AI-powered sentiment extraction
- Batch processing for efficiency
- Extract both positive and negative themes
- Export results with sentiment summaries
"""

import streamlit as st
import pandas as pd
import time
import json

st.set_page_config(page_title="Review Sentiment Extractor", page_icon="💬", layout="wide")

# Check for required packages
try:
    from openai import OpenAI
    PACKAGES_AVAILABLE = True
except ImportError:
    PACKAGES_AVAILABLE = False

st.title("Review Sentiment Extractor")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")

if not PACKAGES_AVAILABLE:
    st.error("""
    Required packages not installed. Run:
    ```
    pip install openai
    ```
    """)
    st.stop()

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Analyzes customer reviews using AI
    - Extracts positive sentiments (praise points)
    - Extracts negative sentiments (pain points)
    - Provides concise summaries of each

    **Data requirements:**
    - CSV with a column containing review text
    - Optional: ID column to track reviews

    **Output includes:**
    - Original review
    - Positive sentiment summary
    - Negative sentiment summary
    - Overall sentiment classification

    **Cost estimate:**
    - GPT-3.5-turbo: ~$0.001-0.002 per review
    - GPT-4o-mini: ~$0.0015-0.003 per review
    - Batch processing reduces costs

    **Tips:**
    - Use batch size of 5-10 for optimal cost/speed balance
    - Clean your review data before processing
    - Remove duplicates to save on API costs
    """)

# Sidebar settings
st.sidebar.header("OpenAI Settings")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Your OpenAI API key from platform.openai.com"
)

model = st.sidebar.selectbox(
    "Model",
    ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"],
    help="GPT-4o-mini is recommended for cost-effectiveness"
)

st.sidebar.markdown("---")
st.sidebar.header("Processing Settings")

batch_size = st.sidebar.slider(
    "Batch size",
    min_value=1,
    max_value=20,
    value=5,
    help="Number of reviews to process in each API call"
)

delay = st.sidebar.slider(
    "Delay between batches (seconds)",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.5
)

st.sidebar.markdown("---")
st.sidebar.header("Extraction Options")

extract_positive = st.sidebar.checkbox("Extract positive sentiments", value=True)
extract_negative = st.sidebar.checkbox("Extract negative sentiments", value=True)

product_context = st.sidebar.text_input(
    "Product/Service context (optional)",
    value="",
    help="e.g., 'curtains' - helps AI understand domain-specific positives/negatives"
)

# System prompts
def get_system_prompt(positive=True, negative=True, context=""):
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
    {{"id": "1", "positive": "summary of positives", "negative": "summary of negatives", "sentiment": "positive|negative|mixed|neutral"}}
  ]
}}"""
    elif positive:
        return f"""You are a review analyst focused on POSITIVE aspects only.
For each review, extract what customers liked, praised, or found valuable.
If a review has no positive aspects, use "N/A".
Keep summaries concise (1-2 sentences).{context_hint}

Respond ONLY with valid JSON in this format:
{{
  "reviews": [
    {{"id": "1", "positive": "summary of positives"}}
  ]
}}"""
    else:
        return f"""You are a review analyst focused on NEGATIVE aspects only.
For each review, extract complaints, pain points, and issues customers mentioned.
If a review has no negative aspects (is neutral/positive), use "N/A".
Keep summaries concise (1-2 sentences).{context_hint}

Respond ONLY with valid JSON in this format:
{{
  "reviews": [
    {{"id": "1", "negative": "summary of negatives"}}
  ]
}}"""


# File upload
st.subheader("Upload Review Data")

review_file = st.file_uploader(
    "Upload CSV with reviews",
    type=['csv'],
    help="CSV file with a column containing review text"
)

if review_file is not None:
    try:
        try:
            df = pd.read_csv(review_file, encoding='utf-8')
        except:
            review_file.seek(0)
            df = pd.read_csv(review_file, encoding='latin-1')

        st.success(f"Loaded {len(df):,} reviews")

        col1, col2 = st.columns(2)
        with col1:
            review_col = st.selectbox(
                "Select review text column",
                df.columns.tolist()
            )
        with col2:
            id_col = st.selectbox(
                "Select ID column (optional)",
                ["(Auto-generate)"] + df.columns.tolist()
            )

        with st.expander("Preview data"):
            st.dataframe(df.head(20))

        # Sample size option
        max_reviews = st.number_input(
            "Maximum reviews to process",
            min_value=1,
            max_value=len(df),
            value=min(100, len(df)),
            help="Limit processing to save API costs during testing"
        )

        if st.button("Extract Sentiments", type="primary"):
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar")
                st.stop()

            if not extract_positive and not extract_negative:
                st.error("Please select at least one sentiment type to extract")
                st.stop()

            # Initialize OpenAI client
            try:
                client = OpenAI(api_key=api_key)
            except Exception as e:
                st.error(f"Error initializing OpenAI client: {str(e)}")
                st.stop()

            # Prepare data
            df_work = df.head(max_reviews).copy()

            if id_col == "(Auto-generate)":
                df_work['_id'] = range(1, len(df_work) + 1)
                id_column = '_id'
            else:
                id_column = id_col

            progress_bar = st.progress(0)
            status_text = st.empty()

            results = []
            total_batches = (len(df_work) + batch_size - 1) // batch_size

            system_prompt = get_system_prompt(extract_positive, extract_negative, product_context)

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(df_work))

                status_text.text(f"Processing batch {batch_idx + 1}/{total_batches}...")
                progress_bar.progress((batch_idx + 1) / total_batches)

                # Prepare batch data
                batch_df = df_work.iloc[start_idx:end_idx]
                reviews_data = []
                for _, row in batch_df.iterrows():
                    reviews_data.append({
                        "id": str(row[id_column]),
                        "review": str(row[review_col])[:1000]  # Truncate long reviews
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
                    st.warning(f"Error processing batch {batch_idx + 1}: {str(e)}")
                    # Add empty results for failed batch
                    for _, row in batch_df.iterrows():
                        result = {"id": str(row[id_column]), "error": str(e)}
                        results.append(result)

                time.sleep(delay)

            status_text.text("Extraction complete!")

            # Create results DataFrame
            df_results = pd.DataFrame(results)

            # Merge with original data
            df_work[id_column] = df_work[id_column].astype(str)
            df_results['id'] = df_results['id'].astype(str)
            df_final = pd.merge(
                df_work,
                df_results,
                left_on=id_column,
                right_on='id',
                how='left'
            )

            # Display results
            st.subheader("Extraction Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Reviews Processed", len(df_final))

            if extract_positive and extract_negative:
                with col2:
                    if 'sentiment' in df_final.columns:
                        positive_count = (df_final['sentiment'] == 'positive').sum()
                        st.metric("Positive Reviews", positive_count)
                with col3:
                    if 'sentiment' in df_final.columns:
                        negative_count = (df_final['sentiment'] == 'negative').sum()
                        st.metric("Negative Reviews", negative_count)

            # Sentiment distribution
            if extract_positive and extract_negative and 'sentiment' in df_final.columns:
                st.subheader("Sentiment Distribution")
                sentiment_counts = df_final['sentiment'].value_counts()
                st.bar_chart(sentiment_counts)

            # Results table
            st.subheader("Detailed Results")
            display_cols = [review_col]
            if extract_positive:
                display_cols.append('positive')
            if extract_negative:
                display_cols.append('negative')
            if extract_positive and extract_negative and 'sentiment' in df_final.columns:
                display_cols.append('sentiment')

            st.dataframe(df_final[display_cols].head(100), use_container_width=True)

            # Common themes (word cloud alternative)
            if extract_negative and 'negative' in df_final.columns:
                st.subheader("Sample Negative Themes")
                negatives = df_final[df_final['negative'] != 'N/A']['negative'].head(10)
                for i, neg in enumerate(negatives, 1):
                    st.write(f"{i}. {neg}")

            if extract_positive and 'positive' in df_final.columns:
                st.subheader("Sample Positive Themes")
                positives = df_final[df_final['positive'] != 'N/A']['positive'].head(10)
                for i, pos in enumerate(positives, 1):
                    st.write(f"{i}. {pos}")

            # Download
            st.subheader("Download")
            csv_output = df_final.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="Download Results (CSV)",
                data=csv_output,
                file_name="review_sentiments.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("Upload a CSV file with review data to begin")

    st.subheader("Example Input")
    example_input = {
        "Review ID": [1, 2, 3],
        "Review Text": [
            "Love these curtains! They block out all the light and look great.",
            "Poor quality, fabric is thin and doesn't block light at all.",
            "Good value for money but took ages to arrive."
        ]
    }
    st.dataframe(pd.DataFrame(example_input))

    st.subheader("Example Output")
    example_output = {
        "Review ID": [1, 2, 3],
        "Positive": ["Excellent light blocking, attractive appearance", "N/A", "Good value for money"],
        "Negative": ["N/A", "Poor fabric quality, fails to block light", "Slow delivery"],
        "Sentiment": ["positive", "negative", "mixed"]
    }
    st.dataframe(pd.DataFrame(example_output))
