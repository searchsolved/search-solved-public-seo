####################################################################################
#                                                                                  #
#  Micro-Moments Classifier                                                        #
#                                                                                  #
#  Classify keywords/queries into Google's 4 micro-moments using OpenAI.           #
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
Micro-Moments Classifier

Classifies keywords/queries into Google's 4 micro-moments using OpenAI:
- I-want-to-BUY (transactional intent)
- I-want-to-KNOW (informational intent)
- I-want-to-DO (instructional/tutorial intent)
- I-want-to-GO (navigational/local intent)

Features:
- Upload CSV of keywords
- Batch classification using GPT
- Confidence scores
- Export with classifications
"""

import streamlit as st
import pandas as pd
import json
from io import BytesIO
from openai import OpenAI

st.set_page_config(page_title="Micro-Moments Classifier", page_icon="🎯", layout="wide")

st.title("Micro-Moments Classifier")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Classifies keywords into Google's 4 micro-moments
    - Uses AI to understand search intent

    **The 4 Micro-Moments:**
    - **I-want-to-BUY** - Transactional intent, ready to purchase
    - **I-want-to-KNOW** - Informational intent, seeking knowledge
    - **I-want-to-DO** - Instructional intent, seeking how-to guidance
    - **I-want-to-GO** - Navigational/local intent, seeking a place/brand

    **Requirements:**
    - OpenAI API key
    - CSV with keywords

    **How to use:**
    1. Enter your OpenAI API key
    2. Upload keywords CSV
    3. Set batch size (more = faster but higher API cost)
    4. Click "Classify Keywords"
    5. Download results

    **Note:** Each batch of keywords uses one API call.
    """)

# Sidebar settings
st.sidebar.header("API Settings")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Your OpenAI API key"
)

st.sidebar.markdown("---")
st.sidebar.header("Classification Settings")

batch_size = st.sidebar.slider(
    "Batch size",
    min_value=10,
    max_value=100,
    value=50,
    help="Keywords per API call (higher = faster but may hit token limits)"
)

model = st.sidebar.selectbox(
    "Model",
    ["gpt-4o-mini", "gpt-4o", "gpt-4.1"],
    index=0,
    help="GPT model to use"
)

include_confidence = st.sidebar.checkbox(
    "Include confidence scores",
    value=True,
    help="Ask the model for confidence scores (1-5)"
)


def classify_keywords_batch(keywords, api_key, model, include_confidence):
    """Classify a batch of keywords using OpenAI."""
    client = OpenAI(api_key=api_key)

    keyword_list = "\n".join([f"- {kw}" for kw in keywords])

    confidence_instruction = ""
    if include_confidence:
        confidence_instruction = "Also provide a confidence score from 1-5 for each classification."

    messages = [
        {
            "role": "system",
            "content": """You are an SEO expert that classifies search queries into Google's 4 micro-moments:

1. I-want-to-BUY - Transactional intent, user wants to purchase something
2. I-want-to-KNOW - Informational intent, user wants to learn something
3. I-want-to-DO - Instructional intent, user wants to accomplish a task
4. I-want-to-GO - Navigational/local intent, user wants to find a specific place or website

Return ONLY valid JSON, no explanations."""
        },
        {
            "role": "user",
            "content": f"""Classify each of these keywords into one of the 4 micro-moments:

{keyword_list}

{confidence_instruction}

Return JSON in this exact format:
{{
  "classifications": [
    {{"keyword": "keyword1", "micro_moment": "I-want-to-BUY", "confidence": 5}},
    {{"keyword": "keyword2", "micro_moment": "I-want-to-KNOW", "confidence": 4}}
  ]
}}

Include ALL keywords in your response."""
        }
    ]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=4000
        )

        response_content = completion.choices[0].message.content
        result = json.loads(response_content)

        return result.get('classifications', []), None

    except Exception as e:
        return None, str(e)


# File upload
st.subheader("Upload Keywords")

uploaded_file = st.file_uploader(
    "Upload CSV with keywords",
    type=['csv'],
    help="CSV with a 'keyword' column"
)

keywords = []

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} rows")

        with st.expander("Column Selection"):
            cols = df.columns.tolist()
            kw_options = [c for c in cols if 'keyword' in c.lower() or 'query' in c.lower()]
            default_idx = cols.index(kw_options[0]) if kw_options else 0
            keyword_col = st.selectbox(
                "Keyword column",
                cols,
                index=default_idx
            )

        keywords = df[keyword_col].dropna().astype(str).tolist()
        st.info(f"Found {len(keywords)} keywords to classify")

        with st.expander("Preview keywords"):
            st.write(keywords[:20])

    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")

# Manual input alternative
with st.expander("Or enter keywords manually"):
    manual_input = st.text_area(
        "Enter keywords (one per line)",
        height=150,
        placeholder="buy running shoes\nhow to tie shoelaces\nwhat is seo\nnike store near me"
    )

    if manual_input and not uploaded_file:
        keywords = [kw.strip() for kw in manual_input.strip().split('\n') if kw.strip()]
        st.info(f"Entered {len(keywords)} keywords")

if keywords:
    num_batches = (len(keywords) + batch_size - 1) // batch_size
    st.caption(f"Will process in {num_batches} batch(es) = ~{num_batches} API call(s)")

if st.button("Classify Keywords", type="primary", disabled=not api_key or not keywords):
    if not api_key:
        st.error("Please enter your OpenAI API key")
    elif not keywords:
        st.error("Please upload keywords or enter them manually")
    else:
        all_results = []
        progress_bar = st.progress(0)

        num_batches = (len(keywords) + batch_size - 1) // batch_size

        for i in range(0, len(keywords), batch_size):
            batch = keywords[i:i + batch_size]
            batch_num = i // batch_size + 1

            st.text(f"Processing batch {batch_num}/{num_batches}...")

            results, error = classify_keywords_batch(batch, api_key, model, include_confidence)

            if error:
                st.error(f"Error in batch {batch_num}: {error}")
            elif results:
                all_results.extend(results)

            progress_bar.progress(batch_num / num_batches)

        if all_results:
            df_results = pd.DataFrame(all_results)

            # Ensure all original keywords are in results
            classified_kws = set(df_results['keyword'].str.lower())
            missing_kws = [kw for kw in keywords if kw.lower() not in classified_kws]

            if missing_kws:
                st.warning(f"{len(missing_kws)} keywords couldn't be classified")
                for kw in missing_kws:
                    df_results = pd.concat([df_results, pd.DataFrame([{
                        'keyword': kw,
                        'micro_moment': 'Unclassified',
                        'confidence': 0
                    }])], ignore_index=True)

            # Store results
            st.session_state['classifications'] = df_results

            st.success(f"Classified {len(df_results)} keywords!")

# Display results
if 'classifications' in st.session_state:
    df_results = st.session_state['classifications']

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Keywords", len(df_results))
    with col2:
        buy_count = len(df_results[df_results['micro_moment'] == 'I-want-to-BUY'])
        st.metric("I-want-to-BUY", buy_count)
    with col3:
        know_count = len(df_results[df_results['micro_moment'] == 'I-want-to-KNOW'])
        st.metric("I-want-to-KNOW", know_count)
    with col4:
        do_count = len(df_results[df_results['micro_moment'] == 'I-want-to-DO'])
        st.metric("I-want-to-DO", do_count)

    # Distribution chart
    st.subheader("Micro-Moment Distribution")
    moment_counts = df_results['micro_moment'].value_counts()
    st.bar_chart(moment_counts)

    # Results by moment
    st.subheader("Keywords by Micro-Moment")

    tabs = st.tabs(["All", "BUY", "KNOW", "DO", "GO"])

    with tabs[0]:
        st.dataframe(df_results, use_container_width=True)

    with tabs[1]:
        buy_df = df_results[df_results['micro_moment'] == 'I-want-to-BUY']
        st.dataframe(buy_df, use_container_width=True)

    with tabs[2]:
        know_df = df_results[df_results['micro_moment'] == 'I-want-to-KNOW']
        st.dataframe(know_df, use_container_width=True)

    with tabs[3]:
        do_df = df_results[df_results['micro_moment'] == 'I-want-to-DO']
        st.dataframe(do_df, use_container_width=True)

    with tabs[4]:
        go_df = df_results[df_results['micro_moment'] == 'I-want-to-GO']
        st.dataframe(go_df, use_container_width=True)

    # Downloads
    st.subheader("Download")

    col1, col2 = st.columns(2)

    with col1:
        csv_data = df_results.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="micro_moments_classified.csv",
            mime="text/csv"
        )

    with col2:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_results.to_excel(writer, sheet_name='All Keywords', index=False)

            # Sheet per moment
            for moment in df_results['micro_moment'].unique():
                moment_df = df_results[df_results['micro_moment'] == moment]
                sheet_name = moment.replace('I-want-to-', '')[:31]  # Excel sheet name limit
                moment_df.to_excel(writer, sheet_name=sheet_name, index=False)

        st.download_button(
            label="Download Excel",
            data=output.getvalue(),
            file_name="micro_moments_classified.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    if not api_key:
        st.warning("Enter your OpenAI API key in the sidebar to get started")

    st.subheader("Example Output")

    example_data = {
        "keyword": ["buy nike air max", "what is seo", "how to tie a tie", "apple store near me"],
        "micro_moment": ["I-want-to-BUY", "I-want-to-KNOW", "I-want-to-DO", "I-want-to-GO"],
        "confidence": [5, 5, 5, 5]
    }
    st.dataframe(pd.DataFrame(example_data))

    st.markdown("""
    **Use Cases:**
    - Segment your keyword list by intent
    - Prioritize transactional keywords for product pages
    - Create content clusters based on informational queries
    - Optimize local SEO for GO queries
    """)
