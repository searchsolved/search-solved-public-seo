"""
Representative Keyword Normaliser - Streamlit App
Suggest a cleaner, more descriptive representative keyword for each keyword in a CSV.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import pandas as pd
import streamlit as st

from core import (DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL,
                  build_client, process_dataframe)

st.set_page_config(
    page_title="Representative Keyword Normaliser",
    page_icon="🔑",
    layout="wide"
)

st.title("🔑 Representative Keyword Normaliser")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Reads keywords from a CSV file
    - Asks an LLM to suggest a cleaner, more descriptive representative keyword for each
    - Keeps the output in the same language as the source keyword
    - Useful as a pre-processing step before SERP clustering or semantic clustering

    **How to use:**
    1. Point the sidebar at any OpenAI-compatible endpoint (Ollama by default)
    2. Upload a CSV of keywords
    3. Choose the keyword column
    4. Run the normaliser and download the results

    **Best for:**
    - Cleaning messy or truncated keyword exports before clustering
    - Normalising scraped page titles into search-style keywords
    - Multi-language keyword lists
    """)
st.markdown("Suggest a cleaner, more descriptive representative keyword for each keyword in your list.")

# Sidebar configuration
with st.sidebar:
    st.header("API Configuration")

    base_url = st.text_input("API Base URL", value=DEFAULT_BASE_URL,
                             help="Any OpenAI-compatible endpoint. Ollama by default, "
                                  "or https://api.openai.com/v1 for OpenAI.")
    model = st.text_input("Model", value=DEFAULT_MODEL,
                          help="Model name as exposed by your endpoint, e.g. a local Llama "
                               "model in Ollama or gpt-4o-mini on OpenAI.")
    api_key = st.text_input("API Key", type="password", value="",
                            help="Leave blank for local servers such as Ollama.")

# Main interface
uploaded_file = st.file_uploader("Upload CSV with keywords", type=['csv'])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        st.write(f"Loaded {len(df)} rows")
        st.dataframe(df.head(), use_container_width=True)

        keyword_col = st.selectbox("Keyword Column", list(df.columns))

        if st.button("Suggest Representative Keywords", type="primary"):
            client = build_client(base_url=base_url,
                                  api_key=api_key if api_key else DEFAULT_API_KEY)

            progress = st.progress(0)
            status = st.empty()

            def progress_callback(done, total):
                progress.progress(done / total)
                status.text(f"Processing keyword {done}/{total}")

            with st.spinner("Processing keywords..."):
                results_df = process_dataframe(df, client, model=model,
                                               column=keyword_col,
                                               progress_callback=progress_callback)

            status.text("Complete!")

            error_count = results_df['Suggested Keyword'].astype(str).str.startswith('Error:').sum()
            changed_count = (results_df['Suggested Keyword'].astype(str).str.strip().str.lower()
                             != results_df[keyword_col].astype(str).str.strip().str.lower()).sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Keywords Processed", len(results_df))
            col2.metric("Suggestions Changed", int(changed_count))
            col3.metric("Errors", int(error_count))

            st.dataframe(results_df, use_container_width=True)

            st.download_button("Download CSV",
                               results_df.to_csv(index=False).encode('utf-8-sig'),
                               "representative_keywords.csv",
                               "text/csv")

    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Upload a CSV file to get started. The keyword column can be selected after upload.")

# Footer
st.markdown("---")
