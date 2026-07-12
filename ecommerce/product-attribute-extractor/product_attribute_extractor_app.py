# Author: Lee Foot
# Website: https://leefoot.com
"""
Product Attribute Extractor - Streamlit App

Upload a product CSV, select the text column, and extract structured attributes
using any OpenAI-compatible LLM. Attributes are iteratively discovered across
the catalogue for consistency.

Author: Lee Foot
Website: https://leefoot.com
"""

import streamlit as st
import pandas as pd
from product_attribute_extractor import (
    create_client,
    extract_attributes,
    sort_columns_by_frequency,
)

st.set_page_config(
    page_title="Product Attribute Extractor",
    page_icon="🏷️",
    layout="wide",
)

st.title("🏷️ Product Attribute Extractor")
st.markdown(
    "*Created by* "
    "[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)]"
    "(https://www.leefoot.com) · "
    "[![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)]"
    "(https://www.leefoot.com/contact) · "
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)]"
    "(https://www.linkedin.com/in/lee-foot/) · "
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)]"
    "(https://bsky.app/profile/leefootseo.bsky.social) · "
    "[![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)]"
    "(https://leefoot.com/tools) · "
    "[![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)]"
    "(https://github.com/searchsolved/search-solved-public-seo)"
)

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Extracts structured product attributes from titles and descriptions using an LLM
    - Iteratively discovers new attribute types across your catalogue
    - Outputs an enriched CSV with one column per attribute

    **How to use:**
    1. Enter your OpenAI API key (or compatible endpoint key)
    2. Upload a CSV containing product titles or descriptions
    3. Select the column containing product text
    4. Choose a model and optionally set a custom base URL for local LLMs
    5. Click Extract and download the enriched CSV

    **Best for:**
    - Product data enrichment and normalisation
    - Faceted navigation planning
    - Product feed optimisation
    - Structured data preparation
    """)

st.markdown("Extract structured attributes from product titles using an LLM.")

# --- Sidebar configuration ---
st.sidebar.header("Configuration")

api_key = st.sidebar.text_input(
    "API Key",
    type="password",
    help="Your OpenAI API key or compatible endpoint key.",
)

base_url = st.sidebar.text_input(
    "Base URL",
    value="https://api.openai.com/v1",
    help="For local LLMs (e.g. LM Studio), set to http://localhost:1234/v1",
)

model = st.sidebar.text_input(
    "Model",
    value="gpt-4o-mini",
    help="Model identifier. Use gpt-4o-mini for cost-effective extraction.",
)

# --- File upload ---
uploaded_file = st.file_uploader("Upload product CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, on_bad_lines="skip")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    st.success(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")

    text_column = st.selectbox(
        "Select the column containing product text",
        options=df.columns.tolist(),
        help="Typically the product title, H1, or name column.",
    )

    st.dataframe(df[[text_column]].head(10), use_container_width=True)

    if st.button("Extract Attributes", type="primary"):
        if not api_key:
            st.error("Please enter an API key in the sidebar.")
            st.stop()

        client = create_client(api_key=api_key, base_url=base_url)
        known_attributes = set()
        results = []

        progress_bar = st.progress(0)
        status_text = st.empty()
        total = len(df)

        for idx, row in df.iterrows():
            product_text = str(row[text_column])
            if not product_text or product_text == "nan":
                results.append({})
                continue

            attrs = extract_attributes(client, model, product_text, known_attributes)

            # Update known attributes with any new discoveries
            for attr_name in attrs:
                known_attributes.add(attr_name)

            record = {"product_text": product_text}
            record.update(attrs)
            results.append(record)

            progress = (idx + 1) / total
            progress_bar.progress(progress)
            status_text.text(
                f"Processed {idx + 1:,} of {total:,} products "
                f"({len(known_attributes)} attributes discovered)"
            )

        progress_bar.progress(1.0)
        status_text.text(
            f"Complete. {total:,} products processed, "
            f"{len(known_attributes)} unique attributes discovered."
        )

        # Build output DataFrame
        output_df = pd.DataFrame(results)
        output_df = sort_columns_by_frequency(output_df)

        # Ensure product_text is first column
        cols = ["product_text"] + [c for c in output_df.columns if c != "product_text"]
        output_df = output_df[cols]

        st.subheader("Extracted Attributes")
        st.dataframe(output_df, use_container_width=True)

        csv_data = output_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="Download Enriched CSV",
            data=csv_data,
            file_name="product_attributes_extracted.csv",
            mime="text/csv",
        )
