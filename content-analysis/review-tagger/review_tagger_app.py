"""
Review Tagger - Streamlit App

Upload a CSV of reviews and use OpenAI to tag each review with a one or
two-word descriptive label capturing its primary topic.

Author: Lee Foot
Website: https://leefoot.com
"""

import pandas as pd
import streamlit as st

try:
    from openai import OpenAI  # noqa: F401
except ImportError:
    st.error("Please install openai: pip install openai")
    st.stop()

from review_tagger import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLUMN,
    DEFAULT_MODEL,
    tag_reviews,
)

st.set_page_config(
    page_title="Review Tagger",
    page_icon="🏷️",
    layout="wide",
)

st.title("Review Tagger")
st.markdown(
    "*Created by* "
    "[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) "
    "[![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) "
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) "
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) "
    "[![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) "
    "[![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)"
)

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Takes a CSV containing review text
    - Sends reviews to OpenAI in batches
    - Tags each review with a one or two-word descriptive label (e.g. "Delivery", "Build Quality", "Sizing")
    - Returns the original data with a new 'Tag' column

    **Data requirements:**
    - A CSV with at least one column containing review text

    **Tips:**
    - gpt-4o-mini is recommended for cost-effectiveness
    - Larger batch sizes reduce the number of API calls but increase token usage per call
    - After tagging, use the **Tag Consolidator** tool to group granular tags into broader categories
    """)

# Sidebar settings
st.sidebar.header("OpenAI Settings")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Your OpenAI API key from platform.openai.com",
)

model = st.sidebar.selectbox(
    "Model",
    ["gpt-4o-mini", "gpt-4o", "gpt-4.1"],
    help="gpt-4o-mini is recommended for cost-effectiveness",
)

batch_size = st.sidebar.number_input(
    "Batch Size",
    min_value=5,
    max_value=100,
    value=DEFAULT_BATCH_SIZE,
    step=5,
    help="Number of reviews to send per API call",
)

# File upload
st.subheader("Upload Review Data")

review_file = st.file_uploader(
    "Upload CSV with reviews",
    type=["csv"],
    help="CSV file with a column containing review text",
)

if review_file is not None:
    try:
        try:
            df = pd.read_csv(review_file, encoding="utf-8")
        except Exception:
            review_file.seek(0)
            df = pd.read_csv(review_file, encoding="latin-1")

        st.success(f"Loaded {len(df):,} rows")

        columns = df.columns.tolist()

        review_column = st.selectbox(
            "Review text column",
            columns,
            index=columns.index(DEFAULT_COLUMN)
            if DEFAULT_COLUMN in columns
            else 0,
        )

        with st.expander("Preview data"):
            st.dataframe(df.head(20))

        total_batches = (len(df) + batch_size - 1) // batch_size
        st.info(f"{len(df):,} reviews will be processed in {total_batches} batches")

        if st.button("Tag Reviews", type="primary"):
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar")
                st.stop()

            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(processed, total):
                progress_bar.progress(processed / total)
                status_text.text(f"Processed batch {processed}/{total}")

            try:
                df_tagged = tag_reviews(
                    df,
                    api_key=api_key,
                    review_column=review_column,
                    model=model,
                    batch_size=batch_size,
                    progress_callback=update_progress,
                )
            except Exception as e:
                st.error(f"Error during tagging: {str(e)}")
                st.stop()

            status_text.text("Tagging complete!")

            # Display results
            st.subheader("Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Reviews Tagged", len(df_tagged))
            with col2:
                tagged_count = df_tagged["Tag"].notna().sum()
                st.metric("Successfully Tagged", tagged_count)
            with col3:
                st.metric("Unique Tags", df_tagged["Tag"].nunique())

            untagged = df_tagged["Tag"].isna().sum()
            if untagged > 0:
                st.warning(
                    f"{untagged} reviews could not be tagged. "
                    "Review these rows manually or re-run the tool."
                )

            st.dataframe(
                df_tagged[[review_column, "Tag"]].head(100),
                use_container_width=True,
            )

            # Download
            st.subheader("Download")
            csv_output = df_tagged.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Download Results (CSV)",
                data=csv_output,
                file_name="tagged_reviews.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

else:
    st.info("Upload a CSV file with review data to begin")

    st.subheader("Example Input")
    example_input = {
        "Review": [
            "Delivery was late and the box was damaged",
            "Great build quality, very sturdy",
            "Sizing was off, had to return it",
            "Easy to install, took five minutes",
            "Good value for money overall",
        ],
    }
    st.dataframe(pd.DataFrame(example_input))

    st.subheader("Example Output")
    example_output = {
        "Review": [
            "Delivery was late and the box was damaged",
            "Great build quality, very sturdy",
            "Sizing was off, had to return it",
            "Easy to install, took five minutes",
            "Good value for money overall",
        ],
        "Tag": [
            "Delivery",
            "Build Quality",
            "Sizing",
            "Installation",
            "Value",
        ],
    }
    st.dataframe(pd.DataFrame(example_output))

# Footer
st.markdown("---")
st.markdown(
    "Built by "
    "[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) "
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) "
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)"
)
