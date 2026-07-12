"""
Tag Consolidator - Streamlit App

Use OpenAI to consolidate granular secondary tags into broader generic
categories, grouped by primary tag.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import pandas as pd
import streamlit as st

try:
    from openai import OpenAI  # noqa: F401
except ImportError:
    st.error("Please install openai: pip install openai")
    st.stop()

from tag_consolidator import (
    DEFAULT_PRIMARY_COLUMN,
    DEFAULT_SECONDARY_COLUMN,
    GENERIC_TAG_COLUMN,
    consolidate_tags,
)

st.set_page_config(
    page_title="Tag Consolidator",
    page_icon="🏷️",
    layout="wide",
)

st.title("🏷️ Tag Consolidator")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Takes a CSV of granular tags (for example, tags mined from customer feedback)
    - Groups rows by their primary tag
    - Uses AI to consolidate each group's secondary tags into broader generic categories
    - Adds a 'Generic Tag' column mapping every secondary tag to its category

    **Data requirements:**
    - CSV with a primary tag column and a secondary tag column
    - Example: primary tag "Delivery" with secondary tags such as
      "arrived late", "left in wrong place", "damaged in transit"

    **Output includes:**
    - The original data with a new 'Generic Tag' column

    **Tips:**
    - Each primary tag group is one API call, so cost scales with the number of groups
    - Review the consolidated categories before using them downstream
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
    help="GPT-4o-mini is recommended for cost-effectiveness",
)

# File upload
st.subheader("Upload Tag Data")

tag_file = st.file_uploader(
    "Upload CSV with tags",
    type=["csv"],
    help="CSV file with primary and secondary tag columns",
)

if tag_file is not None:
    try:
        try:
            df = pd.read_csv(tag_file, encoding="utf-8")
        except Exception:
            tag_file.seek(0)
            df = pd.read_csv(tag_file, encoding="latin-1")

        st.success(f"Loaded {len(df):,} rows")

        columns = df.columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            primary_column = st.selectbox(
                "Primary tag column",
                columns,
                index=columns.index(DEFAULT_PRIMARY_COLUMN)
                if DEFAULT_PRIMARY_COLUMN in columns
                else 0,
            )
        with col2:
            secondary_column = st.selectbox(
                "Secondary tag column",
                columns,
                index=columns.index(DEFAULT_SECONDARY_COLUMN)
                if DEFAULT_SECONDARY_COLUMN in columns
                else min(1, len(columns) - 1),
            )

        with st.expander("Preview data"):
            st.dataframe(df.head(20))

        total_groups = df[primary_column].nunique()
        st.info(f"{total_groups} primary tag groups will be processed ({total_groups} API calls)")

        if st.button("Consolidate Tags", type="primary"):
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar")
                st.stop()

            if primary_column == secondary_column:
                st.error("Primary and secondary tag columns must be different")
                st.stop()

            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(processed, total, group_name):
                progress_bar.progress(processed / total)
                status_text.text(f"Processed {processed}/{total} groups.")

            try:
                df_final = consolidate_tags(
                    df,
                    api_key=api_key,
                    model=model,
                    primary_column=primary_column,
                    secondary_column=secondary_column,
                    progress_callback=update_progress,
                )
            except Exception as e:
                st.error(f"Error during consolidation: {str(e)}")
                st.stop()

            status_text.text("Consolidation complete!")

            # Display results
            st.subheader("Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows Processed", len(df_final))
            with col2:
                st.metric(
                    "Secondary Tags",
                    df_final[secondary_column].nunique(),
                )
            with col3:
                st.metric(
                    "Generic Categories",
                    df_final[GENERIC_TAG_COLUMN].nunique(),
                )

            unmapped = df_final[GENERIC_TAG_COLUMN].isna().sum()
            if unmapped > 0:
                st.warning(
                    f"{unmapped} rows could not be mapped to a generic category. "
                    "Review these rows manually or re-run the tool."
                )

            st.dataframe(
                df_final[[primary_column, secondary_column, GENERIC_TAG_COLUMN]].head(100),
                use_container_width=True,
            )

            # Download
            st.subheader("Download")
            csv_output = df_final.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Download Results (CSV)",
                data=csv_output,
                file_name="consolidated_tags.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

else:
    st.info("Upload a CSV file with tag data to begin")

    st.subheader("Example Input")
    example_input = {
        "Primary Tag": ["Delivery", "Delivery", "Sizing", "Sizing", "Build Quality"],
        "Secondary Tag": [
            "arrived late",
            "left in wrong place",
            "runs small",
            "inconsistent measurements",
            "flimsy material",
        ],
    }
    st.dataframe(pd.DataFrame(example_input))

    st.subheader("Example Output")
    example_output = {
        "Primary Tag": ["Delivery", "Delivery", "Sizing", "Sizing", "Build Quality"],
        "Secondary Tag": [
            "arrived late",
            "left in wrong place",
            "runs small",
            "inconsistent measurements",
            "flimsy material",
        ],
        "Generic Tag": [
            "Late Delivery",
            "Delivery Handling",
            "Fit Issues",
            "Fit Issues",
            "Material Quality",
        ],
    }
    st.dataframe(pd.DataFrame(example_output))

# Footer
st.markdown("---")
st.markdown(
    "Built by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · "
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · "
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)"
)
