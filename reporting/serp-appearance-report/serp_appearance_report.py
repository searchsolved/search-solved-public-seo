"""
SERP Appearance Report - Parse ValueSERP Batch JSON
Filters organic results from ValueSERP batch exports to those containing
your domain and reports the query, position, title and snippet for each
appearance.

Author: Lee Foot
"""

from io import BytesIO

import streamlit as st

from serp_appearance_core import extract_appearances, load_json_bytes, results_to_dataframe

st.set_page_config(
    page_title="SERP Appearance Report",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 SERP Appearance Report")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Parses ValueSERP batch mode JSON exports
    - Filters organic results to those containing your domain
    - Reports the query, position, title and snippet for each appearance

    **How to use:**
    1. Run a batch of searches in ValueSERP and download the JSON result sets
    2. Upload one or more JSON files below
    3. Enter your domain (e.g. example.com)
    4. Review the results table and download the CSV

    **Best for:**
    - Checking how your site appears in the SERPs for your top queries
    - Auditing titles and snippets at scale
    - Building a SERP appearance report from rank tracking batches
    """)

st.markdown("""
Upload ValueSERP batch JSON exports and enter a domain to see every organic
appearance, with the query, position, title and snippet for each result.
""")

domain = st.text_input(
    "Domain to filter by",
    placeholder="example.com",
    help="Organic results are kept when the result URL contains this domain"
)

uploaded_files = st.file_uploader(
    "Upload ValueSERP batch JSON file(s)",
    type=['json'],
    accept_multiple_files=True,
    help="Upload one or more JSON result set files downloaded from a ValueSERP batch"
)

if uploaded_files and domain.strip():
    all_results = []
    all_warnings = []

    with st.spinner("Parsing JSON files..."):
        for uploaded_file in uploaded_files:
            try:
                data = load_json_bytes(uploaded_file.getvalue())
            except ValueError as e:
                st.error(f"Could not read {uploaded_file.name}: {e}")
                continue

            results, warnings = extract_appearances(data, domain)
            all_results.extend(results)
            all_warnings.extend(f"{uploaded_file.name}: {w}" for w in warnings)

    for warning in all_warnings:
        st.warning(warning)

    df = results_to_dataframe(all_results)

    st.subheader("Results")
    st.success(f"Found **{len(df)}** appearances of **{domain.strip()}** "
               f"across {len(uploaded_files)} file(s).")

    if df.empty:
        st.info("No organic results contained that domain. Check the domain "
                "spelling and that the files are ValueSERP batch result sets.")
    else:
        st.dataframe(df, use_container_width=True)

        buffer = BytesIO()
        df.to_csv(buffer, index=False, encoding='utf-8-sig')
        buffer.seek(0)

        st.download_button(
            label="📥 Download SERP Appearance Report (CSV)",
            data=buffer,
            file_name="serp_appearance_report.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Upload at least one ValueSERP batch JSON file and enter a domain to get started.")

    st.markdown("""
    ### Expected JSON Format
    Each file should be a ValueSERP batch result set: a JSON array where each
    item contains a `result` object with `search_parameters` (including the
    query `q`) and an `organic_results` array. A single (non-array) result
    object also works.

    ### How it Works
    1. Upload one or more batch JSON exports
    2. The tool loops through every search in each file
    3. Organic results whose URL contains your domain are kept
    4. The output lists the query, position, link, title and snippet per appearance

    ### Use Cases
    - Reviewing how your top queries render in the SERPs
    - Spotting title rewrites and weak snippets
    - Reporting SERP appearances to stakeholders
    """)
