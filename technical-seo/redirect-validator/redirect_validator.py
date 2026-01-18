"""
Redirect/URL Mapping Validator - Streamlit App
Validates that implemented redirects match your redirect mapping specification.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
import io
from datetime import datetime

st.set_page_config(
    page_title="Redirect Validator",
    page_icon="🔀",
    layout="wide"
)

st.title("🔀 Redirect/URL Mapping Validator")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-181717?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")
st.markdown("Compare your redirect mapping specification against actual crawled redirects.")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Validates that implemented redirects match your specification
    - Identifies mismatches, missing, and extra redirects
    - Compares crawled redirects against your mapping document

    **Files needed:**
    1. **Crawled Redirects File**: Export from your crawler (e.g., Screaming Frog)
       - Should contain: Source URL (Address) and Redirect URL columns
    2. **Redirect Mapping File**: Your redirect specification document
       - Should contain: Source URL and Destination URL columns

    **The tool checks for:**
    - **Matches**: Redirects correctly implemented as specified
    - **Mismatches**: Redirects going to wrong destinations
    - **Missing**: Specified redirects not found in crawl
    - **Extra**: Redirects in crawl not in your specification

    **Tips:**
    - URLs are normalized (lowercase, no trailing slashes, no query params)
    - Make sure your column mapping is correct before validating
    - Export the full report for stakeholder sharing
    """)


def clean_url(url):
    """Clean and standardize a URL for comparison."""
    if pd.isna(url) or url is None:
        return ""
    url = str(url).strip().lower()
    # Remove URL parameters
    url = url.split('?')[0]
    # Remove trailing slashes
    url = url.rstrip('/')
    return url


def verify_mappings(crawled_df, source_df, crawled_source_col, crawled_dest_col,
                   source_source_col, source_dest_col):
    """Verify that source mappings match crawled mappings."""

    # Clean URLs in both dataframes
    crawled_clean = crawled_df.copy()
    source_clean = source_df.copy()

    crawled_clean['_clean_source'] = crawled_clean[crawled_source_col].apply(clean_url)
    crawled_clean['_clean_dest'] = crawled_clean[crawled_dest_col].apply(clean_url)
    source_clean['_clean_source'] = source_clean[source_source_col].apply(clean_url)
    source_clean['_clean_dest'] = source_clean[source_dest_col].apply(clean_url)

    # Create dictionaries for comparison
    crawled_mapping = dict(zip(crawled_clean['_clean_source'], crawled_clean['_clean_dest']))
    source_mapping = dict(zip(source_clean['_clean_source'], source_clean['_clean_dest']))

    # Find matches and mismatches
    matches = []
    mismatches = []
    missing_in_crawled = []
    extra_in_crawled = []

    # Check source mappings against crawled mappings
    for source_url, expected_dest in source_mapping.items():
        if not source_url:  # Skip empty URLs
            continue
        if source_url in crawled_mapping:
            actual_dest = crawled_mapping[source_url]
            if actual_dest == expected_dest:
                matches.append({
                    'source_url': source_url,
                    'expected_destination': expected_dest,
                    'actual_destination': expected_dest,
                    'status': 'MATCH'
                })
            else:
                mismatches.append({
                    'source_url': source_url,
                    'expected_destination': expected_dest,
                    'actual_destination': actual_dest,
                    'status': 'MISMATCH'
                })
        else:
            missing_in_crawled.append({
                'source_url': source_url,
                'expected_destination': expected_dest,
                'actual_destination': 'NOT_FOUND',
                'status': 'MISSING'
            })

    # Check for extra mappings in crawled that aren't in source
    for crawled_url, dest in crawled_mapping.items():
        if not crawled_url:  # Skip empty URLs
            continue
        if crawled_url not in source_mapping:
            extra_in_crawled.append({
                'source_url': crawled_url,
                'expected_destination': 'NOT_IN_SOURCE',
                'actual_destination': dest,
                'status': 'EXTRA'
            })

    return {
        'matches': matches,
        'mismatches': mismatches,
        'missing': missing_in_crawled,
        'extra': extra_in_crawled
    }


# File upload section
st.subheader("1. Upload Files")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Crawled Redirects File**")
    st.markdown("*File containing actual redirects from a crawl (e.g., Screaming Frog)*")
    crawled_file = st.file_uploader(
        "Upload crawled redirects",
        type=['csv', 'xlsx', 'xls'],
        key="crawled"
    )

with col2:
    st.markdown("**Redirect Mapping File**")
    st.markdown("*Your redirect specification/mapping document*")
    source_file = st.file_uploader(
        "Upload redirect mapping",
        type=['csv', 'xlsx', 'xls'],
        key="source"
    )

# Process files if uploaded
if crawled_file and source_file:
    # Load files
    try:
        if crawled_file.name.endswith('.csv'):
            crawled_df = pd.read_csv(crawled_file)
        else:
            crawled_df = pd.read_excel(crawled_file)

        if source_file.name.endswith('.csv'):
            source_df = pd.read_csv(source_file)
        else:
            source_df = pd.read_excel(source_file)

        st.success(f"Loaded {len(crawled_df):,} crawled redirects and {len(source_df):,} mapping entries.")

        # Column mapping section
        st.subheader("2. Map Columns")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Crawled File Columns**")
            crawled_source_col = st.selectbox(
                "Source URL column (Address)",
                options=crawled_df.columns.tolist(),
                index=0 if len(crawled_df.columns) > 0 else None,
                key="crawled_source"
            )
            crawled_dest_col = st.selectbox(
                "Destination URL column (Redirect URL)",
                options=crawled_df.columns.tolist(),
                index=min(1, len(crawled_df.columns) - 1) if len(crawled_df.columns) > 1 else 0,
                key="crawled_dest"
            )

        with col2:
            st.markdown("**Mapping File Columns**")
            source_source_col = st.selectbox(
                "Source URL column",
                options=source_df.columns.tolist(),
                index=0 if len(source_df.columns) > 0 else None,
                key="source_source"
            )
            source_dest_col = st.selectbox(
                "Destination URL column",
                options=source_df.columns.tolist(),
                index=min(1, len(source_df.columns) - 1) if len(source_df.columns) > 1 else 0,
                key="source_dest"
            )

        # Validation button
        if st.button("Validate Redirects", type="primary"):
            with st.spinner("Validating redirects..."):
                results = verify_mappings(
                    crawled_df, source_df,
                    crawled_source_col, crawled_dest_col,
                    source_source_col, source_dest_col
                )

            # Calculate statistics
            total_source = len(results['matches']) + len(results['mismatches']) + len(results['missing'])
            accuracy = (len(results['matches']) / total_source * 100) if total_source > 0 else 0

            # Display metrics
            st.subheader("3. Validation Results")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Matches", f"{len(results['matches']):,}", delta=None)
            with col2:
                st.metric("Mismatches", f"{len(results['mismatches']):,}",
                         delta=f"-{len(results['mismatches'])}" if results['mismatches'] else None,
                         delta_color="inverse")
            with col3:
                st.metric("Missing", f"{len(results['missing']):,}",
                         delta=f"-{len(results['missing'])}" if results['missing'] else None,
                         delta_color="inverse")
            with col4:
                st.metric("Extra", f"{len(results['extra']):,}")
            with col5:
                color = "normal" if accuracy >= 95 else "off" if accuracy >= 80 else "inverse"
                st.metric("Accuracy", f"{accuracy:.1f}%")

            # Status message
            if accuracy == 100 and not results['extra']:
                st.success("Perfect match! All redirects are correctly implemented.")
            elif accuracy >= 95:
                st.info("Very good match. Minor issues to address.")
            elif accuracy >= 80:
                st.warning("Good match with some issues to review.")
            else:
                st.error("Significant discrepancies found. Review needed.")

            # Detailed results tabs
            st.subheader("4. Detailed Results")

            tab1, tab2, tab3, tab4 = st.tabs([
                f"Mismatches ({len(results['mismatches'])})",
                f"Missing ({len(results['missing'])})",
                f"Extra ({len(results['extra'])})",
                f"Matches ({len(results['matches'])})"
            ])

            with tab1:
                if results['mismatches']:
                    st.markdown("**Redirects that go to the wrong destination:**")
                    mismatch_df = pd.DataFrame(results['mismatches'])
                    st.dataframe(mismatch_df, use_container_width=True, height=300)
                else:
                    st.success("No mismatches found!")

            with tab2:
                if results['missing']:
                    st.markdown("**Redirects in your mapping that were not found in the crawl:**")
                    missing_df = pd.DataFrame(results['missing'])
                    st.dataframe(missing_df, use_container_width=True, height=300)
                else:
                    st.success("No missing redirects!")

            with tab3:
                if results['extra']:
                    st.markdown("**Redirects found in crawl that are not in your mapping:**")
                    extra_df = pd.DataFrame(results['extra'])
                    st.dataframe(extra_df, use_container_width=True, height=300)
                else:
                    st.info("No extra redirects found.")

            with tab4:
                if results['matches']:
                    st.markdown("**Correctly implemented redirects:**")
                    matches_df = pd.DataFrame(results['matches'])
                    st.dataframe(matches_df, use_container_width=True, height=300)
                else:
                    st.warning("No matching redirects found.")

            # Download section
            st.subheader("5. Download Reports")

            # Create comprehensive report
            all_results = (
                results['matches'] +
                results['mismatches'] +
                results['missing'] +
                results['extra']
            )
            full_report_df = pd.DataFrame(all_results)

            col1, col2, col3 = st.columns(3)

            with col1:
                csv = full_report_df.to_csv(index=False)
                st.download_button(
                    "Download Full Report (CSV)",
                    csv,
                    file_name=f"redirect_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            with col2:
                if results['mismatches']:
                    mismatch_csv = pd.DataFrame(results['mismatches']).to_csv(index=False)
                    st.download_button(
                        "Download Mismatches (CSV)",
                        mismatch_csv,
                        file_name=f"redirect_mismatches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

            with col3:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    full_report_df.to_excel(writer, index=False, sheet_name='Full Report')
                    if results['mismatches']:
                        pd.DataFrame(results['mismatches']).to_excel(writer, index=False, sheet_name='Mismatches')
                    if results['missing']:
                        pd.DataFrame(results['missing']).to_excel(writer, index=False, sheet_name='Missing')
                    if results['extra']:
                        pd.DataFrame(results['extra']).to_excel(writer, index=False, sheet_name='Extra')
                excel_data = output.getvalue()
                st.download_button(
                    "Download Full Report (Excel)",
                    excel_data,
                    file_name=f"redirect_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"Error loading files: {str(e)}")

else:
    # Show instructions
    st.info("Upload both files to begin validation.")

# Footer
st.markdown("---")
