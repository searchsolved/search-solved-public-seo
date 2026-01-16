import streamlit as st
import pandas as pd
import re
from io import BytesIO

st.set_page_config(page_title="URL Breadcrumb Extractor", page_icon="🔗", layout="wide")

st.title("URL Breadcrumb Extractor")
st.markdown("*Created by [Lee Foot](https://leefoot.com)*")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Extracts URLs from breadcrumb HTML extracted via Screaming Frog
    - Returns the last URL in each breadcrumb chain (typically the parent category)
    - Useful for mapping products to their parent categories

    **Files needed:**
    - Screaming Frog crawl export with a breadcrumb extraction column
    - Set up a custom extraction in Screaming Frog to capture breadcrumb HTML

    **How to set up extraction:**
    1. In Screaming Frog, go to Configuration > Custom > Extraction
    2. Add extraction for your breadcrumb element (e.g., CSS selector: `.breadcrumb`)
    3. Crawl your site
    4. Export Internal > HTML
    """)

# Sidebar settings
st.sidebar.header("Settings")

url_filter = st.sidebar.text_input(
    "URLs to exclude (comma-separated)",
    placeholder="browse, brochure, help",
    help="Exclude URLs containing these patterns"
)

extract_position = st.sidebar.selectbox(
    "Which URL to extract?",
    ["Last URL (parent category)", "First URL", "All URLs"],
    help="Which URL from the breadcrumb to extract"
)

# File upload
st.subheader("Upload Your Data")

crawl_file = st.file_uploader(
    "Screaming Frog Crawl with Breadcrumb Extraction",
    type=['csv'],
    help="Export from Screaming Frog with custom breadcrumb extraction"
)

if crawl_file:
    try:
        # Load crawl data
        try:
            df = pd.read_csv(crawl_file, encoding='utf-8')
        except:
            crawl_file.seek(0)
            df = pd.read_csv(crawl_file, encoding='latin-1')

        st.success(f"Loaded {len(df):,} rows")

        # Let user select columns
        cols = df.columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            url_col = st.selectbox(
                "URL column",
                cols,
                index=cols.index("Address") if "Address" in cols else 0
            )
        with col2:
            # Find likely breadcrumb column
            breadcrumb_cols = [c for c in cols if 'breadcrumb' in c.lower() or 'extraction' in c.lower()]
            default_bc_idx = cols.index(breadcrumb_cols[0]) if breadcrumb_cols else 0
            breadcrumb_col = st.selectbox(
                "Breadcrumb column",
                cols,
                index=default_bc_idx
            )

        if st.button("Extract URLs", type="primary"):
            with st.spinner("Extracting URLs from breadcrumbs..."):
                # Comprehensive URL regex pattern
                url_pattern = r'https?://[^\s<>"\']+|(?:href=["\'])([^"\']+)["\']'

                def extract_urls(text):
                    if pd.isna(text):
                        return []
                    text = str(text)
                    # Find all URLs
                    urls = re.findall(url_pattern, text, re.IGNORECASE)
                    # Flatten and clean
                    clean_urls = []
                    for url in urls:
                        if isinstance(url, tuple):
                            url = [u for u in url if u][0] if any(url) else ''
                        if url and url.startswith(('http', '/')):
                            clean_urls.append(url.strip())
                    return clean_urls

                # Extract URLs
                df['extracted_urls'] = df[breadcrumb_col].apply(extract_urls)

                # Get the desired URL position
                if extract_position == "Last URL (parent category)":
                    df['breadcrumb_url'] = df['extracted_urls'].apply(
                        lambda x: x[-1] if len(x) > 0 else None
                    )
                elif extract_position == "First URL":
                    df['breadcrumb_url'] = df['extracted_urls'].apply(
                        lambda x: x[0] if len(x) > 0 else None
                    )
                else:
                    df['breadcrumb_url'] = df['extracted_urls'].apply(
                        lambda x: ' | '.join(x) if x else None
                    )

                # Apply URL filters
                if url_filter:
                    filters = [f.strip() for f in url_filter.split(',')]
                    for f in filters:
                        if f:
                            df = df[~df[url_col].str.contains(f, na=False, case=False)]

                # Clean up - remove self-references
                df = df[df[url_col] != df['breadcrumb_url']]

                # Create result dataframe
                df_result = df[[url_col, breadcrumb_col, 'breadcrumb_url']].copy()
                df_result = df_result[df_result['breadcrumb_url'].notna()]
                df_result.columns = ['Page URL', 'Breadcrumb HTML', 'Extracted URL']

                # Display summary
                st.subheader("Results Summary")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Pages", f"{len(df):,}")
                with col2:
                    st.metric("URLs Extracted", f"{len(df_result):,}")
                with col3:
                    st.metric("Unique Parent URLs", f"{df_result['Extracted URL'].nunique():,}")

                # Display results
                st.subheader("Extracted URLs")
                st.dataframe(df_result[['Page URL', 'Extracted URL']], use_container_width=True)

                # Top parent categories
                st.subheader("Top Parent Categories")
                parent_counts = df_result['Extracted URL'].value_counts().head(20).reset_index()
                parent_counts.columns = ['Parent URL', 'Child Pages']
                st.dataframe(parent_counts, use_container_width=True)

                # Download
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_result.to_excel(writer, sheet_name='Extracted URLs', index=False)
                    parent_counts.to_excel(writer, sheet_name='Parent Summary', index=False)

                st.download_button(
                    label="Download Excel Report",
                    data=output.getvalue(),
                    file_name="breadcrumb_extraction.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("Upload a Screaming Frog crawl file with breadcrumb extraction to get started")

    st.subheader("Example Output")
    example_data = {
        "Page URL": ["/product/blue-shoes/", "/product/red-dress/", "/product/green-hat/"],
        "Extracted URL": ["/category/shoes/", "/category/dresses/", "/category/hats/"]
    }
    st.dataframe(pd.DataFrame(example_data))
