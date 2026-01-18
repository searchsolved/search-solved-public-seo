####################################################################################
#                                                                                  #
#  Hreflang Generator                                                              #
#                                                                                  #
#  Generate hreflang XML tags from Screaming Frog crawl data.                      #
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
Hreflang Generator

Generates hreflang XML sitemap tags from Screaming Frog crawl data. Perfect for
multi-language sites that need to generate or audit hreflang implementation.

Features:
- Upload Screaming Frog crawl export (internal_html.csv)
- Configurable language code detection
- Filter non-indexable URLs
- Generate valid hreflang XML tags
- Download as CSV or ready-to-use XML
"""

import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Hreflang Generator", page_icon="🌍", layout="wide")

st.title("Hreflang Generator")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Generates hreflang XML tags from your Screaming Frog crawl data
    - Automatically detects language codes from URL structure
    - Filters out non-indexable URLs
    - Outputs ready-to-use hreflang XML tags

    **How to get the data:**
    1. Crawl your multi-language site with Screaming Frog
    2. Export `internal_html.csv` from the Internal tab
    3. Upload the file here

    **URL Structure Requirements:**
    - URLs should contain language codes (e.g., `/en/`, `/de/`, `/fr/`)
    - Configure which position in the URL path contains the language code

    **Example URL Structures:**
    - `https://example.com/en/page` (position 1)
    - `https://example.com/uk/en/page` (position 2)
    """)

# Sidebar settings
st.sidebar.header("Settings")

lang_position = st.sidebar.number_input(
    "Language code position in URL path",
    min_value=1,
    max_value=5,
    value=1,
    help="Which folder in the URL contains the language code? (1 = first folder after domain)"
)

default_lang = st.sidebar.text_input(
    "Default language (when no code found)",
    value="en",
    help="Language code to use when URL doesn't contain a language folder"
)

max_folder_length = st.sidebar.number_input(
    "Max language code length",
    min_value=2,
    max_value=10,
    value=5,
    help="Folders longer than this will be treated as content, not language codes"
)

filter_non_indexable = st.sidebar.checkbox(
    "Filter non-indexable URLs",
    value=True,
    help="Exclude URLs marked as non-indexable in the crawl"
)

exclude_patterns = st.sidebar.text_input(
    "URL patterns to exclude (comma-separated)",
    value="/page",
    help="Exclude URLs containing these patterns (e.g., pagination)"
)

# File upload
st.subheader("Upload Screaming Frog Crawl")
uploaded_file = st.file_uploader(
    "Upload internal_html.csv from Screaming Frog",
    type=['csv'],
    help="Export from Screaming Frog: Internal > HTML > Export"
)

if uploaded_file is not None:
    try:
        # Load the crawl data
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin-1')

        st.success(f"Loaded {len(df):,} URLs")

        # Find URL column
        url_col = None
        for col in df.columns:
            col_lower = col.lower()
            if 'address' in col_lower or 'url' in col_lower:
                url_col = col
                break

        if url_col is None:
            url_col = st.selectbox("Select URL column", df.columns.tolist())

        # Find Indexability column
        index_col = None
        for col in df.columns:
            if 'indexability' in col.lower():
                index_col = col
                break

        with st.expander("Preview uploaded data"):
            st.dataframe(df.head(20))

        if st.button("Generate Hreflang Tags", type="primary"):
            with st.spinner("Generating hreflang tags..."):
                df_work = df.copy()

                # Filter non-indexable URLs
                if filter_non_indexable and index_col:
                    original_count = len(df_work)
                    df_work = df_work[~df_work[index_col].str.contains("Non-Indexable", na=False, case=False)]
                    st.info(f"Filtered {original_count - len(df_work)} non-indexable URLs")

                # Exclude patterns
                if exclude_patterns.strip():
                    patterns = [p.strip() for p in exclude_patterns.split(',') if p.strip()]
                    original_count = len(df_work)
                    for pattern in patterns:
                        df_work = df_work[~df_work[url_col].str.contains(pattern, na=False, case=False)]
                    st.info(f"Filtered {original_count - len(df_work)} URLs matching exclusion patterns")

                # Extract language code from URL
                def extract_language_code(url):
                    try:
                        parts = str(url).split('/')
                        if len(parts) > 2 + lang_position:
                            folder = parts[2 + lang_position]  # Skip protocol and domain
                            if len(folder) <= max_folder_length and folder:
                                return folder.lower()
                        return default_lang
                    except:
                        return default_lang

                df_work['Language'] = df_work[url_col].apply(extract_language_code)

                # Generate hreflang XML tag
                df_work['Hreflang XML'] = (
                    '<xhtml:link rel="alternate" hreflang="' +
                    df_work['Language'] +
                    '" href="' +
                    df_work[url_col].astype(str) +
                    '"/>'
                )

                # Keep relevant columns
                output_cols = [url_col, 'Language', 'Hreflang XML']
                df_output = df_work[output_cols].copy()

                # Display summary
                st.subheader("Results")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total URLs", f"{len(df_output):,}")
                with col2:
                    unique_langs = df_output['Language'].nunique()
                    st.metric("Unique Languages", unique_langs)
                with col3:
                    top_lang = df_output['Language'].mode().iloc[0] if len(df_output) > 0 else "N/A"
                    st.metric("Most Common", top_lang)

                # Show language distribution
                st.subheader("Language Distribution")
                lang_counts = df_output['Language'].value_counts()
                st.bar_chart(lang_counts)

                # Show sample output
                st.subheader("Sample Output")
                st.dataframe(df_output.head(50), use_container_width=True)

                # Download options
                st.subheader("Download")
                col1, col2 = st.columns(2)

                with col1:
                    # CSV download
                    csv_output = df_output.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv_output,
                        file_name="hreflang_tags.csv",
                        mime="text/csv"
                    )

                with col2:
                    # XML sitemap format download
                    xml_header = '''<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:xhtml="http://www.w3.org/1999/xhtml">
'''
                    xml_footer = '</urlset>'

                    xml_urls = []
                    for url in df_output[url_col].unique():
                        url_tags = df_output[df_output[url_col] == url]['Hreflang XML'].tolist()
                        xml_urls.append(f'''  <url>
    <loc>{url}</loc>
    {chr(10).join('    ' + tag for tag in url_tags)}
  </url>''')

                    xml_content = xml_header + '\n'.join(xml_urls) + '\n' + xml_footer

                    st.download_button(
                        label="Download XML Sitemap",
                        data=xml_content.encode('utf-8'),
                        file_name="hreflang_sitemap.xml",
                        mime="application/xml"
                    )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("Upload a Screaming Frog crawl export to get started")

    st.subheader("Example Output")
    example_data = {
        "URL": [
            "https://example.com/en/products",
            "https://example.com/de/products",
            "https://example.com/fr/products"
        ],
        "Language": ["en", "de", "fr"],
        "Hreflang XML": [
            '<xhtml:link rel="alternate" hreflang="en" href="https://example.com/en/products"/>',
            '<xhtml:link rel="alternate" hreflang="de" href="https://example.com/de/products"/>',
            '<xhtml:link rel="alternate" hreflang="fr" href="https://example.com/fr/products"/>'
        ]
    }
    st.dataframe(pd.DataFrame(example_data))
