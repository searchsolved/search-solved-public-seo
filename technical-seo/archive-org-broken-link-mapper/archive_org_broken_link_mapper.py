import streamlit as st
import pandas as pd
import requests
from io import BytesIO, StringIO
from urllib.parse import urlparse
from polyfuzz import PolyFuzz

st.set_page_config(page_title="Archive.org Broken Link Mapper", page_icon="🗄️", layout="wide")

st.title("Archive.org Broken Link Mapper")
st.markdown("*Created by [Lee Foot](https://leefoot.co.uk)*")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Downloads historical URLs from Archive.org (Wayback Machine)
    - Compares with your current crawl to find URLs that no longer exist
    - Automatically maps old URLs to new URLs using fuzzy matching
    - Helps create redirect maps for broken backlinks

    **Files needed:**
    - Screaming Frog crawl export (internal_html.csv) with Address, Status Code columns

    **Process:**
    1. Upload your crawl file
    2. Tool fetches all historical URLs from Archive.org
    3. Identifies URLs not in your crawl (potential broken links)
    4. Fuzzy matches old URLs to current URLs for redirect suggestions
    """)

# Sidebar settings
st.sidebar.header("Settings")

filter_patterns = st.sidebar.text_area(
    "URL patterns to exclude (one per line)",
    value="utm\ngclid\nbasket\ncheckout\naccount\n/page",
    help="Exclude URLs containing these patterns"
)

content_types = st.sidebar.multiselect(
    "Content types to include",
    ["text/html", "text/plain"],
    default=["text/html"],
    help="Only include URLs with these content types"
)

similarity_threshold = st.sidebar.slider(
    "Minimum match similarity",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    help="Minimum similarity score for redirect suggestions"
)

# File upload
st.subheader("Upload Your Data")

crawl_file = st.file_uploader(
    "Screaming Frog Crawl (internal_html.csv)",
    type=['csv'],
    help="Export from Screaming Frog: Internal > HTML"
)

# Option to manually enter domain
domain_input = st.text_input(
    "Or enter domain manually",
    placeholder="example.com",
    help="If not uploading a crawl, enter your domain to fetch Archive.org data"
)

if crawl_file or domain_input:
    try:
        if crawl_file:
            # Load crawl data
            try:
                df_crawl = pd.read_csv(crawl_file, encoding='utf-8')
            except:
                crawl_file.seek(0)
                df_crawl = pd.read_csv(crawl_file, encoding='latin-1')

            # Auto-detect URL column
            cols = df_crawl.columns.tolist()
            url_col = "Address" if "Address" in cols else cols[0]

            # Extract domain from crawl
            first_url = df_crawl[url_col].iloc[0]
            parsed = urlparse(first_url)
            domain = parsed.netloc

            st.success(f"Loaded {len(df_crawl):,} URLs from crawl. Detected domain: {domain}")
        else:
            domain = domain_input.replace("https://", "").replace("http://", "").replace("www.", "").strip("/")
            df_crawl = None
            st.info(f"Using domain: {domain}")

        if st.button("Fetch Archive.org Data", type="primary"):
            with st.spinner(f"Downloading URLs from Archive.org for {domain}..."):
                # Build Archive.org CDX API URL
                archive_url = f"http://web.archive.org/cdx/search/cdx?url={domain}/*&output=txt&fl=original,mimetype,statuscode&collapse=urlkey"

                try:
                    resp = requests.get(archive_url, timeout=120)
                    resp.raise_for_status()

                    # Parse response
                    df_archive = pd.read_csv(
                        StringIO(resp.text),
                        sep=' ',
                        names=["Address", "Content Type", "Status Code"],
                        on_bad_lines='skip'
                    )

                    st.success(f"Downloaded {len(df_archive):,} URLs from Archive.org")

                    # Clean up Archive.org data
                    # Remove port 80
                    df_archive["Address"] = df_archive["Address"].str.replace(r':80(?=/|$)', '', regex=True)

                    # Remove query parameters (optional)
                    df_archive["Address_clean"] = df_archive["Address"].str.split('?').str[0]

                    # Drop duplicates
                    df_archive = df_archive.drop_duplicates(subset="Address_clean")

                    # Filter by content type
                    if content_types:
                        df_archive = df_archive[df_archive["Content Type"].isin(content_types)]

                    # Apply exclusion filters
                    if filter_patterns:
                        patterns = [p.strip() for p in filter_patterns.split('\n') if p.strip()]
                        for pattern in patterns:
                            df_archive = df_archive[~df_archive["Address"].str.contains(pattern, na=False, case=False)]

                    # Filter out common non-page extensions
                    exclude_ext = r'\.(css|js|jpg|jpeg|png|gif|pdf|ico|svg|woff|woff2|ttf|eot|mp4|mp3|zip|xml)(\?|$)'
                    df_archive = df_archive[~df_archive["Address"].str.contains(exclude_ext, na=False, case=False)]

                    st.info(f"After filtering: {len(df_archive):,} URLs")

                    if df_crawl is not None:
                        # Find URLs not in current crawl
                        df_crawl["Address_lower"] = df_crawl[url_col].str.lower()
                        df_archive["Address_lower"] = df_archive["Address"].str.lower()

                        current_urls = set(df_crawl["Address_lower"].unique())
                        df_archive["in_crawl"] = df_archive["Address_lower"].isin(current_urls)

                        df_missing = df_archive[~df_archive["in_crawl"]].copy()
                        st.warning(f"Found {len(df_missing):,} URLs not in current crawl")

                        if len(df_missing) > 0 and len(df_crawl) > 0:
                            with st.spinner("Fuzzy matching old URLs to current URLs..."):
                                # Prepare lists for PolyFuzz
                                old_urls = df_missing["Address"].tolist()[:1000]  # Limit for performance
                                current_urls_list = df_crawl[url_col].tolist()[:5000]

                                # Run fuzzy matching
                                model = PolyFuzz("TF-IDF").match(old_urls, current_urls_list)
                                df_matches = model.get_matches()

                                # Filter by similarity threshold
                                df_matches = df_matches[df_matches["Similarity"] >= similarity_threshold]

                                st.success(f"Generated {len(df_matches):,} redirect suggestions")

                                # Display results
                                st.subheader("Results Summary")

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Archive.org URLs", f"{len(df_archive):,}")
                                with col2:
                                    st.metric("Missing from Crawl", f"{len(df_missing):,}")
                                with col3:
                                    st.metric("Redirect Suggestions", f"{len(df_matches):,}")

                                st.subheader("Redirect Suggestions")
                                df_matches = df_matches.rename(columns={
                                    "From": "Old URL",
                                    "To": "Suggested Redirect",
                                    "Similarity": "Match Score"
                                })
                                df_matches = df_matches.sort_values("Match Score", ascending=False)
                                st.dataframe(df_matches, use_container_width=True)

                                # Download
                                output = BytesIO()
                                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                    df_matches.to_excel(writer, sheet_name='Redirect Suggestions', index=False)
                                    df_missing[["Address"]].to_excel(writer, sheet_name='All Missing URLs', index=False)
                                    df_archive[["Address", "Content Type"]].to_excel(writer, sheet_name='All Archive URLs', index=False)

                                st.download_button(
                                    label="Download Excel Report",
                                    data=output.getvalue(),
                                    file_name="archive_org_redirect_map.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        else:
                            st.info("No missing URLs found or crawl is empty")
                    else:
                        # Just show Archive.org data without crawl comparison
                        st.subheader("Archive.org URLs")
                        st.dataframe(df_archive[["Address", "Content Type"]], use_container_width=True)

                        output = BytesIO()
                        df_archive[["Address", "Content Type"]].to_excel(output, index=False)

                        st.download_button(
                            label="Download Archive URLs",
                            data=output.getvalue(),
                            file_name="archive_org_urls.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                except requests.exceptions.Timeout:
                    st.error("Request timed out. The domain may have too many URLs. Try a smaller site.")
                except Exception as e:
                    st.error(f"Error fetching Archive.org data: {str(e)}")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("Upload a Screaming Frog crawl file or enter a domain to get started")

    st.subheader("Example Output")
    example_data = {
        "Old URL": ["example.com/old-product", "example.com/discontinued-category"],
        "Suggested Redirect": ["example.com/new-product", "example.com/category"],
        "Match Score": [0.85, 0.72]
    }
    st.dataframe(pd.DataFrame(example_data))
