import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="E-commerce Page Title Optimizer", page_icon="🏷️", layout="wide")

st.title("E-commerce Page Title Optimizer")
st.markdown("*Created by [Lee Foot](https://leefoot.com)*")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Analyzes your page titles against GSC keyword data
    - Identifies keywords you rank for but aren't in your title
    - Suggests keywords to add based on click/impression data
    - Helps optimize category and product page titles

    **Files needed:**
    1. **Screaming Frog crawl** (internal_html.csv) with Address and Title columns
    2. **GSC keyword export** with Query, Page URL, Clicks, Impressions columns

    **How to get GSC data:**
    - Use the GSC Data Exporter tool, or
    - Export from GSC > Performance > Pages > Export
    """)

# Sidebar settings
st.sidebar.header("Settings")

delimiter = st.sidebar.selectbox(
    "Title delimiter",
    ["|", "-", ":", "—"],
    help="Character used to separate title parts (e.g., 'Product Name | Brand')"
)

branding = st.sidebar.text_input(
    "Brand name to exclude",
    placeholder="e.g., My Store",
    help="Brand text to remove from title analysis"
)

url_filter = st.sidebar.text_input(
    "URL path filter (optional)",
    placeholder="e.g., /category/",
    help="Only analyze URLs containing this path"
)

max_suggestions = st.sidebar.slider(
    "Max keyword suggestions per page",
    min_value=3,
    max_value=20,
    value=10,
    help="Maximum number of keyword suggestions to show per page"
)

# File uploads
st.subheader("Upload Your Data")

col1, col2 = st.columns(2)

with col1:
    crawl_file = st.file_uploader(
        "Screaming Frog Crawl (internal_html.csv)",
        type=['csv'],
        help="Export from Screaming Frog: Bulk Export > All > Internal HTML"
    )

with col2:
    gsc_file = st.file_uploader(
        "GSC Keyword Data",
        type=['csv'],
        help="GSC export with Query, Page, Clicks, Impressions columns"
    )

if crawl_file and gsc_file:
    try:
        # Load crawl data
        try:
            df_crawl = pd.read_csv(crawl_file, encoding='utf-8')
        except:
            crawl_file.seek(0)
            df_crawl = pd.read_csv(crawl_file, encoding='latin-1')

        # Load GSC data
        try:
            df_gsc = pd.read_csv(gsc_file, encoding='utf-8')
        except:
            gsc_file.seek(0)
            df_gsc = pd.read_csv(gsc_file, encoding='latin-1')

        # Identify columns
        crawl_cols = df_crawl.columns.tolist()
        gsc_cols = df_gsc.columns.tolist()

        st.success(f"Loaded {len(df_crawl):,} pages and {len(df_gsc):,} keyword rows")

        # Column mapping
        with st.expander("Map columns (if needed)"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Crawl file columns:**")
                url_col = st.selectbox(
                    "URL column",
                    crawl_cols,
                    index=crawl_cols.index("Address") if "Address" in crawl_cols else 0
                )
                title_col = st.selectbox(
                    "Title column",
                    crawl_cols,
                    index=crawl_cols.index("Title 1") if "Title 1" in crawl_cols else 0
                )

            with col2:
                st.markdown("**GSC file columns:**")
                query_col = st.selectbox(
                    "Query column",
                    gsc_cols,
                    index=gsc_cols.index("Top queries") if "Top queries" in gsc_cols else (
                        gsc_cols.index("Query") if "Query" in gsc_cols else (
                            gsc_cols.index("query") if "query" in gsc_cols else 0
                        )
                    )
                )
                page_col = st.selectbox(
                    "Page URL column",
                    gsc_cols,
                    index=gsc_cols.index("Top pages") if "Top pages" in gsc_cols else (
                        gsc_cols.index("Page") if "Page" in gsc_cols else (
                            gsc_cols.index("page") if "page" in gsc_cols else 0
                        )
                    )
                )
                clicks_col = st.selectbox(
                    "Clicks column",
                    gsc_cols,
                    index=gsc_cols.index("Clicks") if "Clicks" in gsc_cols else (
                        gsc_cols.index("clicks") if "clicks" in gsc_cols else 0
                    )
                )
                impressions_col = st.selectbox(
                    "Impressions column",
                    gsc_cols,
                    index=gsc_cols.index("Impressions") if "Impressions" in gsc_cols else (
                        gsc_cols.index("impressions") if "impressions" in gsc_cols else 0
                    )
                )

        if st.button("Analyze Page Titles", type="primary"):
            with st.spinner("Analyzing page titles..."):
                # Normalize column names
                df_crawl = df_crawl.rename(columns={url_col: "page", title_col: "title"})
                df_gsc = df_gsc.rename(columns={
                    query_col: "query",
                    page_col: "page",
                    clicks_col: "clicks",
                    impressions_col: "impressions"
                })

                # Filter to required columns
                df_crawl = df_crawl[["page", "title"]].copy()
                df_gsc = df_gsc[["query", "page", "clicks", "impressions"]].copy()

                # Clean data
                df_crawl = df_crawl[df_crawl["title"].notna()]
                df_crawl = df_crawl[df_crawl["page"].notna()]
                df_gsc = df_gsc[df_gsc["query"].notna()]
                df_gsc = df_gsc[df_gsc["page"].notna()]

                # Ensure numeric columns
                df_gsc["clicks"] = pd.to_numeric(df_gsc["clicks"], errors='coerce').fillna(0)
                df_gsc["impressions"] = pd.to_numeric(df_gsc["impressions"], errors='coerce').fillna(0)

                # Apply URL filter
                if url_filter:
                    df_crawl = df_crawl[df_crawl["page"].str.contains(url_filter, na=False)]
                    df_gsc = df_gsc[df_gsc["page"].str.contains(url_filter, na=False)]

                # Remove branding from queries
                if branding:
                    df_gsc = df_gsc[~df_gsc["query"].str.lower().str.contains(branding.lower(), na=False)]

                # Extract keywords from page titles
                df_titles = df_crawl.copy()
                df_titles["title_keywords"] = df_titles["title"].str.split(delimiter)
                df_titles = df_titles.explode("title_keywords")
                df_titles["title_keywords"] = df_titles["title_keywords"].str.strip().str.lower()
                df_titles = df_titles[df_titles["title_keywords"].notna()]
                df_titles = df_titles[df_titles["title_keywords"] != ""]

                # Remove branding from title keywords
                if branding:
                    df_titles = df_titles[~df_titles["title_keywords"].str.contains(branding.lower(), na=False)]

                # Create set of title keywords per page
                title_kw_sets = df_titles.groupby("page")["title_keywords"].apply(set).to_dict()

                # Normalize GSC queries
                df_gsc["query_lower"] = df_gsc["query"].str.lower()

                # Find keywords in GSC that are NOT in page title
                def check_in_title(row):
                    page = row["page"]
                    query = row["query_lower"]
                    if page in title_kw_sets:
                        # Check if any word in query appears in title keywords
                        query_words = set(query.split())
                        title_words = title_kw_sets[page]
                        return not bool(query_words & title_words)
                    return True

                df_gsc["missing_from_title"] = df_gsc.apply(check_in_title, axis=1)

                # Filter to missing keywords with clicks
                df_missing = df_gsc[df_gsc["missing_from_title"] & (df_gsc["clicks"] > 0)].copy()

                # Sort by clicks and take top N per page
                df_missing = df_missing.sort_values("clicks", ascending=False)
                df_suggestions = df_missing.groupby("page").head(max_suggestions)

                # Merge with crawl data to get titles
                df_result = df_suggestions.merge(
                    df_crawl[["page", "title"]],
                    on="page",
                    how="left"
                )

                # Add summary stats
                df_result["total_page_clicks"] = df_result.groupby("page")["clicks"].transform("sum")
                df_result["suggestion_count"] = df_result.groupby("page")["query"].transform("count")

                # Clean up columns
                df_result = df_result[[
                    "page", "title", "query", "clicks", "impressions",
                    "total_page_clicks", "suggestion_count"
                ]].sort_values(["total_page_clicks", "clicks"], ascending=[False, False])

                # Display summary
                st.subheader("Results Summary")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Pages Analyzed", f"{df_crawl['page'].nunique():,}")
                with col2:
                    st.metric("Pages with Suggestions", f"{df_result['page'].nunique():,}")
                with col3:
                    st.metric("Total Suggestions", f"{len(df_result):,}")
                with col4:
                    st.metric("Potential Clicks", f"{int(df_result['clicks'].sum()):,}")

                # Display results
                st.subheader("Keyword Suggestions by Page")
                st.dataframe(df_result, use_container_width=True)

                # Top opportunities
                st.subheader("Top 20 Keyword Opportunities")
                top_opps = df_result.nlargest(20, "clicks")[["page", "title", "query", "clicks", "impressions"]]
                st.dataframe(top_opps, use_container_width=True)

                # Download
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_result.to_excel(writer, sheet_name='All Suggestions', index=False)
                    top_opps.to_excel(writer, sheet_name='Top Opportunities', index=False)

                    # Summary by page
                    df_summary = df_result.groupby(["page", "title"]).agg({
                        "query": lambda x: " | ".join(x.head(5)),
                        "clicks": "sum",
                        "impressions": "sum",
                        "suggestion_count": "first"
                    }).reset_index()
                    df_summary.columns = ["Page", "Current Title", "Top Missing Keywords", "Total Clicks", "Total Impressions", "Suggestion Count"]
                    df_summary = df_summary.sort_values("Total Clicks", ascending=False)
                    df_summary.to_excel(writer, sheet_name='Summary by Page', index=False)

                st.download_button(
                    label="Download Excel Report",
                    data=output.getvalue(),
                    file_name="page_title_optimization.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("Upload both files to get started")

    st.subheader("Example Output")
    example_data = {
        "Page": ["/category/shoes/", "/category/shoes/", "/product/nike-air-max/"],
        "Current Title": ["Shoes | My Store", "Shoes | My Store", "Nike Air Max | My Store"],
        "Missing Keyword": ["running shoes", "trainers", "air max 90"],
        "Clicks": [150, 85, 200],
        "Impressions": [3500, 2000, 5000]
    }
    st.dataframe(pd.DataFrame(example_data))
