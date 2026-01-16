####################################################################################
#                                                                                  #
#  Breadcrumb Relevancy Checker                                                    #
#                                                                                  #
#  Check if products are assigned to the most relevant categories using fuzzy      #
#  matching between product titles and breadcrumb paths.                           #
#                                                                                  #
####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Contact  : https://www.leefoot.com/contact                                       #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Twitter  : https://x.com/LeeFootSEO                                              #
####################################################################################

"""
Breadcrumb Relevancy Checker - Streamlit App

Uses PolyFuzz TF-IDF matching to check if products are in the most relevant
categories based on their titles vs breadcrumb paths. Identifies products that
might be miscategorized.

Requirements:
    pip install streamlit pandas polyfuzz
"""

import streamlit as st
import pandas as pd
from polyfuzz import PolyFuzz
from io import BytesIO

# App Configuration
st.set_page_config(
    page_title="Breadcrumb Relevancy Checker",
    page_icon="🥖",
    layout="wide"
)

st.title("🥖 Breadcrumb Relevancy Checker")
st.markdown("""
Check if your products are assigned to the most relevant categories.
Uses TF-IDF fuzzy matching to compare product titles against breadcrumb paths.
""")

# Sidebar configuration
st.sidebar.header("Settings")

product_url_pattern = st.sidebar.text_input(
    "Product URL Pattern",
    value="/product/",
    help="URL pattern to identify product pages (e.g., /product/, /p/, /products/)"
)

category_url_pattern = st.sidebar.text_input(
    "Category URL Pattern",
    value="/category/",
    help="URL pattern to identify category pages (e.g., /category/, /c/, /collections/)"
)

similarity_threshold = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
    help="Minimum similarity difference to flag as potential miscategorization"
)

# File uploader
st.header("Upload Crawl Data")
uploaded_file = st.file_uploader(
    "Upload your Screaming Frog crawl export (CSV)",
    type=["csv"],
    help="Export should contain: Address, H1-1, and Breadcrumb columns"
)

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file, dtype=str)

        st.success(f"Loaded {len(df):,} rows of crawl data")

        # Show column selection
        st.subheader("Map Columns")
        col1, col2, col3 = st.columns(3)

        with col1:
            url_col = st.selectbox(
                "URL/Address Column",
                options=df.columns.tolist(),
                index=df.columns.tolist().index("Address") if "Address" in df.columns else 0
            )

        with col2:
            h1_col = st.selectbox(
                "H1/Title Column",
                options=df.columns.tolist(),
                index=df.columns.tolist().index("H1-1") if "H1-1" in df.columns else 0
            )

        with col3:
            breadcrumb_options = [c for c in df.columns if 'breadcrumb' in c.lower() or 'Breadcrumb' in c]
            default_bread = breadcrumb_options[0] if breadcrumb_options else df.columns.tolist()[0]
            breadcrumb_col = st.selectbox(
                "Breadcrumb Column",
                options=df.columns.tolist(),
                index=df.columns.tolist().index(default_bread) if default_bread in df.columns else 0
            )

        # Show raw data preview
        with st.expander("Preview Raw Data"):
            st.dataframe(df[[url_col, h1_col, breadcrumb_col]].head(20))

        # Process button
        if st.button("Check Breadcrumb Relevancy", type="primary"):
            with st.spinner("Processing... this may take a few minutes for large datasets"):

                # Clean the data
                df_clean = df[[url_col, h1_col, breadcrumb_col]].copy()
                df_clean.columns = ['Address', 'H1-1', 'Breadcrumb']

                # Remove rows with missing values
                df_clean = df_clean[df_clean["H1-1"].notna()]
                df_clean = df_clean[df_clean["Breadcrumb"].notna()]

                if len(df_clean) == 0:
                    st.error("No valid rows found after filtering. Check your column mapping.")
                    st.stop()

                # Create category dataframe for matching
                df_cats = df_clean[df_clean["Address"].str.contains(category_url_pattern, na=False)].copy()

                if len(df_cats) == 0:
                    st.warning(f"No category pages found matching pattern: {category_url_pattern}")
                    st.info("Continuing with all pages for best match comparison...")
                    df_cats = df_clean.copy()

                # Filter to product pages only
                df_products = df_clean[df_clean["Address"].str.contains(product_url_pattern, na=False)].copy()

                if len(df_products) == 0:
                    st.error(f"No product pages found matching pattern: {product_url_pattern}")
                    st.stop()

                st.info(f"Found {len(df_products):,} products and {len(df_cats):,} categories")

                # Get unique H1s and breadcrumbs
                h1_list = df_products["H1-1"].tolist()
                bread_list = df_products["Breadcrumb"].tolist()

                # Calculate similarity between H1 and existing breadcrumb
                progress_bar = st.progress(0)
                status_text = st.empty()

                dfs = []
                total = len(h1_list)

                for idx, (h1, bread) in enumerate(zip(h1_list, bread_list)):
                    try:
                        pf_model = PolyFuzz("TF-IDF").match([h1], [bread])
                        df_fuzzed = pf_model.get_matches()
                        dfs.append(df_fuzzed)
                    except Exception:
                        dfs.append(pd.DataFrame({'From': [h1], 'To': [bread], 'Similarity': [0.0]}))

                    if (idx + 1) % 50 == 0 or idx == total - 1:
                        progress_bar.progress((idx + 1) / total)
                        status_text.text(f"Processing existing breadcrumbs: {idx + 1}/{total}")

                df_concat = pd.concat(dfs)
                df_concat.rename(columns={"From": "H1-1"}, inplace=True)
                df_products = pd.merge(df_products, df_concat[['H1-1', 'Similarity']], on='H1-1', how='left')

                # Find the highest matching category using PolyFuzz
                status_text.text("Finding best category matches...")
                progress_bar.progress(0)

                try:
                    pf_model = PolyFuzz("TF-IDF").match(
                        list(df_products["H1-1"].unique()),
                        list(df_cats['Breadcrumb'].unique())
                    )
                    df_fuzzed = pf_model.get_matches()
                    df_products = pd.merge(df_products, df_fuzzed, left_on="H1-1", right_on="From", how="left")
                except Exception as e:
                    st.error(f"Error during category matching: {str(e)}")
                    st.stop()

                progress_bar.progress(1.0)
                status_text.text("Analysis complete!")

                # Clean up the final dataframe
                df_products = df_products.rename(columns={
                    "Similarity_x": "Similarity (Existing Breadcrumb)",
                    "Similarity_y": "Similarity (Highest Match)",
                    "To": "Breadcrumb (Best Match)",
                    "Breadcrumb": "Breadcrumb (Existing)"
                })

                if 'From' in df_products.columns:
                    del df_products['From']

                # Calculate differences
                df_products["Similarity (Highest Match)"] = df_products["Similarity (Highest Match)"].fillna(0)
                df_products["Similarity (Existing Breadcrumb)"] = df_products["Similarity (Existing Breadcrumb)"].fillna(0)

                # Check for breadcrumbs containing /all pattern
                df_products.loc[df_products["Breadcrumb (Existing)"].str.contains(r"\/all[\s\/]?", na=False, regex=True), "Category Assigned to All"] = True
                df_products["Category Assigned to All"] = df_products["Category Assigned to All"].fillna(False)

                # Calculate breadcrumb depths
                df_products["Breadcrumb Depth (Existing)"] = df_products["Breadcrumb (Existing)"].str.count("/")
                df_products["Breadcrumb Depth (Best Match)"] = df_products["Breadcrumb (Best Match)"].str.count("/")

                # Calculate similarity and depth differences
                df_products['Similarity Diff'] = df_products["Similarity (Highest Match)"] - df_products["Similarity (Existing Breadcrumb)"]
                df_products['Breadcrumb Depth Diff'] = df_products["Breadcrumb Depth (Best Match)"] - df_products["Breadcrumb Depth (Existing)"]

                # Fill NaNs
                df_products["Breadcrumb Depth (Best Match)"] = df_products["Breadcrumb Depth (Best Match)"].fillna(0)
                df_products["Similarity Diff"] = df_products["Similarity Diff"].fillna(0)
                df_products["Breadcrumb Depth Diff"] = df_products["Breadcrumb Depth Diff"].fillna(0)

                # Add previous folder for /all categories
                df_products.loc[df_products["Category Assigned to All"] == True, "Previous Folder"] = \
                    df_products['Breadcrumb (Existing)'].str.rsplit('/', n=1).str.get(0)

                # Round similarity scores
                for col in df_products.columns:
                    if 'Similarity' in col and df_products[col].dtype in ['float64', 'float32']:
                        df_products[col] = df_products[col].round(3)

                # Display results
                st.header("Breadcrumb Relevancy Results")

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Products Analyzed", f"{len(df_products):,}")
                with col2:
                    potentially_wrong = len(df_products[df_products['Similarity Diff'] >= similarity_threshold])
                    st.metric("Potential Miscategorizations", f"{potentially_wrong:,}")
                with col3:
                    assigned_to_all = len(df_products[df_products['Category Assigned to All'] == True])
                    st.metric("Assigned to 'All' Categories", f"{assigned_to_all:,}")
                with col4:
                    avg_similarity = df_products["Similarity (Existing Breadcrumb)"].mean()
                    st.metric("Avg. Breadcrumb Similarity", f"{avg_similarity:.2%}")

                # Filter options
                st.subheader("Filter Results")
                filter_col1, filter_col2 = st.columns(2)

                with filter_col1:
                    show_only_issues = st.checkbox("Show only potential miscategorizations", value=True)

                with filter_col2:
                    show_all_categories = st.checkbox("Show 'All' category assignments", value=False)

                # Apply filters
                df_display = df_products.copy()

                if show_only_issues:
                    df_display = df_display[df_display['Similarity Diff'] >= similarity_threshold]

                if show_all_categories:
                    df_display = df_display[df_display['Category Assigned to All'] == True]

                # Sort by similarity diff (highest potential issues first)
                df_display = df_display.sort_values('Similarity Diff', ascending=False)

                # Display columns
                display_cols = [
                    'Address', 'H1-1', 'Breadcrumb (Existing)', 'Breadcrumb (Best Match)',
                    'Similarity (Existing Breadcrumb)', 'Similarity (Highest Match)', 'Similarity Diff',
                    'Category Assigned to All'
                ]
                display_cols = [c for c in display_cols if c in df_display.columns]

                st.dataframe(
                    df_display[display_cols],
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(f"Showing {len(df_display):,} of {len(df_products):,} products")

                # Download full results
                output = BytesIO()
                df_products.to_csv(output, index=False, encoding='utf-8-sig')
                output.seek(0)

                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=output,
                    file_name="breadcrumb_relevancy_results.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Upload a Screaming Frog crawl export to get started.")

    st.markdown("""
    ### How to export data from Screaming Frog:

    1. Crawl your website with Screaming Frog Spider
    2. Configure custom extraction for breadcrumbs (if not already extracted)
    3. Go to **Bulk Export > All > Internal HTML**
    4. Save as CSV and upload here

    ### Required columns:

    - **Address/URL**: The page URL
    - **H1-1**: The page's H1 heading (product title)
    - **Breadcrumb**: The breadcrumb text or path

    ### What this tool does:

    - Compares product H1s to their current breadcrumb paths
    - Finds the best matching category for each product
    - Identifies products that might be in the wrong category
    - Flags products assigned to generic "/all" categories
    """)
