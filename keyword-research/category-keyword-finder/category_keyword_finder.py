import streamlit as st
import pandas as pd
import re
import string
import collections
from nltk.util import ngrams
from io import BytesIO

st.set_page_config(page_title="Category Keyword Finder", page_icon="🏷️", layout="wide")

st.title("Category Keyword Finder")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-Follow-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Extracts n-gram keyword opportunities from product titles grouped by category
    - Identifies common phrases across products that could become new category pages
    - Helps discover long-tail category opportunities

    **Files needed:**
    - Screaming Frog crawl with product URLs and H1s/Titles
    - Must be able to identify products vs categories by URL pattern

    **How it works:**
    1. Groups products by parent category (using URL patterns)
    2. Extracts common n-gram phrases from product H1s/Titles
    3. Shows phrases that appear across multiple products
    4. Suggests potential new category keywords

    **Optional:**
    - Upload keyword data with search volumes to filter by actual search demand
    """)

# Sidebar settings
st.sidebar.header("Settings")

product_pattern = st.sidebar.text_input(
    "Product URL pattern",
    value="/product/",
    help="URL pattern to identify product pages (e.g., /product/, /p/, /pdp/)"
)

category_pattern = st.sidebar.text_input(
    "Category URL pattern",
    value="/category/",
    help="URL pattern to identify category pages"
)

min_ngram = st.sidebar.slider(
    "Minimum n-gram length",
    min_value=1,
    max_value=5,
    value=2,
    help="Minimum number of words in phrases"
)

max_ngram = st.sidebar.slider(
    "Maximum n-gram length",
    min_value=2,
    max_value=8,
    value=5,
    help="Maximum number of words in phrases"
)

min_products = st.sidebar.slider(
    "Minimum matching products",
    min_value=2,
    max_value=20,
    value=3,
    help="Phrase must appear in at least this many product titles"
)

# File uploads
st.subheader("Upload Your Data")

col1, col2 = st.columns(2)

with col1:
    crawl_file = st.file_uploader(
        "Screaming Frog Crawl",
        type=['csv'],
        help="Export with Address and H1-1 or Title columns"
    )

with col2:
    keyword_file = st.file_uploader(
        "Keyword Data with Search Volumes (optional)",
        type=['csv'],
        help="Optional: keyword data to match volumes"
    )

if crawl_file:
    try:
        # Load crawl data
        try:
            df_crawl = pd.read_csv(crawl_file, encoding='utf-8')
        except:
            crawl_file.seek(0)
            df_crawl = pd.read_csv(crawl_file, encoding='latin-1')

        st.success(f"Loaded {len(df_crawl):,} URLs")

        # Column mapping
        cols = df_crawl.columns.tolist()

        with st.expander("Map columns (if needed)"):
            url_col = st.selectbox(
                "URL column",
                cols,
                index=cols.index("Address") if "Address" in cols else 0
            )
            h1_col = st.selectbox(
                "H1/Title column",
                cols,
                index=cols.index("H1-1") if "H1-1" in cols else (
                    cols.index("Title 1") if "Title 1" in cols else 0
                )
            )

        # Load optional keyword data
        df_keywords = None
        if keyword_file:
            try:
                df_keywords = pd.read_csv(keyword_file, encoding='utf-8')
            except:
                keyword_file.seek(0)
                df_keywords = pd.read_csv(keyword_file, encoding='latin-1')

            kw_cols = df_keywords.columns.tolist()
            with st.expander("Map keyword columns"):
                kw_col = st.selectbox(
                    "Keyword column",
                    kw_cols,
                    index=kw_cols.index("Keyword") if "Keyword" in kw_cols else 0
                )
                vol_col = st.selectbox(
                    "Volume column",
                    kw_cols,
                    index=kw_cols.index("Volume") if "Volume" in kw_cols else (
                        kw_cols.index("Search Volume") if "Search Volume" in kw_cols else 0
                    )
                )

        if st.button("Find Category Keywords", type="primary"):
            with st.spinner("Extracting n-grams from product titles..."):
                # Filter to products
                df_products = df_crawl[df_crawl[url_col].str.contains(product_pattern, na=False)].copy()

                if len(df_products) == 0:
                    st.error(f"No products found matching pattern '{product_pattern}'. Check your URL pattern.")
                else:
                    st.info(f"Found {len(df_products):,} product pages")

                    # Extract parent category from URL
                    def get_parent_category(url):
                        # Remove product pattern and get parent path
                        parts = url.split('/')
                        # Find the part before 'product' pattern
                        for i, part in enumerate(parts):
                            if product_pattern.strip('/') in part.lower():
                                # Return the path before this
                                return '/'.join(parts[:i]) + '/'
                        return '/'.join(parts[:-1]) + '/'

                    df_products["parent_category"] = df_products[url_col].apply(get_parent_category)
                    df_products["h1_clean"] = df_products[h1_col].fillna("").str.lower()

                    # Get unique parent categories
                    categories = df_products["parent_category"].unique()
                    st.info(f"Found {len(categories):,} parent categories")

                    # Process each category
                    all_ngrams = []

                    for category in categories:
                        df_cat = df_products[df_products["parent_category"] == category]

                        if len(df_cat) < min_products:
                            continue

                        # Combine all H1s into text corpus
                        text = " ".join(df_cat["h1_clean"].tolist())

                        # Clean text
                        text = re.sub(r'\d+', '', text)  # Remove numbers
                        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)  # Remove punctuation
                        text = re.sub(r'\s+', ' ', text).strip()  # Clean whitespace

                        # Tokenize
                        tokens = text.split()

                        # Generate n-grams
                        for n in range(min_ngram, max_ngram + 1):
                            if len(tokens) >= n:
                                n_grams = list(ngrams(tokens, n))
                                counts = collections.Counter(n_grams)

                                for gram, count in counts.most_common(50):
                                    if count >= min_products:
                                        phrase = ' '.join(gram)
                                        all_ngrams.append({
                                            "parent_category": category,
                                            "keyword": phrase,
                                            "frequency": count,
                                            "n_gram_length": n
                                        })

                    if all_ngrams:
                        df_ngrams = pd.DataFrame(all_ngrams)

                        # Count how many products actually contain each phrase
                        def count_product_matches(row):
                            kw = row["keyword"]
                            cat = row["parent_category"]
                            cat_products = df_products[df_products["parent_category"] == cat]
                            matches = cat_products["h1_clean"].str.contains(kw, regex=False, na=False).sum()
                            return matches

                        df_ngrams["matching_products"] = df_ngrams.apply(count_product_matches, axis=1)
                        df_ngrams = df_ngrams[df_ngrams["matching_products"] >= min_products]

                        # Remove duplicates - keep longest n-gram
                        df_ngrams = df_ngrams.sort_values("n_gram_length", ascending=False)
                        df_ngrams = df_ngrams.drop_duplicates(subset=["parent_category", "keyword"], keep="first")

                        # Match with keyword data if provided
                        if df_keywords is not None:
                            df_keywords["kw_lower"] = df_keywords[kw_col].str.lower().str.strip()
                            df_ngrams = df_ngrams.merge(
                                df_keywords[["kw_lower", vol_col]].rename(columns={"kw_lower": "keyword", vol_col: "search_volume"}),
                                on="keyword",
                                how="left"
                            )
                            df_ngrams["search_volume"] = df_ngrams["search_volume"].fillna(0)
                            df_ngrams = df_ngrams.sort_values("search_volume", ascending=False)
                        else:
                            df_ngrams["search_volume"] = "N/A"
                            df_ngrams = df_ngrams.sort_values("matching_products", ascending=False)

                        # Check if keyword matches existing category
                        existing_cats = df_crawl[df_crawl[url_col].str.contains(category_pattern, na=False)]
                        if len(existing_cats) > 0:
                            existing_h1s = existing_cats[h1_col].str.lower().unique()
                            df_ngrams["exists_as_category"] = df_ngrams["keyword"].isin(existing_h1s)
                        else:
                            df_ngrams["exists_as_category"] = False

                        # Display results
                        st.subheader("Results Summary")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Keywords Found", f"{len(df_ngrams):,}")
                        with col2:
                            st.metric("New Opportunities", f"{(~df_ngrams['exists_as_category']).sum():,}")
                        with col3:
                            st.metric("Categories Analyzed", f"{df_ngrams['parent_category'].nunique():,}")
                        with col4:
                            if df_keywords is not None:
                                st.metric("With Search Volume", f"{(df_ngrams['search_volume'] > 0).sum():,}")

                        # Show new opportunities
                        st.subheader("New Category Opportunities")
                        df_new = df_ngrams[~df_ngrams["exists_as_category"]]
                        display_cols = ["parent_category", "keyword", "matching_products", "search_volume"]
                        st.dataframe(df_new[display_cols].head(100), use_container_width=True)

                        # Download
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df_ngrams.to_excel(writer, sheet_name='All Keywords', index=False)
                            df_new.to_excel(writer, sheet_name='New Opportunities', index=False)

                            # Summary by category
                            df_summary = df_ngrams.groupby("parent_category").agg({
                                "keyword": "count",
                                "matching_products": "sum"
                            }).reset_index()
                            df_summary.columns = ["Parent Category", "Keywords Found", "Total Product Matches"]
                            df_summary = df_summary.sort_values("Keywords Found", ascending=False)
                            df_summary.to_excel(writer, sheet_name='By Category', index=False)

                        st.download_button(
                            label="Download Excel Report",
                            data=output.getvalue(),
                            file_name="category_keyword_opportunities.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.warning("No n-grams found meeting the minimum product threshold. Try lowering the threshold.")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("Upload a Screaming Frog crawl file to get started")

    st.subheader("Example Output")
    example_data = {
        "Parent Category": ["/category/sofas/", "/category/sofas/", "/category/beds/"],
        "Keyword": ["velvet sofa", "corner sofa", "storage bed"],
        "Matching Products": [12, 8, 15],
        "Search Volume": [2400, 5600, 1900]
    }
    st.dataframe(pd.DataFrame(example_data))
