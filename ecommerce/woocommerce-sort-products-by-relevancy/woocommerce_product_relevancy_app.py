"""
WooCommerce Product Relevancy Sorter - Streamlit App

Sorts WooCommerce products by relevancy to their category using fuzzy matching.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd

try:
    from woocommerce import API
    from fuzzywuzzy import fuzz
except ImportError:
    st.error("Please install: pip install woocommerce fuzzywuzzy python-Levenshtein")
    st.stop()

st.set_page_config(
    page_title="WooCommerce Product Relevancy Sorter",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 WooCommerce Product Relevancy Sorter")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-Follow-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Sorts WooCommerce products by keyword relevancy
    - Optimizes category page product ordering
    - Uses semantic matching for sorting

    **How to use:**
    1. Export WooCommerce product data
    2. Enter target keywords for category
    3. Calculate relevancy scores
    4. Get optimized sort order

    **Best for:**
    - Category page optimization
    - Product merchandising
    - Search-aligned product ordering
    """)
st.markdown("Sort products by relevancy to their category using fuzzy matching.")


def get_woo_api(url, consumer_key, consumer_secret):
    """Create WooCommerce API connection."""
    return API(
        url=url,
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        version="wc/v3",
        timeout=(120, 120)
    )


def fetch_categories(api):
    """Fetch all product categories."""
    categories = []
    page = 1
    while True:
        response = api.get("products/categories", params={"per_page": 100, "page": page})
        if not response.ok or not response.json():
            break
        categories += response.json()
        page += 1
    return categories


def fetch_products_by_category(api, category_id):
    """Fetch products for a specific category."""
    return api.get(f"products?category={category_id}&per_page=100").json()


def calculate_relevancy(products, category_name):
    """Calculate relevancy scores for products."""
    results = []
    for product in products:
        score = fuzz.token_sort_ratio(product["name"], category_name)
        results.append({
            'id': product['id'],
            'name': product['name'],
            'score': score,
            'current_order': product.get('menu_order', 0)
        })
    return sorted(results, key=lambda x: (-x['score'], x['name']))


def update_product_order(api, product_updates):
    """Batch update product menu order."""
    response = api.post("products/batch", {"update": product_updates})
    return response.ok


# Sidebar
with st.sidebar:
    st.header("🔑 WooCommerce API Credentials")

    woo_url = st.text_input(
        "Store URL",
        placeholder="https://yourstore.com",
        help="Your WooCommerce store URL"
    )

    consumer_key = st.text_input(
        "Consumer Key",
        type="password",
        help="WooCommerce API consumer key"
    )

    consumer_secret = st.text_input(
        "Consumer Secret",
        type="password",
        help="WooCommerce API consumer secret"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    min_score = st.slider(
        "Minimum Score to Display",
        min_value=0,
        max_value=100,
        value=0,
        help="Only show products above this relevancy score"
    )

    st.markdown("---")
    st.markdown("### 📖 API Setup")
    st.markdown("""
    1. Go to WooCommerce → Settings → Advanced → REST API
    2. Click 'Add Key'
    3. Set permissions to 'Read/Write'
    4. Copy Consumer Key and Secret
    """)

# Main content
if woo_url and consumer_key and consumer_secret:
    try:
        api = get_woo_api(woo_url, consumer_key, consumer_secret)

        st.markdown("### 📁 Categories")

        with st.spinner("Fetching categories..."):
            categories = fetch_categories(api)

        if categories:
            st.success(f"✅ Found {len(categories)} categories")

            # Category selector
            category_options = {f"{c['name']} (ID: {c['id']})": c for c in categories}
            selected_category_key = st.selectbox(
                "Select a category",
                list(category_options.keys())
            )

            if selected_category_key:
                selected_category = category_options[selected_category_key]
                category_id = selected_category['id']
                category_name = selected_category['name']

                st.markdown("---")
                st.markdown(f"### 📦 Products in '{category_name}'")

                if st.button("🔍 Analyze Relevancy", type="primary", use_container_width=True):
                    with st.spinner("Fetching products..."):
                        products = fetch_products_by_category(api, category_id)

                    if products:
                        st.info(f"Found {len(products)} products")

                        # Calculate relevancy
                        relevancy_data = calculate_relevancy(products, category_name)

                        # Filter by minimum score
                        relevancy_data = [r for r in relevancy_data if r['score'] >= min_score]

                        if relevancy_data:
                            # Create dataframe
                            df = pd.DataFrame(relevancy_data)
                            df['new_order'] = range(len(df))

                            # Results tabs
                            tab1, tab2 = st.tabs(["📊 Results", "🔄 Apply Changes"])

                            with tab1:
                                st.subheader("Relevancy Scores")

                                # Show color-coded results
                                def highlight_score(val):
                                    if val >= 80:
                                        return 'background-color: #28a745; color: white'
                                    elif val >= 50:
                                        return 'background-color: #ffc107'
                                    else:
                                        return 'background-color: #dc3545; color: white'

                                styled_df = df.style.applymap(
                                    highlight_score,
                                    subset=['score']
                                )
                                st.dataframe(styled_df, use_container_width=True, height=400)

                                # Summary stats
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Products", len(df))
                                with col2:
                                    st.metric("Avg Score", f"{df['score'].mean():.1f}")
                                with col3:
                                    high_relevancy = len(df[df['score'] >= 70])
                                    st.metric("High Relevancy (≥70)", high_relevancy)

                                # Score distribution
                                import plotly.express as px
                                fig = px.histogram(
                                    df,
                                    x='score',
                                    nbins=20,
                                    title='Relevancy Score Distribution'
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                # Download
                                csv = df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    "📥 Download Results CSV",
                                    data=csv,
                                    file_name=f"relevancy_{category_name}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )

                            with tab2:
                                st.subheader("Apply Sorting to WooCommerce")

                                st.warning("""
                                ⚠️ **Warning:** This will update the menu_order for all products
                                in this category, changing their display order on your store.
                                """)

                                st.markdown("**Preview of changes:**")
                                preview_df = df[['name', 'current_order', 'new_order', 'score']].head(10)
                                st.dataframe(preview_df, use_container_width=True)

                                confirm = st.checkbox(
                                    "I understand this will change product order on my live store"
                                )

                                if confirm:
                                    if st.button("🚀 Apply Sorting", type="primary"):
                                        with st.spinner("Updating product order..."):
                                            updates = [
                                                {"id": row['id'], "menu_order": row['new_order']}
                                                for _, row in df.iterrows()
                                            ]

                                            success = update_product_order(api, updates)

                                        if success:
                                            st.success("✅ Product order updated successfully!")
                                        else:
                                            st.error("Failed to update product order")
                        else:
                            st.warning(f"No products with relevancy score ≥ {min_score}")
                    else:
                        st.warning("No products found in this category")
        else:
            st.warning("No categories found")

    except Exception as e:
        st.error(f"Error connecting to WooCommerce: {str(e)}")

else:
    st.info("👆 Enter your WooCommerce API credentials in the sidebar")

    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        This tool helps you **automatically sort products** in WooCommerce categories
        by their relevancy to the category name.

        **How it works:**
        1. Connects to your WooCommerce store via API
        2. Fetches products from a selected category
        3. Uses fuzzy matching to calculate relevancy scores
        4. Sorts products so the most relevant appear first
        5. Optionally applies the new order to your store

        **Benefits:**
        - Better user experience
        - Improved category page conversions
        - Automatic maintenance as products are added
        """)

    with st.expander("Example Relevancy Scoring"):
        st.markdown("""
        **Category: "Red Sneakers"**

        | Product Name | Score |
        |--------------|-------|
        | Red Running Sneakers | 85 |
        | Classic Red Sneakers | 82 |
        | Red Sports Shoes | 65 |
        | Blue Sneakers | 50 |
        | Black Boots | 20 |

        Products with higher scores (more similar to category name)
        will be sorted to the top.
        """)

# Footer
st.markdown("---")
st.markdown(
    "Built by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · "
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · "
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)"
)
