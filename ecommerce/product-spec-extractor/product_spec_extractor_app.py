"""
Product Spec Extractor - Streamlit App

Scrapes product specifications from e-commerce pages.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

st.set_page_config(
    page_title="Product Spec Extractor",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Product Spec Extractor")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Extracts product specifications from pages
    - Structures spec data into consistent format
    - Identifies missing specifications

    **How to use:**
    1. Upload URLs or product page HTML
    2. Configure extraction settings
    3. Extract and normalize specs
    4. Download structured data

    **Best for:**
    - Product data enrichment
    - Comparison page creation
    - Structured data preparation
    """)
st.markdown("Scrape product specifications from e-commerce pages.")


def extract_specs(url, dt_selector, dd_selector, parent_selector, headers, timeout):
    """Extract specifications from a single page."""
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        soup = BeautifulSoup(response.text, 'html.parser')

        if parent_selector:
            container = soup.select_one(parent_selector)
            if container:
                soup = container

        specs = {'URL': url}

        dt_tags = soup.select(dt_selector)
        dd_tags = soup.select(dd_selector)

        for j, dt in enumerate(dt_tags):
            key = dt.get_text(strip=True)
            if j < len(dd_tags):
                value = ' '.join(dd_tags[j].stripped_strings)
            else:
                value = ""
            if key:
                specs[key] = value

        return specs, None

    except Exception as e:
        return {'URL': url, 'Error': str(e)}, str(e)


def process_urls(urls, dt_selector, dd_selector, parent_selector, headers, timeout, delay, progress_bar, status_text):
    """Process multiple URLs."""
    all_specs = []
    errors = []

    for i, url in enumerate(urls):
        progress_bar.progress((i + 1) / len(urls))
        status_text.text(f"Processing {i + 1}/{len(urls)}: {url[:50]}...")

        specs, error = extract_specs(
            url, dt_selector, dd_selector, parent_selector, headers, timeout
        )

        all_specs.append(specs)
        if error:
            errors.append({'URL': url, 'Error': error})

        time.sleep(delay)

    return all_specs, errors


# Sidebar
with st.sidebar:
    st.header("⚙️ CSS Selectors")

    dt_selector = st.text_input(
        "Key Selector",
        value="dt",
        help="CSS selector for specification keys/labels"
    )

    dd_selector = st.text_input(
        "Value Selector",
        value="dd",
        help="CSS selector for specification values"
    )

    parent_selector = st.text_input(
        "Parent Selector (optional)",
        value="",
        help="Limit extraction to a specific container"
    )

    st.markdown("---")
    st.subheader("⚡ Request Settings")

    delay = st.slider(
        "Delay (seconds)",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.5
    )

    timeout = st.slider(
        "Timeout (seconds)",
        min_value=5,
        max_value=30,
        value=15
    )

    user_agent = st.text_input(
        "User Agent",
        value="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )

    st.markdown("---")
    st.markdown("### 💡 Selector Examples")
    st.markdown("""
    **Definition lists:**
    - Key: `dt`
    - Value: `dd`

    **Tables:**
    - Key: `th`
    - Value: `td`

    **Custom classes:**
    - Key: `.spec-name`
    - Value: `.spec-value`

    **With parent:**
    - Parent: `#specifications`
    - Key: `dt`
    - Value: `dd`
    """)

# Main content
input_method = st.radio(
    "Input Method",
    ["Single URL", "Multiple URLs", "Upload CSV"],
    horizontal=True
)

urls = []

if input_method == "Single URL":
    url = st.text_input(
        "Product URL",
        placeholder="https://example.com/product/123",
        help="Enter a product page URL"
    )
    if url:
        urls = [url]

elif input_method == "Multiple URLs":
    urls_text = st.text_area(
        "Enter URLs (one per line)",
        height=150,
        placeholder="https://example.com/product/1\nhttps://example.com/product/2"
    )
    if urls_text:
        urls = [u.strip() for u in urls_text.split('\n') if u.strip().startswith('http')]

else:  # Upload CSV
    uploaded_file = st.file_uploader("Upload CSV with URLs", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Auto-detect URL column
        url_columns = [c for c in df.columns if 'url' in c.lower() or 'address' in c.lower()]
        if url_columns:
            url_col = st.selectbox("URL Column", url_columns)
        else:
            url_col = st.selectbox("URL Column", df.columns.tolist())

        urls = df[url_col].dropna().tolist()
        urls = [u for u in urls if isinstance(u, str) and u.startswith('http')]

if urls:
    st.info(f"Ready to process **{len(urls)}** URL(s)")

    # Preview URLs
    with st.expander("Preview URLs"):
        for url in urls[:10]:
            st.markdown(f"- {url}")
        if len(urls) > 10:
            st.markdown(f"... and {len(urls) - 10} more")

    if st.button("📋 Extract Specifications", type="primary", use_container_width=True):
        headers = {'User-Agent': user_agent}
        progress_bar = st.progress(0)
        status_text = st.empty()

        all_specs, errors = process_urls(
            urls, dt_selector, dd_selector, parent_selector,
            headers, timeout, delay, progress_bar, status_text
        )

        progress_bar.empty()
        status_text.empty()

        # Create DataFrame
        df_results = pd.DataFrame(all_specs)

        if not df_results.empty:
            # Count non-null values per column (excluding URL and Error)
            spec_columns = [c for c in df_results.columns if c not in ['URL', 'Error']]

            st.success(f"✅ Extracted specs from {len(df_results)} pages!")
            st.info(f"Found **{len(spec_columns)}** unique specification fields")

            # Results tabs
            tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Field Coverage", "🔍 Single URL Preview"])

            with tab1:
                # Sort columns by frequency
                if spec_columns:
                    col_counts = df_results[spec_columns].notna().sum().sort_values(ascending=False)
                    sorted_cols = ['URL'] + col_counts.index.tolist()
                    if 'Error' in df_results.columns:
                        sorted_cols.append('Error')
                    df_display = df_results[[c for c in sorted_cols if c in df_results.columns]]
                else:
                    df_display = df_results

                st.dataframe(df_display, use_container_width=True, height=400)

                csv = df_results.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "📥 Download Specs CSV",
                    data=csv,
                    file_name="product_specs.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with tab2:
                st.subheader("Field Coverage")

                if spec_columns:
                    coverage_data = []
                    for col in spec_columns:
                        count = df_results[col].notna().sum()
                        pct = count / len(df_results) * 100
                        coverage_data.append({
                            'Field': col,
                            'Count': count,
                            'Coverage %': round(pct, 1)
                        })

                    coverage_df = pd.DataFrame(coverage_data)
                    coverage_df = coverage_df.sort_values('Count', ascending=False)

                    st.dataframe(coverage_df, use_container_width=True, height=400)

                    # Chart
                    import plotly.express as px
                    fig = px.bar(
                        coverage_df.head(20),
                        x='Field',
                        y='Coverage %',
                        title='Top 20 Specification Fields by Coverage'
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)

            with tab3:
                if len(urls) > 0:
                    st.subheader("Single URL Preview")

                    selected_idx = st.selectbox(
                        "Select URL",
                        range(len(urls)),
                        format_func=lambda i: urls[i][:60] + '...'
                    )

                    if selected_idx < len(all_specs):
                        specs = all_specs[selected_idx]

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Extracted Specifications:**")
                            for key, value in specs.items():
                                if key not in ['URL', 'Error'] and value:
                                    st.markdown(f"- **{key}:** {value}")

                        with col2:
                            if 'Error' in specs and specs['Error']:
                                st.error(f"Error: {specs['Error']}")
                            else:
                                st.success("Extraction successful")

        else:
            st.warning("No specifications could be extracted")

        # Show errors
        if errors:
            with st.expander(f"⚠️ Errors ({len(errors)})"):
                for err in errors[:20]:
                    st.markdown(f"- {err['URL'][:50]}: {err['Error']}")

else:
    st.info("👆 Enter URL(s) to extract product specifications")

    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        This tool helps you **extract product specifications** from e-commerce pages.

        **How it works:**
        1. Fetches the HTML from each URL
        2. Uses CSS selectors to find key-value pairs
        3. Extracts and structures the specifications

        **Common use cases:**
        - Competitor product analysis
        - Product catalog enrichment
        - Price comparison data collection
        - Specification compliance audits
        """)

    with st.expander("🔧 Selector Troubleshooting"):
        st.markdown("""
        **If extraction fails:**

        1. **Inspect the page** in your browser (F12)
        2. Find the specifications section
        3. Identify the HTML structure:
           - `<dt>` / `<dd>` for definition lists
           - `<th>` / `<td>` for tables
           - Custom classes like `.spec-label` / `.spec-value`

        4. Use a **parent selector** to limit the search area
           (e.g., `#product-specs` or `.specifications`)

        **Testing tip:** Start with a single URL to find
        the right selectors before batch processing.
        """)

# Footer
st.markdown("---")
st.markdown(
    "Built by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · "
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · "
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)"
)
