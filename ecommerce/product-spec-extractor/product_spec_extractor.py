####################################################################################
#                                                                                  #
#  Product Spec Extractor                                                          #
#                                                                                  #
#  Scrape product specifications from e-commerce pages.                            #
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
Product Spec Extractor

Extracts product specifications from e-commerce product pages. Scrapes structured
specification tables and outputs clean CSV data for analysis, matching, or import.

Features:
- Configurable CSS selectors for different site structures
- Rate limiting to avoid blocking
- Progress tracking
- Structured CSV output
"""

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from io import StringIO

st.set_page_config(page_title="Product Spec Extractor", page_icon="📋", layout="wide")

st.title("Product Spec Extractor")
st.markdown("*Created by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)*")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Scrapes product specification tables from product pages
    - Extracts key-value pairs from structured data
    - Outputs clean CSV with all specs as columns

    **How to prepare:**
    1. Get a list of product URLs you want to scrape
    2. Identify the CSS selector for the specification table/section
    3. Configure rate limiting to avoid being blocked

    **Common CSS selectors:**
    - Definition lists: `dl.specs dt, dl.specs dd`
    - Tables: `table.specifications tr`
    - Key-value divs: `.product-specs .spec-row`

    **Tips:**
    - Use browser DevTools to find the right selectors
    - Start with a small sample to test selectors
    - Use appropriate delays to avoid IP blocking
    """)

# Sidebar settings
st.sidebar.header("Extraction Settings")

dt_selector = st.sidebar.text_input(
    "Key selector (dt/th)",
    value="dt",
    help="CSS selector for specification names (e.g., 'dt', 'th', '.spec-name')"
)

dd_selector = st.sidebar.text_input(
    "Value selector (dd/td)",
    value="dd",
    help="CSS selector for specification values (e.g., 'dd', 'td', '.spec-value')"
)

parent_selector = st.sidebar.text_input(
    "Parent container selector (optional)",
    value="",
    help="CSS selector for the specs container (e.g., '.product-specs', '#specifications')"
)

st.sidebar.markdown("---")
st.sidebar.header("Request Settings")

delay = st.sidebar.slider(
    "Delay between requests (seconds)",
    min_value=0.5,
    max_value=10.0,
    value=1.0,
    step=0.5,
    help="Time to wait between page requests"
)

timeout = st.sidebar.number_input(
    "Request timeout (seconds)",
    min_value=5,
    max_value=60,
    value=15,
    help="Maximum time to wait for page response"
)

user_agent = st.sidebar.text_input(
    "User Agent",
    value="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    help="Browser user agent string"
)

# URL input
st.subheader("Enter Product URLs")

input_method = st.radio(
    "Input method",
    ["Paste URLs", "Upload CSV"],
    horizontal=True
)

urls = []

if input_method == "Paste URLs":
    url_text = st.text_area(
        "Paste URLs (one per line)",
        height=200,
        help="Enter product page URLs, one per line"
    )
    if url_text:
        urls = [u.strip() for u in url_text.strip().split('\n') if u.strip()]
        st.info(f"Found {len(urls)} URLs")

else:
    url_file = st.file_uploader(
        "Upload CSV with URLs",
        type=['csv', 'txt'],
        help="CSV file with a column containing URLs"
    )

    if url_file is not None:
        try:
            df_urls = pd.read_csv(url_file)
            url_col = st.selectbox("Select URL column", df_urls.columns.tolist())
            urls = df_urls[url_col].dropna().tolist()
            st.info(f"Found {len(urls)} URLs")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# Test single URL
st.subheader("Test Extraction")
test_url = st.text_input("Test URL (optional)", help="Test your selectors on a single URL first")

if test_url and st.button("Test Selectors"):
    with st.spinner("Testing extraction..."):
        try:
            headers = {'User-Agent': user_agent}
            response = requests.get(test_url, headers=headers, timeout=timeout)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find parent container if specified
            if parent_selector.strip():
                container = soup.select_one(parent_selector)
                if container:
                    soup = container
                else:
                    st.warning(f"Parent container '{parent_selector}' not found, searching whole page")

            # Find specs
            dt_tags = soup.select(dt_selector)
            dd_tags = soup.select(dd_selector)

            st.write(f"**Found {len(dt_tags)} keys and {len(dd_tags)} values**")

            if dt_tags:
                specs = {}
                for i, dt in enumerate(dt_tags):
                    key = dt.get_text(strip=True)
                    if i < len(dd_tags):
                        value = ' '.join(dd_tags[i].stripped_strings)
                    else:
                        value = ""
                    specs[key] = value

                st.subheader("Extracted Specifications")
                st.json(specs)
            else:
                st.warning("No specifications found. Try adjusting your selectors.")

                # Show page structure hint
                with st.expander("Page structure hints"):
                    # Find common spec patterns
                    dls = soup.find_all('dl')
                    tables = soup.find_all('table')
                    st.write(f"Found {len(dls)} definition lists (<dl>)")
                    st.write(f"Found {len(tables)} tables")

                    if dls:
                        st.write("Try: `dt` for keys, `dd` for values")
                    if tables:
                        st.write("Try: `th` or `td:first-child` for keys, `td:last-child` for values")

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {str(e)}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Main extraction
st.markdown("---")
st.subheader("Extract Specifications")

if urls and st.button("Extract All Specifications", type="primary"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    all_specs = []
    errors = []

    headers = {'User-Agent': user_agent}

    for i, url in enumerate(urls):
        status_text.text(f"Processing {i+1}/{len(urls)}: {url[:60]}...")
        progress_bar.progress((i + 1) / len(urls))

        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find parent container if specified
            if parent_selector.strip():
                container = soup.select_one(parent_selector)
                if container:
                    soup = container

            # Extract specs
            product_specs = {'URL': url}

            dt_tags = soup.select(dt_selector)
            dd_tags = soup.select(dd_selector)

            for j, dt in enumerate(dt_tags):
                key = dt.get_text(strip=True)
                if j < len(dd_tags):
                    value = ' '.join(dd_tags[j].stripped_strings)
                else:
                    value = ""
                product_specs[key] = value

            all_specs.append(product_specs)

            # Rate limiting
            time.sleep(delay)

        except Exception as e:
            errors.append({'URL': url, 'Error': str(e)})
            all_specs.append({'URL': url, 'Error': str(e)})

    status_text.text("Extraction complete!")

    # Create DataFrame
    df_results = pd.DataFrame(all_specs)

    if not df_results.empty:
        # Sort columns by frequency (most common specs first)
        col_counts = df_results.notna().sum().sort_values(ascending=False)
        sorted_cols = ['URL'] + [c for c in col_counts.index if c != 'URL']
        df_results = df_results.reindex(columns=sorted_cols)

        # Display results
        st.subheader("Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("URLs Processed", len(urls))
        with col2:
            st.metric("Successful", len(urls) - len(errors))
        with col3:
            st.metric("Unique Specs Found", len(df_results.columns) - 1)

        # Show specs by frequency
        st.subheader("Specification Coverage")
        spec_coverage = pd.DataFrame({
            'Specification': [c for c in df_results.columns if c not in ['URL', 'Error']],
            'Products with Spec': [df_results[c].notna().sum() for c in df_results.columns if c not in ['URL', 'Error']]
        })
        spec_coverage = spec_coverage.sort_values('Products with Spec', ascending=False)
        st.dataframe(spec_coverage.head(30), use_container_width=True)

        # Show data
        st.subheader("Extracted Data")
        st.dataframe(df_results.head(50), use_container_width=True)

        # Errors
        if errors:
            with st.expander(f"Errors ({len(errors)})"):
                st.dataframe(pd.DataFrame(errors))

        # Download
        st.subheader("Download")
        csv_output = df_results.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="Download CSV",
            data=csv_output,
            file_name="product_specifications.csv",
            mime="text/csv"
        )
    else:
        st.warning("No data extracted. Check your selectors and URLs.")

elif not urls:
    st.info("Enter product URLs above to begin extraction")

# Example output
if not urls and not test_url:
    st.subheader("Example Output")
    example_data = {
        "URL": ["https://example.com/product1", "https://example.com/product2"],
        "Brand": ["ACME", "ACME"],
        "Material": ["Steel", "Aluminum"],
        "Weight": ["2.5 kg", "1.8 kg"],
        "Dimensions": ["10 x 5 x 3 cm", "8 x 4 x 2 cm"],
        "Color": ["Black", "Silver"]
    }
    st.dataframe(pd.DataFrame(example_data))
