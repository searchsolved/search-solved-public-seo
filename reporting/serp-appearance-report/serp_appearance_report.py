"""
SERP Appearance Report - Live Google Results via DataForSEO
Fetches live Google results for your keyword list from the DataForSEO SERP
API and reports the query, position, link, title and snippet for every
organic appearance of your domain.

Author: Lee Foot
Website: https://leefoot.com
"""

import os
from io import BytesIO

import streamlit as st

from serp_appearance_core import (
    DEVICES,
    LOCATIONS,
    estimate_cost,
    keywords_from_bytes,
    normalise_domain,
    parse_keyword_lines,
    results_to_dataframe,
    run_report,
)

st.set_page_config(
    page_title="SERP Appearance Report",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 SERP Appearance Report")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Fetches the live Google results for each of your keywords using the DataForSEO SERP API
    - Keeps the organic results on your domain (subdomains included)
    - Reports the query, position, link, title and snippet for each appearance

    **Requirements:**
    - DataForSEO account (sign up at [dataforseo.com](https://dataforseo.com/))

    **How to use:**
    1. Enter your DataForSEO login and password in the sidebar
    2. Choose the location, device and number of results per keyword
    3. Enter your domain (e.g. example.com)
    4. Paste your keywords (one per line) or upload a CSV with a keyword column
    5. Check the estimated cost, click "Fetch SERPs", then download the CSV

    **Best for:**
    - Checking how your site appears in the SERPs for your top queries
    - Auditing titles and snippets at scale
    - Building a SERP appearance report from your Google Search Console top queries
    """)

# Sidebar settings
st.sidebar.header("API Settings")

dataforseo_login = st.sidebar.text_input(
    "DataForSEO Login",
    type="password",
    value=os.environ.get('DATAFORSEO_LOGIN', ''),
    help="Your login from dataforseo.com"
)

dataforseo_password = st.sidebar.text_input(
    "DataForSEO Password",
    type="password",
    value=os.environ.get('DATAFORSEO_PASSWORD', ''),
    help="Your password from dataforseo.com"
)

st.sidebar.markdown("---")
st.sidebar.header("Search Settings")

location = st.sidebar.selectbox("Location", list(LOCATIONS.keys()), index=0)

device = st.sidebar.selectbox("Device", [d.title() for d in DEVICES], index=0)
st.sidebar.caption("DataForSEO supports desktop and mobile results only.")

depth = st.sidebar.slider(
    "Results per keyword",
    min_value=10,
    max_value=100,
    value=10,
    step=10,
    help="Number of Google results to fetch per keyword. DataForSEO bills per 10 results."
)

delay = st.sidebar.slider(
    "Seconds between requests",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5,
    help="Requests are sent one at a time with this pause between them"
)

# Main inputs
domain = st.text_input(
    "Your domain",
    placeholder="example.com",
    help="Organic results on this domain or its subdomains are reported"
)

input_method = st.radio("Keywords", ["Paste keywords", "Upload CSV"], horizontal=True)

keywords = []
if input_method == "Paste keywords":
    keyword_text = st.text_area(
        "Keywords (one per line)",
        height=150,
        placeholder="seo tools\nkeyword research\ncontent marketing"
    )
    keywords = parse_keyword_lines(keyword_text) if keyword_text else []
else:
    uploaded_file = st.file_uploader(
        "Upload a CSV of keywords",
        type=['csv', 'txt'],
        help="Uses the column named keyword, keywords, query, queries, search term or "
             "top queries; otherwise the first column. A .txt file with one keyword per line also works."
    )
    if uploaded_file is not None:
        try:
            keywords = keywords_from_bytes(uploaded_file.getvalue(), uploaded_file.name)
        except Exception as e:
            st.error(f"Could not read {uploaded_file.name}: {e}")

has_credentials = bool(dataforseo_login and dataforseo_password)
has_domain = bool(normalise_domain(domain))

if keywords:
    estimated_cost = estimate_cost(len(keywords), depth)
    st.info(f"Ready to fetch {len(keywords)} keyword(s) at {depth} results each. "
            f"Estimated cost: ${estimated_cost:.3f}")

if st.button("Fetch SERPs", type="primary",
             disabled=not has_credentials or not keywords or not has_domain):
    location_code, language_code = LOCATIONS[location]
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(done, total, keyword):
        progress_bar.progress(done / total if total else 1.0)
        status_text.text(f"Fetching SERP {done + 1}/{total}: {keyword}" if keyword else "")

    rows, warnings, total_cost = run_report(
        keywords, domain, dataforseo_login, dataforseo_password,
        location_code, language_code,
        device=device.lower(), depth=depth, delay=delay,
        progress_callback=update_progress
    )

    st.session_state['appearance_df'] = results_to_dataframe(rows)
    st.session_state['appearance_warnings'] = warnings
    st.session_state['appearance_cost'] = total_cost
    st.session_state['appearance_domain'] = normalise_domain(domain)
    st.session_state['appearance_keywords'] = len(keywords)

if 'appearance_df' in st.session_state:
    df = st.session_state['appearance_df']
    warnings = st.session_state['appearance_warnings']

    st.subheader("Results")
    st.success(f"Found **{len(df)}** appearances of **{st.session_state['appearance_domain']}** "
               f"across {st.session_state['appearance_keywords']} keyword(s).")
    st.caption(f"Cost reported by DataForSEO: ${st.session_state['appearance_cost']:.4f}")

    if warnings:
        with st.expander(f"{len(warnings)} keyword(s) had issues"):
            for warning in warnings:
                st.text(warning)

    if df.empty:
        st.info("No organic results were on that domain. Check the domain spelling, "
                "or fetch more results per keyword.")
    else:
        st.dataframe(df, use_container_width=True)

        buffer = BytesIO()
        df.to_csv(buffer, index=False, encoding='utf-8-sig')
        buffer.seek(0)

        st.download_button(
            label="📥 Download SERP Appearance Report (CSV)",
            data=buffer,
            file_name="serp_appearance_report.csv",
            mime="text/csv"
        )

else:
    if not has_credentials:
        st.warning("Enter your DataForSEO login and password in the sidebar to get started")

    st.markdown("""
    ### How it Works
    1. Enter your domain and your keywords
    2. The tool fetches the live Google results for each keyword, one request at a time
    3. Organic results on your domain (or its subdomains) are kept
    4. The output lists the query, position, link, title and snippet per appearance

    ### Keyword CSV Format
    Any CSV with a column named `keyword`, `keywords`, `query`, `queries`,
    `search term` or `top queries` (any case). A Google Search Console
    queries export works as is. Without one of those names, the first column is used.

    ### Use Cases
    - Reviewing how your top queries render in the SERPs
    - Spotting title rewrites and weak snippets
    - Reporting SERP appearances to stakeholders
    """)
