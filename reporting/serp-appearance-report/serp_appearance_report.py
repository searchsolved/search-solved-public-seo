"""
SERP Appearance Report - Google Results via DataForSEO
Sends your keyword list to the DataForSEO SERP API (Standard queue), then
collects the Google results and reports the query, position, link, title and
snippet for every organic appearance of your domain.

Author: Lee Foot
Website: https://leefoot.com
"""

import os
from io import BytesIO

import streamlit as st

from dataforseo_queue import QueueError, collect_results, estimate_cost, job_from_csv, job_to_csv, send_searches
from serp_appearance_core import (
    DEVICES,
    LOCATIONS,
    build_report,
    decode_bytes,
    keywords_from_bytes,
    normalise_domain,
    parse_keyword_lines,
    results_to_dataframe,
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
    - Gets the Google results for each of your keywords from the DataForSEO SERP API
    - Keeps the organic results on your domain (subdomains included)
    - Reports the query, position, link, title and snippet for each appearance

    **Why two steps:** Google now routes result links through google.com/goto. DataForSEO's
    Standard queue turns those back into real URLs, but a search can take 25 minutes or
    more to finish. So you send the searches, keep the job file, and collect the results later.

    **Requirements:**
    - DataForSEO account (sign up at [dataforseo.com](https://dataforseo.com/))

    **How to use:**
    1. Enter your DataForSEO login and password in the sidebar
    2. **Send searches:** choose the location, device and results per keyword, add your keywords,
       check the estimated cost and click "Send searches". Download the job file.
    3. **Collect results** (about 30 minutes later): upload the job file, enter your domain
       and click "Collect results", then download the CSV

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

has_credentials = bool(dataforseo_login and dataforseo_password)
if not has_credentials:
    st.warning("Enter your DataForSEO login and password in the sidebar to get started")

send_tab, collect_tab = st.tabs(["1. Send searches", "2. Collect results"])

with send_tab:
    col1, col2, col3 = st.columns(3)
    with col1:
        location = st.selectbox("Location", list(LOCATIONS.keys()), index=0)
    with col2:
        device = st.selectbox("Device", [d.title() for d in DEVICES], index=0,
                              help="DataForSEO supports desktop and mobile results only.")
    with col3:
        depth = st.slider("Results per keyword", min_value=10, max_value=100, value=10, step=10,
                          help="Number of Google results to fetch per keyword. DataForSEO bills per 10 results.")

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

    if keywords:
        st.info(f"Ready to send {len(keywords)} keyword(s) at {depth} results each. "
                f"Estimated cost: ${estimate_cost(len(keywords), depth):.4f}")

    if st.button("Send searches", type="primary", disabled=not has_credentials or not keywords):
        location_code, language_code = LOCATIONS[location]
        try:
            with st.spinner("Sending searches to DataForSEO..."):
                job_rows, send_warnings, send_cost = send_searches(
                    keywords, dataforseo_login, dataforseo_password, location_code, language_code,
                    device=device.lower(), depth=depth)
        except QueueError as e:
            st.error(str(e))
        else:
            for warning in send_warnings:
                st.warning(warning)
            if job_rows:
                st.session_state['appearance_job'] = job_rows
                st.session_state['appearance_send_cost'] = send_cost

    if st.session_state.get('appearance_job'):
        job_rows = st.session_state['appearance_job']
        st.success(f"Queued {len(job_rows)} search(es). Cost reported by DataForSEO: "
                   f"${st.session_state['appearance_send_cost']:.4f}")
        st.markdown("DataForSEO can take **25 minutes or more** to finish. Download the job file, "
                    "then use **2. Collect results**. While this page stays open, the job is kept "
                    "here too, so you can collect without uploading it.")
        st.download_button(
            label="📥 Download job file (CSV)",
            data=job_to_csv(job_rows).encode('utf-8'),
            file_name="serp_appearance_job.csv",
            mime="text/csv"
        )

with collect_tab:
    job_rows = st.session_state.get('appearance_job')
    job_file = st.file_uploader("Job file from step 1", type=['csv'],
                                help="The serp_appearance_job.csv you downloaded when you sent the searches")
    if job_file is not None:
        try:
            job_rows = job_from_csv(decode_bytes(job_file.getvalue()))
        except ValueError as e:
            st.error(str(e))
            job_rows = None
    elif job_rows:
        st.caption(f"Using the job sent from this page ({len(job_rows)} search(es)).")

    domain = st.text_input(
        "Your domain",
        placeholder="example.com",
        help="Organic results on this domain or its subdomains are reported"
    )
    has_domain = bool(normalise_domain(domain))

    if st.button("Collect results", type="primary",
                 disabled=not has_credentials or not job_rows or not has_domain):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(done, total, keyword):
            progress_bar.progress(done / total if total else 1.0)
            status_text.text(f"Checking {done + 1}/{total}: {keyword}" if keyword else "")

        try:
            result = collect_results(job_rows, dataforseo_login, dataforseo_password,
                                     progress_callback=update_progress)
        except QueueError as e:
            st.error(str(e))
        else:
            rows, unresolved = build_report(result['ready'], domain)
            st.session_state['appearance_result'] = {
                'df': results_to_dataframe(rows),
                'ready': len(result['ready']),
                'pending': result['pending'],
                'errors': result['errors'],
                'unresolved': unresolved,
                'total': len(job_rows),
                'domain': normalise_domain(domain),
            }

    report = st.session_state.get('appearance_result')
    if report:
        if report['pending']:
            st.warning(f"{report['ready']} of {report['total']} search(es) have finished; "
                       f"{len(report['pending'])} are still running at DataForSEO. "
                       "Try again in a few minutes. The table below covers the finished ones only.")
        if report['errors']:
            with st.expander(f"{len(report['errors'])} search(es) failed"):
                for error in report['errors']:
                    st.text(error)
        if report['unresolved']:
            st.warning(f"{report['unresolved']} result(s) came back as unresolved google.com/goto links "
                       "and could not be checked against your domain.")

        if report['ready']:
            df = report['df']
            st.subheader("Results")
            st.success(f"Found **{len(df)}** appearances of **{report['domain']}** "
                       f"across {report['ready']} keyword(s).")
            if df.empty:
                st.info("No organic results were on that domain. Check the domain spelling, "
                        "or send the searches again with more results per keyword.")
            else:
                st.dataframe(df, width="stretch")
                buffer = BytesIO()
                df.to_csv(buffer, index=False, encoding='utf-8-sig')
                buffer.seek(0)
                st.download_button(
                    label="📥 Download SERP Appearance Report (CSV)",
                    data=buffer,
                    file_name="serp_appearance_report.csv",
                    mime="text/csv"
                )

with st.expander("Keyword CSV format and use cases"):
    st.markdown("""
    **Keyword CSV format:** any CSV with a column named `keyword`, `keywords`, `query`, `queries`,
    `search term` or `top queries` (any case). A Google Search Console queries export works as is.
    Without one of those names, the first column is used.

    **Use cases:**
    - Reviewing how your top queries render in the SERPs
    - Spotting title rewrites and weak snippets
    - Reporting SERP appearances to stakeholders
    """)
