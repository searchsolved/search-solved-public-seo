# Author: Lee Foot
# Website: https://leefoot.com

"""
SERP Crossover Analyzer

Sends SERP searches for multiple keywords to DataForSEO (Standard queue), then
collects the results and analyses URL crossover.
Identifies keyword cannibalisation and topic overlap opportunities.

Features:
- Send SERP searches for multiple keywords via the DataForSEO SERP API (Standard queue)
- Collect the results later from a job file
- Calculate URL crossover percentage between keywords
- Identify which URLs rank for multiple keywords
- Visual crossover matrix
- Export detailed results
"""

import os
import streamlit as st
import pandas as pd
from io import BytesIO
from urllib.parse import urlparse

from dataforseo_queue import (
    QueueError,
    collect_results,
    estimate_cost,
    is_unresolved,
    job_from_csv,
    job_to_csv,
    organic_items,
    send_searches,
)

st.set_page_config(page_title="SERP Crossover Analyzer", page_icon="🔀", layout="wide")

st.title("SERP Crossover Analyzer")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Gets the Google results for multiple keywords from the DataForSEO SERP API
    - Calculates URL crossover percentage between keywords
    - Identifies cannibalisation and topic clustering opportunities

    **Why two steps:** Google now routes result links through google.com/goto. DataForSEO's
    Standard queue turns those back into real URLs, but a search can take 25 minutes or
    more to finish. So you send the searches, keep the job file, and collect the results later.

    **Requirements:**
    - DataForSEO account (sign up at [dataforseo.com](https://dataforseo.com/))

    **How to use:**
    1. Enter your DataForSEO login and password in the sidebar
    2. **Send searches:** choose the location, device and results per SERP, enter the keywords
       to compare (one per line), check the estimated cost and click "Send searches".
       Download the job file.
    3. **Collect results** (about 30 minutes later): upload the job file and click
       "Collect results"
    4. Review the crossover matrix and overlapping URLs

    **Use cases:**
    - Identify keyword cannibalisation (same URL ranking for multiple keywords)
    - Find topic clustering opportunities
    - Analyse SERP similarity for keyword grouping
    """)

LOCATION_CODES = {
    "United Kingdom": 2826,
    "United States": 2840,
    "Australia": 2036,
    "Canada": 2124,
    "Germany": 2276,
    "France": 2250,
    "Spain": 2724,
    "Italy": 2380,
    "Netherlands": 2528,
    "India": 2356,
}

LANGUAGE_CODE = "en"

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


def extract_domain(url):
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc.replace('www.', '')
    except Exception:
        return url


def build_serp_data(ready, depths):
    """Turn collected searches into {keyword: [results]} for the crossover analysis.

    ready is {keyword: items} from collect_results(); depths is {keyword: results per SERP}.
    Organic results that came back as unresolved google.com/goto links are left out and
    counted, since their real URL and domain are unknown. Returns (serp_data, unresolved).
    """
    serp_data, unresolved = {}, 0
    for keyword, items in ready.items():
        organic = organic_items(items)[:depths.get(keyword, len(items))]
        results = []
        for i, result in enumerate(organic):
            url = result.get('url') or ''
            if is_unresolved(url):
                unresolved += 1
                continue
            results.append({
                'position': result.get('rank_group', i + 1),
                'title': result.get('title', ''),
                'link': url,
                'domain': extract_domain(url)
            })
        if results:
            serp_data[keyword] = results
    return serp_data, unresolved


def calculate_crossover(serp_data):
    """Calculate crossover matrix between keywords."""
    keywords = list(serp_data.keys())

    # Create URL sets for each keyword
    url_sets = {kw: set(r['link'] for r in results) for kw, results in serp_data.items()}

    # Calculate crossover matrix
    matrix = pd.DataFrame(index=keywords, columns=keywords, dtype=float)

    for i, kw1 in enumerate(keywords):
        for j, kw2 in enumerate(keywords):
            if i == j:
                matrix.loc[kw1, kw2] = 100.0
            else:
                common = url_sets[kw1].intersection(url_sets[kw2])
                union = url_sets[kw1].union(url_sets[kw2])
                if union:
                    crossover = (len(common) / len(union)) * 100
                else:
                    crossover = 0.0
                matrix.loc[kw1, kw2] = round(crossover, 1)

    return matrix


def find_overlapping_urls(serp_data):
    """Find URLs that appear in multiple SERPs."""
    url_keywords = {}

    for keyword, results in serp_data.items():
        for result in results:
            url = result['link']
            if url not in url_keywords:
                url_keywords[url] = {
                    'title': result['title'],
                    'domain': result['domain'],
                    'keywords': [],
                    'positions': []
                }
            url_keywords[url]['keywords'].append(keyword)
            url_keywords[url]['positions'].append(result['position'])

    # Filter to URLs appearing in multiple SERPs
    overlapping = []
    for url, data in url_keywords.items():
        if len(data['keywords']) > 1:
            overlapping.append({
                'URL': url,
                'Domain': data['domain'],
                'Title': data['title'],
                'Keywords Count': len(data['keywords']),
                'Keywords': ', '.join(data['keywords']),
                'Positions': ', '.join(map(str, data['positions']))
            })

    df = pd.DataFrame(overlapping)
    if not df.empty:
        df = df.sort_values('Keywords Count', ascending=False)

    return df


def decode_bytes(raw_bytes):
    """Decode an uploaded job file."""
    for encoding in ('utf-8', 'utf-8-sig', 'latin-1'):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Unable to read the job file.")


has_credentials = bool(dataforseo_login and dataforseo_password)
if not has_credentials:
    st.warning("Enter your DataForSEO login and password in the sidebar to get started")

send_tab, collect_tab = st.tabs(["1. Send searches", "2. Collect results"])

with send_tab:
    st.subheader("Search Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        location = st.selectbox(
            "Location",
            list(LOCATION_CODES.keys()),
            index=0
        )
    with col2:
        device = st.selectbox(
            "Device",
            ["Desktop", "Mobile"],
            index=0,
            help="DataForSEO returns desktop or mobile results only."
        )
    with col3:
        num_results = st.slider(
            "Results per SERP",
            min_value=5,
            max_value=100,
            value=10,
            help="Number of organic results to fetch per keyword. DataForSEO bills per 10 results."
        )

    st.subheader("Enter Keywords")

    keyword_input = st.text_area(
        "Keywords to compare (one per line, 2-20 keywords)",
        height=150,
        placeholder="seo tools\nkeyword research\ncontent marketing"
    )

    keywords = list(dict.fromkeys(kw.strip() for kw in keyword_input.strip().split('\n') if kw.strip())) if keyword_input else []

    if keywords:
        if len(keywords) < 2:
            st.warning("Please enter at least 2 keywords to compare")
        elif len(keywords) > 20:
            st.warning("Please enter no more than 20 keywords")
        else:
            st.info(f"Ready to send {len(keywords)} keywords. "
                    f"Estimated cost: ${estimate_cost(len(keywords), num_results):.4f}")

    if st.button("Send searches", type="primary",
                 disabled=not has_credentials or len(keywords) < 2 or len(keywords) > 20):
        try:
            with st.spinner("Sending searches to DataForSEO..."):
                job_rows, send_warnings, send_cost = send_searches(
                    keywords, dataforseo_login, dataforseo_password, LOCATION_CODES[location],
                    LANGUAGE_CODE, device=device.lower(), depth=num_results)
        except QueueError as e:
            st.error(str(e))
        else:
            for warning in send_warnings:
                st.warning(warning)
            if job_rows:
                st.session_state['crossover_job'] = job_rows
                st.session_state['crossover_send_cost'] = send_cost

    if st.session_state.get('crossover_job'):
        job_rows = st.session_state['crossover_job']
        st.success(f"Queued {len(job_rows)} search(es). Cost reported by DataForSEO: "
                   f"${st.session_state['crossover_send_cost']:.4f}")
        st.markdown("DataForSEO can take **25 minutes or more** to finish. Download the job file, "
                    "then use **2. Collect results**. While this page stays open, the job is kept "
                    "here too, so you can collect without uploading it.")
        st.download_button(
            label="Download job file (CSV)",
            data=job_to_csv(job_rows).encode('utf-8'),
            file_name="serp_crossover_job.csv",
            mime="text/csv"
        )

with collect_tab:
    job_rows = st.session_state.get('crossover_job')
    job_file = st.file_uploader("Job file from step 1", type=['csv'],
                                help="The serp_crossover_job.csv you downloaded when you sent the searches")
    if job_file is not None:
        try:
            job_rows = job_from_csv(decode_bytes(job_file.getvalue()))
        except ValueError as e:
            st.error(str(e))
            job_rows = None
    elif job_rows:
        st.caption(f"Using the job sent from this page ({len(job_rows)} search(es)).")

    if st.button("Collect results", type="primary", disabled=not has_credentials or not job_rows):
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
            depths = {}
            for row in job_rows:
                try:
                    depths[row['keyword']] = int(row['depth'])
                except (KeyError, TypeError, ValueError):
                    pass
            serp_data, unresolved = build_serp_data(result['ready'], depths)
            st.session_state['crossover_collect'] = {
                'ready': len(result['ready']),
                'pending': result['pending'],
                'errors': result['errors'],
                'unresolved': unresolved,
                'total': len(job_rows),
            }
            for key in ('serp_data', 'crossover_matrix', 'overlapping_urls'):
                st.session_state.pop(key, None)
            if len(serp_data) >= 2:
                st.session_state['serp_data'] = serp_data
                st.session_state['crossover_matrix'] = calculate_crossover(serp_data)
                st.session_state['overlapping_urls'] = find_overlapping_urls(serp_data)

    collected = st.session_state.get('crossover_collect')
    if collected:
        if collected['pending']:
            st.warning(f"{collected['ready']} of {collected['total']} search(es) have finished; "
                       f"{len(collected['pending'])} are still running at DataForSEO. "
                       "Try again in a few minutes. The analysis below covers the finished ones only.")
        if collected['errors']:
            with st.expander(f"{len(collected['errors'])} search(es) failed"):
                for error in collected['errors']:
                    st.text(error)
        if collected['unresolved']:
            st.warning(f"{collected['unresolved']} result(s) came back as unresolved google.com/goto links "
                       "and are left out of the crossover, as their real URL is unknown.")
        if 'crossover_matrix' in st.session_state:
            st.success("Analysis complete!")
        else:
            st.error("Need at least 2 finished searches with results to analyse")

    # Display results
    if 'crossover_matrix' in st.session_state:
        matrix = st.session_state['crossover_matrix']
        overlapping_df = st.session_state['overlapping_urls']
        serp_data = st.session_state['serp_data']

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Keywords Analysed", len(serp_data))
        with col2:
            total_urls = sum(len(r) for r in serp_data.values())
            st.metric("Total SERP Results", total_urls)
        with col3:
            st.metric("Overlapping URLs", len(overlapping_df))
        with col4:
            # Average crossover (excluding diagonal)
            values = []
            for i, row in enumerate(matrix.values):
                for j, val in enumerate(row):
                    if i != j:
                        values.append(val)
            avg_crossover = sum(values) / len(values) if values else 0
            st.metric("Avg Crossover %", f"{avg_crossover:.1f}%")

        # Crossover matrix
        st.subheader("Crossover Matrix")
        st.caption("Percentage of URL overlap between keyword SERPs (Jaccard similarity)")

        # Style matrix with heatmap
        def color_crossover(val):
            if val == 100:
                return 'background-color: #d9d9d9'
            elif val >= 50:
                return 'background-color: #ff6b6b; color: white'
            elif val >= 25:
                return 'background-color: #ffd93d'
            elif val > 0:
                return 'background-color: #c9e4c5'
            else:
                return ''

        styled_matrix = matrix.style.map(color_crossover)
        st.dataframe(styled_matrix, width="stretch")

        st.markdown("""
        **Colour key:**
        - Red (50%+): High crossover - potential cannibalisation
        - Yellow (25-50%): Moderate crossover - related topics
        - Green (1-25%): Low crossover - different intent
        - White (0%): No crossover
        """)

        # Overlapping URLs
        if not overlapping_df.empty:
            st.subheader("Overlapping URLs")
            st.caption("URLs that appear in multiple keyword SERPs")
            st.dataframe(overlapping_df, width="stretch")
        else:
            st.info("No overlapping URLs found between these keywords")

        # Raw SERP data
        with st.expander("View Raw SERP Data"):
            all_serp = []
            for keyword, results in serp_data.items():
                for r in results:
                    all_serp.append({
                        'Keyword': keyword,
                        'Position': r['position'],
                        'Title': r['title'],
                        'URL': r['link'],
                        'Domain': r['domain']
                    })
            st.dataframe(pd.DataFrame(all_serp), width="stretch")

        # Downloads
        st.subheader("Download")

        col1, col2, col3 = st.columns(3)

        with col1:
            matrix_csv = matrix.to_csv(encoding='utf-8-sig')
            st.download_button(
                label="Download Matrix (CSV)",
                data=matrix_csv,
                file_name="crossover_matrix.csv",
                mime="text/csv"
            )

        with col2:
            if not overlapping_df.empty:
                overlap_csv = overlapping_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="Download Overlaps (CSV)",
                    data=overlap_csv,
                    file_name="overlapping_urls.csv",
                    mime="text/csv"
                )

        with col3:
            # Full Excel export
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                matrix.to_excel(writer, sheet_name='Crossover Matrix')
                if not overlapping_df.empty:
                    overlapping_df.to_excel(writer, sheet_name='Overlapping URLs', index=False)
                # All SERP data
                all_serp_df = pd.DataFrame(all_serp)
                all_serp_df.to_excel(writer, sheet_name='Raw SERP Data', index=False)

            st.download_button(
                label="Download Excel Report",
                data=output.getvalue(),
                file_name="serp_crossover_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    else:
        st.subheader("Example Output")

        example_matrix = pd.DataFrame({
            "seo tools": [100.0, 30.0, 10.0],
            "keyword research": [30.0, 100.0, 20.0],
            "content marketing": [10.0, 20.0, 100.0]
        }, index=["seo tools", "keyword research", "content marketing"])

        st.dataframe(example_matrix)

        example_overlap = {
            "URL": ["https://ahrefs.com/seo", "https://semrush.com/"],
            "Domain": ["ahrefs.com", "semrush.com"],
            "Keywords Count": [3, 2],
            "Keywords": ["seo tools, keyword research, content marketing", "seo tools, keyword research"]
        }
        st.dataframe(pd.DataFrame(example_overlap))
