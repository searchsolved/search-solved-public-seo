# Author: Lee Foot
# Website: https://leefoot.com

"""
Share of Voice Calculator (CTR Curve Based)

Calculates estimated organic traffic share using ranking positions and
industry-standard CTR curves (Sistrix, Advanced Web Ranking, etc.).
Supports CSV upload of pre-scraped ranking data, or Google results from
the DataForSEO SERP API (Standard queue: send the searches, collect later).

Features:
- Upload CSV with keyword, volume, position, domain data
- SERP fetching from DataForSEO with user-supplied keywords and volumes
- Configurable CTR curves (Sistrix, AWR, Custom)
- Group by domain or by category
- Export with traffic estimates
"""

import csv
import io
import os
from io import BytesIO

import pandas as pd
import streamlit as st

from dataforseo_queue import (
    JOB_COLUMNS,
    QueueError,
    collect_results,
    estimate_cost,
    is_unresolved,
    job_from_csv,
    organic_items,
    send_searches,
)

# Results per keyword: the SOV calculation only uses positions 1 to 10
SERP_DEPTH = 10

# The job file adds each keyword's search volume to the standard job columns
SOV_JOB_COLUMNS = JOB_COLUMNS + ['volume']

st.set_page_config(page_title="Share of Voice Calculator", page_icon="📊", layout="wide")

st.title("Share of Voice Calculator")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Calculates Share of Voice (SOV) from ranking data
    - Uses CTR curves to estimate organic traffic
    - Aggregates by domain to show market share

    **Two input methods:**

    1. **CSV Upload** - Upload pre-scraped ranking data with columns:
       - `keyword` - The search query
       - `volume` - Monthly search volume
       - `position` - Ranking position (1-10)
       - `domain` - The domain ranking
       - `category` (optional) - For grouping analysis

    2. **Fetch SERPs (DataForSEO)** - Provide keywords and search volumes, and the
       tool gets the Google results from DataForSEO, then calculates SOV from the rankings.
       This works in two steps:
       - **1. Send searches:** choose the location and device, add your keywords and
         volumes, check the estimated cost and click "Send searches". Download the job file.
       - **2. Collect results** (about 30 minutes later): upload the job file and click
         "Collect results". The CTR curve and Top N settings in the sidebar apply here,
         so you can change them and look again without sending new searches.

    **Why two steps:** Google now routes result links through google.com/goto. DataForSEO's
    Standard queue turns those back into real URLs, but a search can take 25 minutes or
    more to finish. So you send the searches, keep the job file, and collect the results later.

    **Where to get ranking data for CSV upload:**
    - SERP API exports (DataForSEO, etc.)
    - Rank tracking tools (Ahrefs, SEMrush, Sistrix)
    - Custom crawl data

    **CTR Curves:**
    - Position 1: ~28-35% of clicks
    - Position 2: ~15-17% of clicks
    - Position 3: ~10-12% of clicks
    - And so on...
    """)

# Location code mapping
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
    "Ireland": 2372,
}

# CTR curve presets
CTR_CURVES = {
    "Sistrix (2020)": {
        1: 0.2848, 2: 0.1548, 3: 0.1100, 4: 0.0765, 5: 0.0535,
        6: 0.0491, 7: 0.0306, 8: 0.0313, 9: 0.0279, 10: 0.0274
    },
    "Advanced Web Ranking (Desktop)": {
        1: 0.3100, 2: 0.1560, 3: 0.0990, 4: 0.0700, 5: 0.0512,
        6: 0.0390, 7: 0.0305, 8: 0.0248, 9: 0.0209, 10: 0.0182
    },
    "Backlinko (2023)": {
        1: 0.2750, 2: 0.1510, 3: 0.1120, 4: 0.0810, 5: 0.0740,
        6: 0.0510, 7: 0.0410, 8: 0.0330, 9: 0.0290, 10: 0.0260
    },
    "Conservative Estimate": {
        1: 0.2000, 2: 0.1000, 3: 0.0800, 4: 0.0600, 5: 0.0500,
        6: 0.0400, 7: 0.0350, 8: 0.0300, 9: 0.0250, 10: 0.0200
    }
}

# Sidebar settings
st.sidebar.header("CTR Curve Settings")

ctr_preset = st.sidebar.selectbox(
    "CTR Curve Preset",
    list(CTR_CURVES.keys()),
    help="Select a CTR curve based on industry research"
)

selected_ctr = CTR_CURVES[ctr_preset].copy()

# Custom CTR adjustment
st.sidebar.markdown("---")
st.sidebar.subheader("Custom CTR Values")
use_custom = st.sidebar.checkbox("Customise CTR values")

if use_custom:
    for pos in range(1, 11):
        selected_ctr[pos] = st.sidebar.number_input(
            f"Position {pos} CTR",
            min_value=0.0,
            max_value=1.0,
            value=selected_ctr[pos],
            format="%.4f",
            key=f"ctr_{pos}"
        )

st.sidebar.markdown("---")
st.sidebar.header("Analysis Settings")

top_n = st.sidebar.slider(
    "Top N domains to show",
    min_value=5,
    max_value=100,
    value=20
)

group_by_category = st.sidebar.checkbox(
    "Group by category",
    value=False,
    help="Show SOV within each category"
)

# DataForSEO settings in sidebar
st.sidebar.markdown("---")
st.sidebar.header("DataForSEO Settings")

dataforseo_login = st.sidebar.text_input(
    "DataForSEO Login",
    type="password",
    value=os.environ.get('DATAFORSEO_LOGIN', ''),
    help="Your login from dataforseo.com (only needed for fetching SERPs)"
)

dataforseo_password = st.sidebar.text_input(
    "DataForSEO Password",
    type="password",
    value=os.environ.get('DATAFORSEO_PASSWORD', ''),
    help="Your password from dataforseo.com (only needed for fetching SERPs)"
)

has_credentials = bool(dataforseo_login and dataforseo_password)


def parse_keyword_volumes(text):
    """Parse 'keyword, volume' lines. Returns (pairs, problems); repeated keywords keep the first line."""
    pairs, problems, seen = [], [], set()
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2:
            try:
                kw = parts[0]
                vol = int(parts[1])
            except ValueError:
                problems.append(f"Could not parse volume for line: '{line}'. Expected format: keyword, volume")
                continue
            if kw in seen:
                problems.append(f"'{kw}' is listed more than once; the first volume is used.")
                continue
            seen.add(kw)
            pairs.append((kw, vol))
        else:
            problems.append(f"Could not parse line: '{line}'. Expected format: keyword, volume")
    return pairs, problems


def sov_job_to_csv(job_rows):
    """Job file text (CSV): the standard job columns plus each keyword's search volume."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=SOV_JOB_COLUMNS)
    writer.writeheader()
    writer.writerows(job_rows)
    return buffer.getvalue()


def job_volumes(job_rows):
    """{keyword: volume} from a job. Raises ValueError if the job has no usable volumes."""
    volumes = {}
    for row in job_rows:
        try:
            volumes[row['keyword']] = int(float(row.get('volume')))
        except (TypeError, ValueError):
            raise ValueError("This job file has no search volume for "
                             f"'{row['keyword']}'. Upload the job file this tool gave you "
                             "when the searches were sent.")
    return volumes


def serp_rows(ready, volumes):
    """Organic results from collected searches as SOV rows.

    ready is {keyword: items} from collect_results(). Returns (rows, unresolved), where
    unresolved counts organic results that came back as google.com/goto links; those are
    left out, as they cannot be credited to a domain.
    """
    rows, unresolved = [], 0
    for kw, items in ready.items():
        for item in organic_items(items):
            url = item.get("url") or ""
            if is_unresolved(url):
                unresolved += 1
                continue
            rows.append({
                'keyword': kw,
                'volume': volumes.get(kw),
                'position': item.get("rank_group", 0),
                'domain': item.get("domain", ""),
                'url': url,
                'title': item.get("title", ""),
            })
    return rows, unresolved


def find_column(df, possible_names):
    """Find a column by trying multiple possible names."""
    for name in possible_names:
        for col in df.columns:
            if name.lower() == col.lower() or name.lower() in col.lower():
                return col
    return None


def calculate_and_display_sov(df_work, keyword_col, volume_col, position_col, domain_col,
                               category_col, selected_ctr_curve, top_n_domains, do_group_by_category):
    """Run the SOV calculation and display results. Shared by both input methods."""
    # Ensure numeric types
    df_work[volume_col] = pd.to_numeric(df_work[volume_col], errors='coerce')
    df_work[position_col] = pd.to_numeric(df_work[position_col], errors='coerce')

    # Filter to top 10 positions only
    df_work = df_work[(df_work[position_col] >= 1) & (df_work[position_col] <= 10)]
    df_work[position_col] = df_work[position_col].astype(int)

    # Apply CTR curve
    df_work['ctr'] = df_work[position_col].map(selected_ctr_curve)

    # Calculate estimated traffic
    df_work['estimated_traffic'] = (df_work['ctr'] * df_work[volume_col]).round(0)

    # Display CTR curve used
    st.subheader("CTR Curve Applied")
    ctr_df = pd.DataFrame({
        'Position': list(selected_ctr_curve.keys()),
        'CTR (%)': [f"{v*100:.2f}%" for v in selected_ctr_curve.values()]
    })
    st.dataframe(ctr_df.T, width="stretch")

    if do_group_by_category and category_col and category_col in df_work.columns:
        # Group by category and domain
        grouped = df_work.groupby([category_col, domain_col]).agg({
            'estimated_traffic': 'sum',
            keyword_col: 'count'
        }).reset_index()
        grouped.columns = [category_col, 'Domain', 'Estimated Traffic', 'Keywords']

        # Calculate SOV within each category
        category_totals = grouped.groupby(category_col)['Estimated Traffic'].transform('sum')
        grouped['SOV (%)'] = (grouped['Estimated Traffic'] / category_totals * 100).round(2)

        # Sort within categories
        grouped = grouped.sort_values([category_col, 'Estimated Traffic'], ascending=[True, False])
        grouped = grouped.groupby(category_col).head(top_n_domains)

        st.subheader("Share of Voice by Category")

        # Show each category
        for category in grouped[category_col].unique():
            st.markdown(f"### {category}")
            cat_data = grouped[grouped[category_col] == category]
            st.dataframe(cat_data, width="stretch")
            st.bar_chart(cat_data.set_index('Domain')['SOV (%)'].head(10))

    else:
        # Group by domain only
        grouped = df_work.groupby(domain_col).agg({
            'estimated_traffic': 'sum',
            keyword_col: 'count'
        }).reset_index()
        grouped.columns = ['Domain', 'Estimated Traffic', 'Keywords']

        # Calculate total for SOV
        total_traffic = grouped['Estimated Traffic'].sum()
        grouped['SOV (%)'] = (grouped['Estimated Traffic'] / total_traffic * 100).round(2)

        # Sort and limit
        grouped = grouped.sort_values('Estimated Traffic', ascending=False)
        grouped = grouped.head(top_n_domains)

        # Display results
        st.subheader("Share of Voice Results")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Domains Analysed", f"{len(grouped):,}")
        with col2:
            st.metric("Total Est. Traffic", f"{int(total_traffic):,}")
        with col3:
            unique_kws = df_work[keyword_col].nunique()
            st.metric("Keywords Tracked", f"{unique_kws:,}")
        with col4:
            leader_sov = grouped.iloc[0]['SOV (%)'] if len(grouped) > 0 else 0
            st.metric("Leader SOV", f"{leader_sov}%")

        # Bar chart
        st.subheader("SOV Distribution")
        chart_data = grouped.head(15).set_index('Domain')['SOV (%)']
        st.bar_chart(chart_data)

        # Full table
        st.subheader("Detailed Results")
        grouped.index = range(1, len(grouped) + 1)
        st.dataframe(grouped, width="stretch")

    # Download
    st.subheader("Download")

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        grouped.to_excel(writer, sheet_name='Share of Voice', index=True)

        # Add raw data sheet
        df_work.to_excel(writer, sheet_name='Raw Data', index=False)

    st.download_button(
        label="Download Excel Report",
        data=output.getvalue(),
        file_name="share_of_voice_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# Input method selection
st.subheader("Input Method")
input_method = st.radio(
    "Choose how to provide ranking data:",
    ["Upload CSV", "Fetch SERPs (DataForSEO)"],
    horizontal=True
)

if input_method == "Upload CSV":
    # ---- CSV Upload path (unchanged logic) ----
    st.subheader("Upload Ranking Data")
    ranking_file = st.file_uploader(
        "Upload CSV with ranking data",
        type=['csv'],
        help="CSV with keyword, volume, position, domain columns"
    )

    if ranking_file is not None:
        try:
            # Load data
            try:
                df = pd.read_csv(ranking_file, encoding='utf-8')
            except Exception:
                ranking_file.seek(0)
                df = pd.read_csv(ranking_file, encoding='latin-1')

            st.success(f"Loaded {len(df):,} rows")

            # Find columns
            keyword_col = find_column(df, ['keyword', 'query', 'search term'])
            volume_col = find_column(df, ['volume', 'search volume', 'sv', 'monthly volume'])
            position_col = find_column(df, ['position', 'rank', 'ranking', 'pos'])
            domain_col = find_column(df, ['domain', 'url', 'site', 'website'])
            category_col = find_column(df, ['category', 'group', 'vertical', 'topic'])

            with st.expander("Column Mapping"):
                col1, col2 = st.columns(2)
                with col1:
                    keyword_col = st.selectbox("Keyword column", df.columns.tolist(),
                                               index=df.columns.tolist().index(keyword_col) if keyword_col else 0)
                    volume_col = st.selectbox("Volume column", df.columns.tolist(),
                                              index=df.columns.tolist().index(volume_col) if volume_col else 0)
                with col2:
                    position_col = st.selectbox("Position column", df.columns.tolist(),
                                                index=df.columns.tolist().index(position_col) if position_col else 0)
                    domain_col = st.selectbox("Domain column", df.columns.tolist(),
                                              index=df.columns.tolist().index(domain_col) if domain_col else 0)

                if group_by_category:
                    category_col = st.selectbox("Category column", df.columns.tolist(),
                                                index=df.columns.tolist().index(category_col) if category_col else 0)

            with st.expander("Preview data"):
                st.dataframe(df.head(20))

            if st.button("Calculate Share of Voice", type="primary"):
                with st.spinner("Calculating SOV..."):
                    calculate_and_display_sov(
                        df.copy(), keyword_col, volume_col, position_col, domain_col,
                        category_col, selected_ctr, top_n, group_by_category
                    )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    else:
        st.info("Upload a CSV with ranking data to begin")

        st.subheader("Required Data Format")
        example_data = {
            "keyword": ["seo tools", "seo tools", "seo tools", "keyword research"],
            "volume": [12100, 12100, 12100, 8100],
            "position": [1, 2, 3, 1],
            "domain": ["ahrefs.com", "semrush.com", "moz.com", "ahrefs.com"],
            "category": ["SEO Software", "SEO Software", "SEO Software", "SEO Software"]
        }
        st.dataframe(pd.DataFrame(example_data))

        st.subheader("Example Output")
        output_example = {
            "Domain": ["ahrefs.com", "semrush.com", "moz.com"],
            "Estimated Traffic": [5752, 1873, 1331],
            "Keywords": [2, 1, 1],
            "SOV (%)": [64.2, 20.9, 14.9]
        }
        st.dataframe(pd.DataFrame(output_example))

else:
    # ---- DataForSEO path: send searches now, collect results later ----
    st.subheader("Fetch SERPs via DataForSEO")

    if not has_credentials:
        st.warning("Enter your DataForSEO login and password in the sidebar to fetch SERPs.")

    st.markdown(
        "Google results come from DataForSEO's Standard queue, which can take **25 minutes or more** "
        "per search. Send the searches in step 1, keep the job file, then collect the results "
        "and calculate SOV in step 2."
    )

    send_tab, collect_tab = st.tabs(["1. Send searches", "2. Collect results"])

    with send_tab:
        col1, col2 = st.columns(2)
        with col1:
            live_location = st.selectbox(
                "Location",
                list(LOCATION_CODES.keys()),
                index=0,
                help="Location for the Google searches"
            )
        with col2:
            live_device = st.selectbox(
                "Device",
                ["desktop", "mobile"],
                index=0,
                help="Device type for the Google searches"
            )

        keyword_volume_input = st.text_area(
            "Keywords and volumes (format: keyword, volume - one per line)",
            height=200,
            placeholder="seo tools, 12100\nkeyword research, 8100\nbacklink checker, 6600"
        )

        # Parse input
        keyword_volume_pairs = []
        if keyword_volume_input:
            keyword_volume_pairs, parse_problems = parse_keyword_volumes(keyword_volume_input)
            for problem in parse_problems:
                st.warning(problem)

        if keyword_volume_pairs:
            num_keywords = len(keyword_volume_pairs)
            est_cost = estimate_cost(num_keywords, SERP_DEPTH)
            st.info(f"{num_keywords} keyword(s) to send at {SERP_DEPTH} results each. "
                    f"Estimated DataForSEO cost: ${est_cost:.4f}")

        if st.button("Send searches", type="primary",
                     disabled=not has_credentials or not keyword_volume_pairs):
            volumes = dict(keyword_volume_pairs)
            try:
                with st.spinner("Sending searches to DataForSEO..."):
                    job_rows, send_warnings, send_cost = send_searches(
                        [kw for kw, _ in keyword_volume_pairs], dataforseo_login, dataforseo_password,
                        LOCATION_CODES[live_location], "en", device=live_device, depth=SERP_DEPTH)
            except QueueError as e:
                st.error(str(e))
            else:
                for warning in send_warnings:
                    st.warning(warning)
                if job_rows:
                    for row in job_rows:
                        row['volume'] = volumes[row['keyword']]
                    st.session_state['sov_job'] = job_rows
                    st.session_state['sov_send_cost'] = send_cost
                    st.session_state.pop('sov_result', None)

        if st.session_state.get('sov_job'):
            job_rows = st.session_state['sov_job']
            st.success(f"Queued {len(job_rows)} search(es). Cost reported by DataForSEO: "
                       f"${st.session_state['sov_send_cost']:.4f}")
            st.markdown("DataForSEO can take **25 minutes or more** to finish. Download the job file, "
                        "then use **2. Collect results**. While this page stays open, the job is kept "
                        "here too, so you can collect without uploading it.")
            st.download_button(
                label="Download job file (CSV)",
                data=sov_job_to_csv(job_rows).encode('utf-8'),
                file_name="share_of_voice_job.csv",
                mime="text/csv"
            )

    with collect_tab:
        job_rows = st.session_state.get('sov_job')
        job_file = st.file_uploader("Job file from step 1", type=['csv'],
                                    help="The share_of_voice_job.csv you downloaded when you sent the searches")
        if job_file is not None:
            try:
                job_rows = job_from_csv(job_file.getvalue().decode('utf-8-sig', errors='replace'))
            except ValueError as e:
                st.error(str(e))
                job_rows = None
        elif job_rows:
            st.caption(f"Using the job sent from this page ({len(job_rows)} search(es)).")

        job_volume_map = None
        if job_rows:
            try:
                job_volume_map = job_volumes(job_rows)
            except ValueError as e:
                st.error(str(e))

        if st.button("Collect results", type="primary",
                     disabled=not has_credentials or not job_volume_map):
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
                status_text.empty()
                all_rows, unresolved = serp_rows(result['ready'], job_volume_map)
                st.session_state['sov_result'] = {
                    'rows': all_rows,
                    'ready': len(result['ready']),
                    'pending': result['pending'],
                    'errors': result['errors'],
                    'unresolved': unresolved,
                    'total': len(job_rows),
                }

        report = st.session_state.get('sov_result')
        if report:
            if report['pending']:
                st.warning(f"{report['ready']} of {report['total']} search(es) have finished; "
                           f"{len(report['pending'])} are still running at DataForSEO. "
                           "Try again in a few minutes. The results below cover the finished ones only.")
            if report['errors']:
                with st.expander(f"{len(report['errors'])} search(es) failed"):
                    for error in report['errors']:
                        st.text(error)
            if report['unresolved']:
                st.warning(f"{report['unresolved']} result(s) came back as unresolved google.com/goto links. "
                           "They are left out of the SOV calculation, as they cannot be credited to a domain.")

            if report['rows']:
                df_live = pd.DataFrame(report['rows'])
                st.success(f"Collected {len(df_live):,} organic results across {report['ready']} keywords")

                with st.expander("Preview fetched data"):
                    st.dataframe(df_live.head(30))

                calculate_and_display_sov(
                    df_live.copy(), 'keyword', 'volume', 'position', 'domain',
                    None, selected_ctr, top_n, False
                )
            elif report['ready']:
                st.warning("No organic results came back for the finished searches.")
