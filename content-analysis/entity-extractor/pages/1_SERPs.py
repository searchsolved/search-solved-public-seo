# Author: Lee Foot
# Website: https://leefoot.com

####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://www.leefoot.com                                               #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                                              #
####################################################################################

import os

import altair as alt
import pandas as pd
import streamlit as st
import trafilatura
from dandelion import DataTXT
from stqdm import stqdm
from trafilatura import extract
from trafilatura.settings import use_config

# dataforseo_queue.py sits next to Home.py; Streamlit puts that folder on the import path
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

st.set_page_config(page_title="SERP Entity Extractor", page_icon="mag",
                   layout="wide")

newconfig = use_config()
newconfig.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")

# Region name -> (DataForSEO location_code, language_code)
LOCATIONS = {
    "United Kingdom": (2826, "en"),
    "United States": (2840, "en"),
    "Australia": (2036, "en"),
    "Brazil": (2076, "pt"),
    "France": (2250, "fr"),
    "Canada": (2124, "en"),
    "Germany": (2276, "de"),
    "Italy": (2380, "it"),
    "Spain": (2724, "es"),
    "New Zealand": (2554, "en"),
}


def result_urls(items):
    """Organic result titles and URLs, leaving out unresolved google.com/goto links.

    Returns (serp_title, link, unresolved_count).
    """
    serp_title, link, unresolved = [], [], 0
    for var in organic_items(items):
        if not var.get('url'):
            continue
        if is_unresolved(var['url']):
            unresolved += 1
            continue
        serp_title.append(var.get('title') or '')
        link.append(var['url'])
    return serp_title, link, unresolved


def decode_job_file(raw):
    """Job file bytes as text."""
    for encoding in ('utf-8-sig', 'cp1252', 'latin-1'):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', errors='replace')


st.title("SERP Entity Extractor")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Extracts named entities from the pages ranking for a keyword, using the Dandelion.eu API
    - Shows the most frequent entities across those pages in a chart and a table

    **Why two steps:** Google now routes result links through google.com/goto. DataForSEO's
    Standard queue turns those back into real URLs, but a search can take 25 minutes or
    more to finish. So you send the search, keep the job file, and collect the results later.

    **Requirements:**
    - DataForSEO account (sign up at [dataforseo.com](https://dataforseo.com/), pay as you go)
    - Dandelion.eu API key (dandelion.eu, 1,000 free requests per day)

    **How to use:**
    1. Enter your DataForSEO login and password and your Dandelion.eu key in the sidebar
    2. **Send search:** choose the region, device and number of results pages to analyse,
       enter a seed keyword, check the estimated cost and click "Send search". Download the job file.
    3. **Collect results** (about 30 minutes later): upload the job file, set the entity accuracy
       and number of entities to display, and click "Collect results"
    4. Identify entity gaps vs competitors

    **Best for:**
    - SERP entity analysis
    - Content gap identification
    - Topical authority research
    """)

dataforseo_login = st.sidebar.text_input(
    'DataForSEO Login',
    type='password',
    value=os.environ.get('DATAFORSEO_LOGIN', ''),
    help="Your login from dataforseo.com"
)

dataforseo_password = st.sidebar.text_input(
    'DataForSEO Password',
    type='password',
    value=os.environ.get('DATAFORSEO_PASSWORD', ''),
    help="Your password from dataforseo.com"
)

dandelion_key = st.sidebar.text_input('Input your Dandelion.eu API Key')

has_credentials = bool(dataforseo_login and dataforseo_password)
if not has_credentials:
    st.warning("Enter your DataForSEO login and password in the sidebar to get started")

send_tab, collect_tab = st.tabs(["1. Send search", "2. Collect results"])

with send_tab:
    # streamlit variables
    q = st.text_input('Input Your Search Keyword')

    col1, col2, col3 = st.columns(3)
    with col1:
        location_select = st.selectbox(
            "Select The Region To Search Google From",
            list(LOCATIONS.keys()),
        )
    with col2:
        device_select = st.selectbox(
            "Select The Host Device To Use To Search Google",
            (
                "Desktop",
                "Mobile",
            ),
            help="Tablet is no longer offered: DataForSEO only supports desktop and mobile results.",
        )
    with col3:
        num_pages = st.slider("Set Number of Pages to Analyse", min_value=1, max_value=10, value=1,
                              help="Each page is 10 Google results. DataForSEO bills per 10 results.")

    depth = num_pages * 10

    # Show estimated cost (DataForSEO Standard queue, billed per 10 results)
    st.caption(f"Estimated DataForSEO cost for this search: ${estimate_cost(1, depth):.4f}")

    with st.form(key='columns_in_form_2'):
        submitted = st.form_submit_button('Send search', type="primary")

    if submitted:
        if not dataforseo_login or not dataforseo_password:
            st.error("Please enter your DataForSEO login and password in the sidebar")
        elif not q.strip():
            st.info("No search sent. Did you enter a keyword to search?")
        else:
            location_code, language_code = LOCATIONS[location_select]
            try:
                with st.spinner('Sending search to DataForSEO...'):
                    job_rows, send_warnings, send_cost = send_searches(
                        [q.strip()], dataforseo_login, dataforseo_password, location_code, language_code,
                        device=device_select.lower(), depth=depth)
            except QueueError as e:
                st.error(str(e))
            else:
                for warning in send_warnings:
                    st.warning(warning)
                if job_rows:
                    st.session_state['entity_serp_job'] = job_rows
                    st.session_state['entity_serp_send_cost'] = send_cost

    if st.session_state.get('entity_serp_job'):
        job_rows = st.session_state['entity_serp_job']
        st.success(f"Queued the search for: {job_rows[0]['keyword']}. Cost reported by DataForSEO: "
                   f"${st.session_state['entity_serp_send_cost']:.4f}")
        st.markdown("DataForSEO can take **25 minutes or more** to finish. Download the job file, "
                    "then use **2. Collect results**. While this page stays open, the job is kept "
                    "here too, so you can collect without uploading it.")
        st.download_button(
            label="Download job file (CSV)",
            data=job_to_csv(job_rows).encode('utf-8'),
            file_name="serp_entity_job.csv",
            mime="text/csv"
        )

with collect_tab:
    job_rows = st.session_state.get('entity_serp_job')
    job_file = st.file_uploader("Job file from step 1", type=['csv'],
                                help="The serp_entity_job.csv you downloaded when you sent the search")
    if job_file is not None:
        try:
            job_rows = job_from_csv(decode_job_file(job_file.getvalue()))
        except ValueError as e:
            st.error(str(e))
            job_rows = None
    elif job_rows:
        st.caption(f"Using the search sent from this page: {job_rows[0]['keyword']}")

    if job_rows and len(job_rows) > 1:
        st.error(f"This page analyses one keyword per job, but this job file has {len(job_rows)} searches. "
                 "Upload the job file from this page's step 1.")
        job_rows = None

    col1, col2 = st.columns(2)
    with col1:
        accuracy_pct = st.slider("Set Entity Accuracy", min_value=10, max_value=100, value=80)
    with col2:
        top_entities = st.slider("Set The Number of Entities to Display", min_value=5, max_value=25, value=15)

    accuracy = accuracy_pct / 100

    collected = st.button("Collect results", type="primary", disabled=not has_credentials or not job_rows)

    if collected:
        try:
            datatxt = DataTXT(token=dandelion_key)
        except Exception:
            st.info("Missing Dandelion.eu API key! Please visit www.dandelion.eu to get your free API key")
            st.stop()

        q = job_rows[0]['keyword']
        try:
            with st.spinner('Checking DataForSEO for the results...'):
                result = collect_results(job_rows, dataforseo_login, dataforseo_password)
        except QueueError as e:
            st.error(str(e))
            st.stop()

        report = {'query': q, 'pending': result['pending'], 'errors': result['errors'],
                  'ready': q in result['ready'], 'unresolved': 0, 'urls': 0, 'entities': None}

        if report['ready']:
            # store the SERP Data
            serp_title, link, report['unresolved'] = result_urls(result['ready'][q])
            report['urls'] = len(link)

            if link:
                st.write("Analysing the Google results for: %s" % q)

                # store the entity data
                entity = []
                confidence = []
                title = []
                wiki_url = []
                categories = []
                url_list = []

                # make the serp df
                df = pd.DataFrame(None)
                df['Keyword'] = [q] * len(link)
                df['Current URL'] = link
                df['title'] = serp_title

                # get the entities
                df['Keyword'] = df['Keyword'].astype(str)
                unique_urls = list(set(df['Current URL']))

                for url in stqdm(unique_urls):

                    try:
                        downloaded = trafilatura.fetch_url(url)
                        txt = extract(downloaded, config=newconfig)

                        response = datatxt.nex(txt)

                        for annotation in response.annotations:
                            entity.append(annotation['spot'])
                            confidence.append(annotation['confidence'])
                            title.append(annotation['title'])
                            wiki_url.append(annotation['uri'])
                            categories.append(annotation['label'])
                            url_list.append(str(url))
                    except Exception:
                        pass

                df = pd.DataFrame(None)
                df['url'] = url_list
                try:
                    df['entity'] = entity
                except Exception:
                    df = None
                if df is not None:
                    df['confidence'] = confidence
                    df['entity_title'] = title
                    df['category'] = categories
                    df['wiki_url'] = wiki_url

                    # drop duplicates
                    df['entity'] = df['entity'].str.lower()
                    df['entity_title'] = df['entity_title'].str.lower()
                    df['category'] = df['category'].str.lower()
                    df.drop_duplicates(subset=["url", "entity"], keep="first", inplace=True)

                    df['most_frequent'] = df['entity'].map(df.groupby('entity')['entity'].count())
                    df.sort_values(["most_frequent", "entity"], ascending=[False, True], inplace=True)
                report['entities'] = df

        st.session_state['entity_serp_result'] = report

    report = st.session_state.get('entity_serp_result')
    if report:
        if report['pending']:
            st.warning("The search for **%s** is still running at DataForSEO. "
                       "Try again in a few minutes." % report['query'])
        if report['errors']:
            with st.expander(f"{len(report['errors'])} search(es) failed"):
                for error in report['errors']:
                    st.text(error)
        if report['unresolved']:
            st.warning(f"{report['unresolved']} result(s) came back as unresolved google.com/goto links "
                       "and were skipped.")

        if report['ready'] and not report['urls']:
            st.info("No organic results found for this keyword.")
        elif report['ready'] and report['entities'] is None:
            st.info("No Entities Were Found! Please increase the Number of Pages to Search, or adjust your search term")
        elif report['ready']:
            q = report['query']
            df = report['entities'].copy()

            df_chart = df[["entity", "most_frequent", "confidence"]].copy()
            df_chart = df_chart.sort_values(by="most_frequent", ascending=False)
            df_chart.drop_duplicates(subset="entity", inplace=True)
            df_chart = df_chart[df_chart.confidence >= accuracy]
            df_chart = df_chart.iloc[:top_entities]
            df_chart.drop_duplicates(subset=['entity'], keep="first", inplace=True)

            st.subheader("Top Entities")
            chart = (
                alt.Chart(df_chart)
                    .mark_bar()
                    .encode(
                    alt.X("entity:O", sort=alt.EncodingSortField(field="entity", op="count", order='ascending')),
                    alt.Y("most_frequent"),
                    alt.Color("entity:O", sort=alt.SortField('count', order='descending')),
                    alt.Tooltip(["entity", "most_frequent"]),
                )
                    .properties(width=800, height=400)
                    .interactive()
            )
            st.altair_chart(chart)

            df["query"] = q
            df = df[["query", "entity", "confidence", "entity_title", "category", "wiki_url", "most_frequent"]]
            df = df[df.confidence >= accuracy]
            df = df.sort_values(by="confidence", ascending=False)
            df.sort_values(["most_frequent", "confidence"], ascending=[False, False], inplace=True)
            df.drop_duplicates(subset="entity", inplace=True)

            def convert_df(df):
                return df.to_csv(index=False).encode("utf-8")

            # download results
            st.write("")
            csv = convert_df(df)
            st.download_button(
                label="Download The Data!",
                data=csv,
                file_name="related_serp_keywords.csv",
                mime="text/csv",
            )

            st.subheader("Related Keywords Table")
            st.dataframe(df, width="stretch", hide_index=True)
