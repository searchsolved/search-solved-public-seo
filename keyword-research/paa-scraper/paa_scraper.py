# Author: Lee Foot
# Website: https://leefoot.com
"""
People Also Ask (PAA) Scraper - Streamlit App
Extracts PAA questions from search results using the DataForSEO SERP API.
DataForSEO handles recursive PAA expansion server-side via people_also_ask_click_depth,
so only one API call is needed per seed keyword.
"""

import streamlit as st
import pandas as pd
import requests
import time
import os
from base64 import b64encode
from datetime import datetime
import io

st.set_page_config(
    page_title="People Also Ask Scraper",
    page_icon="❓",
    layout="wide"
)

st.title("❓ People Also Ask (PAA) Scraper")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")
st.markdown("Extract 'People Also Ask' questions using the DataForSEO SERP API.")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Extracts PAA questions from Google search results via DataForSEO
    - Recursively expands each PAA box server-side (up to 4 levels deep)
    - Captures answer snippets and source URLs
    - Only one API call per seed keyword (DataForSEO handles the recursive expansion)

    **How to use:**
    1. Get a DataForSEO account from [dataforseo.com](https://dataforseo.com/)
    2. Enter your login and password in the sidebar
    3. Configure search settings (location, language, device)
    4. Enter seed keywords (one per line)
    5. Click "Extract PAA Questions"

    **Output columns:**
    - **original_query**: Your seed keyword
    - **level**: Depth (1 = direct PAA, 2 = expanded PAA, etc.)
    - **parent_query**: The query or question that produced this result
    - **question**: The PAA question text
    - **answer_snippet**: The snippet answer shown in search
    - **source_url/source_title**: The answer source

    **Cost:**
    - Approximately $0.002 per keyword
    - The click depth parameter does not add extra cost per level

    **Best for:**
    - Building comprehensive FAQ pages
    - Discovering content topic ideas
    - Finding featured snippet opportunities
    - Understanding user search intent
    """)

# Location codes for DataForSEO
LOCATION_CODES = {
    "United Kingdom": 2826,
    "United States": 2840,
    "Canada": 2124,
    "Australia": 2036,
    "Germany": 2276,
    "France": 2250,
    "Spain": 2724,
    "Italy": 2380,
    "Netherlands": 2528,
    "Brazil": 2076,
    "Mexico": 2484,
    "India": 2356,
    "Japan": 2392,
}

# Sidebar configuration
with st.sidebar:
    st.header("API Configuration")

    dataforseo_login = st.text_input(
        "DataForSEO Login",
        value=os.environ.get("DATAFORSEO_LOGIN", ""),
        type="password",
        help="Your DataForSEO account login (email)"
    )

    dataforseo_password = st.text_input(
        "DataForSEO Password",
        value=os.environ.get("DATAFORSEO_PASSWORD", ""),
        type="password",
        help="Your DataForSEO account password"
    )

    has_credentials = bool(dataforseo_login and dataforseo_password)

    st.markdown("---")
    st.header("Search Settings")

    location = st.selectbox(
        "Location",
        options=list(LOCATION_CODES.keys()),
        index=0,
        help="Location for search results"
    )

    language = st.selectbox(
        "Language",
        options=["en", "de", "fr", "es", "it", "nl", "pt", "ja"],
        index=0,
        help="Language for search results"
    )

    device = st.selectbox(
        "Device",
        options=["desktop", "mobile"],
        index=0
    )

    st.markdown("---")
    st.header("Scrape Settings")

    max_depth = st.slider(
        "PAA Click Depth",
        min_value=1,
        max_value=4,
        value=2,
        help="How many levels deep to expand PAA questions (1-4). DataForSEO handles expansion server-side."
    )

    request_delay = st.slider(
        "Request Delay (seconds)",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.5,
        help="Delay between API requests for each keyword"
    )


def build_auth_header(login, password):
    """Build the Basic Auth header for DataForSEO."""
    cred = b64encode(f"{login}:{password}".encode()).decode()
    return {
        "Authorization": f"Basic {cred}",
        "Content-Type": "application/json"
    }


def extract_paa_items(items, original_query, parent_query, level, all_questions, seen_questions):
    """
    Recursively extract PAA questions from the DataForSEO response items.
    Walks through top-level items and any nested expanded_element lists.
    """
    if items is None:
        return

    for item in items:
        if not isinstance(item, dict):
            continue

        # Check for PAA container at top level
        if item.get("type") == "people_also_ask":
            paa_questions = item.get("items", [])
            for q in paa_questions:
                if not isinstance(q, dict):
                    continue
                question_text = q.get("title", "")
                if not question_text or question_text in seen_questions:
                    continue
                seen_questions.add(question_text)

                question_data = {
                    "original_query": original_query,
                    "level": level,
                    "parent_query": parent_query,
                    "question": question_text,
                    "answer_snippet": q.get("snippet", ""),
                    "source_url": q.get("url", ""),
                    "source_title": q.get("domain", ""),
                }
                all_questions.append(question_data)

                # Recurse into expanded elements (deeper PAA levels)
                expanded = q.get("expanded_element", [])
                if expanded:
                    extract_paa_items(
                        expanded, original_query, question_text,
                        level + 1, all_questions, seen_questions
                    )


def fetch_paa_for_keyword(keyword, headers, location_code, language_code, device_type,
                          click_depth, progress_callback=None):
    """
    Fetch PAA questions for a single keyword using DataForSEO.
    One API call per keyword; DataForSEO handles recursive expansion.
    """
    if progress_callback:
        progress_callback(f"Querying DataForSEO for '{keyword}'")

    payload = [{
        "keyword": keyword,
        "location_code": location_code,
        "language_code": language_code,
        "device": device_type,
        "depth": 10,
        "people_also_ask_click_depth": click_depth,
    }]

    try:
        response = requests.post(
            "https://api.dataforseo.com/v3/serp/google/organic/live/advanced",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        # Validate response
        if data.get("status_code") != 20000:
            msg = data.get("status_message", "Unknown API error")
            if progress_callback:
                progress_callback(f"API error: {msg}")
            return []

        tasks = data.get("tasks", [])
        if not tasks:
            return []

        task = tasks[0]
        if task.get("status_code") != 20000:
            msg = task.get("status_message", "Task error")
            if progress_callback:
                progress_callback(f"Task error: {msg}")
            return []

        results = task.get("result", [])
        if not results:
            return []

        items = results[0].get("items", [])
        all_questions = []
        seen_questions = set()
        extract_paa_items(items, keyword, keyword, 1, all_questions, seen_questions)
        return all_questions

    except requests.exceptions.RequestException as e:
        if progress_callback:
            progress_callback(f"Request error: {str(e)}")
        return []


# Main app
st.subheader("Enter Keywords")

keywords_input = st.text_area(
    "Keywords (one per line)",
    height=150,
    placeholder="Enter your seed keywords, one per line:\n\nwhat is SEO\nhow to rank on Google\nbest keyword research tools"
)

# Parse keywords for cost estimate
keywords_for_estimate = [k.strip() for k in keywords_input.strip().split('\n') if k.strip()] if keywords_input else []
if keywords_for_estimate:
    estimated_cost = len(keywords_for_estimate) * 0.002
    st.info(f"Estimated cost: **${estimated_cost:.3f}** for {len(keywords_for_estimate)} keyword(s) at ~$0.002 each")

col1, col2 = st.columns([1, 3])
with col1:
    run_button = st.button("Extract PAA Questions", type="primary", disabled=not has_credentials)

if not has_credentials:
    st.warning("Please enter your DataForSEO login and password in the sidebar.")

if run_button and keywords_input and has_credentials:
    keywords = [k.strip() for k in keywords_input.strip().split('\n') if k.strip()]

    if not keywords:
        st.error("Please enter at least one keyword.")
    else:
        location_code = LOCATION_CODES[location]
        headers = build_auth_header(dataforseo_login, dataforseo_password)

        all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(message):
            status_text.text(message)

        for i, keyword in enumerate(keywords):
            update_progress(f"Processing keyword {i+1}/{len(keywords)}: {keyword}")

            results = fetch_paa_for_keyword(
                keyword,
                headers,
                location_code,
                language,
                device,
                max_depth,
                progress_callback=update_progress
            )

            all_results.extend(results)
            progress_bar.progress((i + 1) / len(keywords))

            # Rate limit between keywords
            if i < len(keywords) - 1 and request_delay > 0:
                time.sleep(request_delay)

        progress_bar.progress(100)

        if all_results:
            df = pd.DataFrame(all_results)

            # Reorder columns
            columns = ['original_query', 'level', 'parent_query', 'question',
                      'answer_snippet', 'source_url', 'source_title']
            columns = [c for c in columns if c in df.columns]
            df = df[columns]

            actual_cost = len(keywords) * 0.002
            st.success(f"Found {len(df):,} unique PAA questions! Estimated API cost: ${actual_cost:.3f}")

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Questions", f"{len(df):,}")
            with col2:
                st.metric("Seed Keywords", len(keywords))
            with col3:
                st.metric("Avg Questions/Keyword", f"{len(df)/len(keywords):.1f}")
            with col4:
                st.metric("Max Depth Used", df['level'].max())

            # Questions by level
            st.subheader("Questions by Level")
            level_df = df['level'].value_counts().sort_index().reset_index()
            level_df.columns = ['Level', 'Count']
            st.bar_chart(level_df.set_index('Level'))

            # Show data
            st.subheader("All PAA Questions")
            st.dataframe(df, use_container_width=True, height=400)

            # Download options
            st.subheader("Download Results")
            col1, col2 = st.columns(2)

            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    file_name=f"paa_questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            with col2:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='PAA Questions')
                excel_data = output.getvalue()
                st.download_button(
                    "Download Excel",
                    excel_data,
                    file_name=f"paa_questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        else:
            st.warning("No PAA questions found for the entered keywords.")


# Footer
st.markdown("---")
