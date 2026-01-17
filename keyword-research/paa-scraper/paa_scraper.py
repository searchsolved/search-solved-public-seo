"""
People Also Ask (PAA) Scraper - Streamlit App
Recursively extracts PAA questions from search results using ValueSERP API.

Author: Lee Foot
Website: https://leefoot.com
"""

import streamlit as st
import pandas as pd
import requests
import json
import time
from datetime import datetime
import io

st.set_page_config(
    page_title="People Also Ask Scraper",
    page_icon="❓",
    layout="wide"
)

st.title("❓ People Also Ask (PAA) Scraper")
st.markdown("Extract 'People Also Ask' questions recursively using the ValueSERP API.")

# Sidebar configuration
with st.sidebar:
    st.header("API Configuration")

    api_key = st.text_input(
        "ValueSERP API Key",
        type="password",
        help="Get your API key from https://www.valueserp.com/"
    )

    st.markdown("---")
    st.header("Search Settings")

    # Location settings
    location = st.selectbox(
        "Location",
        options=[
            "United States", "United Kingdom", "Canada", "Australia",
            "Germany", "France", "Spain", "Italy", "Netherlands",
            "Brazil", "Mexico", "India", "Japan"
        ],
        index=0,
        help="Location for search results"
    )

    country_codes = {
        "United States": ("us", "google.com"),
        "United Kingdom": ("uk", "google.co.uk"),
        "Canada": ("ca", "google.ca"),
        "Australia": ("au", "google.com.au"),
        "Germany": ("de", "google.de"),
        "France": ("fr", "google.fr"),
        "Spain": ("es", "google.es"),
        "Italy": ("it", "google.it"),
        "Netherlands": ("nl", "google.nl"),
        "Brazil": ("br", "google.com.br"),
        "Mexico": ("mx", "google.com.mx"),
        "India": ("in", "google.co.in"),
        "Japan": ("jp", "google.co.jp")
    }

    language = st.selectbox(
        "Language",
        options=["en", "de", "fr", "es", "it", "nl", "pt", "ja"],
        index=0,
        help="Language for search results"
    )

    device = st.selectbox(
        "Device",
        options=["Desktop", "Mobile", "Tablet"],
        index=0
    )

    st.markdown("---")
    st.header("Scrape Settings")

    max_depth = st.slider(
        "Max Depth",
        min_value=1,
        max_value=5,
        value=2,
        help="How many levels deep to follow PAA questions"
    )

    request_delay = st.slider(
        "Request Delay (seconds)",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.5,
        help="Delay between API requests"
    )


def get_related_questions(query, api_key, location, country_code, google_domain,
                         language, device, max_depth, delay,
                         level=1, all_questions=None, parent=None,
                         original_query=None, progress_callback=None):
    """Recursively fetch related questions from ValueSERP API."""
    if all_questions is None:
        all_questions = []
        original_query = query

    if level > max_depth:
        return all_questions

    if progress_callback:
        progress_callback(f"Level {level}: Querying '{query[:50]}...'")

    params = {
        'api_key': api_key,
        'q': query,
        'gl': country_code,
        'hl': language,
        'location': location,
        'google_domain': google_domain,
        'device': device.lower(),
        'output': 'json',
        'page': '1',
        'num': '10',
        'include_fields': 'related_questions'
    }

    try:
        response = requests.get('https://api.valueserp.com/search', params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        questions = data.get('related_questions', [])

        if not questions:
            return all_questions

        for q in questions:
            question_text = q.get('question', '')

            if not question_text:
                continue

            question_data = {
                'original_query': original_query,
                'level': level,
                'parent_query': parent if parent else query,
                'question': question_text,
                'answer_snippet': q.get('answer', {}).get('text', '') if isinstance(q.get('answer'), dict) else '',
                'source_url': q.get('answer', {}).get('link', '') if isinstance(q.get('answer'), dict) else '',
                'source_title': q.get('answer', {}).get('title', '') if isinstance(q.get('answer'), dict) else ''
            }

            # Check for duplicates
            if not any(d.get('question') == question_text for d in all_questions):
                all_questions.append(question_data)

                # Recursively query next level
                if level < max_depth:
                    time.sleep(delay)
                    get_related_questions(
                        question_text,
                        api_key, location, country_code, google_domain,
                        language, device, max_depth, delay,
                        level=level + 1,
                        all_questions=all_questions,
                        parent=question_text,
                        original_query=original_query,
                        progress_callback=progress_callback
                    )

    except requests.exceptions.RequestException as e:
        if progress_callback:
            progress_callback(f"Error: {str(e)}")

    return all_questions


# Main app
st.subheader("Enter Keywords")

keywords_input = st.text_area(
    "Keywords (one per line)",
    height=150,
    placeholder="Enter your seed keywords, one per line:\n\nwhat is SEO\nhow to rank on Google\nbest keyword research tools"
)

col1, col2 = st.columns([1, 3])
with col1:
    run_button = st.button("Extract PAA Questions", type="primary", disabled=not api_key)

if not api_key:
    st.warning("Please enter your ValueSERP API key in the sidebar.")

if run_button and keywords_input and api_key:
    keywords = [k.strip() for k in keywords_input.strip().split('\n') if k.strip()]

    if not keywords:
        st.error("Please enter at least one keyword.")
    else:
        country_code, google_domain = country_codes[location]

        all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(message):
            status_text.text(message)

        for i, keyword in enumerate(keywords):
            update_progress(f"Processing keyword {i+1}/{len(keywords)}: {keyword}")

            results = get_related_questions(
                keyword,
                api_key,
                location,
                country_code,
                google_domain,
                language,
                device,
                max_depth,
                request_delay,
                progress_callback=update_progress
            )

            all_results.extend(results)
            progress_bar.progress((i + 1) / len(keywords))

        progress_bar.progress(100)

        if all_results:
            df = pd.DataFrame(all_results)

            # Reorder columns
            columns = ['original_query', 'level', 'parent_query', 'question',
                      'answer_snippet', 'source_url', 'source_title']
            columns = [c for c in columns if c in df.columns]
            df = df[columns]

            st.success(f"Found {len(df):,} unique PAA questions!")

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Questions", f"{len(df):,}")
            with col2:
                st.metric("Seed Keywords", len(keywords))
            with col3:
                level_counts = df['level'].value_counts().sort_index()
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

# Help section
with st.expander("How to use this tool"):
    st.markdown("""
    ### Getting Started

    1. **Get a ValueSERP API key** from [valueserp.com](https://www.valueserp.com/)
    2. **Enter your API key** in the sidebar
    3. **Configure your search settings** (location, language, device)
    4. **Enter your seed keywords** (one per line)
    5. **Click "Extract PAA Questions"**

    ### Understanding the Results

    - **original_query**: Your seed keyword
    - **level**: Depth level (1 = direct PAA, 2 = PAA of PAA, etc.)
    - **parent_query**: The query that triggered this question
    - **question**: The PAA question text
    - **answer_snippet**: The snippet answer shown in search results
    - **source_url**: The URL of the answer source
    - **source_title**: The title of the answer source

    ### Use Cases

    - Build comprehensive FAQ pages
    - Discover content topic ideas
    - Understand user search intent
    - Find featured snippet opportunities
    - Expand keyword research
    """)

# Footer
st.markdown("---")
st.markdown("Built by [Lee Foot](https://leefoot.com) | [GitHub](https://github.com/searchsolved/search-solved-public-seo)")
