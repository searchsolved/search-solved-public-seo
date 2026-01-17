####################################################################################
#                                                                                  #
#  Reading Score Analyzer                                                          #
#                                                                                  #
#  Analyze content readability from XML sitemaps or CSV files.                     #
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
Reading Score Analyzer

Fetches URLs from an XML sitemap or CSV, extracts main content using Trafilatura,
and calculates various readability scores including Flesch Reading Ease.

Features:
- XML sitemap or CSV URL input
- Content extraction with Trafilatura
- Multiple readability metrics
- Progress tracking with rate limiting
- Export to CSV
"""

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from io import StringIO

st.set_page_config(page_title="Reading Score Analyzer", page_icon="📖", layout="wide")

# Check for required packages
try:
    import trafilatura
    import textstat
    from fake_useragent import UserAgent
    PACKAGES_AVAILABLE = True
except ImportError:
    PACKAGES_AVAILABLE = False

st.title("Reading Score Analyzer")
st.markdown("*Created by 🌐 [Lee Foot](https://www.leefoot.com) · [LinkedIn](https://www.linkedin.com/in/lee-foot/) · 🦋 [Bluesky](https://bsky.app/profile/leefootseo.bsky.social)*")

if not PACKAGES_AVAILABLE:
    st.error("""
    Required packages not installed. Run:
    ```
    pip install trafilatura textstat fake-useragent
    ```
    """)
    st.stop()

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Fetches URLs from an XML sitemap or CSV file
    - Extracts main content from each page using Trafilatura
    - Calculates readability scores (Flesch Reading Ease, Grade Level, etc.)

    **Readability Scores Explained:**
    - **Flesch Reading Ease**: 0-100 scale (higher = easier to read)
      - 90-100: Very easy (5th grade)
      - 80-90: Easy (6th grade)
      - 70-80: Fairly easy (7th grade)
      - 60-70: Standard (8th-9th grade)
      - 50-60: Fairly difficult (10th-12th grade)
      - 30-50: Difficult (college)
      - 0-30: Very difficult (college graduate)

    - **Flesch-Kincaid Grade**: US school grade level needed to understand
    - **Gunning Fog Index**: Years of formal education needed
    - **SMOG Index**: Years of education needed to understand

    **Tips:**
    - Use appropriate delays to avoid being blocked
    - Start with a small sample to test
    - Image URLs are automatically filtered out
    """)

# Sidebar settings
st.sidebar.header("Settings")

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
    value=15
)

max_urls = st.sidebar.number_input(
    "Maximum URLs to process",
    min_value=1,
    max_value=10000,
    value=100,
    help="Limit the number of URLs to process"
)

include_content = st.sidebar.checkbox(
    "Include content in export",
    value=False,
    help="Include extracted text in the CSV output"
)

# URL input
st.subheader("Enter URLs")

input_method = st.radio(
    "Input method",
    ["XML Sitemap URL", "Upload CSV", "Paste URLs"],
    horizontal=True
)

urls = []


def get_random_user_agent():
    """Get a random user agent string."""
    try:
        ua = UserAgent()
        return ua.random
    except:
        return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def fetch_urls_from_sitemap(sitemap_url):
    """Fetch URLs from an XML sitemap."""
    try:
        headers = {'User-Agent': get_random_user_agent()}
        response = requests.get(sitemap_url, headers=headers, timeout=30)
        soup = BeautifulSoup(response.content, 'xml')
        urls = [loc.text for loc in soup.find_all('loc')]

        # Filter out non-HTML URLs
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.svg', '.pdf', '.webp')
        filtered_urls = [url for url in urls if not url.lower().endswith(image_extensions)]

        return filtered_urls
    except Exception as e:
        st.error(f"Error fetching sitemap: {str(e)}")
        return []


def extract_content(url):
    """Extract main content from a URL using Trafilatura."""
    try:
        headers = {'User-Agent': get_random_user_agent()}
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            content = trafilatura.extract(
                response.content,
                include_comments=False,
                include_tables=False
            )
            return content
        return None
    except Exception as e:
        return None


def calculate_reading_scores(text):
    """Calculate various readability scores."""
    if not text or len(text.split()) < 100:
        return None

    try:
        return {
            'flesch_reading_ease': round(textstat.flesch_reading_ease(text), 2),
            'flesch_kincaid_grade': round(textstat.flesch_kincaid_grade(text), 2),
            'gunning_fog': round(textstat.gunning_fog(text), 2),
            'smog_index': round(textstat.smog_index(text), 2),
            'automated_readability_index': round(textstat.automated_readability_index(text), 2),
            'coleman_liau_index': round(textstat.coleman_liau_index(text), 2),
            'linsear_write_formula': round(textstat.linsear_write_formula(text), 2),
            'dale_chall_readability_score': round(textstat.dale_chall_readability_score(text), 2),
            'word_count': textstat.lexicon_count(text, removepunct=True),
            'sentence_count': textstat.sentence_count(text),
            'avg_sentence_length': round(textstat.avg_sentence_length(text), 2),
            'difficult_words': textstat.difficult_words(text),
            'reading_time_mins': round(textstat.lexicon_count(text, removepunct=True) / 200, 1)
        }
    except Exception as e:
        return None


if input_method == "XML Sitemap URL":
    sitemap_url = st.text_input(
        "Enter XML Sitemap URL",
        placeholder="https://example.com/sitemap.xml"
    )
    if sitemap_url and st.button("Fetch URLs from Sitemap"):
        with st.spinner("Fetching URLs from sitemap..."):
            urls = fetch_urls_from_sitemap(sitemap_url)
            if urls:
                st.session_state['urls'] = urls
                st.success(f"Found {len(urls)} URLs in sitemap")

    if 'urls' in st.session_state:
        urls = st.session_state['urls']
        st.info(f"{len(urls)} URLs ready to analyze")

elif input_method == "Upload CSV":
    url_file = st.file_uploader(
        "Upload CSV with URLs",
        type=['csv'],
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

else:  # Paste URLs
    url_text = st.text_area(
        "Paste URLs (one per line)",
        height=200
    )
    if url_text:
        urls = [u.strip() for u in url_text.strip().split('\n') if u.strip()]
        st.info(f"Found {len(urls)} URLs")


# Main processing
if urls and st.button("Analyze Reading Scores", type="primary"):
    # Limit URLs
    urls_to_process = urls[:max_urls]
    if len(urls) > max_urls:
        st.warning(f"Processing first {max_urls} URLs (total: {len(urls)})")

    progress_bar = st.progress(0)
    status_text = st.empty()

    results = []
    errors = []

    for i, url in enumerate(urls_to_process):
        status_text.text(f"Processing {i+1}/{len(urls_to_process)}: {url[:60]}...")
        progress_bar.progress((i + 1) / len(urls_to_process))

        content = extract_content(url)

        if content:
            scores = calculate_reading_scores(content)
            if scores:
                result = {'url': url}
                result.update(scores)
                if include_content:
                    result['content'] = content
                results.append(result)
            else:
                errors.append({'url': url, 'error': 'Content too short for analysis'})
        else:
            errors.append({'url': url, 'error': 'Could not extract content'})

        time.sleep(delay)

    status_text.text("Analysis complete!")

    if results:
        df_results = pd.DataFrame(results)

        # Display summary
        st.subheader("Results Summary")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pages Analyzed", len(results))
        with col2:
            avg_score = df_results['flesch_reading_ease'].mean()
            st.metric("Avg Flesch Score", f"{avg_score:.1f}")
        with col3:
            avg_grade = df_results['flesch_kincaid_grade'].mean()
            st.metric("Avg Grade Level", f"{avg_grade:.1f}")
        with col4:
            total_words = df_results['word_count'].sum()
            st.metric("Total Words", f"{total_words:,}")

        # Reading ease distribution
        st.subheader("Flesch Reading Ease Distribution")

        df_results['readability_category'] = pd.cut(
            df_results['flesch_reading_ease'],
            bins=[-float('inf'), 30, 50, 60, 70, 80, 90, float('inf')],
            labels=['Very Difficult', 'Difficult', 'Fairly Difficult', 'Standard',
                    'Fairly Easy', 'Easy', 'Very Easy']
        )
        category_counts = df_results['readability_category'].value_counts()
        st.bar_chart(category_counts)

        # Full results table
        st.subheader("Detailed Results")

        # Select display columns
        display_cols = ['url', 'flesch_reading_ease', 'flesch_kincaid_grade',
                        'word_count', 'avg_sentence_length', 'reading_time_mins']
        st.dataframe(
            df_results[display_cols].sort_values('flesch_reading_ease'),
            use_container_width=True
        )

        # Show hardest to read pages
        st.subheader("Hardest to Read Pages")
        hardest = df_results.nsmallest(10, 'flesch_reading_ease')[display_cols]
        st.dataframe(hardest, use_container_width=True)

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
            file_name="reading_scores.csv",
            mime="text/csv"
        )

    else:
        st.warning("No pages could be analyzed. Check the errors below.")
        if errors:
            st.dataframe(pd.DataFrame(errors))

elif not urls:
    st.info("Enter URLs to analyze using one of the methods above")

    st.subheader("Example Output")
    example_data = {
        "URL": ["/about", "/services", "/blog/technical-post"],
        "Flesch Reading Ease": [72.5, 65.3, 42.8],
        "Grade Level": [7.2, 8.5, 12.1],
        "Word Count": [850, 1200, 2500],
        "Reading Time": [4.3, 6.0, 12.5]
    }
    st.dataframe(pd.DataFrame(example_data))
