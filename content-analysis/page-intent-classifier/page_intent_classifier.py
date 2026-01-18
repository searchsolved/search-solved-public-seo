####################################################################################
#                                                                                  #
#  Page Intent Classifier                                                          #
#                                                                                  #
#  Use OpenAI to classify page intent and expected user actions.                   #
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
Page Intent Classifier

Uses OpenAI's GPT models to analyze web pages and classify their intent,
identifying the primary purpose and expected user action.

Features:
- Fetch and extract content from URLs
- AI-powered intent classification
- Batch processing with progress tracking
- Configurable content extraction
- Export results to CSV
"""

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import json

st.set_page_config(page_title="Page Intent Classifier", page_icon="🎯", layout="wide")

# Check for required packages
try:
    import html2text
    from openai import OpenAI
    PACKAGES_AVAILABLE = True
except ImportError:
    PACKAGES_AVAILABLE = False

st.title("Page Intent Classifier")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-181717?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

if not PACKAGES_AVAILABLE:
    st.error("""
    Required packages not installed. Run:
    ```
    pip install openai html2text
    ```
    """)
    st.stop()

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Fetches content from web pages
    - Uses OpenAI GPT to classify the page intent
    - Identifies the primary purpose and expected user action

    **Intent categories include:**
    - Sign up / Register
    - Purchase / Buy
    - Browse / Explore
    - Learn / Educate
    - Contact / Support
    - Compare / Evaluate
    - Download / Get resource

    **Requirements:**
    - OpenAI API key (get one at platform.openai.com)
    - List of URLs to analyze

    **Cost estimate:**
    - GPT-3.5-turbo: ~$0.001-0.002 per page
    - GPT-4o-mini: ~$0.0015-0.003 per page
    """)

# Sidebar settings
st.sidebar.header("OpenAI Settings")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Your OpenAI API key from platform.openai.com"
)

model = st.sidebar.selectbox(
    "Model",
    ["gpt-4o-mini", "gpt-4o", "gpt-4.1"],
    help="GPT-4o-mini is recommended for cost-effectiveness"
)

st.sidebar.markdown("---")
st.sidebar.header("Content Extraction")

content_selector = st.sidebar.text_input(
    "Content CSS selector (optional)",
    value="",
    help="CSS selector for main content area. Leave blank to use entire body."
)

max_chars = st.sidebar.number_input(
    "Max characters per page",
    min_value=500,
    max_value=10000,
    value=3000,
    help="Limit content length to reduce API costs"
)

st.sidebar.markdown("---")
st.sidebar.header("Request Settings")

delay = st.sidebar.slider(
    "Delay between requests (seconds)",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.5
)

timeout = st.sidebar.number_input(
    "Request timeout (seconds)",
    min_value=5,
    max_value=60,
    value=15
)

# Intent classification prompt
SYSTEM_PROMPT = """You are a web page intent analyzer. Analyze the provided web page content to determine:
1. The primary PURPOSE of the page (what the page is designed to achieve)
2. The expected USER ACTION (what the user is supposed to do)

Respond ONLY with valid JSON in this exact format:
{
  "intent": "brief description of page purpose (6 words or fewer)",
  "action": "expected user action (3 words or fewer)",
  "category": "one of: signup, purchase, browse, learn, contact, compare, download, other"
}

Examples:
- {"intent": "Subscribe to email newsletter", "action": "Sign up", "category": "signup"}
- {"intent": "Browse product catalog", "action": "View items", "category": "browse"}
- {"intent": "Learn about services", "action": "Read content", "category": "learn"}
- {"intent": "Purchase software license", "action": "Buy now", "category": "purchase"}
- {"intent": "Get support help", "action": "Contact us", "category": "contact"}
"""

# URL input
st.subheader("Enter URLs to Classify")

input_method = st.radio(
    "Input method",
    ["Paste URLs", "Upload CSV"],
    horizontal=True
)

urls = []

if input_method == "Paste URLs":
    url_text = st.text_area(
        "Paste URLs (one per line)",
        height=200
    )
    if url_text:
        urls = [u.strip() for u in url_text.strip().split('\n') if u.strip()]
        st.info(f"Found {len(urls)} URLs")

else:
    url_file = st.file_uploader("Upload CSV with URLs", type=['csv'])
    if url_file is not None:
        try:
            df_urls = pd.read_csv(url_file)
            url_col = st.selectbox("Select URL column", df_urls.columns.tolist())
            urls = df_urls[url_col].dropna().tolist()
            st.info(f"Found {len(urls)} URLs")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")


def fetch_page_content(url, selector, max_length):
    """Fetch and extract content from a URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Get content from selector or body
        if selector:
            content_el = soup.select_one(selector)
            if content_el:
                html_content = str(content_el)
            else:
                html_content = str(soup.body) if soup.body else str(soup)
        else:
            html_content = str(soup.body) if soup.body else str(soup)

        # Convert to clean text
        text_maker = html2text.HTML2Text()
        text_maker.ignore_links = True
        text_maker.ignore_images = True
        text_maker.bypass_tables = True
        text_maker.ignore_emphasis = True

        text = text_maker.handle(html_content)

        # Truncate if needed
        if len(text) > max_length:
            text = text[:max_length] + "..."

        return text, None

    except Exception as e:
        return None, str(e)


def classify_intent(client, model_name, content):
    """Use OpenAI to classify page intent."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ],
            temperature=0.3,
            max_tokens=150,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return result, None

    except Exception as e:
        return None, str(e)


# Main processing
if urls and st.button("Classify Page Intents", type="primary"):
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar")
        st.stop()

    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        st.stop()

    progress_bar = st.progress(0)
    status_text = st.empty()

    results = []

    for i, url in enumerate(urls):
        status_text.text(f"Processing {i+1}/{len(urls)}: {url[:60]}...")
        progress_bar.progress((i + 1) / len(urls))

        # Fetch content
        content, fetch_error = fetch_page_content(url, content_selector, max_chars)

        if fetch_error:
            results.append({
                'url': url,
                'intent': None,
                'action': None,
                'category': None,
                'error': fetch_error
            })
            continue

        # Classify intent
        classification, classify_error = classify_intent(client, model, content)

        if classify_error:
            results.append({
                'url': url,
                'intent': None,
                'action': None,
                'category': None,
                'error': classify_error
            })
        else:
            results.append({
                'url': url,
                'intent': classification.get('intent', ''),
                'action': classification.get('action', ''),
                'category': classification.get('category', ''),
                'error': None
            })

        time.sleep(delay)

    status_text.text("Classification complete!")

    # Create DataFrame
    df_results = pd.DataFrame(results)

    # Display results
    st.subheader("Classification Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pages Processed", len(df_results))
    with col2:
        successful = df_results['intent'].notna().sum()
        st.metric("Successfully Classified", successful)
    with col3:
        errors = df_results['error'].notna().sum()
        st.metric("Errors", errors)

    # Category distribution
    if df_results['category'].notna().any():
        st.subheader("Intent Category Distribution")
        category_counts = df_results['category'].value_counts()
        st.bar_chart(category_counts)

    # Full results
    st.subheader("Detailed Results")
    st.dataframe(df_results, use_container_width=True)

    # Errors
    errors_df = df_results[df_results['error'].notna()]
    if len(errors_df) > 0:
        with st.expander(f"Errors ({len(errors_df)})"):
            st.dataframe(errors_df[['url', 'error']])

    # Download
    st.subheader("Download")
    csv_output = df_results.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="Download Results (CSV)",
        data=csv_output,
        file_name="page_intent_classification.csv",
        mime="text/csv"
    )

elif not urls:
    st.info("Enter URLs to classify page intent")

    st.subheader("Example Output")
    example = {
        "URL": ["/pricing", "/blog/how-to-guide", "/products", "/contact"],
        "Intent": ["View subscription options", "Learn about topic", "Browse products", "Get in touch"],
        "Action": ["Subscribe", "Read", "Shop", "Contact"],
        "Category": ["purchase", "learn", "browse", "contact"]
    }
    st.dataframe(pd.DataFrame(example))
