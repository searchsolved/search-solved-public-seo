####################################################################################
#                                                                                  #
#  Product Q&A Extractor                                                           #
#                                                                                  #
#  Extract product reviews and Q&A from e-commerce pages.                          #
#                                                                                  #
####################################################################################
# Author   : Lee Foot                                                              #
# Website  : https://leefoot.com                                                   #
# Contact  : https://leefoot.com/contact                                           #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                       #
####################################################################################

"""
Product Q&A Extractor

Extracts product reviews, ratings, and Q&A content from e-commerce product pages.
Supports configurable CSS selectors for different site structures.

Features:
- Configurable CSS selectors for reviews and Q&A
- Star rating extraction
- Review count aggregation
- Q&A content extraction
- Export to CSV
"""

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re

st.set_page_config(page_title="Product Q&A Extractor", page_icon="💬", layout="wide")

st.title("Product Q&A Extractor")
st.markdown("*Created by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://leefoot.com) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · 🦋 [Bluesky](https://bsky.app/profile/leefootseo.bsky.social)*")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Extracts review ratings and counts from product pages
    - Extracts Q&A content (questions and answers)
    - Works with any e-commerce site using configurable selectors

    **How to prepare:**
    1. Get a list of product URLs
    2. Use browser DevTools to find the CSS selectors for:
       - Overall rating (e.g., ".rating-value")
       - Review count (e.g., ".review-count")
       - Individual reviews (e.g., ".review-item")
       - Q&A sections (e.g., ".qa-item")

    **Common patterns:**
    - Trustpilot: `.tp-widget__star-rating`
    - Yotpo: `.yotpo-review-stars`
    - Bazaarvoice: `.bv-rating`
    - Shopify Product Reviews: `.spr-summary-starrating`

    **Tips:**
    - Test with a single URL first to verify selectors
    - Use appropriate delays to avoid blocking
    - Some sites require JavaScript rendering (not supported)
    """)

# Sidebar settings
st.sidebar.header("Review Selectors")

rating_selector = st.sidebar.text_input(
    "Overall rating selector",
    value=".rating, [itemprop='ratingValue'], .star-rating",
    help="CSS selector for the aggregate rating element"
)

review_count_selector = st.sidebar.text_input(
    "Review count selector",
    value=".review-count, [itemprop='reviewCount'], .reviews-count",
    help="CSS selector for the review count element"
)

individual_review_selector = st.sidebar.text_input(
    "Individual review container selector",
    value=".review, .review-item, [itemprop='review']",
    help="CSS selector for each review container"
)

review_star_selector = st.sidebar.text_input(
    "Individual review star selector",
    value=".review-stars, .star-rating",
    help="CSS selector for star rating within a review"
)

review_text_selector = st.sidebar.text_input(
    "Review text selector",
    value=".review-text, .review-content, [itemprop='reviewBody']",
    help="CSS selector for review text content"
)

st.sidebar.markdown("---")
st.sidebar.header("Q&A Selectors")

qa_container_selector = st.sidebar.text_input(
    "Q&A container selector",
    value=".qa-item, .question-answer, .faq-item",
    help="CSS selector for each Q&A pair"
)

question_selector = st.sidebar.text_input(
    "Question selector (within container)",
    value=".question, .qa-question, dt",
    help="CSS selector for the question text"
)

answer_selector = st.sidebar.text_input(
    "Answer selector (within container)",
    value=".answer, .qa-answer, dd",
    help="CSS selector for the answer text"
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

user_agent = st.sidebar.text_input(
    "User Agent",
    value="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
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
        "Paste product URLs (one per line)",
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

# Test single URL
st.subheader("Test Extraction")
test_url = st.text_input("Test URL (optional)")

if test_url and st.button("Test Selectors"):
    with st.spinner("Testing extraction..."):
        try:
            headers = {'User-Agent': user_agent}
            response = requests.get(test_url, headers=headers, timeout=timeout)
            soup = BeautifulSoup(response.text, 'html.parser')

            st.markdown("**Rating:**")
            rating_el = soup.select_one(rating_selector)
            st.write(rating_el.get_text(strip=True) if rating_el else "Not found")

            st.markdown("**Review Count:**")
            count_el = soup.select_one(review_count_selector)
            st.write(count_el.get_text(strip=True) if count_el else "Not found")

            st.markdown("**Individual Reviews:**")
            reviews = soup.select(individual_review_selector)
            st.write(f"Found {len(reviews)} review containers")

            if reviews:
                for i, rev in enumerate(reviews[:3]):
                    st.markdown(f"*Review {i+1}:*")
                    text_el = rev.select_one(review_text_selector)
                    st.write(text_el.get_text(strip=True)[:200] + "..." if text_el else "No text found")

            st.markdown("**Q&A Items:**")
            qa_items = soup.select(qa_container_selector)
            st.write(f"Found {len(qa_items)} Q&A items")

            if qa_items:
                for i, qa in enumerate(qa_items[:3]):
                    q = qa.select_one(question_selector)
                    a = qa.select_one(answer_selector)
                    st.markdown(f"*Q{i+1}:* {q.get_text(strip=True)[:100] if q else 'N/A'}")
                    st.markdown(f"*A{i+1}:* {a.get_text(strip=True)[:100] if a else 'N/A'}")

        except Exception as e:
            st.error(f"Error: {str(e)}")


def extract_number(text):
    """Extract first number from text."""
    if not text:
        return None
    match = re.search(r'[\d.]+', text.replace(',', ''))
    if match:
        try:
            return float(match.group())
        except:
            return None
    return None


def extract_product_data(url, soup):
    """Extract all review and Q&A data from a page."""
    data = {
        'url': url,
        'rating': None,
        'review_count': None,
        'reviews': [],
        'qa_items': []
    }

    # Extract overall rating
    rating_el = soup.select_one(rating_selector)
    if rating_el:
        data['rating'] = extract_number(rating_el.get_text(strip=True))
        # Also check for content/value attributes
        if data['rating'] is None:
            for attr in ['content', 'value', 'data-rating']:
                if rating_el.get(attr):
                    data['rating'] = extract_number(rating_el.get(attr))
                    break

    # Extract review count
    count_el = soup.select_one(review_count_selector)
    if count_el:
        data['review_count'] = extract_number(count_el.get_text(strip=True))
        if data['review_count'] is None:
            for attr in ['content', 'value', 'data-count']:
                if count_el.get(attr):
                    data['review_count'] = extract_number(count_el.get(attr))
                    break

    # Extract individual reviews
    reviews = soup.select(individual_review_selector)
    for rev in reviews:
        review_data = {}

        # Get review text
        text_el = rev.select_one(review_text_selector)
        if text_el:
            review_data['text'] = text_el.get_text(strip=True)

        # Get review rating
        star_el = rev.select_one(review_star_selector)
        if star_el:
            review_data['rating'] = extract_number(star_el.get_text(strip=True))

        if review_data:
            data['reviews'].append(review_data)

    # Extract Q&A
    qa_items = soup.select(qa_container_selector)
    for qa in qa_items:
        q_el = qa.select_one(question_selector)
        a_el = qa.select_one(answer_selector)

        if q_el or a_el:
            data['qa_items'].append({
                'question': q_el.get_text(strip=True) if q_el else None,
                'answer': a_el.get_text(strip=True) if a_el else None
            })

    return data


# Main processing
if urls and st.button("Extract Reviews & Q&A", type="primary"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    all_data = []
    all_reviews = []
    all_qa = []

    headers = {'User-Agent': user_agent}

    for i, url in enumerate(urls):
        status_text.text(f"Processing {i+1}/{len(urls)}: {url[:60]}...")
        progress_bar.progress((i + 1) / len(urls))

        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            soup = BeautifulSoup(response.text, 'html.parser')

            data = extract_product_data(url, soup)
            all_data.append({
                'url': data['url'],
                'rating': data['rating'],
                'review_count': data['review_count'],
                'reviews_extracted': len(data['reviews']),
                'qa_extracted': len(data['qa_items'])
            })

            # Store individual reviews
            for rev in data['reviews']:
                all_reviews.append({
                    'url': url,
                    'review_rating': rev.get('rating'),
                    'review_text': rev.get('text')
                })

            # Store Q&A items
            for qa in data['qa_items']:
                all_qa.append({
                    'url': url,
                    'question': qa.get('question'),
                    'answer': qa.get('answer')
                })

        except Exception as e:
            all_data.append({
                'url': url,
                'rating': None,
                'review_count': None,
                'reviews_extracted': 0,
                'qa_extracted': 0,
                'error': str(e)
            })

        time.sleep(delay)

    status_text.text("Extraction complete!")

    # Create DataFrames
    df_summary = pd.DataFrame(all_data)
    df_reviews = pd.DataFrame(all_reviews)
    df_qa = pd.DataFrame(all_qa)

    # Display results
    st.subheader("Extraction Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Products Processed", len(df_summary))
    with col2:
        with_rating = df_summary['rating'].notna().sum()
        st.metric("With Ratings", with_rating)
    with col3:
        st.metric("Reviews Extracted", len(df_reviews))
    with col4:
        st.metric("Q&A Extracted", len(df_qa))

    # Summary table
    st.subheader("Product Summary")
    st.dataframe(df_summary, use_container_width=True)

    # Rating distribution
    if df_summary['rating'].notna().any():
        st.subheader("Rating Distribution")
        rating_counts = df_summary['rating'].dropna().value_counts().sort_index()
        st.bar_chart(rating_counts)

    # Reviews table
    if len(df_reviews) > 0:
        st.subheader("Extracted Reviews")
        st.dataframe(df_reviews.head(50), use_container_width=True)

    # Q&A table
    if len(df_qa) > 0:
        st.subheader("Extracted Q&A")
        st.dataframe(df_qa.head(50), use_container_width=True)

    # Download
    st.subheader("Download")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_summary = df_summary.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="Download Summary (CSV)",
            data=csv_summary,
            file_name="product_qa_summary.csv",
            mime="text/csv"
        )

    with col2:
        if len(df_reviews) > 0:
            csv_reviews = df_reviews.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="Download Reviews (CSV)",
                data=csv_reviews,
                file_name="product_reviews.csv",
                mime="text/csv"
            )

    with col3:
        if len(df_qa) > 0:
            csv_qa = df_qa.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="Download Q&A (CSV)",
                data=csv_qa,
                file_name="product_qa.csv",
                mime="text/csv"
            )

elif not urls:
    st.info("Enter product URLs to begin extraction")

    st.subheader("Example Output")
    example = {
        "URL": ["/product/widget-1", "/product/widget-2"],
        "Rating": [4.5, 4.2],
        "Review Count": [128, 45],
        "Reviews Extracted": [10, 8],
        "Q&A Extracted": [5, 3]
    }
    st.dataframe(pd.DataFrame(example))
