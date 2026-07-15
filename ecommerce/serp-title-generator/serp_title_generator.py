# Author: Lee Foot
# Website: https://leefoot.com

import os
import math
import streamlit as st
import pandas as pd
import requests
import json
from base64 import b64encode
from nltk.corpus import stopwords
import nltk

st.set_page_config(page_title="SERP Title Generator", page_icon="📝", layout="wide")

# Download stopwords if not present
try:
    stop = stopwords.words('english')
except Exception:
    nltk.download('stopwords')
    stop = stopwords.words('english')

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

st.title("SERP Title Generator")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Searches Google for your product code/MPN/keyword
    - Extracts and analyses titles from top-ranking pages
    - Identifies the most common words used in successful titles
    - Suggests an optimised title based on SERP patterns

    **Best for:**
    - Optimising product page titles
    - Finding the best title format for MPNs/product codes
    - Understanding what title patterns rank well

    **Requirements:**
    - DataForSEO account (sign up at [dataforseo.com](https://dataforseo.com/))
    """)

# Sidebar settings
st.sidebar.header("Settings")

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

location_select = st.sidebar.selectbox("Search Location", list(LOCATION_CODES.keys()))
device_select = st.sidebar.selectbox("Device", ["Desktop", "Mobile", "Tablet"])
num_results = st.sidebar.slider("Number of Results to Analyse", min_value=10, max_value=50, value=20)

# Show estimated cost
estimated_cost = 0.002 * math.ceil(num_results / 10)
st.sidebar.caption(f"Estimated cost per search: ${estimated_cost:.3f}")

# Main input
query = st.text_input("Enter Product Code / MPN / Keyword", placeholder="e.g., EZC16 or iPhone 15 Pro case")

if st.button("Generate Title Suggestions", type="primary"):
    if not dataforseo_login or not dataforseo_password:
        st.error("Please enter your DataForSEO login and password in the sidebar")
        st.stop()

    if not query:
        st.error("Please enter a search query")
        st.stop()

    with st.spinner(f"Searching Google for '{query}'..."):
        # DataForSEO auth
        cred = b64encode(f"{dataforseo_login}:{dataforseo_password}".encode()).decode()
        headers = {
            'Authorization': f'Basic {cred}',
            'Content-Type': 'application/json'
        }

        payload = [{
            "keyword": query,
            "location_code": LOCATION_CODES[location_select],
            "language_code": "en",
            "device": device_select.lower(),
            "depth": num_results
        }]

        try:
            response = requests.post(
                'https://api.dataforseo.com/v3/serp/google/organic/live/advanced',
                headers=headers,
                json=payload,
                timeout=60
            )
            response_data = response.json()

            # Check for API-level errors
            if response_data.get('status_code') and response_data['status_code'] != 20000:
                st.error(f"API Error: {response_data.get('status_message', 'Unknown error')}")
                st.stop()

            try:
                items = response_data["tasks"][0]["result"][0]["items"]
                organic_results = [item for item in items if item["type"] == "organic"]
            except (KeyError, TypeError, IndexError):
                st.warning("No search results found")
                st.stop()

            if not organic_results:
                st.warning("No organic results found")
                st.stop()

            # Extract titles
            titles = []
            urls = []
            for result in organic_results:
                if 'title' in result:
                    titles.append(result['title'])
                    urls.append(result.get('url', ''))

            st.success(f"Analysed {len(titles)} search results")

            # Create results dataframe
            df = pd.DataFrame({'title': titles, 'url': urls})

            # Show raw titles
            with st.expander("View Extracted Titles"):
                st.dataframe(df)

            # Process titles
            df_processed = df.copy()

            # Standardize delimiters and take first part
            df_processed['title'] = df_processed['title'].str.replace(" - ", "|")
            df_processed['title'] = df_processed['title'].str.replace(" | ", "|")
            df_processed['title_part'] = df_processed['title'].str.split("|").str[0]

            # Clean titles
            df_processed['title_clean'] = df_processed['title_part'].str.strip()
            df_processed = df_processed.drop_duplicates(subset=['title_clean'], keep="first")

            df_processed['title_clean'] = df_processed['title_clean'].str.replace("'", '')
            df_processed['title_clean'] = df_processed['title_clean'].str.replace(",", ' ')
            df_processed['title_clean'] = df_processed['title_clean'].str.replace(".", '', regex=False)
            df_processed['title_clean'] = df_processed['title_clean'].apply(
                lambda x: ' '.join([word for word in str(x).split() if word.lower() not in stop])
            )
            df_processed['title_clean'] = df_processed['title_clean'].str.lower()

            # Count word frequencies
            all_words = ' '.join(df_processed['title_clean'].dropna()).split()
            word_counts = pd.Series(all_words).value_counts().reset_index()
            word_counts.columns = ['word', 'frequency']
            word_counts = word_counts[word_counts['frequency'] > 1]

            # Display word frequency
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Most Common Title Words")
                st.dataframe(word_counts.head(20), use_container_width=True)

            # Generate suggested title
            words_df = df_processed['title_clean'].str.split(' ', expand=True)

            if len(words_df.columns) > 0:
                # Rename columns in reverse order (most important first)
                max_cols = len(words_df.columns)
                cols = list(range(max_cols - 1, -1, -1))
                words_df.columns = range(len(words_df.columns))

                # Find most common word in each position
                suggested_words = []
                for col in words_df.columns:
                    try:
                        most_common = words_df[col].value_counts().idxmax()
                        if most_common and pd.notna(most_common) and most_common.strip():
                            suggested_words.append(most_common)
                    except Exception:
                        pass

                # Remove duplicates while preserving order
                seen = set()
                unique_words = []
                for word in suggested_words:
                    if word and word not in seen:
                        seen.add(word)
                        unique_words.append(word)

                suggested_title = ' '.join(unique_words).title()

                with col2:
                    st.subheader("Suggested Title")
                    st.info(f"**{suggested_title}**")

                    st.markdown("---")
                    st.markdown("**Title Components (by position importance):**")
                    for i, word in enumerate(unique_words[:10], 1):
                        st.write(f"{i}. {word}")

            # Download results
            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                csv_words = word_counts.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Word Frequency",
                    data=csv_words,
                    file_name=f"title_words_{query.replace(' ', '_')}.csv",
                    mime="text/csv"
                )

            with col2:
                csv_titles = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download All Titles",
                    data=csv_titles,
                    file_name=f"serp_titles_{query.replace(' ', '_')}.csv",
                    mime="text/csv"
                )

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {str(e)}")
        except json.JSONDecodeError:
            st.error("Failed to parse API response")
        except Exception as e:
            st.error(f"Error: {str(e)}")

else:
    st.info("Enter a product code or keyword and click 'Generate Title Suggestions'")

    st.subheader("Example Use Cases")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Product MPNs**")
        st.code("EZC16\nSamsung UN55TU7000\nBosch GSB 18V-55")

    with col2:
        st.markdown("**Product Names**")
        st.code("iPhone 15 Pro case\nNike Air Max 90\nDyson V15 Detect")

    with col3:
        st.markdown("**Generic Products**")
        st.code("wireless earbuds\ngaming mouse\nstanding desk")
