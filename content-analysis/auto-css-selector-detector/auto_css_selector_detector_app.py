# Author   : Lee Foot
# Website  : https://leefoot.com
"""
Auto CSS Selector Detector - Streamlit App

Automatically identifies the best CSS selector for a page's main content
using an LLM, then extracts and converts the content to Markdown.

Author: Lee Foot
Website: https://leefoot.com
"""

import streamlit as st

from auto_css_selector_detector import detect_and_extract

st.set_page_config(
    page_title="Auto CSS Selector Detector",
    page_icon="🎯",
    layout="wide",
)

st.title("Auto CSS Selector Detector")
st.markdown(
    "*Created by* "
    "[![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) "
    "· [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) "
    "· [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) "
    "· [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) "
    "· [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) "
    "· [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)"
)

with st.expander("How to use this tool"):
    st.markdown("""
**What this tool does:**
- Fetches the HTML of any URL
- Sends a structural summary to an LLM to identify the best CSS selector for the main content area
- Extracts the content using that selector and converts it to Markdown

**How to use:**
1. Enter your OpenAI API key (or compatible provider key)
2. Optionally change the model or base URL for local LLM support
3. Paste a URL and click **Detect & Extract**
4. Review the detected selector, reasoning, and extracted content
""")

# Sidebar configuration
st.sidebar.header("Configuration")

api_key = st.sidebar.text_input(
    "API Key",
    type="password",
    help="Your OpenAI API key (or compatible provider key). Not stored.",
)

model = st.sidebar.text_input(
    "Model",
    value="gpt-4o-mini",
    help="Model name to use for selector detection.",
)

base_url = st.sidebar.text_input(
    "Base URL",
    value="https://api.openai.com/v1",
    help="API base URL. Change for local LLM servers (e.g. http://localhost:1234/v1).",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "The API key is used only for this request and is never stored or logged."
)

# Main input
url = st.text_input(
    "URL to analyse",
    placeholder="https://example.com/page",
    help="The full URL of the page whose content you want to extract.",
)

if st.button("Detect & Extract", type="primary"):
    if not api_key:
        st.error("Please enter your API key in the sidebar.")
    elif not url:
        st.error("Please enter a URL.")
    elif not url.startswith(("http://", "https://")):
        st.error("URL must start with http:// or https://")
    else:
        with st.spinner("Fetching page and querying LLM..."):
            try:
                result = detect_and_extract(
                    url=url,
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                )

                st.success("Content extracted successfully.")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Detection Results")
                    st.markdown(f"**H1:** {result.get('h1', 'N/A')}")
                    st.markdown(f"**Initial Selector:** `{result.get('selector', 'N/A')}`")
                    st.markdown(
                        f"**Specific Selector:** `{result.get('specific_selector', 'N/A')}`"
                    )
                    st.markdown(f"**Reasoning:** {result.get('reasoning', 'N/A')}")

                with col2:
                    st.subheader("Internal Links Found")
                    links_str = result.get("links", "[]")
                    st.code(links_str, language="python")

                st.subheader("Extracted Content (Markdown)")
                extracted = result.get("extracted_text", "")
                st.text_area(
                    "Content",
                    value=extracted,
                    height=400,
                    label_visibility="collapsed",
                )

                # Copy button via download
                st.download_button(
                    label="Download as .txt",
                    data=extracted,
                    file_name="extracted_content.txt",
                    mime="text/plain",
                )

            except Exception as e:
                st.error(f"Error: {e}")
