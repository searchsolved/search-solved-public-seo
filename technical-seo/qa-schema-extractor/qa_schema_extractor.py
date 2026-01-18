"""
Q&A Schema Extractor - Streamlit App
Extract Question/Answer pairs from JSON-LD schema markup.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
import json
import re
import requests
from bs4 import BeautifulSoup
from io import BytesIO

st.set_page_config(
    page_title="Q&A Schema Extractor",
    page_icon="❓",
    layout="wide"
)

st.title("❓ Q&A Schema Extractor")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-181717?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Extracts Q&A structured data from pages
    - Validates QAPage schema markup
    - Identifies FAQ schema opportunities

    **How to use:**
    1. Upload URLs or paste HTML
    2. Extract Q&A schema
    3. Validate existing markup
    4. Export extraction results

    **Best for:**
    - FAQ schema audits
    - Structured data validation
    - Rich result optimization
    """)
st.markdown("Extract FAQ and Q&A structured data from websites or crawl exports.")


def extract_qa_pairs(json_data):
    """Extract Q&A pairs from JSON-LD data."""
    qa_pairs = []

    def traverse(obj):
        if isinstance(obj, dict):
            obj_type = obj.get('@type', '')

            # Handle Question type
            if obj_type == 'Question' or 'Question' in str(obj_type):
                question = obj.get('name', obj.get('text', ''))
                answer_obj = obj.get('acceptedAnswer', obj.get('suggestedAnswer', {}))

                if isinstance(answer_obj, dict):
                    answer = answer_obj.get('text', answer_obj.get('name', ''))
                elif isinstance(answer_obj, list) and answer_obj:
                    answer = answer_obj[0].get('text', answer_obj[0].get('name', ''))
                else:
                    answer = ''

                if question:
                    qa_pairs.append({
                        'question': question,
                        'answer': answer
                    })

            # Handle FAQPage type
            if obj_type == 'FAQPage' or 'FAQPage' in str(obj_type):
                main_entity = obj.get('mainEntity', [])
                if isinstance(main_entity, list):
                    for item in main_entity:
                        traverse(item)

            # Continue traversing
            for value in obj.values():
                traverse(value)

        elif isinstance(obj, list):
            for item in obj:
                traverse(item)

    traverse(json_data)
    return qa_pairs


def extract_schema_from_html(html_content):
    """Extract JSON-LD schema from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    schemas = []

    # Find all JSON-LD script tags
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            # Clean the script content
            script_text = script.string
            if script_text:
                # Remove comments
                script_text = re.sub(r'/\*\*/', '', script_text)
                schema = json.loads(script_text)
                schemas.append(schema)
        except json.JSONDecodeError:
            continue

    return schemas


def fetch_url_schema(url):
    """Fetch a URL and extract its schema."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return extract_schema_from_html(response.text), None
    except Exception as e:
        return [], str(e)


# Main interface
tab1, tab2, tab3 = st.tabs(["🌐 From URL", "📄 From JSON", "📊 Bulk Extract"])

with tab1:
    st.subheader("Extract from URL")

    url = st.text_input("Enter URL", placeholder="https://example.com/faq")

    if st.button("Extract Q&A", type="primary", disabled=not url):
        with st.spinner("Fetching and parsing..."):
            schemas, error = fetch_url_schema(url)

        if error:
            st.error(f"Error fetching URL: {error}")
        elif schemas:
            all_qa = []
            for schema in schemas:
                qa_pairs = extract_qa_pairs(schema)
                all_qa.extend(qa_pairs)

            if all_qa:
                st.success(f"Found {len(all_qa)} Q&A pairs!")

                # Display Q&A pairs
                for i, qa in enumerate(all_qa, 1):
                    with st.expander(f"Q{i}: {qa['question'][:80]}..."):
                        st.markdown(f"**Question:** {qa['question']}")
                        st.markdown(f"**Answer:** {qa['answer']}")

                # Create DataFrame for download
                df = pd.DataFrame(all_qa)
                df['source_url'] = url

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("Download CSV", df.to_csv(index=False),
                                       "qa_extracted.csv", "text/csv")
                with col2:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False)
                    st.download_button("Download Excel", output.getvalue(),
                                       "qa_extracted.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.warning("No Q&A schema found on this page.")

            # Show raw schema for debugging
            with st.expander("View Raw Schema"):
                st.json(schemas)
        else:
            st.warning("No JSON-LD schema found on this page.")

with tab2:
    st.subheader("Extract from JSON")

    json_input = st.text_area("Paste JSON-LD Schema", height=300,
                               placeholder='{"@type": "FAQPage", "mainEntity": [...]}')

    if st.button("Parse JSON", type="primary", disabled=not json_input):
        try:
            # Clean input
            json_input = re.sub(r'/\*\*/', '', json_input)
            schema = json.loads(json_input)

            qa_pairs = extract_qa_pairs(schema)

            if qa_pairs:
                st.success(f"Found {len(qa_pairs)} Q&A pairs!")

                # Display Q&A pairs
                for i, qa in enumerate(qa_pairs, 1):
                    with st.expander(f"Q{i}: {qa['question'][:80]}..."):
                        st.markdown(f"**Question:** {qa['question']}")
                        st.markdown(f"**Answer:** {qa['answer']}")

                # Download
                df = pd.DataFrame(qa_pairs)
                st.download_button("Download CSV", df.to_csv(index=False),
                                   "qa_extracted.csv", "text/csv")
            else:
                st.warning("No Q&A data found in the schema.")

        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")

with tab3:
    st.subheader("Bulk Extract from Crawl Data")

    st.markdown("""
    Upload a CSV/Excel file with schema data from a crawl tool (like Screaming Frog).
    The file should have a column containing JSON-LD schema data.
    """)

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write(f"Loaded {len(df)} rows")
            st.dataframe(df.head(), use_container_width=True)

            # Column mapping
            col1, col2 = st.columns(2)

            with col1:
                schema_col = st.selectbox("Schema/JSON Column", list(df.columns))
            with col2:
                url_col = st.selectbox("URL Column (optional)", ["(None)"] + list(df.columns))

            if st.button("Extract All Q&A", type="primary"):
                results = []
                progress_bar = st.progress(0)

                for idx, row in df.iterrows():
                    url = row[url_col] if url_col != "(None)" else ""
                    schema_str = str(row.get(schema_col, "")) if pd.notna(row.get(schema_col)) else ""

                    if schema_str and schema_str != 'nan':
                        try:
                            # Clean and parse
                            schema_str = re.sub(r'/\*\*/', '', schema_str)
                            schema = json.loads(schema_str)
                            qa_pairs = extract_qa_pairs(schema)

                            for qa in qa_pairs:
                                results.append({
                                    'source_url': url,
                                    'question': qa['question'],
                                    'answer': qa['answer']
                                })
                        except:
                            pass

                    progress_bar.progress((idx + 1) / len(df))

                if results:
                    results_df = pd.DataFrame(results)
                    st.success(f"Extracted {len(results)} Q&A pairs from {len(df)} pages!")

                    st.dataframe(results_df, use_container_width=True)

                    # Stats
                    col1, col2 = st.columns(2)
                    col1.metric("Total Q&A Pairs", len(results))
                    col2.metric("Pages with Q&A", results_df['source_url'].nunique())

                    # Download
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button("Download CSV", results_df.to_csv(index=False),
                                           "bulk_qa_extracted.csv", "text/csv")
                    with col2:
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            results_df.to_excel(writer, index=False)
                        st.download_button("Download Excel", output.getvalue(),
                                           "bulk_qa_extracted.xlsx",
                                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else:
                    st.warning("No Q&A data found in the uploaded file.")

        except Exception as e:
            st.error(f"Error loading file: {e}")

# Example schema
with st.expander("📖 Example FAQPage Schema"):
    example = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": "What is an LLC?",
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": "An LLC (Limited Liability Company) is a business structure..."
                }
            },
            {
                "@type": "Question",
                "name": "How do I form an LLC?",
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": "To form an LLC, you need to file articles of organization..."
                }
            }
        ]
    }
    st.json(example)

# Footer
st.markdown("---")
