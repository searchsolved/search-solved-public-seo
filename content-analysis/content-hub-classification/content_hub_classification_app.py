"""
Content Hub Classification - Streamlit App

Classify article content into content hub categories using OpenAI GPT.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
import json
import time

try:
    from openai import OpenAI
except ImportError:
    st.error("Please install openai: pip install openai")
    st.stop()

st.set_page_config(
    page_title="Content Hub Classification",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Content Hub Classification")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Classifies content into hub/spoke relationships
    - Identifies topical clusters in your content
    - Maps content hierarchy using semantic analysis

    **How to use:**
    1. Upload a CSV with URLs and content/titles
    2. Configure similarity threshold
    3. Click "Classify Content"
    4. Review hub and spoke relationships

    **Best for:**
    - Content strategy planning
    - Internal linking optimization
    - Identifying content gaps
    """)
st.markdown("Classify article content into content hub categories using AI.")


def analyze_article(article_text, client, model):
    """Analyzes article and returns structured content analysis."""
    messages = [
        {
            "role": "system",
            "content": (
                "Analyze the following article and provide a structured summary based on the specified JSON schema. "
                "Select the most specific and relevant single content hub category that directly relates to the article's primary topic. "
                "Avoid broad or general categories. Use UK English."
            )
        },
        {
            "role": "user",
            "content": f"Analyze this article:\n\n{article_text}"
        }
    ]

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "content_analysis_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "content_analysis": {
                        "type": "object",
                        "properties": {
                            "primary_topic": {"type": "string"},
                            "content_hub_category": {"type": "string"},
                            "key_subtopics": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "recommended_products": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": [
                            "primary_topic",
                            "content_hub_category",
                            "key_subtopics",
                            "recommended_products"
                        ],
                        "additionalProperties": False
                    }
                },
                "required": ["content_analysis"],
                "additionalProperties": False
            }
        }
    }

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format
        )
        return json.loads(completion.choices[0].message.content), None
    except Exception as e:
        return None, str(e)


def process_batch(articles, client, model, progress_bar, status_text):
    """Process multiple articles."""
    results = []

    for i, article in enumerate(articles):
        progress_bar.progress((i + 1) / len(articles))
        status_text.text(f"Analyzing {i + 1}/{len(articles)}...")

        result, error = analyze_article(article, client, model)

        if result:
            analysis = result.get("content_analysis", {})
            results.append({
                'Article': article[:200] + '...' if len(article) > 200 else article,
                'Primary Topic': analysis.get('primary_topic', ''),
                'Content Hub Category': analysis.get('content_hub_category', ''),
                'Key Subtopics': ', '.join(analysis.get('key_subtopics', [])),
                'Recommended Products': ', '.join(analysis.get('recommended_products', [])),
                'Status': 'Success'
            })
        else:
            results.append({
                'Article': article[:200] + '...' if len(article) > 200 else article,
                'Primary Topic': '',
                'Content Hub Category': '',
                'Key Subtopics': '',
                'Recommended Products': '',
                'Status': f'Error: {error}'
            })

        time.sleep(0.5)  # Rate limiting

    return pd.DataFrame(results)


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key"
    )

    model = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1"],
        help="Select the OpenAI model to use"
    )

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    This tool uses AI to classify articles into:
    - **Primary Topic**: Main subject
    - **Content Hub Category**: Best-fit category
    - **Key Subtopics**: Related topics
    - **Recommended Products**: Relevant products
    """)

# Main content
input_method = st.radio(
    "Input Method",
    ["Single Article", "Batch Upload (CSV)"],
    horizontal=True
)

if input_method == "Single Article":
    st.markdown("### 📝 Enter Article Content")

    article_text = st.text_area(
        "Article Text",
        height=300,
        placeholder="Paste your article content here...",
        help="Enter the full text of the article to analyze"
    )

    if article_text and api_key:
        if st.button("🔍 Analyze Article", type="primary", use_container_width=True):
            with st.spinner("Analyzing article..."):
                client = OpenAI(api_key=api_key)
                result, error = analyze_article(article_text, client, model)

            if result:
                st.success("✅ Analysis Complete!")

                analysis = result.get("content_analysis", {})

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🎯 Primary Topic")
                    st.info(analysis.get('primary_topic', 'N/A'))

                    st.markdown("### 📁 Content Hub Category")
                    st.success(analysis.get('content_hub_category', 'N/A'))

                with col2:
                    st.markdown("### 📋 Key Subtopics")
                    subtopics = analysis.get('key_subtopics', [])
                    for topic in subtopics:
                        st.markdown(f"- {topic}")

                    st.markdown("### 🛒 Recommended Products")
                    products = analysis.get('recommended_products', [])
                    for product in products:
                        st.markdown(f"- {product}")

                # Download JSON
                st.markdown("---")
                json_output = json.dumps(result, indent=2)
                st.download_button(
                    "📥 Download JSON",
                    data=json_output,
                    file_name="content_analysis.json",
                    mime="application/json"
                )
            else:
                st.error(f"Error: {error}")
    elif not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar")

else:  # Batch Upload
    st.markdown("### 📤 Upload Articles CSV")

    uploaded_file = st.file_uploader(
        "Upload CSV with articles",
        type=["csv"],
        help="CSV should contain a column with article text"
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Select content column
        content_column = st.selectbox(
            "Select article content column",
            df.columns.tolist(),
            help="Choose the column containing article text"
        )

        articles = df[content_column].dropna().tolist()
        st.info(f"Found **{len(articles)}** articles to analyze")

        # Preview
        with st.expander("Preview Articles"):
            for i, article in enumerate(articles[:3]):
                st.markdown(f"**Article {i+1}:**")
                st.text(article[:300] + "..." if len(article) > 300 else article)
                st.markdown("---")

        if api_key:
            if st.button("🚀 Analyze All Articles", type="primary", use_container_width=True):
                client = OpenAI(api_key=api_key)
                progress_bar = st.progress(0)
                status_text = st.empty()

                results_df = process_batch(articles, client, model, progress_bar, status_text)

                progress_bar.empty()
                status_text.empty()

                st.success(f"✅ Analyzed {len(results_df)} articles!")

                # Results
                st.dataframe(results_df, use_container_width=True, height=400)

                # Summary stats
                st.markdown("### 📊 Summary")
                col1, col2, col3 = st.columns(3)

                with col1:
                    success_count = (results_df['Status'] == 'Success').sum()
                    st.metric("Successful", success_count)

                with col2:
                    categories = results_df['Content Hub Category'].nunique()
                    st.metric("Unique Categories", categories)

                with col3:
                    error_count = (results_df['Status'] != 'Success').sum()
                    st.metric("Errors", error_count)

                # Category distribution
                if success_count > 0:
                    st.markdown("### 📁 Category Distribution")
                    category_counts = results_df[results_df['Status'] == 'Success']['Content Hub Category'].value_counts()
                    st.bar_chart(category_counts)

                # Download
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Results CSV",
                    data=csv,
                    file_name="content_hub_classifications.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar")
    else:
        with st.expander("Example CSV Format"):
            example_df = pd.DataFrame({
                'article_content': [
                    'A Complete Guide to Sensor Cables and Connectors...',
                    'How to Choose the Right Industrial Pump...',
                ]
            })
            st.dataframe(example_df)

# Footer
st.markdown("---")
st.markdown(
    "Built by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · "
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · "
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)"
)
