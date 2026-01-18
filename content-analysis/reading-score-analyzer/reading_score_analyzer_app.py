"""
Reading Score Analyzer - Streamlit App

Analyze content readability from URLs using Flesch scores and other metrics.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

try:
    import trafilatura
    import textstat
except ImportError:
    st.error("Please install: pip install trafilatura textstat")
    st.stop()

st.set_page_config(
    page_title="Reading Score Analyzer",
    page_icon="📖",
    layout="wide"
)

st.title("📖 Reading Score Analyzer")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Calculates readability scores (Flesch, Gunning Fog, etc.)
    - Analyzes content complexity
    - Provides grade-level recommendations

    **How to use:**
    1. Upload a CSV with content or paste text
    2. Select readability metrics to calculate
    3. Analyze content scores
    4. Download readability report

    **Best for:**
    - Content accessibility audits
    - Matching content to audience level
    - Editorial guidelines compliance
    """)
st.markdown("Analyze content readability from URLs using multiple readability metrics.")


def get_random_user_agent():
    """Get a user agent string."""
    return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def fetch_urls_from_sitemap(sitemap_url, timeout=30):
    """Fetch URLs from an XML sitemap."""
    try:
        headers = {'User-Agent': get_random_user_agent()}
        response = requests.get(sitemap_url, headers=headers, timeout=timeout)
        soup = BeautifulSoup(response.content, 'xml')
        urls = [loc.text for loc in soup.find_all('loc')]

        # Filter out non-HTML URLs
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.svg', '.pdf', '.webp')
        urls = [url for url in urls if not url.lower().endswith(image_extensions)]

        return urls, None
    except Exception as e:
        return [], str(e)


def extract_content(url, timeout=15):
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
            return content, None
        return None, f"HTTP {response.status_code}"
    except Exception as e:
        return None, str(e)


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
            'word_count': textstat.lexicon_count(text, removepunct=True),
            'sentence_count': textstat.sentence_count(text),
            'avg_sentence_length': round(textstat.avg_sentence_length(text), 2),
            'difficult_words': textstat.difficult_words(text),
            'reading_time_mins': round(textstat.lexicon_count(text, removepunct=True) / 200, 1)
        }
    except Exception:
        return None


def get_flesch_interpretation(score):
    """Get interpretation of Flesch Reading Ease score."""
    if score >= 90:
        return "Very Easy", "5th grade", "#28a745"
    elif score >= 80:
        return "Easy", "6th grade", "#5cb85c"
    elif score >= 70:
        return "Fairly Easy", "7th grade", "#8bc34a"
    elif score >= 60:
        return "Standard", "8th-9th grade", "#ffc107"
    elif score >= 50:
        return "Fairly Difficult", "10th-12th grade", "#ff9800"
    elif score >= 30:
        return "Difficult", "College", "#ff5722"
    else:
        return "Very Difficult", "College Graduate", "#dc3545"


def process_urls(urls, delay, timeout, include_content, progress_bar, status_text):
    """Process URLs and calculate reading scores."""
    results = []
    errors = []

    for i, url in enumerate(urls):
        progress_bar.progress((i + 1) / len(urls))
        status_text.text(f"Processing {i + 1}/{len(urls)}: {url[:50]}...")

        content, error = extract_content(url, timeout)

        if content:
            scores = calculate_reading_scores(content)
            if scores:
                result = {'url': url}
                result.update(scores)
                if include_content:
                    result['content'] = content[:500] + '...' if len(content) > 500 else content
                results.append(result)
            else:
                errors.append({'url': url, 'error': 'Content too short'})
        else:
            errors.append({'url': url, 'error': error or 'Could not extract content'})

        time.sleep(delay)

    return pd.DataFrame(results), errors


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    delay = st.slider(
        "Delay (seconds)",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.5,
        help="Delay between requests"
    )

    timeout = st.slider(
        "Timeout (seconds)",
        min_value=5,
        max_value=30,
        value=15,
        help="Request timeout"
    )

    max_urls = st.number_input(
        "Max URLs",
        min_value=1,
        max_value=500,
        value=50,
        help="Maximum URLs to process"
    )

    include_content = st.checkbox(
        "Include content preview",
        value=False,
        help="Include extracted content in output"
    )

    st.markdown("---")
    st.markdown("### 📊 Score Guide")
    st.markdown("""
    **Flesch Reading Ease:**
    - 90-100: Very Easy (5th grade)
    - 80-89: Easy (6th grade)
    - 70-79: Fairly Easy (7th grade)
    - 60-69: Standard (8th-9th grade)
    - 50-59: Fairly Difficult (10th-12th)
    - 30-49: Difficult (College)
    - 0-29: Very Difficult (Graduate)
    """)

# Main content
input_method = st.radio(
    "Input Method",
    ["Sitemap URL", "Manual URLs", "Upload CSV"],
    horizontal=True
)

urls = []

if input_method == "Sitemap URL":
    sitemap_url = st.text_input(
        "Sitemap URL",
        placeholder="https://example.com/sitemap.xml",
        help="Enter your XML sitemap URL"
    )

    if sitemap_url:
        if st.button("🔍 Fetch URLs from Sitemap"):
            with st.spinner("Fetching sitemap..."):
                urls, error = fetch_urls_from_sitemap(sitemap_url, timeout)

            if error:
                st.error(f"Error: {error}")
            elif urls:
                st.success(f"✅ Found {len(urls)} URLs")
                st.session_state['urls'] = urls[:max_urls]

    if 'urls' in st.session_state:
        urls = st.session_state['urls']
        with st.expander(f"Preview URLs ({len(urls)})"):
            for url in urls[:20]:
                st.markdown(f"- {url}")

elif input_method == "Manual URLs":
    urls_text = st.text_area(
        "Enter URLs (one per line)",
        height=150,
        placeholder="https://example.com/page1\nhttps://example.com/page2"
    )
    if urls_text:
        urls = [u.strip() for u in urls_text.split('\n') if u.strip().startswith('http')]
        urls = urls[:max_urls]
        st.info(f"Found {len(urls)} URLs")

else:  # Upload CSV
    uploaded_file = st.file_uploader("Upload CSV with URLs", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        url_col = st.selectbox("URL Column", df.columns.tolist())
        urls = df[url_col].dropna().tolist()
        urls = [u for u in urls if isinstance(u, str) and u.startswith('http')][:max_urls]
        st.info(f"Found {len(urls)} URLs")

if urls:
    if st.button("📖 Analyze Readability", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        results_df, errors = process_urls(
            urls, delay, timeout, include_content, progress_bar, status_text
        )

        progress_bar.empty()
        status_text.empty()

        if len(results_df) > 0:
            st.success(f"✅ Analyzed {len(results_df)} pages!")

            # Results tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Results", "📈 Overview", "🔍 Problem Pages", "📉 Distribution"
            ])

            with tab1:
                display_cols = [
                    'url', 'flesch_reading_ease', 'flesch_kincaid_grade',
                    'word_count', 'reading_time_mins'
                ]
                st.dataframe(results_df[display_cols], use_container_width=True, height=400)

                csv = results_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "📥 Download Results CSV",
                    data=csv,
                    file_name="reading_scores.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with tab2:
                col1, col2, col3, col4 = st.columns(4)

                avg_flesch = results_df['flesch_reading_ease'].mean()
                interpretation, grade, color = get_flesch_interpretation(avg_flesch)

                with col1:
                    st.metric("Avg Flesch Score", f"{avg_flesch:.1f}")
                    st.markdown(f"**{interpretation}** ({grade})")

                with col2:
                    avg_grade = results_df['flesch_kincaid_grade'].mean()
                    st.metric("Avg Grade Level", f"{avg_grade:.1f}")

                with col3:
                    avg_words = results_df['word_count'].mean()
                    st.metric("Avg Word Count", f"{avg_words:.0f}")

                with col4:
                    avg_time = results_df['reading_time_mins'].mean()
                    st.metric("Avg Read Time", f"{avg_time:.1f} min")

                # Metric explanations
                st.markdown("---")
                st.subheader("📚 All Metrics")

                metrics_df = results_df[[
                    'flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog',
                    'smog_index', 'automated_readability_index', 'coleman_liau_index'
                ]].mean().round(2)

                for metric, value in metrics_df.items():
                    st.markdown(f"- **{metric.replace('_', ' ').title()}**: {value}")

            with tab3:
                st.subheader("🔍 Hardest to Read Pages")

                hardest = results_df.nsmallest(10, 'flesch_reading_ease')

                for _, row in hardest.iterrows():
                    score = row['flesch_reading_ease']
                    interpretation, grade, color = get_flesch_interpretation(score)

                    st.markdown(
                        f"**[{score:.0f}]** {row['url'][:60]}... "
                        f"<span style='color:{color}'>{interpretation}</span>",
                        unsafe_allow_html=True
                    )

                st.markdown("---")
                st.subheader("📊 Easy to Read Pages")

                easiest = results_df.nlargest(10, 'flesch_reading_ease')
                for _, row in easiest.iterrows():
                    score = row['flesch_reading_ease']
                    interpretation, grade, color = get_flesch_interpretation(score)
                    st.markdown(
                        f"**[{score:.0f}]** {row['url'][:60]}... "
                        f"<span style='color:{color}'>{interpretation}</span>",
                        unsafe_allow_html=True
                    )

            with tab4:
                import plotly.express as px

                st.subheader("Flesch Score Distribution")
                fig = px.histogram(
                    results_df,
                    x='flesch_reading_ease',
                    nbins=20,
                    title='Flesch Reading Ease Score Distribution'
                )
                fig.add_vline(x=60, line_dash="dash", annotation_text="Standard")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Grade Level Distribution")
                fig2 = px.histogram(
                    results_df,
                    x='flesch_kincaid_grade',
                    nbins=15,
                    title='Flesch-Kincaid Grade Level Distribution'
                )
                st.plotly_chart(fig2, use_container_width=True)

                st.subheader("Word Count vs Reading Ease")
                fig3 = px.scatter(
                    results_df,
                    x='word_count',
                    y='flesch_reading_ease',
                    hover_data=['url'],
                    title='Word Count vs Readability'
                )
                st.plotly_chart(fig3, use_container_width=True)

            # Show errors
            if errors:
                with st.expander(f"⚠️ Errors ({len(errors)})"):
                    for err in errors[:20]:
                        st.markdown(f"- {err['url'][:50]}: {err['error']}")

        else:
            st.warning("No pages could be analyzed")

else:
    st.info("👆 Enter URLs to analyze their readability")

# Footer
st.markdown("---")
st.markdown(
    "Built by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · "
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · "
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)"
)
