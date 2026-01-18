"""
Keyword Gap Analyzer - Streamlit App
Compare keyword lists to find gap opportunities with content matching.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re
from io import BytesIO

st.set_page_config(
    page_title="Keyword Gap Analyzer",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Keyword Gap Analyzer")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Identifies keyword gaps vs competitors
    - Finds keywords competitors rank for that you don't
    - Prioritizes gap opportunities by potential

    **How to use:**
    1. Upload your keyword rankings
    2. Upload competitor keyword data
    3. Configure gap analysis
    4. Review and export opportunities

    **Best for:**
    - Competitive keyword research
    - Content gap identification
    - SEO opportunity discovery
    """)
st.markdown("Find keyword gap opportunities by comparing competitor data and matching to existing content.")


def preprocess_content(content_df, url_col, content_col, h1_col=None):
    """Preprocess content data for efficient matching."""

    processed_df = content_df[[url_col, content_col]].copy()
    processed_df.columns = ['url', 'content']

    if h1_col and h1_col in content_df.columns:
        processed_df['h1'] = content_df[h1_col].fillna('')
    else:
        processed_df['h1'] = ''

    # Normalize text
    processed_df['content_lower'] = processed_df['content'].astype(str).str.lower()
    processed_df['h1_lower'] = processed_df['h1'].astype(str).str.lower()

    # Build keyword index
    keyword_index = defaultdict(list)
    for idx, row in processed_df.iterrows():
        words = set(re.findall(r'\b\w+\b', row['content_lower']))
        for word in words:
            if len(word) > 2:
                keyword_index[word].append(idx)

    # Build TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    content_matrix = vectorizer.fit_transform(processed_df['content_lower'])

    return {
        'df': processed_df,
        'keyword_index': keyword_index,
        'vectorizer': vectorizer,
        'content_matrix': content_matrix
    }


def find_content_matches(keywords, processed_content, top_n=3):
    """Find content matches for keywords using exact + semantic matching."""

    df = processed_content['df']
    keyword_index = processed_content['keyword_index']
    vectorizer = processed_content['vectorizer']
    content_matrix = processed_content['content_matrix']

    results = {}

    for keyword in keywords:
        keyword_lower = keyword.lower()
        results[keyword] = []

        # Exact match search using keyword index
        potential_matches = set()
        for word in re.findall(r'\b\w+\b', keyword_lower):
            if len(word) > 2 and word in keyword_index:
                potential_matches.update(keyword_index[word])

        # Check for exact keyword presence
        for idx in potential_matches:
            if keyword_lower in df.iloc[idx]['content_lower']:
                # Calculate H1 relevance
                h1_score = 0.0
                h1_text = df.iloc[idx]['h1_lower']
                if h1_text and keyword_lower in h1_text:
                    h1_score = 1.0
                elif h1_text:
                    kw_words = set(re.findall(r'\b\w+\b', keyword_lower))
                    h1_words = set(re.findall(r'\b\w+\b', h1_text))
                    common = kw_words.intersection(h1_words)
                    if common:
                        h1_score = len(common) / len(kw_words)

                results[keyword].append({
                    'url': df.iloc[idx]['url'],
                    'match_type': 'exact',
                    'h1_score': h1_score,
                    'h1': df.iloc[idx]['h1']
                })

        # Semantic match using TF-IDF
        try:
            kw_vector = vectorizer.transform([keyword_lower])
            similarities = cosine_similarity(kw_vector, content_matrix)[0]
            top_indices = np.argsort(similarities)[-top_n:][::-1]

            for idx in top_indices:
                if similarities[idx] > 0.1:
                    url = df.iloc[idx]['url']
                    # Skip if already an exact match
                    if any(r['url'] == url and r['match_type'] == 'exact' for r in results[keyword]):
                        continue

                    results[keyword].append({
                        'url': url,
                        'match_type': 'semantic',
                        'similarity': float(similarities[idx]),
                        'h1': df.iloc[idx]['h1']
                    })
        except:
            pass

        # Sort by match quality
        results[keyword] = sorted(
            results[keyword],
            key=lambda x: (x['match_type'] == 'exact', x.get('h1_score', 0), x.get('similarity', 0)),
            reverse=True
        )[:top_n]

    return results


# Main interface
st.markdown("### 1. Upload Keyword Data")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Your Keywords (Site A)**")
    your_file = st.file_uploader("Upload your keyword data", type=['csv', 'xlsx'], key="your")

with col2:
    st.markdown("**Competitor Keywords (Site B)**")
    competitor_file = st.file_uploader("Upload competitor keyword data", type=['csv', 'xlsx'], key="competitor")

# Optional content file
st.markdown("### 2. Content Matching (Optional)")
content_file = st.file_uploader("Upload content data for matching keywords to existing pages",
                                 type=['csv', 'xlsx'], key="content")

if your_file and competitor_file:
    # Load data
    try:
        if your_file.name.endswith('.csv'):
            your_df = pd.read_csv(your_file)
        else:
            your_df = pd.read_excel(your_file)

        if competitor_file.name.endswith('.csv'):
            competitor_df = pd.read_csv(competitor_file)
        else:
            competitor_df = pd.read_excel(competitor_file)

        st.success(f"Loaded: Your data ({len(your_df)} rows), Competitor data ({len(competitor_df)} rows)")

        # Column mapping
        st.markdown("### 3. Map Columns")

        col1, col2 = st.columns(2)

        with col1:
            your_kw_col = st.selectbox("Your Keyword Column", list(your_df.columns), key="your_kw")
            your_vol_col = st.selectbox("Your Volume Column (optional)",
                                        ["(None)"] + list(your_df.columns), key="your_vol")

        with col2:
            comp_kw_col = st.selectbox("Competitor Keyword Column", list(competitor_df.columns), key="comp_kw")
            comp_vol_col = st.selectbox("Competitor Volume Column (optional)",
                                        ["(None)"] + list(competitor_df.columns), key="comp_vol")
            comp_url_col = st.selectbox("Competitor URL Column (optional)",
                                        ["(None)"] + list(competitor_df.columns), key="comp_url")

        # Content mapping if provided
        content_data = None
        if content_file:
            try:
                if content_file.name.endswith('.csv'):
                    content_df = pd.read_csv(content_file)
                else:
                    content_df = pd.read_excel(content_file)

                st.markdown("**Content Column Mapping:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    content_url_col = st.selectbox("URL Column", list(content_df.columns))
                with col2:
                    content_text_col = st.selectbox("Content/Text Column", list(content_df.columns))
                with col3:
                    content_h1_col = st.selectbox("H1 Column (optional)", ["(None)"] + list(content_df.columns))

                content_data = (content_df, content_url_col, content_text_col, content_h1_col)
            except Exception as e:
                st.error(f"Error loading content file: {e}")

        # Filters
        st.markdown("### 4. Filters")
        col1, col2 = st.columns(2)

        with col1:
            keyword_filter = st.text_input("Filter keywords containing (optional)", placeholder="e.g., llc")
        with col2:
            min_volume = st.number_input("Minimum search volume", 0, 1000000, 0)

        if st.button("Run Gap Analysis", type="primary"):
            with st.spinner("Analyzing keyword gaps..."):

                # Get unique keywords
                your_keywords = set(your_df[your_kw_col].astype(str).str.lower().dropna())
                comp_keywords = set(competitor_df[comp_kw_col].astype(str).str.lower().dropna())

                # Apply keyword filter
                if keyword_filter:
                    filter_lower = keyword_filter.lower()
                    your_keywords = {k for k in your_keywords if filter_lower in k}
                    comp_keywords = {k for k in comp_keywords if filter_lower in k}

                # Find gaps
                gap_keywords = comp_keywords - your_keywords
                common_keywords = your_keywords.intersection(comp_keywords)
                unique_to_you = your_keywords - comp_keywords

                st.markdown("### Results Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Gap Opportunities", len(gap_keywords))
                col2.metric("Keywords in Common", len(common_keywords))
                col3.metric("Unique to You", len(unique_to_you))

                # Build gap opportunities DataFrame
                gap_rows = []
                for kw in gap_keywords:
                    # Find matching row in competitor data
                    match = competitor_df[competitor_df[comp_kw_col].astype(str).str.lower() == kw]
                    if len(match) > 0:
                        row_data = {'keyword': kw}
                        if comp_vol_col != "(None)":
                            row_data['volume'] = match[comp_vol_col].iloc[0]
                        if comp_url_col != "(None)":
                            row_data['competitor_url'] = match[comp_url_col].iloc[0]
                        gap_rows.append(row_data)

                gap_df = pd.DataFrame(gap_rows)

                # Apply volume filter
                if min_volume > 0 and 'volume' in gap_df.columns:
                    gap_df = gap_df[gap_df['volume'] >= min_volume]

                # Sort by volume
                if 'volume' in gap_df.columns:
                    gap_df = gap_df.sort_values('volume', ascending=False)

                # Content matching
                if content_data and len(gap_df) > 0:
                    content_df, url_col, text_col, h1_col = content_data
                    h1_col = h1_col if h1_col != "(None)" else None

                    st.info("Matching gap keywords to existing content...")
                    processed = preprocess_content(content_df, url_col, text_col, h1_col)

                    matches = find_content_matches(gap_df['keyword'].tolist(), processed)

                    # Add matches to DataFrame
                    gap_df['best_match_url'] = gap_df['keyword'].apply(
                        lambda k: matches.get(k, [{}])[0].get('url', '') if matches.get(k) else ''
                    )
                    gap_df['match_type'] = gap_df['keyword'].apply(
                        lambda k: matches.get(k, [{}])[0].get('match_type', '') if matches.get(k) else ''
                    )
                    gap_df['match_h1'] = gap_df['keyword'].apply(
                        lambda k: matches.get(k, [{}])[0].get('h1', '') if matches.get(k) else ''
                    )

                st.markdown("### Gap Opportunities")
                st.dataframe(gap_df, use_container_width=True)

                # Download
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("Download CSV", gap_df.to_csv(index=False),
                                       "keyword_gap_opportunities.csv", "text/csv")
                with col2:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        gap_df.to_excel(writer, index=False, sheet_name='Gap Keywords')
                    st.download_button("Download Excel", output.getvalue(),
                                       "keyword_gap_opportunities.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload both your keyword data and competitor keyword data to begin analysis.")

# Help section
with st.expander("📖 How to Use"):
    st.markdown("""
    **Step 1: Export keyword data**
    - Export keywords from Ahrefs, SEMrush, or similar tools
    - Include columns for keyword and search volume

    **Step 2: Upload files**
    - Your site's keywords in one file
    - Competitor's keywords in another file
    - Optionally add a content file with your page URLs and content

    **Step 3: Map columns**
    - Select which columns contain keywords and volumes

    **Step 4: Run analysis**
    - Gap keywords = competitor has, you don't
    - Content matching finds existing pages that could target gap keywords
    """)

# Footer
st.markdown("---")
