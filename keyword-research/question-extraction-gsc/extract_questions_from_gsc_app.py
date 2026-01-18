"""
Question Extraction from GSC - Streamlit App

Extracts question-type keywords from Google Search Console data using pattern matching.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import streamlit as st
import pandas as pd
import string
import re

st.set_page_config(
    page_title="Question Extraction from GSC",
    page_icon="❓",
    layout="wide"
)

st.title("❓ Question Extraction from GSC")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-181717?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Extracts question queries from GSC data
    - Identifies FAQ opportunities from search data
    - Filters for question-format queries

    **How to use:**
    1. Upload GSC query export
    2. Configure question filters
    3. Extract question queries
    4. Download FAQ opportunities

    **Best for:**
    - FAQ content optimization
    - Featured snippet targeting
    - People Also Ask strategy
    """)
st.markdown("Extract question-type keywords from your Google Search Console data.")

# Default question patterns
DEFAULT_PATTERN = r'\b(?:who|what|when|where|why|how|are|do|did|can|will|is|am|should|may|might|' \
                  r'adjusting|cutting|measuring|weight|height|depth|installing|instalation|best|' \
                  r'types|type|vs|building a|regulations|changing|change|choose|choosing|cleaning|' \
                  r'converting|convert|cost|price|different|measure|measurement|do i|do you|size|' \
                  r'sizes|thickness|dimensions|meaning|definition|terminology|difference|fitting|' \
                  r'slating|tiling|insulating|putting up|draught proofing|fixing|repairing|hanging|' \
                  r'painting|mounting|replacing|resealing|sanding|sealing|trimming|adding|boarding|' \
                  r'laying|is it|making a|mixing|moving|putting|reduce|reducing|replace|rendering|' \
                  r'skimming|options|water proofing|waterproofing|calculating|calculator|alternative|' \
                  r'alternatives|substitute|capping off|planning permission|prevent|preventing|' \
                  r'pros and cons|recoating|re-coating|removing|repair|repointing|retiling|re-tiling|' \
                  r'aligning|welding|using|finishing|preparing|priming)\b'

LOOSE_PATTERN = r'(?i)(\bwhat\b|\bwho\b|\bwhom\b|\bwhose\b|\bwhere\b|\bwhen\b|\bwhy\b|\bhow\b|\bwhich\b|\bwhether\b|\bif\b|\bdo\b|\bdoes\b|\bdid\b|\bcould\b|\bcan\b|\bwill\b|\bwould\b)'


def extract_questions(df, query_col, pattern, include_question_marks=True, clean_punctuation=True):
    """Extract question-type queries from dataframe."""
    df = df.copy()

    # Start with question mark queries if enabled
    questions = pd.DataFrame()
    if include_question_marks:
        question_mark_df = df[df[query_col].str.contains(r'\?', na=False, regex=True)]
        questions = pd.concat([questions, question_mark_df])

    # Filter by pattern
    pattern_matches = df[df[query_col].str.contains(pattern, na=False, regex=True)]
    questions = pd.concat([questions, pattern_matches])

    # Remove duplicates
    questions = questions.drop_duplicates(subset=[query_col])

    # Clean punctuation if enabled
    if clean_punctuation:
        punctuation_pattern = f"[{string.punctuation}]"
        questions[query_col] = questions[query_col].str.replace(punctuation_pattern, '', regex=True)
        questions[query_col] = questions[query_col].str.split().str.join(' ')

    # Remove any duplicates after cleaning
    questions = questions.drop_duplicates(subset=[query_col])

    # Sort by impressions if available
    if 'impressions' in questions.columns:
        questions = questions.sort_values('impressions', ascending=False)

    return questions.reset_index(drop=True)


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    pattern_type = st.radio(
        "Pattern Type",
        ["Strict (Recommended)", "Loose", "Custom"],
        help="Strict has fewer false positives, Loose catches more questions"
    )

    if pattern_type == "Custom":
        custom_pattern = st.text_area(
            "Custom Regex Pattern",
            value=DEFAULT_PATTERN,
            height=150,
            help="Enter a valid regex pattern"
        )
        pattern = custom_pattern
    elif pattern_type == "Loose":
        pattern = LOOSE_PATTERN
    else:
        pattern = DEFAULT_PATTERN

    st.markdown("---")

    include_question_marks = st.checkbox(
        "Include queries with ?",
        value=True,
        help="Include all queries containing question marks"
    )

    clean_punctuation = st.checkbox(
        "Clean punctuation",
        value=True,
        help="Remove punctuation from queries"
    )

    st.markdown("---")
    st.markdown("### 📖 Data Export Guide")
    st.markdown("""
    **From GSC API:**
    Export query data with impressions

    **From GSC UI:**
    1. Go to Performance
    2. Click 'Queries'
    3. Export as CSV
    """)

# Main content
st.markdown("### 📤 Upload GSC Query Data")

uploaded_file = st.file_uploader(
    "Upload CSV with queries",
    type=["csv"],
    help="CSV should contain a query column from GSC"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df):,} queries")

    # Find query column
    query_columns = [c for c in df.columns if 'query' in c.lower()]
    if query_columns:
        query_col = st.selectbox("Query Column", query_columns)
    else:
        query_col = st.selectbox("Query Column", df.columns.tolist())

    # Preview
    with st.expander("Preview Data"):
        st.dataframe(df.head(20), use_container_width=True)

    if st.button("🔍 Extract Questions", type="primary", use_container_width=True):
        try:
            with st.spinner("Extracting questions..."):
                questions_df = extract_questions(
                    df, query_col, pattern,
                    include_question_marks, clean_punctuation
                )

            if len(questions_df) > 0:
                st.success(f"✅ Found {len(questions_df):,} question-type queries!")

                # Results tabs
                tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Analysis", "🔧 Pattern Test"])

                with tab1:
                    st.dataframe(questions_df, use_container_width=True, height=400)

                    csv = questions_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Questions CSV",
                        data=csv,
                        file_name="gsc_questions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with tab2:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Total Questions", len(questions_df))
                        st.metric("% of All Queries", f"{len(questions_df) / len(df) * 100:.1f}%")

                    with col2:
                        if 'impressions' in questions_df.columns:
                            total_impressions = questions_df['impressions'].sum()
                            st.metric("Total Impressions", f"{total_impressions:,.0f}")

                        if 'clicks' in questions_df.columns:
                            total_clicks = questions_df['clicks'].sum()
                            st.metric("Total Clicks", f"{total_clicks:,.0f}")

                    # Word frequency in questions
                    st.subheader("Most Common Question Words")
                    all_words = ' '.join(questions_df[query_col].tolist()).lower().split()
                    question_words = ['how', 'what', 'why', 'when', 'where', 'who', 'which', 'can', 'do', 'is', 'are']
                    word_counts = {w: all_words.count(w) for w in question_words if all_words.count(w) > 0}

                    if word_counts:
                        import plotly.express as px
                        fig = px.bar(
                            x=list(word_counts.keys()),
                            y=list(word_counts.values()),
                            labels={'x': 'Question Word', 'y': 'Count'},
                            title='Question Word Frequency'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Top questions by impressions
                    if 'impressions' in questions_df.columns:
                        st.subheader("Top 10 Questions by Impressions")
                        top_10 = questions_df.nlargest(10, 'impressions')
                        st.dataframe(top_10[[query_col, 'impressions']], use_container_width=True)

                with tab3:
                    st.subheader("Test Pattern")
                    test_query = st.text_input("Enter a query to test", "how do I install")

                    if test_query:
                        is_match = bool(re.search(pattern, test_query.lower()))
                        has_question_mark = '?' in test_query

                        if is_match or (include_question_marks and has_question_mark):
                            st.success("✅ This query WOULD be extracted")
                            if is_match:
                                st.info("Matched by pattern")
                            if has_question_mark:
                                st.info("Contains question mark")
                        else:
                            st.warning("❌ This query would NOT be extracted")

                    st.markdown("---")
                    st.markdown("**Current Pattern:**")
                    st.code(pattern, language='regex')

            else:
                st.warning("No questions found with the current settings. Try using a looser pattern.")

        except re.error as e:
            st.error(f"Invalid regex pattern: {str(e)}")

else:
    st.info("👆 Upload a CSV file with GSC query data to get started")

    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        This tool helps you **extract question-type queries** from your GSC data:

        - **Identify** user questions and informational queries
        - **Find** content opportunities for FAQ pages
        - **Discover** topics to address in your content

        **Pattern Types:**

        - **Strict**: Uses a comprehensive list of question indicators with lower false positives
        - **Loose**: Catches more questions but may include some false positives
        - **Custom**: Define your own regex pattern
        """)

    with st.expander("Example Question Patterns"):
        st.markdown("""
        The tool looks for patterns like:

        - **Question words**: who, what, when, where, why, how
        - **Helping verbs**: can, will, should, is, are, do
        - **Action words**: installing, measuring, choosing, comparing
        - **Comparison**: vs, best, types, alternative
        - **Cost-related**: cost, price, calculator
        """)

# Footer
st.markdown("---")
st.markdown(
    "Built by [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · "
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · "
    "[![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)"
)
