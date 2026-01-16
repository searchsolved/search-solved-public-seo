import streamlit as st
import pandas as pd
from polyfuzz import PolyFuzz
from io import BytesIO

st.set_page_config(page_title="Keyword Grouper", page_icon="🔗", layout="wide")

st.title("Keyword Grouper")
st.markdown("*Created by [Lee Foot](https://leefoot.co.uk)*")
st.warning("**Experimental Tool** - This is a proof of concept and may have limitations.")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Groups similar keywords together using TF-IDF similarity matching
    - Uses PolyFuzz for fast fuzzy matching at scale
    - Great for initial keyword clustering before manual review

    **Best for:**
    - Grouping question-based keywords
    - Clustering long-tail keyword variations
    - Quick keyword organization before deeper analysis

    **Note:** This uses TF-IDF matching, not semantic/embedding matching.
    For semantic clustering, use the Semantic Keyword Clustering tool instead.
    """)

# Sidebar settings
st.sidebar.header("Settings")
similarity_threshold = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.5,
    max_value=0.99,
    value=0.80,
    step=0.05,
    help="Higher values = stricter matching, smaller clusters"
)

# File upload
uploaded_file = st.file_uploader("Upload CSV with keywords", type=['csv'])

if uploaded_file is not None:
    try:
        # Try different encodings
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin-1')

        st.success(f"Loaded {len(df):,} rows")

        # Column selection
        columns = df.columns.tolist()
        keyword_col = st.selectbox("Select keyword column", columns)

        with st.expander("Preview uploaded data"):
            st.dataframe(df.head(20))

        if st.button("Group Keywords", type="primary"):
            with st.spinner("Grouping keywords with TF-IDF matching..."):
                # Clean the data
                df_work = df.copy()
                df_work = df_work[df_work[keyword_col].notna()]
                df_work[keyword_col] = df_work[keyword_col].astype(str)

                keyword_list = df_work[keyword_col].tolist()

                if len(keyword_list) < 2:
                    st.error("Need at least 2 keywords to group")
                    st.stop()

                # Do the matching
                model = PolyFuzz("TF-IDF")
                model.match(keyword_list, keyword_list)
                model.group(link_min_similarity=similarity_threshold)
                df_matched = model.get_matches()

                # Clean the results
                df_matched['cluster_size'] = df_matched['Group'].map(
                    df_matched.groupby('Group')['Group'].count()
                )
                df_matched.sort_values(
                    ["cluster_size", "Group"],
                    ascending=[False, True],
                    inplace=True
                )

                # Rename columns for clarity
                df_matched.rename(columns={
                    "From": "Keyword",
                    "To": "Matched To",
                    "Similarity": "Match Score",
                    "Group": "Cluster Name",
                    "cluster_size": "Cluster Size"
                }, inplace=True)

                # Mark unclustered keywords
                df_matched.loc[df_matched["Cluster Size"] == 1, "Cluster Name"] = "[No Cluster]"

                # Display summary
                total_keywords = len(df_matched)
                clustered = len(df_matched[df_matched["Cluster Name"] != "[No Cluster]"])
                num_clusters = df_matched[df_matched["Cluster Name"] != "[No Cluster]"]["Cluster Name"].nunique()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Keywords", f"{total_keywords:,}")
                with col2:
                    st.metric("Clustered", f"{clustered:,}")
                with col3:
                    st.metric("Clusters Found", f"{num_clusters:,}")

                # Show results
                st.subheader("Grouped Keywords")
                st.dataframe(df_matched, use_container_width=True)

                # Download button
                output = BytesIO()
                df_matched.to_csv(output, index=False)
                st.download_button(
                    label="Download Grouped Keywords CSV",
                    data=output.getvalue(),
                    file_name="grouped_keywords.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

else:
    st.info("Upload a CSV file with keywords to get started")

    st.subheader("Example Use Cases")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Question Keywords**")
        st.code("how to clean a carpet\nhow to clean carpets\ncleaning carpet how to\ncarpet cleaning tips")

    with col2:
        st.markdown("**Product Variations**")
        st.code("blue running shoes size 10\nrunning shoes blue size 10\nsize 10 blue running shoes")
