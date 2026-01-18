####################################################################################
#                                                                                  #
#  Keyword-to-Page Mapper                                                          #
#                                                                                  #
#  Semantically match keywords to existing pages using ML embeddings.              #
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
Keyword-to-Page Mapper

Uses Sentence Transformers and PolyFuzz to semantically match competitor keywords
to your existing page H1s/titles. Identifies content opportunities and gaps.

Features:
- Upload your page crawl (URLs with H1s/titles)
- Upload competitor keywords (from Ahrefs, SEMrush, etc.)
- Semantic matching using sentence embeddings
- Configurable similarity threshold
- Find unmapped keywords = content gaps
"""

import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Keyword-to-Page Mapper", page_icon="🗺️", layout="wide")

st.title("Keyword-to-Page Mapper")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-Follow-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Matches external/competitor keywords to your existing pages
    - Uses AI embeddings for semantic similarity (not just exact match)
    - Identifies content gaps (keywords with no good page match)

    **Data requirements:**

    1. **Your pages CSV** with columns:
       - URL/Address
       - H1 or Page Title

    2. **Keywords CSV** with columns:
       - Keyword
       - Volume (optional but recommended)

    **How to use:**
    1. Upload your site crawl export (Screaming Frog, Sitebulb, etc.)
    2. Upload competitor keywords (Ahrefs, SEMrush export)
    3. Map the columns
    4. Set similarity threshold
    5. Click "Map Keywords to Pages"

    **Note:** First run may be slow as the ML model downloads.
    """)

# Sidebar settings
st.sidebar.header("Matching Settings")

similarity_threshold = st.sidebar.slider(
    "Minimum similarity",
    min_value=0.5,
    max_value=0.95,
    value=0.75,
    step=0.05,
    help="Higher = stricter matching, fewer results. Lower = more matches but potentially less relevant."
)

remove_exact_matches = st.sidebar.checkbox(
    "Exclude exact matches",
    value=True,
    help="Skip keywords already found exactly in page content"
)

transformer_model = st.sidebar.selectbox(
    "Embedding model",
    [
        "all-MiniLM-L6-v2",
        "multi-qa-mpnet-base-dot-v1",
        "all-mpnet-base-v2"
    ],
    index=0,
    help="MiniLM is fastest, mpnet-base is most accurate"
)

# Cache the model loading
@st.cache_resource
def load_models(model_name):
    """Load PolyFuzz and embedding models."""
    try:
        from polyfuzz import PolyFuzz
        from polyfuzz.models import SentenceEmbeddings
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer(model_name)
        distance_model = SentenceEmbeddings(embedding_model)
        return PolyFuzz(distance_model), True
    except Exception as e:
        return None, str(e)


# File uploads
st.subheader("1. Upload Your Pages")

pages_file = st.file_uploader(
    "Upload crawl CSV (URL + H1/Title)",
    type=['csv'],
    key="pages_file"
)

pages_df = None
page_url_col = None
page_content_col = None

if pages_file:
    try:
        pages_df = pd.read_csv(pages_file)
        st.success(f"Loaded {len(pages_df)} pages")

        col1, col2 = st.columns(2)
        with col1:
            page_url_col = st.selectbox(
                "URL column",
                pages_df.columns.tolist(),
                index=pages_df.columns.tolist().index('Address') if 'Address' in pages_df.columns else 0
            )
        with col2:
            # Try to find H1 column
            h1_options = [c for c in pages_df.columns if 'h1' in c.lower() or 'title' in c.lower()]
            default_idx = pages_df.columns.tolist().index(h1_options[0]) if h1_options else 0
            page_content_col = st.selectbox(
                "H1/Title column",
                pages_df.columns.tolist(),
                index=default_idx
            )

    except Exception as e:
        st.error(f"Error reading pages CSV: {str(e)}")

st.subheader("2. Upload Keywords")

keywords_file = st.file_uploader(
    "Upload keywords CSV",
    type=['csv'],
    key="keywords_file"
)

keywords_df = None
keyword_col = None
volume_col = None

if keywords_file:
    try:
        keywords_df = pd.read_csv(keywords_file)
        st.success(f"Loaded {len(keywords_df)} keywords")

        col1, col2 = st.columns(2)
        with col1:
            kw_options = [c for c in keywords_df.columns if 'keyword' in c.lower() or 'query' in c.lower()]
            default_kw_idx = keywords_df.columns.tolist().index(kw_options[0]) if kw_options else 0
            keyword_col = st.selectbox(
                "Keyword column",
                keywords_df.columns.tolist(),
                index=default_kw_idx
            )
        with col2:
            vol_options = [c for c in keywords_df.columns if 'vol' in c.lower()]
            default_vol_idx = keywords_df.columns.tolist().index(vol_options[0]) if vol_options else 0
            volume_col = st.selectbox(
                "Volume column (optional)",
                ['None'] + keywords_df.columns.tolist(),
                index=0 if not vol_options else keywords_df.columns.tolist().index(vol_options[0]) + 1
            )
            if volume_col == 'None':
                volume_col = None

    except Exception as e:
        st.error(f"Error reading keywords CSV: {str(e)}")

# Run mapping
if st.button("Map Keywords to Pages", type="primary",
             disabled=pages_df is None or keywords_df is None):

    with st.spinner("Loading embedding model... (first run may download model)"):
        model, status = load_models(transformer_model)

        if model is None:
            st.error(f"Failed to load model: {status}")
            st.info("Install required packages: pip install polyfuzz sentence-transformers")
        else:
            st.success("Model loaded")

            # Prepare data
            pages_df_clean = pages_df[[page_url_col, page_content_col]].copy()
            pages_df_clean = pages_df_clean[pages_df_clean[page_content_col].notna()]
            pages_df_clean[page_content_col] = pages_df_clean[page_content_col].str.lower()

            keywords_df_clean = keywords_df.copy()
            keywords_df_clean = keywords_df_clean[keywords_df_clean[keyword_col].notna()]

            to_list = list(pages_df_clean[page_content_col])
            from_list = list(keywords_df_clean[keyword_col].str.lower())

            st.info(f"Matching {len(from_list)} keywords to {len(to_list)} pages...")

            progress_bar = st.progress(0)

            # Run matching
            with st.spinner("Computing semantic similarity..."):
                model.match(from_list, to_list)
                progress_bar.progress(50)

                df_matches = model.get_matches()
                progress_bar.progress(80)

            # Filter by similarity
            df_matches = df_matches[df_matches['Similarity'] >= similarity_threshold]

            # Remove exact matches if requested
            if remove_exact_matches:
                df_matches['From_lower'] = df_matches['From'].str.lower()
                df_matches['To_lower'] = df_matches['To'].str.lower()
                df_matches['is_exact'] = df_matches.apply(
                    lambda row: row['From_lower'] in row['To_lower'], axis=1
                )
                df_matches = df_matches[~df_matches['is_exact']]
                df_matches = df_matches.drop(columns=['From_lower', 'To_lower', 'is_exact'])

            # Rename columns
            df_matches = df_matches.rename(columns={
                'From': 'Keyword',
                'To': 'Matched H1'
            })

            # Add URL lookup
            h1_to_url = dict(zip(
                pages_df_clean[page_content_col].str.lower(),
                pages_df_clean[page_url_col]
            ))
            df_matches['URL'] = df_matches['Matched H1'].map(h1_to_url)

            # Add volume if available
            if volume_col:
                kw_to_vol = dict(zip(
                    keywords_df_clean[keyword_col].str.lower(),
                    keywords_df_clean[volume_col]
                ))
                df_matches['Volume'] = df_matches['Keyword'].map(kw_to_vol)

            # Count opportunities per page
            df_matches['Opportunity Size'] = df_matches['URL'].map(
                df_matches.groupby('URL').size()
            )

            # Sort
            if volume_col:
                df_matches = df_matches.sort_values(['Volume', 'Similarity'], ascending=[False, False])
            else:
                df_matches = df_matches.sort_values('Similarity', ascending=False)

            progress_bar.progress(100)

            # Store results
            st.session_state['mapped_keywords'] = df_matches
            st.session_state['unmapped_keywords'] = set(from_list) - set(df_matches['Keyword'].str.lower())

            st.success(f"Found {len(df_matches)} keyword-to-page matches!")

# Display results
if 'mapped_keywords' in st.session_state:
    df_matches = st.session_state['mapped_keywords']
    unmapped = st.session_state['unmapped_keywords']

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mapped Keywords", len(df_matches))
    with col2:
        st.metric("Unmapped Keywords", len(unmapped))
    with col3:
        unique_pages = df_matches['URL'].nunique()
        st.metric("Pages with Matches", unique_pages)
    with col4:
        if 'Volume' in df_matches.columns:
            total_vol = df_matches['Volume'].sum()
            st.metric("Total Volume (Mapped)", f"{int(total_vol):,}")

    # Mapped keywords
    st.subheader("Mapped Keywords")
    st.dataframe(df_matches, use_container_width=True)

    # Unmapped keywords (content gaps)
    if unmapped:
        st.subheader("Unmapped Keywords (Content Gaps)")
        st.info("These keywords couldn't be matched to any existing page - potential new content opportunities!")

        unmapped_df = pd.DataFrame({'Keyword': list(unmapped)})

        # Add volume if available
        if volume_col and keywords_df is not None:
            kw_to_vol = dict(zip(
                keywords_df[keyword_col].str.lower(),
                keywords_df[volume_col]
            ))
            unmapped_df['Volume'] = unmapped_df['Keyword'].map(kw_to_vol)
            unmapped_df = unmapped_df.sort_values('Volume', ascending=False)

        st.dataframe(unmapped_df, use_container_width=True)

    # Downloads
    st.subheader("Download")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_data = df_matches.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="Download Mapped (CSV)",
            data=csv_data,
            file_name="keyword_page_mapping.csv",
            mime="text/csv"
        )

    with col2:
        if unmapped:
            unmapped_csv = unmapped_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="Download Unmapped (CSV)",
                data=unmapped_csv,
                file_name="content_gaps.csv",
                mime="text/csv"
            )

    with col3:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_matches.to_excel(writer, sheet_name='Mapped Keywords', index=False)
            if unmapped:
                unmapped_df.to_excel(writer, sheet_name='Content Gaps', index=False)

        st.download_button(
            label="Download Excel Report",
            data=output.getvalue(),
            file_name="keyword_mapping_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.subheader("Example Output")

    example_data = {
        "Keyword": ["best running shoes", "marathon training", "nike vs adidas"],
        "Matched H1": ["top running shoes 2024", "how to train for a marathon", "comparing running shoe brands"],
        "Similarity": [0.89, 0.85, 0.78],
        "URL": ["example.com/shoes", "example.com/training", "example.com/compare"],
        "Volume": [12000, 5000, 3000]
    }
    st.dataframe(pd.DataFrame(example_data))
