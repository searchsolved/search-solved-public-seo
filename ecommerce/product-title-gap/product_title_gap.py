####################################################################################
#                                                                                  #
#  Product Title Gap Analyzer                                                      #
#                                                                                  #
#  Compare product titles with competitors using MPN matching.                     #
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
Product Title Gap Analyzer

Compares your product titles with competitors using MPN (Manufacturer Part Number)
as a common key. Identifies missing words that competitors use in their titles
to help optimize your product pages.

Features:
- Upload your crawl and multiple competitor crawls
- Match products by MPN or custom identifier
- Find missing words with frequency analysis
- Multi-language stopword support
- Export optimization recommendations
"""

import streamlit as st
import pandas as pd
from collections import Counter
import string
import nltk
from nltk.corpus import stopwords
import statistics

st.set_page_config(page_title="Product Title Gap Analyzer", page_icon="🔍", layout="wide")

# Download NLTK resources
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')
    nltk.download('punkt')

st.title("Product Title Gap Analyzer")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logo=bluesky&logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social)")

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Compares your product titles with competitor titles using MPN/SKU matching
    - Identifies words that competitors use but you don't
    - Shows which titles might be missing important keywords
    - Helps optimize product pages for better visibility

    **How to get the data:**
    1. Crawl your site with Screaming Frog with custom extraction for MPN/SKU
    2. Crawl competitor sites with the same extraction
    3. Export crawl CSVs with at least: URL, H1/Title, and MPN columns

    **Required columns:**
    - `url` - The page URL
    - `h1` or `title` - The product title/heading
    - `mpn` - The MPN, SKU, or common product identifier
    """)

# Sidebar settings
st.sidebar.header("Settings")

language = st.sidebar.selectbox(
    "Stopwords Language",
    ["english", "german", "french", "spanish", "italian", "dutch", "portuguese"],
    help="Select the language for stopword removal"
)

try:
    stop_words = set(stopwords.words(language))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words(language))

custom_stopwords = st.sidebar.text_area(
    "Custom stopwords (one per line)",
    help="Add additional words to ignore"
)
if custom_stopwords:
    custom_list = [w.strip().lower() for w in custom_stopwords.split('\n') if w.strip()]
    stop_words.update(custom_list)

min_word_length = st.sidebar.number_input(
    "Minimum word length",
    min_value=1,
    max_value=10,
    value=2,
    help="Ignore words shorter than this"
)

# File uploads
st.subheader("Upload Your Crawl")
your_file = st.file_uploader(
    "Upload your site's crawl CSV",
    type=['csv'],
    key="your_crawl",
    help="Export from Screaming Frog with MPN custom extraction"
)

st.subheader("Upload Competitor Crawls")
competitor_files = st.file_uploader(
    "Upload competitor crawl CSVs",
    type=['csv'],
    accept_multiple_files=True,
    key="competitor_crawls",
    help="Upload one or more competitor crawl exports"
)


def preprocess_text(text, stop_words, min_length):
    """Remove punctuation and stop words from text."""
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) >= min_length]
    return ' '.join(words)


def find_column(df, possible_names):
    """Find a column by trying multiple possible names."""
    for name in possible_names:
        for col in df.columns:
            if name.lower() in col.lower():
                return col
    return None


if your_file is not None and competitor_files:
    try:
        # Load your crawl
        try:
            df_source = pd.read_csv(your_file, encoding='utf-8')
        except:
            your_file.seek(0)
            df_source = pd.read_csv(your_file, encoding='latin-1')

        st.success(f"Loaded your crawl: {len(df_source):,} URLs")

        # Find required columns
        url_col = find_column(df_source, ['url', 'address'])
        title_col = find_column(df_source, ['h1', 'title 1', 'title', 'heading'])
        mpn_col = find_column(df_source, ['mpn', 'sku', 'product_id', 'part_number'])

        with st.expander("Column Mapping"):
            col1, col2, col3 = st.columns(3)
            with col1:
                url_col = st.selectbox("URL column", df_source.columns.tolist(),
                                       index=df_source.columns.tolist().index(url_col) if url_col else 0)
            with col2:
                title_col = st.selectbox("Title/H1 column", df_source.columns.tolist(),
                                         index=df_source.columns.tolist().index(title_col) if title_col else 0)
            with col3:
                mpn_col = st.selectbox("MPN/SKU column", df_source.columns.tolist(),
                                       index=df_source.columns.tolist().index(mpn_col) if mpn_col else 0)

        # Load competitor crawls
        competitor_dfs = []
        for comp_file in competitor_files:
            try:
                try:
                    df_comp = pd.read_csv(comp_file, encoding='utf-8')
                except:
                    comp_file.seek(0)
                    df_comp = pd.read_csv(comp_file, encoding='latin-1')

                # Find MPN column in competitor file
                comp_mpn_col = find_column(df_comp, ['mpn', 'sku', 'product_id', 'part_number'])
                comp_title_col = find_column(df_comp, ['h1', 'title 1', 'title', 'heading'])
                comp_url_col = find_column(df_comp, ['url', 'address'])

                if comp_mpn_col and comp_title_col:
                    df_comp = df_comp.rename(columns={
                        comp_mpn_col: 'mpn',
                        comp_title_col: 'h1',
                        comp_url_col: 'url' if comp_url_col else 'url'
                    })
                    df_comp['mpn'] = df_comp['mpn'].astype(str).str.lower().str.strip()
                    competitor_dfs.append(df_comp)
                    st.success(f"Loaded competitor: {comp_file.name} ({len(df_comp):,} URLs)")
                else:
                    st.warning(f"Could not find required columns in {comp_file.name}")
            except Exception as e:
                st.warning(f"Error loading {comp_file.name}: {str(e)}")

        if competitor_dfs and st.button("Analyze Title Gaps", type="primary"):
            with st.spinner("Analyzing product titles..."):
                # Prepare source data
                df_work = df_source.copy()
                df_work = df_work.rename(columns={
                    url_col: 'url',
                    title_col: 'h1',
                    mpn_col: 'mpn'
                })

                df_work['mpn_matching'] = df_work['mpn'].astype(str).str.lower().str.strip()
                df_work['h1_original'] = df_work['h1']
                df_work['h1'] = df_work['h1'].astype(str).str.lower().str.strip()
                df_work = df_work.dropna(subset=['h1'])
                df_work = df_work[df_work['h1'] != 'nan']

                def find_missing_words(row):
                    words = preprocess_text(row['h1'], stop_words, min_word_length).split()
                    missing_words = []
                    matching_urls = []
                    matching_h1s = []
                    h1_lengths = []
                    freq_source = Counter(words)

                    for comp_df in competitor_dfs:
                        matched_df = comp_df[comp_df['mpn'] == row['mpn_matching']]
                        if not matched_df.empty:
                            if 'url' in matched_df.columns:
                                matching_urls.append(matched_df['url'].tolist()[0])
                            matching_h1s.append(matched_df['h1'].tolist()[0])

                            comparison_h1 = preprocess_text(
                                matched_df['h1'].tolist()[0],
                                stop_words,
                                min_word_length
                            )
                            freq_comparison = Counter(comparison_h1.split())

                            for word in freq_comparison:
                                if word.lower() not in freq_source:
                                    missing_words.append((word, freq_comparison[word]))

                            h1_lengths.append(len(comparison_h1.split()))

                    source_h1_length = len(words)
                    median_comparison_h1_length = statistics.median(h1_lengths) if h1_lengths else 0
                    median_length_difference = median_comparison_h1_length - source_h1_length

                    return (
                        missing_words,
                        matching_urls,
                        matching_h1s,
                        source_h1_length,
                        median_comparison_h1_length,
                        median_length_difference
                    )

                # Apply the analysis
                progress_bar = st.progress(0)
                results = []
                total_rows = len(df_work)

                for idx, (_, row) in enumerate(df_work.iterrows()):
                    result = find_missing_words(row)
                    results.append(result)
                    progress_bar.progress((idx + 1) / total_rows)

                df_work[['missing_words', 'matching_urls', 'matching_h1s',
                         'source_h1_length', 'median_comparison_h1_length',
                         'median_length_difference']] = pd.DataFrame(results, index=df_work.index)

                # Process results
                df_work['missing_words'] = df_work['missing_words'].apply(
                    lambda x: Counter([word for word, freq in x]).most_common()
                )

                # Filter to only products with matches
                df_work = df_work[df_work['matching_urls'].apply(lambda x: len(x) > 0)]

                if len(df_work) == 0:
                    st.warning("No matching products found between your crawl and competitors. "
                               "Check that MPNs match across files.")
                else:
                    # Find most verbose competitor title
                    df_work['most_verbose_h1'] = df_work['matching_h1s'].apply(
                        lambda x: sorted(x, key=lambda h1: len(str(h1).split()), reverse=True)[0] if x else ''
                    )

                    # Sort by gap size
                    df_work = df_work.sort_values(by='median_length_difference', ascending=False)

                    # Restore original title
                    df_work['h1'] = df_work['h1_original']

                    # Clean up output
                    output_df = df_work[[
                        'url', 'mpn', 'h1', 'missing_words', 'median_length_difference',
                        'most_verbose_h1', 'matching_urls'
                    ]].copy()
                    output_df.columns = [
                        'Your URL', 'MPN', 'Your Title', 'Missing Words',
                        'Length Gap', 'Best Competitor Title', 'Competitor URLs'
                    ]

                    # Display results
                    st.subheader("Results")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Products Analyzed", f"{len(output_df):,}")
                    with col2:
                        avg_gap = output_df['Length Gap'].mean()
                        st.metric("Avg Length Gap", f"{avg_gap:.1f} words")
                    with col3:
                        with_gaps = len(output_df[output_df['Length Gap'] > 0])
                        st.metric("Products with Gaps", f"{with_gaps:,}")

                    # Show products with biggest gaps
                    st.subheader("Products with Title Gaps")
                    st.dataframe(output_df.head(100), use_container_width=True)

                    # Word frequency analysis
                    st.subheader("Most Commonly Missing Words")
                    all_missing = []
                    for words in output_df['Missing Words']:
                        if words:
                            for word, count in words:
                                all_missing.append(word)

                    word_freq = pd.DataFrame(Counter(all_missing).most_common(50),
                                             columns=['Word', 'Frequency'])
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(word_freq, use_container_width=True)
                    with col2:
                        st.bar_chart(word_freq.head(20).set_index('Word'))

                    # Download
                    st.subheader("Download")
                    csv_output = output_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="Download Results CSV",
                        data=csv_output,
                        file_name="product_title_gaps.csv",
                        mime="text/csv"
                    )

    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("Upload your crawl and at least one competitor crawl to begin")

    st.subheader("Example Analysis")
    example_data = {
        "Your Title": ["Widget Pro 2000", "Gadget X100"],
        "Competitor Title": ["Widget Pro 2000 Industrial Heavy Duty Steel", "Gadget X100 Professional Grade Aluminum"],
        "Missing Words": ["industrial, heavy, duty, steel", "professional, grade, aluminum"],
        "Length Gap": [4, 3]
    }
    st.dataframe(pd.DataFrame(example_data))
