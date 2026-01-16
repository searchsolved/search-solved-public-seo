"""
Entity Extractor - Extract Named Entities from Text/HTML using SpaCy NLP
Identify people, organizations, locations, and other entities in your content.

Author: Lee Foot
Date: January 2025
"""

import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from io import BytesIO

st.set_page_config(
    page_title="Entity Extractor",
    page_icon="🔍",
    layout="wide"
)

# Check if spaCy is available
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

st.title("🔍 Entity Extractor")
st.markdown("""
Extract named entities (people, organizations, locations) from text using SpaCy NLP.
Identify key topics and entities in your content for semantic SEO analysis.
""")

if not SPACY_AVAILABLE:
    st.error("""
    **SpaCy is not installed.** Please install it with:
    ```bash
    pip install spacy
    python -m spacy download en_core_web_sm
    ```
    """)
    st.stop()

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
model_options = {
    'en_core_web_sm': 'Small (fast, less accurate)',
    'en_core_web_md': 'Medium (balanced)',
    'en_core_web_lg': 'Large (slow, most accurate)'
}

selected_model = st.sidebar.selectbox(
    "SpaCy Model",
    options=list(model_options.keys()),
    format_func=lambda x: f"{x} - {model_options[x]}",
    help="Larger models are more accurate but slower. You need to download the model first."
)

# Entity type filter
entity_types = {
    'PERSON': 'People, including fictional',
    'NORP': 'Nationalities, religious or political groups',
    'FAC': 'Buildings, airports, highways, bridges',
    'ORG': 'Companies, agencies, institutions',
    'GPE': 'Countries, cities, states',
    'LOC': 'Non-GPE locations, mountain ranges, bodies of water',
    'PRODUCT': 'Objects, vehicles, foods (not services)',
    'EVENT': 'Named hurricanes, battles, wars, sports events',
    'WORK_OF_ART': 'Titles of books, songs, etc.',
    'LAW': 'Named documents made into laws',
    'LANGUAGE': 'Any named language',
}

excluded_types = ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']

selected_types = st.sidebar.multiselect(
    "Entity Types to Extract",
    options=list(entity_types.keys()),
    default=list(entity_types.keys()),
    format_func=lambda x: f"{x}: {entity_types[x]}",
    help="Select which entity types to extract"
)

# Load model
@st.cache_resource
def load_spacy_model(model_name):
    """Load spaCy model with caching."""
    try:
        return spacy.load(model_name)
    except OSError:
        return None

nlp = load_spacy_model(selected_model)

if nlp is None:
    st.error(f"""
    **Model '{selected_model}' is not installed.** Download it with:
    ```bash
    python -m spacy download {selected_model}
    ```
    """)
    st.stop()

st.success(f"Using SpaCy model: {selected_model}")


def clean_html(html_content):
    """Clean HTML content and extract text."""
    if pd.isna(html_content):
        return ""

    soup = BeautifulSoup(str(html_content), 'html.parser')

    # Remove specific tags that often contain noise
    for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'ol', 'ul', 'table']):
        tag.decompose()

    return soup.get_text(separator=' ', strip=True)


def extract_entities(text, nlp_model, allowed_types):
    """Extract entities from text using spaCy."""
    if not text or pd.isna(text):
        return []

    # Truncate very long texts to avoid memory issues
    max_chars = 100000
    if len(text) > max_chars:
        text = text[:max_chars]

    doc = nlp_model(text)
    entities = [
        (ent.text.strip(), ent.label_)
        for ent in doc.ents
        if ent.label_ in allowed_types and ent.text.strip()
    ]
    return entities


# Input methods
st.subheader("Input Content")
input_method = st.radio(
    "Choose input method:",
    ["Text Area", "CSV Upload"],
    horizontal=True
)

if input_method == "Text Area":
    text_input = st.text_area(
        "Enter text or HTML content",
        height=300,
        placeholder="Paste your text or HTML content here..."
    )

    if text_input and st.button("🔍 Extract Entities", type="primary"):
        with st.spinner("Extracting entities..."):
            # Clean HTML if present
            cleaned_text = clean_html(text_input)

            # Extract entities
            entities = extract_entities(cleaned_text, nlp, selected_types)

            if entities:
                # Create dataframe
                df = pd.DataFrame(entities, columns=['Entity', 'Label'])

                # Count frequencies
                entity_counts = df.groupby(['Entity', 'Label']).size().reset_index(name='Count')
                entity_counts = entity_counts.sort_values('Count', ascending=False)

                st.subheader("Extracted Entities")

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Entities", len(entities))
                with col2:
                    st.metric("Unique Entities", len(entity_counts))
                with col3:
                    st.metric("Entity Types", entity_counts['Label'].nunique())

                # Display by type
                st.subheader("Entities by Type")
                for label in sorted(entity_counts['Label'].unique()):
                    with st.expander(f"{label} ({len(entity_counts[entity_counts['Label'] == label])})"):
                        type_df = entity_counts[entity_counts['Label'] == label][['Entity', 'Count']]
                        st.dataframe(type_df, use_container_width=True, hide_index=True)

                # Full table
                st.subheader("All Entities")
                st.dataframe(entity_counts, use_container_width=True, hide_index=True)

                # Download
                csv_buffer = BytesIO()
                entity_counts.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)

                st.download_button(
                    label="📥 Download Entities (CSV)",
                    data=csv_buffer,
                    file_name="extracted_entities.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No entities found in the provided text.")

else:
    uploaded_file = st.file_uploader(
        "Upload CSV with content",
        type=['csv', 'xlsx'],
        help="Upload a CSV/Excel file containing text content to analyze"
    )

    if uploaded_file:
        # Load file
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        col1, col2 = st.columns(2)

        with col1:
            content_column = st.selectbox(
                "Select content column",
                options=df.columns.tolist(),
                help="Column containing the text/HTML to analyze"
            )

        with col2:
            id_column = st.selectbox(
                "Select ID column (optional)",
                options=['None'] + df.columns.tolist(),
                help="Column to use as identifier (e.g., URL, Address)"
            )

        if st.button("🔍 Extract Entities", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            all_results = []

            for idx, row in df.iterrows():
                content = row[content_column]
                identifier = row[id_column] if id_column != 'None' else idx

                # Clean and extract
                cleaned = clean_html(content)
                entities = extract_entities(cleaned, nlp, selected_types)

                for entity, label in entities:
                    all_results.append({
                        'Source': identifier,
                        'Entity': entity,
                        'Label': label
                    })

                progress_bar.progress((idx + 1) / len(df))
                status_text.text(f"Processing: {idx + 1}/{len(df)}")

            progress_bar.empty()
            status_text.empty()

            if all_results:
                results_df = pd.DataFrame(all_results)

                # Summary
                st.subheader("Results Summary")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Entities Found", len(results_df))
                with col2:
                    st.metric("Unique Entities", results_df['Entity'].nunique())
                with col3:
                    st.metric("Sources Processed", results_df['Source'].nunique())

                # Entity frequency across all sources
                st.subheader("Top Entities (All Sources)")
                top_entities = results_df.groupby(['Entity', 'Label']).size().reset_index(name='Count')
                top_entities = top_entities.sort_values('Count', ascending=False).head(50)
                st.dataframe(top_entities, use_container_width=True, hide_index=True)

                # Full results
                with st.expander("View All Results"):
                    st.dataframe(results_df, use_container_width=True, hide_index=True)

                # Downloads
                st.subheader("Download Results")

                col1, col2 = st.columns(2)

                with col1:
                    # Full results
                    csv_buffer = BytesIO()
                    results_df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)

                    st.download_button(
                        label="📥 Download Full Results (CSV)",
                        data=csv_buffer,
                        file_name="entities_by_source.csv",
                        mime="text/csv"
                    )

                with col2:
                    # Aggregated counts
                    agg_buffer = BytesIO()
                    top_entities_full = results_df.groupby(['Entity', 'Label']).size().reset_index(name='Count')
                    top_entities_full = top_entities_full.sort_values('Count', ascending=False)
                    top_entities_full.to_csv(agg_buffer, index=False)
                    agg_buffer.seek(0)

                    st.download_button(
                        label="📥 Download Entity Counts (CSV)",
                        data=agg_buffer,
                        file_name="entity_counts.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("No entities found in the uploaded content.")

if input_method == "Text Area" and not text_input:
    st.info("👆 Enter text or upload a file to get started.")

    st.markdown("""
    ### Entity Types Explained
    | Type | Description | Examples |
    |------|-------------|----------|
    | PERSON | People names | Elon Musk, Shakespeare |
    | ORG | Organizations | Google, NASA, WHO |
    | GPE | Countries/Cities | London, United States |
    | LOC | Locations | Mount Everest, Pacific Ocean |
    | PRODUCT | Products | iPhone, Tesla Model S |
    | EVENT | Events | World War II, Olympics |
    | WORK_OF_ART | Creative works | Mona Lisa, Hamlet |

    ### Requirements
    - SpaCy library installed
    - SpaCy English model downloaded

    ```bash
    pip install spacy
    python -m spacy download en_core_web_sm  # Small (fast)
    python -m spacy download en_core_web_md  # Medium
    python -m spacy download en_core_web_lg  # Large (best)
    ```

    ### Use Cases
    - Content entity analysis for semantic SEO
    - Identifying key topics in competitor content
    - Building comprehensive topic coverage
    - Understanding entity relationships in content
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("Built by [Lee Foot](https://leefoot.co.uk)")
