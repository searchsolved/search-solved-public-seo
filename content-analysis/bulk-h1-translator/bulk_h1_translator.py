# Author: Lee Foot
# Website: https://leefoot.com

####################################################################################
#                                                                                  #
#  Bulk H1 Translator                                                              #
#                                                                                  #
#  Translate H1 headings to English in bulk using any OpenAI-compatible API.       #
#                                                                                  #
####################################################################################
# Author: Lee Foot                                                              #
# Website  : https://www.leefoot.com                                                   #
# Contact  : https://www.leefoot.com/contact                                           #
# Email    : hello@leefoot.com                                                     #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Bluesky  : https://bsky.app/profile/leefootseo.bsky.social                       #
####################################################################################

"""
Bulk H1 Translator

Reads a CSV containing H1 headings and their source language (Screaming Frog
format by default, but columns are mappable) and translates each H1 to English
using any OpenAI-compatible chat completions endpoint. Works with a local LLM
(e.g. Ollama) or the OpenAI API.

Features:
- Upload CSV with H1 and Language columns
- Mappable column names (defaults match Screaming Frog exports)
- Any OpenAI-compatible endpoint (local Ollama by default)
- Structured JSON schema responses with retry handling
- Export results with the original columns preserved
"""

import json
import time

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bulk H1 Translator", page_icon="🌍", layout="wide")

# Check for required packages
try:
    from openai import OpenAI
    PACKAGES_AVAILABLE = True
except ImportError:
    PACKAGES_AVAILABLE = False

st.title("Bulk H1 Translator")
st.markdown("*Created by* [![Website](https://img.shields.io/badge/-leefoot.com-2A9D8F?logoColor=white)](https://www.leefoot.com) · [![Hire Me](https://img.shields.io/badge/-Hire%20Me-FF6B6B?logoColor=white)](https://www.leefoot.com/contact) · [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lee-foot/) · [![Bluesky](https://img.shields.io/badge/-Bluesky-0285FF?logoColor=white)](https://bsky.app/profile/leefootseo.bsky.social) · [![More Tools](https://img.shields.io/badge/-More%20Tools-8B5CF6?logoColor=white)](https://leefoot.com/tools) · [![GitHub](https://img.shields.io/badge/-GitHub-6B7280?logoColor=white)](https://github.com/searchsolved/search-solved-public-seo)")

if not PACKAGES_AVAILABLE:
    st.error("""
    Required packages not installed. Run:
    ```
    pip install openai
    ```
    """)
    st.stop()

with st.expander("How to use this tool"):
    st.markdown("""
    **What this tool does:**
    - Translates H1 headings to English in bulk
    - Preserves the original meaning and tone
    - Leaves H1s that are already in English (or empty) unchanged
    - Keeps all of your original CSV columns in the output

    **Data requirements:**
    - CSV with a column containing the H1 text (Screaming Frog exports use `H1-1`)
    - CSV with a column containing the source language (e.g. `Language`)

    **Endpoint options:**
    - **Local LLM (default):** run Ollama (or any OpenAI-compatible server)
      locally and use the default base URL `http://localhost:11434/v1`.
      No real API key is needed; any placeholder value works.
    - **OpenAI:** set the base URL to `https://api.openai.com/v1`, enter your
      API key and choose a model such as `gpt-4o-mini`.

    **Tips:**
    - The endpoint must support structured (JSON schema) responses
    - Test on a small file first to check translation quality
    - Local models vary in quality; larger instruction-tuned models translate best
    """)

# Sidebar settings
st.sidebar.header("API Settings")

base_url = st.sidebar.text_input(
    "Base URL",
    value="http://localhost:11434/v1",
    help="OpenAI-compatible endpoint. Default is Ollama running locally. "
         "Use https://api.openai.com/v1 for OpenAI."
)

model = st.sidebar.text_input(
    "Model name",
    value="local-model",
    help="For Ollama this can be any value (the loaded model is used). "
         "For OpenAI use a model name such as gpt-4o-mini."
)

api_key = st.sidebar.text_input(
    "API Key (optional for local endpoints)",
    type="password",
    help="Required for OpenAI and other hosted APIs. "
         "Leave blank for local endpoints such as Ollama."
)

st.sidebar.markdown("---")
st.sidebar.header("Processing Settings")

max_retries = st.sidebar.slider(
    "Retries per row",
    min_value=1,
    max_value=5,
    value=3,
    help="Number of attempts per H1 before recording an error"
)

# JSON schema response format
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "translation_response",
        "strict": "true",
        "schema": {
            "type": "object",
            "properties": {
                "translated_h1": {"type": "string"}
            },
            "required": ["translated_h1"]
        }
    }
}

SYSTEM_PROMPT = ("You are a translator that always responds with a valid JSON "
                 "object containing only the translated text.")


def create_translation_prompt(h1, language):
    """Build the translation prompt for a single row."""
    prompt = f"""Translate the following text from {language} to English.
    Maintain the original meaning and tone as closely as possible.
    If the text is already in English or empty, return it unchanged.

    H1 to translate: '{h1}'"""
    return prompt


def translate_h1(client, model_name, prompt, retries=3):
    """Call the API for a single prompt, retrying on failure."""
    last_error = "Error: Translation failed"
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=500,
                response_format=RESPONSE_FORMAT
            )
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            last_error = "Error: Invalid JSON response"
        except Exception:
            last_error = "Error: Translation failed"
        if attempt < retries - 1:
            time.sleep(1)
    return {"translated_h1": last_error}


def default_index(columns, preferred):
    """Return the index of a preferred column name if present, else 0."""
    return columns.index(preferred) if preferred in columns else 0


# File upload
st.subheader("Upload CSV")

csv_file = st.file_uploader(
    "Upload CSV with H1 and Language columns",
    type=['csv'],
    help="Screaming Frog exports work out of the box (H1-1 and Language columns)"
)

if csv_file is not None:
    try:
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except Exception:
            csv_file.seek(0)
            df = pd.read_csv(csv_file, encoding='latin-1')

        st.success(f"Loaded {len(df):,} rows")

        columns = df.columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            h1_col = st.selectbox(
                "Select H1 column",
                columns,
                index=default_index(columns, 'H1-1')
            )
        with col2:
            language_col = st.selectbox(
                "Select Language column",
                columns,
                index=default_index(columns, 'Language')
            )

        with st.expander("Preview data"):
            st.dataframe(df.head(20))

        if st.button("Translate H1s", type="primary"):
            # Initialise the client; local endpoints accept any key value
            try:
                client = OpenAI(base_url=base_url, api_key=api_key or "lm-studio")
            except Exception as e:
                st.error(f"Error initialising API client: {str(e)}")
                st.stop()

            df_work = df.copy()

            progress_bar = st.progress(0)
            status_text = st.empty()

            translations = []
            total_rows = len(df_work)

            for i, (_, row) in enumerate(df_work.iterrows()):
                status_text.text(f"Translating row {i + 1}/{total_rows}...")
                progress_bar.progress((i + 1) / total_rows)

                h1 = row[h1_col] if pd.notna(row[h1_col]) else ''
                language = row[language_col]

                prompt = create_translation_prompt(h1, language)
                result = translate_h1(client, model, prompt, retries=max_retries)
                translations.append(result.get('translated_h1', ''))

            status_text.text("Translation complete!")

            df_work['translated_h1'] = translations

            # Display results
            st.subheader("Translation Results")

            error_count = sum(1 for t in translations if str(t).startswith("Error:"))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows Translated", total_rows - error_count)
            with col2:
                st.metric("Errors", error_count)

            st.dataframe(
                df_work[[h1_col, language_col, 'translated_h1']].head(100),
                use_container_width=True
            )

            # Download
            st.subheader("Download")
            csv_output = df_work.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="Download Results (CSV)",
                data=csv_output,
                file_name="translated_h1s.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("Upload a CSV file to begin")

    st.subheader("Example Input")
    example_input = {
        "Address": [
            "https://www.example.com/es/guia",
            "https://www.example.com/de/anleitung",
            "https://www.example.com/guide"
        ],
        "H1-1": [
            "Guía para principiantes",
            "Anleitung für Anfänger",
            "Beginner's Guide"
        ],
        "Language": ["Spanish", "German", "English"]
    }
    st.dataframe(pd.DataFrame(example_input))

    st.subheader("Example Output")
    example_output = {
        "H1-1": [
            "Guía para principiantes",
            "Anleitung für Anfänger",
            "Beginner's Guide"
        ],
        "Language": ["Spanish", "German", "English"],
        "translated_h1": [
            "Beginner's Guide",
            "Beginner's Guide",
            "Beginner's Guide"
        ]
    }
    st.dataframe(pd.DataFrame(example_output))
