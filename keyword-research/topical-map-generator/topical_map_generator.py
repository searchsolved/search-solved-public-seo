"""
Topical Map Generator - Use AI to Organize Keywords into Hierarchical Maps
Great for content strategy planning.

Author: Lee Foot
Date: January 2025
"""

import streamlit as st
import pandas as pd
import json
from io import BytesIO

st.set_page_config(
    page_title="Topical Map Generator",
    page_icon="🗺️",
    layout="wide"
)

# Check if OpenAI is available
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

st.title("🗺️ Topical Map Generator")
st.markdown("""
Use AI (GPT-4o) to organize keywords into hierarchical topical maps.
Great for content strategy planning and building comprehensive topic coverage.
""")

if not OPENAI_AVAILABLE:
    st.error("""
    **OpenAI library is not installed.** Please install it with:
    ```bash
    pip install openai
    ```
    """)
    st.stop()

# Sidebar configuration
st.sidebar.header("Configuration")

# API Key input (secure)
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Your OpenAI API key. Get one at https://platform.openai.com/api-keys"
)

# Model selection
model_options = {
    'gpt-4o': 'GPT-4o (Recommended)',
    'gpt-4o-mini': 'GPT-4o Mini (Faster/Cheaper)',
    'gpt-4-turbo': 'GPT-4 Turbo',
    'gpt-3.5-turbo': 'GPT-3.5 Turbo (Budget)'
}

selected_model = st.sidebar.selectbox(
    "AI Model",
    options=list(model_options.keys()),
    format_func=lambda x: model_options[x],
    help="GPT-4o provides best results for topical mapping"
)

# Hierarchy depth
hierarchy_depth = st.sidebar.slider(
    "Hierarchy Depth",
    min_value=2,
    max_value=5,
    value=4,
    help="Number of levels in the topical hierarchy"
)

# Level names
st.sidebar.subheader("Level Names")
level_names = []
for i in range(hierarchy_depth):
    default_names = ['Parent Topic', 'Niche Topic 1', 'Niche Topic 2', 'Niche Topic 3', 'Niche Topic 4']
    name = st.sidebar.text_input(
        f"Level {i + 1} Name",
        value=default_names[i] if i < len(default_names) else f"Level {i + 1}",
        key=f"level_{i}"
    )
    level_names.append(name)


def create_topical_map(keywords, api_key, model, depth, levels):
    """Generate a topical map using OpenAI API."""
    client = OpenAI(api_key=api_key)

    # Build level description
    level_desc = "\n".join([f"{i+1}. {levels[i]}: {'The broadest category' if i == 0 else 'More specific sub-categories' if i == 1 else 'Further specific sub-categories' if i < depth - 1 else 'The most specific topics'}" for i in range(depth)])

    prompt_content = f"""Create a detailed topical map from the following database of keywords: {keywords}.
The topical map should organize keywords into a hierarchical structure with {depth} levels:
{level_desc}

Group related keywords together logically. Each keyword should appear in only one place in the hierarchy.
The output should be in JSON format with the following structure:
{{
    "topical_map": [
        {{
            "{levels[0]}": "Topic Name",
            "subtopics": [
                {{
                    "{levels[1]}": "Subtopic Name",
                    {"'subtopics': [...]," if depth > 2 else ""}
                    "keywords": ["keyword1", "keyword2"]
                }}
            ]
        }}
    ]
}}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to organize keywords into a detailed topical map for SEO purposes and output JSON in a specific format. Create logical groupings that would help inform content strategy."
                },
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.7
        )

        content = response.choices[0].message.content
        return json.loads(content), None

    except Exception as e:
        return None, str(e)


def flatten_topical_map(data, levels, parent_path=None):
    """Flatten the nested topical map into rows for a DataFrame."""
    if parent_path is None:
        parent_path = {}

    rows = []

    if isinstance(data, dict):
        if 'topical_map' in data:
            for item in data['topical_map']:
                rows.extend(flatten_topical_map(item, levels, parent_path.copy()))
        else:
            current_path = parent_path.copy()

            # Find the current level
            for level in levels:
                if level in data:
                    current_path[level] = data[level]

            # If there are keywords at this level
            if 'keywords' in data:
                for kw in data['keywords']:
                    row = current_path.copy()
                    row['Keyword'] = kw
                    rows.append(row)

            # Process subtopics
            if 'subtopics' in data:
                for subtopic in data['subtopics']:
                    rows.extend(flatten_topical_map(subtopic, levels, current_path.copy()))

    elif isinstance(data, list):
        for item in data:
            rows.extend(flatten_topical_map(item, levels, parent_path.copy()))

    return rows


# Input methods
st.subheader("Input Keywords")
input_method = st.radio(
    "Choose input method:",
    ["Text Area", "CSV Upload"],
    horizontal=True
)

keywords = []

if input_method == "Text Area":
    keyword_input = st.text_area(
        "Enter keywords (one per line or comma-separated)",
        height=200,
        placeholder="keyword 1\nkeyword 2\nkeyword 3\n\nor\n\nkeyword 1, keyword 2, keyword 3"
    )

    if keyword_input:
        # Handle both newline and comma separation
        if ',' in keyword_input and '\n' not in keyword_input.replace(',', ''):
            keywords = [kw.strip() for kw in keyword_input.split(',') if kw.strip()]
        else:
            keywords = [kw.strip() for kw in keyword_input.strip().split('\n') if kw.strip()]

else:
    uploaded_file = st.file_uploader(
        "Upload CSV with keywords",
        type=['csv', 'xlsx', 'txt'],
        help="Upload a file containing keywords"
    )

    if uploaded_file:
        if uploaded_file.name.endswith('.txt'):
            content = uploaded_file.read().decode('utf-8')
            keywords = [kw.strip() for kw in content.split('\n') if kw.strip()]
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
            st.dataframe(df.head())
            kw_col = st.selectbox("Select keyword column", options=df.columns.tolist())
            keywords = df[kw_col].dropna().astype(str).tolist()
        else:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            kw_col = st.selectbox("Select keyword column", options=df.columns.tolist())
            keywords = df[kw_col].dropna().astype(str).tolist()

if keywords:
    st.info(f"Found {len(keywords)} keywords")

    # Preview keywords
    with st.expander("Preview Keywords"):
        st.write(keywords[:50])
        if len(keywords) > 50:
            st.write(f"... and {len(keywords) - 50} more")

    # Limit warning
    if len(keywords) > 200:
        st.warning(f"You have {len(keywords)} keywords. For best results, consider limiting to 200 keywords or fewer. Large lists may hit token limits.")

    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to generate the topical map.")
    else:
        if st.button("🗺️ Generate Topical Map", type="primary"):
            with st.spinner("Generating topical map... This may take a minute."):
                result, error = create_topical_map(
                    keywords,
                    api_key,
                    selected_model,
                    hierarchy_depth,
                    level_names
                )

                if error:
                    st.error(f"Error generating topical map: {error}")
                else:
                    st.success("Topical map generated successfully!")

                    # Display raw JSON
                    with st.expander("View Raw JSON"):
                        st.json(result)

                    # Flatten to DataFrame
                    try:
                        rows = flatten_topical_map(result, level_names)

                        if rows:
                            df = pd.DataFrame(rows)

                            # Reorder columns
                            cols = level_names + ['Keyword']
                            cols = [c for c in cols if c in df.columns]
                            df = df[cols]

                            st.subheader("Topical Map Table")
                            st.dataframe(df, use_container_width=True, hide_index=True)

                            # Summary stats
                            st.subheader("Summary")
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Total Keywords Mapped", len(df))
                            with col2:
                                if level_names[0] in df.columns:
                                    st.metric(f"Unique {level_names[0]}s", df[level_names[0]].nunique())
                            with col3:
                                if len(level_names) > 1 and level_names[1] in df.columns:
                                    st.metric(f"Unique {level_names[1]}s", df[level_names[1]].nunique())

                            # Downloads
                            st.subheader("Download Results")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                # CSV download
                                csv_buffer = BytesIO()
                                df.to_csv(csv_buffer, index=False)
                                csv_buffer.seek(0)

                                st.download_button(
                                    label="📥 Download CSV",
                                    data=csv_buffer,
                                    file_name="topical_map.csv",
                                    mime="text/csv"
                                )

                            with col2:
                                # Excel download
                                excel_buffer = BytesIO()
                                df.to_excel(excel_buffer, index=False, engine='openpyxl')
                                excel_buffer.seek(0)

                                st.download_button(
                                    label="📥 Download Excel",
                                    data=excel_buffer,
                                    file_name="topical_map.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )

                            with col3:
                                # JSON download
                                json_str = json.dumps(result, indent=2)

                                st.download_button(
                                    label="📥 Download JSON",
                                    data=json_str,
                                    file_name="topical_map.json",
                                    mime="application/json"
                                )
                        else:
                            st.warning("Could not parse the topical map into a table. Please download the JSON.")

                    except Exception as e:
                        st.warning(f"Could not flatten results to table: {e}")
                        st.info("You can still download the raw JSON result.")

else:
    st.info("👆 Enter keywords above to get started.")

    st.markdown("""
    ### How it Works
    1. Enter your keywords (one per line or upload a CSV)
    2. Configure the hierarchy depth and level names
    3. Enter your OpenAI API key
    4. Click "Generate Topical Map"
    5. The AI will organize keywords into logical topic clusters
    6. Download results as CSV, Excel, or JSON

    ### Best Practices
    - **Keep keyword lists focused** - Related keywords work better than random mixes
    - **Limit to ~200 keywords** - Larger lists may hit token limits
    - **Use descriptive level names** - Helps the AI understand your desired structure
    - **Review and refine** - AI suggestions are a starting point

    ### Use Cases
    - Content strategy planning
    - Building topic clusters and pillar pages
    - Organizing keyword research
    - Identifying content gaps
    - Planning site architecture

    ### API Key
    You need an OpenAI API key to use this tool. Get one at:
    https://platform.openai.com/api-keys
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("Built by [Lee Foot](https://leefoot.co.uk) · [Bluesky](https://bsky.app/profile/leefootseo.bsky.social) · [LinkedIn](https://www.linkedin.com/in/lee-foot/)")
